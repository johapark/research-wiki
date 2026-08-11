"""Import a reference-manager library (Zotero / Paperpile / Mendeley / ReadCube).

✅ Use when: the user has an existing corpus in a reference manager and wants it
   in the wiki. Their export already carries curated DOI, title, authors and
   year — the fields `agent ingest` otherwise rediscovers through its most
   failure-prone stretch.
❌ Don't use: to import markdown pages from an older wiki (that's `migrate`), or
   to fix metadata on a page that already exists (that's `prompts/recovery.md`).

    researchwiki import preflight <export>              # parse only
    researchwiki import inspect   <export> [pdf-root]   # triage

Both phases cost **zero tokens** and write nothing outside their run directory.
`inspect` is where the value is: it pairs records to PDFs, runs every gate, and
writes a manifest plus a report you read *before* deciding what to import.

**This module is named for a Python keyword, deliberately.** `__main__` derives
each CLI name from its module name by a pure transformation
(`_discover_tasks`: underscores → dashes), and both discovery
(`pkgutil.iter_modules`) and dispatch (`importlib.import_module`) work on
strings, so a keyword filename is fine for them. The cost is that this is the
one task module no `import` statement can reach — tests and any other caller
must use `importlib.import_module("researchwiki.tasks.import")`. That was
preferred over an alias table in `__main__`, which would have bought a
conventional filename by making CLI names no longer derivable from module
names. Don't "fix" the filename; the CLI name would silently become
`import-library` again.

`<pdf-root>` is optional. With no PDFs at all the report still lists every
record that clears every gate except having a file, with its DOI — which on a
cloud-hosted library is the most useful thing this command produces.

Exit codes: 0 success (including "nothing is importable" — a triage result is
the deliverable), 1 bad input, 2 environment.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from ..paths import wiki_dir


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _load_export(path: Path):
    """Parse, or print the reason and return None. Callers map None → exit 1."""
    from ..refimport.parse import parse_export

    if not path.is_file():
        print(f"researchwiki import: no such file: {path}", file=sys.stderr)
        return None
    try:
        return parse_export(path)
    except ValueError as e:
        print(f"researchwiki import: {e}", file=sys.stderr)
        return None


def _embedding_status() -> tuple[bool, str]:
    """`(ok, detail)` for the local embedding model.

    Probes an actual **encode**, not `is_available()`. `is_available()` only
    proves the model constructs, and `prompts/migration-backfill.md` documents
    the trap that makes the distinction matter: a torch/NumPy ABI mismatch
    loads fine, prints OK, and dies on the first real call.

    This is reported, never fatal, in these two phases. Parsing a `.ris` file
    has no dependency on a 133 MB bi-encoder, and refusing to read an export
    because a torch wheel is wrong would block the phase a user runs first —
    before they own any PDFs at all. `apply` is where the dependency binds.
    """
    try:
        from ..index import embeddings
        shape = embeddings.embed_texts(["probe"]).shape
        return True, f"{embeddings.DEFAULT_MODEL} ({shape[-1]}d)"
    except Exception as e:  # noqa: BLE001 - any failure means "unusable"
        return False, f"{type(e).__name__}: {e}"


# ---------- preflight ----------

def _run_preflight(args: argparse.Namespace) -> int:
    loaded = _load_export(Path(args.export))
    if loaded is None:
        return 1
    fmt, items = loaded

    print(f"# import preflight — {args.export}\n")
    print(f"  format             {fmt}")
    print(f"  records            {len(items)}")
    if not items:
        print("\nNo records parsed. Nothing to import.")
        return 1

    n = len(items)
    def pct(k):
        return f"{k:>5}  ({100 * k / n:.0f}%)"

    print(f"  with a usable DOI  {pct(sum(1 for i in items if i.has_usable_doi))}")
    print(f"  with title+year    {pct(sum(1 for i in items if i.title and i.year))}")
    print(f"  with an author     {pct(sum(1 for i in items if i.authors))}")
    declared = sum(1 for i in items if i.declared_files)
    print(f"  naming a file      {pct(declared)}")

    types: dict[str, int] = {}
    for i in items:
        types[i.item_type] = types.get(i.item_type, 0) + 1
    print("\n  item types         " + ", ".join(
        f"{k} {v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])))

    if not declared:
        print("\n  NOTE This export names no attachment paths, which is normal for\n"
              "       ReadCube and for every CSL-JSON exporter. PDFs will be matched\n"
              "       by the DOI printed in them, then by title. Point `inspect` at\n"
              "       whatever directory holds them.")

    ok, detail = _embedding_status()
    print(f"\n  embedding model    {'OK   ' + detail if ok else 'WARN unusable — ' + detail}")
    if not ok:
        print("       Not needed to inspect an export, but `agent ingest` grades\n"
              "       against it and degrades to BM25-only without it — which would\n"
              "       mean re-grading the whole import later. Fix before importing.")

    print(f"\nNext: `researchwiki import inspect {args.export!r} <pdf-root>`")
    return 0


# ---------- inspect ----------

def _run_inspect(args: argparse.Namespace) -> int:
    from ..refimport.manifest import new_run_dir
    from ..refimport.pair import Pairing, build_pdf_index, pair_items
    from ..refimport.triage import (
        assess_all,
        missing_pdf_fetch_list,
        summarize,
    )
    from ..wiki import find_stem_collision, read_wiki_dois

    export = Path(args.export)
    loaded = _load_export(export)
    if loaded is None:
        return 1
    fmt, items = loaded
    if not items:
        print(f"No records parsed from {export}. Nothing to inspect.", file=sys.stderr)
        return 1
    if args.limit:
        items = items[: args.limit]

    if args.category and not (wiki_dir() / args.category).is_dir():
        print(f"researchwiki import inspect: category '{args.category}' has "
              f"no wiki/{args.category}/ directory — create it first (categories "
              f"are explicit; see CLAUDE.md → Categories).", file=sys.stderr)
        return 1

    pdf_root = Path(args.pdf_root) if args.pdf_root else None
    if pdf_root is not None and not pdf_root.is_dir():
        # Exit 1, not 2: a mistyped argument, not a broken environment.
        print(f"researchwiki import inspect: no such directory: {pdf_root}",
              file=sys.stderr)
        return 1

    if pdf_root is not None:
        print(f"Indexing PDFs under {pdf_root} …", file=sys.stderr)
        facts = build_pdf_index(pdf_root)
        pairings, unclaimed = pair_items(items, facts, pdf_root=pdf_root,
                                         export_dir=export.parent)
    else:
        facts, unclaimed = [], []
        pairings = [Pairing(item=i) for i in items]
    facts_by_path = {f.path: f for f in facts}

    assessments = assess_all(
        items, pairings, facts_by_path,
        known_dois=read_wiki_dois(),
        stem_exists=lambda s: find_stem_collision(s) is not None,
    )
    summary = summarize(assessments)
    fetch = missing_pdf_fetch_list(assessments)
    records = [a.as_dict() for a in assessments]

    run = new_run_dir(_stamp(), base=Path(args.run_dir) if args.run_dir else None)
    run.write_manifest(records, export_path=export, export_format=fmt,
                       pdf_root=pdf_root, category=args.category,
                       created_at=_iso(), summary=summary,
                       unclaimed_pdfs=[str(f.path) for f in unclaimed])
    run.report_path.write_text(
        _render_report(assessments, summary, fetch, unclaimed, export, fmt, pdf_root),
        encoding="utf-8")

    if args.as_json:
        print(json.dumps({
            "run_dir": str(run.root),
            "export_format": fmt,
            "summary": summary,
            "missing_pdf_fetch_list": fetch,
            "unclaimed_pdfs": [str(f.path) for f in unclaimed],
            "items": records,
        }, indent=2))
        return 0

    _print_report(assessments, summary, fetch, unclaimed, run)
    return 0


def _by_verdict(assessments) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in assessments:
        out.setdefault(a.verdict, []).append(a)
    return out


def _print_report(assessments, summary, fetch, unclaimed, run) -> None:
    groups = _by_verdict(assessments)
    print(f"\n# import inspect — {summary['total']} record(s)\n")
    for verdict in ("ready", "review", "skip"):
        group = groups.get(verdict) or []
        print(f"  {verdict:<8} {len(group)}")
    if summary["reasons"]:
        print("\n  reasons")
        for reason, n in summary["reasons"].items():
            print(f"    {reason:<24} {n}")

    for a in (groups.get("review") or [])[:20]:
        title = (a.item.title or "(untitled)")[:64]
        print(f"\n  review  {title}")
        print(f"          {', '.join(a.reasons)}")
        if a.pairing.candidates:
            best = a.pairing.candidates[0]
            print(f"          best PDF candidate: {Path(best[0]).name} ({best[1]})")
    if len(groups.get("review") or []) > 20:
        print(f"\n  … and {len(groups['review']) - 20} more under review "
              f"(full list in {run.report_path.name})")

    if fetch:
        print(f"\n  {len(fetch)} record(s) are importable except that no PDF was "
              f"found for them.\n  Their DOIs are listed in {run.report_path}.")
    if unclaimed:
        print(f"\n  {len(unclaimed)} PDF(s) matched no record.")

    print(f"\n  manifest: {run.manifest_path}")
    print(f"  report:   {run.report_path}")
    n_ready = len(groups.get("ready") or [])
    if n_ready:
        print(f"\nNext: `researchwiki import apply --run {run.root} "
              f"--limit 30 --dry-run`   ({n_ready} ready)")
    else:
        print("\nNothing is ready to import yet — see the reasons above.")


def _render_report(assessments, summary, fetch, unclaimed, export, fmt, pdf_root) -> str:
    """The durable version, including the full fetch list.

    The fetch list is the reason this file exists rather than only terminal
    output: on a cloud-hosted library it is a work item hundreds of lines long,
    and it should be pipeable rather than scrollback.
    """
    groups = _by_verdict(assessments)
    L: list[str] = [
        f"# Import inspection — {export.name}", "",
        f"- Format: `{fmt}`",
        f"- Records: {summary['total']}",
        f"- PDF root: {pdf_root or '(none given)'}",
        "",
        "| Verdict | Count |", "|---|---|",
    ]
    for verdict in ("ready", "review", "skip"):
        L.append(f"| {verdict} | {len(groups.get(verdict) or [])} |")
    L += ["", "| Reason | Count |", "|---|---|"]
    for reason, n in summary["reasons"].items():
        L.append(f"| `{reason}` | {n} |")

    if groups.get("review"):
        L += ["", "## Needs review", ""]
        for a in groups["review"]:
            L.append(f"- **{a.item.title or '(untitled)'}** — {', '.join(a.reasons)}")
            if a.item.doi:
                L.append(f"  - DOI: `{a.item.doi}`")
            if a.pairing.candidates:
                L.append("  - candidate PDFs: " + ", ".join(
                    f"`{Path(p).name}` ({s})" for p, s in a.pairing.candidates[:3]))

    if fetch:
        L += ["", "## Missing PDFs", "",
              f"{len(fetch)} record(s) clear every gate except having a file. "
              "Fetch these and re-run `inspect`:", "", "```"]
        L += [f["doi"] for f in fetch]
        L += ["```", "", "<details><summary>with titles</summary>", ""]
        L += [f"- `{f['doi']}` — {f['title']}" for f in fetch]
        L += ["", "</details>"]

    if unclaimed:
        L += ["", "## PDFs matching no record", ""]
        L += [f"- `{p}`" for p in [str(f.path) for f in unclaimed][:200]]

    if groups.get("ready"):
        L += ["", "## Ready", ""]
        for a in groups["ready"]:
            L.append(f"- `{a.derived_stem or '(stem pending)'}` — {a.item.title}")
    return "\n".join(L) + "\n"


# ---------- entry point ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="researchwiki import",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", required=True, metavar="PHASE")

    pf = sub.add_parser("preflight", help="Parse the export and describe it. No PDFs read.")
    pf.add_argument("export", help="BibTeX / RIS / CSL-JSON export file.")

    ins = sub.add_parser("inspect", help="Pair records to PDFs and triage them.")
    ins.add_argument("export", help="BibTeX / RIS / CSL-JSON export file.")
    ins.add_argument("pdf_root", nargs="?", default=None,
                     help="Directory holding the PDFs. Optional — without it every "
                          "record is triaged as far as its metadata allows.")
    ins.add_argument("--category", default=None,
                     help="Target wiki/<category>/ for the whole run (must exist). "
                          "Omit to let the classifier vote per paper.")
    ins.add_argument("--limit", type=int, default=0, help="Assess at most N records.")
    ins.add_argument("--run-dir", default=None, help="Where to write the run directory.")
    ins.add_argument("--json", dest="as_json", action="store_true")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        return _run_preflight(args)
    if args.phase == "inspect":
        return _run_inspect(args)
    return 1
