"""Import a reference-manager library (Zotero / Paperpile / Mendeley / ReadCube).

✅ Use when: the user has an existing corpus in a reference manager and wants it
   in the wiki. Their export already carries curated DOI, title, authors and
   year — the fields `agent ingest` otherwise rediscovers through its most
   failure-prone stretch.
❌ Don't use: to import markdown pages from an older wiki (that's `migrate`), or
   to fix metadata on a page that already exists (that's `prompts/recovery.md`).

    researchwiki import preflight <export>              # parse only
    researchwiki import inspect   <export> [pdf-root]   # triage
    researchwiki import apply --run <dir> --limit N     # copy + ingest a wave
    researchwiki import verify --run <dir>              # did it land?

`preflight` writes nothing. `inspect` writes only its run directory; `apply` is
the phase that spends and that writes pages.
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

**No `--category`.** There was one; it validated its argument, wrote it to the
manifest, and was then read by nothing — `agent ingest`, which `apply`
dispatches, has no such flag. (The *digest* path `researchwiki ingest` does,
which is what made the promise look plausible.) Category is chosen per paper by
promote's neighbour-vote classifier, which is the better answer for an imported
library anyway: a reference manager's collections rarely map onto wiki
categories, and one global value would flatten a mixed corpus into a single bin.

Exit codes: 0 success (including "nothing is importable" — a triage result is
the deliverable), 1 bad input, 2 environment.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shlex
import sys
from pathlib import Path

from ..errors import EnvironmentFailure
from ..paths import canonical


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

    `preflight` and `inspect` deliberately never call this: parsing and pairing
    have no dependency on a 133 MB bi-encoder. A dry-run reports the result as
    advice; a real apply checks it before copying and fails if it is unusable.
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
    print(
        "\n  item types         "
        + ", ".join(f"{k} {v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1]))
    )

    if not declared:
        print(
            "\n  NOTE This export names no attachment paths, which is normal for\n"
            "       ReadCube and for every CSL-JSON exporter. PDFs will be matched\n"
            "       by the DOI printed in them, then by title. Point `inspect` at\n"
            "       whatever directory holds them."
        )

    print(f"\nNext: `researchwiki import inspect {args.export!r} <pdf-root>`")
    return 0


# ---------- inspect ----------


def _run_inspect(args: argparse.Namespace) -> int:
    from ..refimport.manifest import new_run_dir
    from ..refimport.pair import Pairing, build_pdf_index, pair_items
    from ..refimport.triage import (
        assess_all,
        find_superseded,
        missing_pdf_fetch_list,
        reference_doc_candidates,
        summarize,
    )
    from ..wiki import read_wiki_dois, read_wiki_stems

    # Canonical from here down, so the `pdf_root`, `primary_pdf` and
    # `unclaimed_pdfs` spellings the manifest records all agree with each other
    # and with what `pair` indexed. Also gives `export.parent` a real directory
    # for a bare relative argument, where `Path("lib.ris").parent` is `.`.
    export = canonical(args.export)
    loaded = _load_export(export)
    if loaded is None:
        return 1
    fmt, items = loaded
    if not items:
        print(f"No records parsed from {export}. Nothing to inspect.", file=sys.stderr)
        return 1
    pdf_root = Path(args.pdf_root) if args.pdf_root else None
    if pdf_root is not None and not pdf_root.is_dir():
        # Exit 1, not 2: a mistyped argument, not a broken environment. Reported
        # with the spelling the user typed, which is what they can recognize.
        print(
            f"researchwiki import inspect: no such directory: {pdf_root}",
            file=sys.stderr,
        )
        return 1
    if pdf_root is not None:
        pdf_root = canonical(pdf_root)

    # Dedupe *before* pairing. A record superseded by its own published version
    # is not going to be imported, so letting it compete for PDFs costs twice:
    # it can win a file the survivor should have had, and — because the pair
    # shares a title verbatim — it scores an exact tie against the survivor's
    # own PDF, which the distinctiveness gate then reads as a genuine ambiguity.
    # On the real library that alone accounted for most `ambiguous-pairing`
    # reviews, every one of them spurious.
    superseded = find_superseded(items)
    active = [i for i in items if id(i) not in superseded]

    if pdf_root is not None:
        print(f"Indexing PDFs under {pdf_root} …", file=sys.stderr)
        facts = build_pdf_index(pdf_root)
        pairings, unclaimed = pair_items(
            active, facts, pdf_root=pdf_root, export_dir=export.parent
        )
    else:
        facts, unclaimed = [], []
        pairings = [Pairing(item=i) for i in active]
    facts_by_path = {f.path: f for f in facts}

    assessments = assess_all(
        items,
        pairings,
        facts_by_path,
        known_dois=read_wiki_dois(),
        # One walk, not one per record — see `read_wiki_stems`.
        stem_exists=read_wiki_stems().__contains__,
        superseded=superseded,
    )
    # `--limit` is a report/apply sample, not an identity boundary. Supersede
    # detection, duplicate selection and title-rival scoring must see the full
    # export or a sample can bless a pairing the complete run would reject.
    if args.limit:
        assessments = assessments[: args.limit]
    summary = summarize(assessments)
    fetch = missing_pdf_fetch_list(assessments)
    ref_docs = reference_doc_candidates(assessments)
    records = [a.as_dict() for a in assessments]

    run = new_run_dir(_stamp(), base=Path(args.run_dir) if args.run_dir else None)
    run.write_manifest(
        records,
        export_path=export,
        export_format=fmt,
        pdf_root=pdf_root,
        category=None,
        created_at=_iso(),
        summary=summary,
        unclaimed_pdfs=[str(f.path) for f in unclaimed],
    )
    run.report_path.write_text(
        _render_report(
            assessments, summary, fetch, ref_docs, unclaimed, export, fmt, pdf_root
        ),
        encoding="utf-8",
    )

    if args.as_json:
        print(
            json.dumps(
                {
                    "run_dir": str(run.root),
                    "export_format": fmt,
                    "summary": summary,
                    "missing_pdf_fetch_list": fetch,
                    "reference_doc_candidates": ref_docs,
                    "unclaimed_pdfs": [str(f.path) for f in unclaimed],
                    "items": records,
                },
                indent=2,
            )
        )
        return 0

    _print_report(assessments, summary, fetch, ref_docs, unclaimed, run)
    return 0


def _by_verdict(assessments) -> dict[str, list]:
    out: dict[str, list] = {}
    for a in assessments:
        out.setdefault(a.verdict, []).append(a)
    return out


def _print_report(assessments, summary, fetch, ref_docs, unclaimed, run) -> None:
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
        print(
            f"\n  … and {len(groups['review']) - 20} more under review "
            f"(full list in {run.report_path.name})"
        )

    if fetch:
        print(
            f"\n  {len(fetch)} record(s) are importable except that no PDF was "
            f"found for them.\n  Their DOIs are listed in {run.report_path}."
        )
    if ref_docs:
        print(
            f"\n  {len(ref_docs)} record(s) are reference material (book / "
            f"guidance / thesis),\n  not papers — listed in the report for a "
            f"hand-written wiki/references/ page."
        )
    if unclaimed:
        print(f"\n  {len(unclaimed)} PDF(s) matched no record.")

    print(f"\n  manifest: {run.manifest_path}")
    print(f"  report:   {run.report_path}")
    n_ready = len(groups.get("ready") or [])
    if n_ready:
        print(
            f"\nNext: `researchwiki import apply --run {run.root} "
            f"--limit 30 --dry-run`   ({n_ready} ready)"
        )
    else:
        print("\nNothing is ready to import yet — see the reasons above.")


def _render_report(
    assessments, summary, fetch, ref_docs, unclaimed, export, fmt, pdf_root
) -> str:
    """The durable version, including the full fetch list.

    The fetch list is the reason this file exists rather than only terminal
    output: on a cloud-hosted library it is a work item hundreds of lines long,
    and it should be pipeable rather than scrollback.
    """
    groups = _by_verdict(assessments)
    L: list[str] = [
        f"# Import inspection — {export.name}",
        "",
        f"- Format: `{fmt}`",
        f"- Records: {summary['total']}",
        f"- PDF root: {pdf_root or '(none given)'}",
        "",
        "| Verdict | Count |",
        "|---|---|",
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
                L.append(
                    "  - candidate PDFs: "
                    + ", ".join(
                        f"`{Path(p).name}` ({s})" for p, s in a.pairing.candidates[:3]
                    )
                )

    if fetch:
        L += [
            "",
            "## Missing PDFs",
            "",
            f"{len(fetch)} record(s) clear every gate except having a file. "
            "Fetch these and re-run `inspect`:",
            "",
            "```",
        ]
        L += [f["doi"] for f in fetch]
        L += ["```", "", "<details><summary>with titles</summary>", ""]
        L += [f"- `{f['doi']}` — {f['title']}" for f in fetch]
        L += ["", "</details>"]

    if ref_docs:
        L += [
            "",
            "## Reference material, not papers",
            "",
            "Typed by the exporter as a book, thesis, report or web page. These are "
            "legitimate `wiki/references/` pages — hand-written, not ingested "
            "(CLAUDE.md → Page Types §3). Listed here because a bare count would "
            "leave you no way to find them.",
            "",
        ]
        for r in ref_docs:
            L.append(
                f"- **{r['title'] or '(untitled)'}** — `{r['item_type']}`"
                + (f", DOI `{r['doi']}`" if r["doi"] else "")
            )
            if r["primary_pdf"]:
                L.append(f"  - PDF: `{r['primary_pdf']}`")

    if unclaimed:
        L += ["", "## PDFs matching no record", ""]
        L += [f"- `{p}`" for p in [str(f.path) for f in unclaimed][:200]]

    if groups.get("ready"):
        L += ["", "## Ready", ""]
        for a in groups["ready"]:
            L.append(f"- `{a.derived_stem or '(stem pending)'}` — {a.item.title}")
    return "\n".join(L) + "\n"


# ---------- apply ----------


def _run_apply(args: argparse.Namespace) -> int:
    from . import _ingest_batch
    from ..refimport.apply import dispatch, plan_wave, stage
    from ..refimport.manifest import open_run_dir

    bad_workers = _ingest_batch.invalid_worker_count(args.workers)
    if bad_workers:
        print(f"researchwiki import apply: {bad_workers}", file=sys.stderr)
        return 1

    try:
        run = open_run_dir(Path(args.run))
        data = run.read_manifest()
    except FileNotFoundError as e:
        print(f"researchwiki import apply: {e}", file=sys.stderr)
        return 1

    plan = plan_wave(data["items"], limit=args.limit)
    if plan.already_present:
        print(f"{len(plan.already_present)} record(s) landed since inspect — skipping:")
        for rec in plan.already_present[:10]:
            print(f"    {rec['landed_as']}")
    if plan.already_staged:
        # Not inert, which is why it names both exits rather than just reporting:
        # a PDF left in `inbox/` is the ingest backlog, and the next
        # `agent ingest inbox/*.pdf` picks it up as a paper on its own.
        print(
            f"{len(plan.already_staged)} record(s) already copied into inbox/ by an "
            f"earlier wave — not re-copied:"
        )
        for rec in plan.already_staged[:10]:
            command = [
                "researchwiki",
                "agent",
                "ingest",
                rec["staged_as"],
                *(rec.get("ingest_args") or []),
            ]
            print(f"    {shlex.join(command)}")
        print(
            "  Each is a real backlog entry: run the printed command to preserve "
            "its import metadata, or delete it."
        )
    if plan.missing_pdf:
        print(f"{len(plan.missing_pdf)} record(s) lost their PDF since inspect:")
        for rec in plan.missing_pdf[:10]:
            print(f"    {rec.get('primary_pdf')}")
    if not plan.staged:
        print("Nothing left to import from this manifest.", file=sys.stderr)
        return 1

    # A real run must establish every environment precondition before copying.
    # Otherwise an embedding failure leaves PDFs in inbox/, and the next apply
    # treats them as already staged instead of dispatching their frozen metadata.
    if not args.dry_run:
        ok, detail = _embedding_status()
        if not ok:
            raise EnvironmentFailure(
                f"the local embedding model is unusable ({detail}). `agent ingest` "
                f"grades against it, and without it every claim in this wave would "
                f"grade BM25-only and need re-grading later. See the install notes "
                f"in prompts/migration-backfill.md."
            )

    # Resolve the provider-aware default before the preview: API-backed waves use
    # four workers, while chat-relay stays sequential unless the user supplied -w.
    from . import _ingest_batch

    workers, relay_watch = _ingest_batch.resolve_batch_workers(
        args.workers, subcommand=["agent", "ingest"])

    staged = stage(plan.staged, dry_run=args.dry_run)
    print(
        f"\n# import apply — {len(staged)} paper(s)"
        f"{' (dry run — nothing copied)' if args.dry_run else ''}\n"
    )
    for path, argv in staged:
        # Not truncated. The whole point of `--dry-run` is reading the argv, and
        # a 160-char cut lands mid-`--title` on realistic titles — hiding
        # `--year` and every `--supplementary`, which `build_ingest_args`
        # appends last.
        print(f"  {path.name}")
        print(f"      agent ingest {' '.join(argv)}")

    if args.dry_run:
        ok, detail = _embedding_status()
        if not ok:
            print(
                f"\n  WARNING embedding model unusable — {detail}\n"
                f"          A real run would refuse; fix it before applying."
            )
        print(
            f"\nThen: `researchwiki import apply --run {args.run}"
            f"{f' --limit {args.limit}' if args.limit else ''}`"
        )
        return 0

    code = dispatch(
        staged,
        workers=workers,
        relay_watch=relay_watch,
        workers_explicit=args.workers is not None,
    )
    if code:
        print(
            "\nImport wave failed. Use the batch runner's resume command above; "
            "success follow-ups were not run.",
            file=sys.stderr,
        )
        return code

    print("\nNext — free (no tokens):")
    print("  researchwiki db rebuild && researchwiki reindex")
    print("  researchwiki grade regression --missing-only --no-salience")
    print(f"  researchwiki import verify --run {args.run}")
    print("\nThen, once you're happy with these pages:")
    print(f"  researchwiki import apply --run {args.run} --limit {args.limit or 30}")
    return code


# ---------- verify ----------

#: `lint --json` keys worth reading after a bulk import, and what each means
#: here. Deliberately a subset: `lint` reports ~30 checks and most are about
#: the wiki as a whole, while these are the ones an import can break.
_LINT_KEYS = (
    (
        "invalid_frontmatter",
        0,
        "unparseable YAML — the page is invisible to db rebuild and every query",
    ),
    ("zero_claim_papers", 0, "paper pages producing no citable claims"),
    ("missing_hook", 0, "no catalog gloss"),
    ("missing_doi", None, "no DOI recorded"),
    ("missing_keywords", None, "fewer than 5 keywords"),
    ("ungraded_papers", 0, "claims not yet scored against their PDF"),
    ("broken_wikilinks", 0, "links to pages that didn't come along"),
    ("page_type_mismatches", 0, "type: disagrees with the directory"),
    ("category_yaml_drift", 0, "YAML category ≠ parent dir"),
    (
        "duplicate_claim_sets",
        None,
        "advisory — near-duplicate claim sets; the "
        "signature of a commentary typed as a paper",
    ),
)


def _run_verify(args: argparse.Namespace) -> int:
    """Did the import land, and what does it still need?

    Reads the manifest rather than the batch checkpoint, because the question
    is about *records* ("did this paper make it into the wiki") and the
    checkpoint only knows about files. A record can complete its ingest and
    still not be in `wiki/` — that's the sandbox path, and it is exactly what
    this phase exists to surface.
    """
    import json as _json

    from ..refimport.manifest import open_run_dir
    from ..refimport.parse import clean_doi
    from ..wiki import read_page, read_wiki_dois, read_wiki_stems

    try:
        run = open_run_dir(Path(args.run))
        data = run.read_manifest()
    except FileNotFoundError as e:
        print(f"researchwiki import verify: {e}", file=sys.stderr)
        return 1

    known_dois = {k.lower(): v for k, v in read_wiki_dois().items()}
    known_stems = read_wiki_stems()
    sandbox = Path(".agent-output")
    sandboxed_stems: set[str] = set()
    sandboxed_dois: dict[str, str] = {}
    if sandbox.is_dir():
        for md in sandbox.glob("*.md"):
            sandboxed_stems.add(md.stem)
            page = read_page(md)
            doi = clean_doi(page.str_field("doi")) if page is not None else None
            if doi:
                sandboxed_dois[doi] = md.stem

    landed, in_sandbox, missing = [], [], []
    for rec in data["items"]:
        if rec.get("verdict") != "ready":
            continue
        doi = (rec.get("doi") or "").lower()
        stem = rec.get("derived_stem")
        if doi and doi in known_dois:
            landed.append({**rec, "landed_as": known_dois[doi]})
        elif stem and stem in known_stems:
            landed.append({**rec, "landed_as": stem})
        elif doi and doi in sandboxed_dois:
            in_sandbox.append({**rec, "sandboxed_as": sandboxed_dois[doi]})
        elif stem and stem in sandboxed_stems:
            in_sandbox.append({**rec, "sandboxed_as": stem})
        else:
            missing.append(rec)

    lint = _lint_snapshot()

    if args.as_json:
        print(
            _json.dumps(
                {
                    "run_dir": str(run.root),
                    "ready": len(landed) + len(in_sandbox) + len(missing),
                    "landed": [r["landed_as"] for r in landed],
                    "sandboxed": [r["sandboxed_as"] for r in in_sandbox],
                    "not_imported": [
                        r.get("derived_stem") or r.get("key") for r in missing
                    ],
                    "lint": lint,
                },
                indent=2,
            )
        )
        return 0

    total = len(landed) + len(in_sandbox) + len(missing)
    print(f"# import verify — {run.root.name}\n")
    print(f"  ready in manifest    {total}")
    print(f"  landed in wiki/      {len(landed)}")
    print(f"  in .agent-output/    {len(in_sandbox)}")
    print(f"  not imported yet     {len(missing)}")

    if in_sandbox:
        print("\n  Sandboxed — the gates held these back; review and promote by hand:")
        for rec in in_sandbox[:15]:
            print(f"    .agent-output/{rec['sandboxed_as']}.md")

    if lint is None:
        print(
            "\n  lint: no snapshot — nothing to lint yet, or lint itself "
            "failed.\n        Run `researchwiki lint --json` directly to see which."
        )
    else:
        print("\n  lint")
        for key, want, meaning in _LINT_KEYS:
            n = lint.get(key)
            n = len(n) if isinstance(n, (list, dict)) else n
            if n is None:
                continue
            flag = "  ← " + meaning if want is not None and n != want else ""
            print(f"    {key:<24} {n}{flag}")

    # A bulk import arrives as N disconnected nodes. Nothing above measures
    # that, and it is the difference between a pile of pages and a wiki.
    print("\n  Wiring the new pages into the graph (each is free or cheap):")
    print("    researchwiki claim-overlap --backlog --dry-run   # reciprocal links")
    print("    researchwiki scout --json                        # citation-graph gaps")
    print("    researchwiki candidates concepts --bridges       # cross-category hubs")
    print("    researchwiki candidates synthesis                # dense clusters")
    return 0


def _lint_snapshot() -> dict | None:
    """`lint --json` as a dict, or None.

    Run rather than suggested: a bulk import is exactly when nobody runs the
    follow-ups by hand, and the keys that matter here are a small subset of the
    ~30 `lint` reports.

    None covers two cases deliberately merged, because the caller's response to
    both is the same — go run `lint` yourself. `lint` prints a human message
    instead of JSON when the wiki has no pages at all, and it can also fail
    outright. Neither may take `verify` down with it: this phase is a report,
    and a report that dies while reporting is worse than one with a gap.
    """
    import io
    import json as _json
    from contextlib import redirect_stdout

    try:
        from . import lint as lint_task

        buf = io.StringIO()
        with redirect_stdout(buf):
            lint_task.main(["--json"])
        return _json.loads(buf.getvalue())
    except Exception:  # noqa: BLE001 — verify must never fail on its own report
        return None


# ---------- entry point ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="researchwiki import",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", required=True, metavar="PHASE")

    pf = sub.add_parser(
        "preflight", help="Parse the export and describe it. No PDFs read."
    )
    pf.add_argument("export", help="BibTeX / RIS / CSL-JSON export file.")

    ins = sub.add_parser("inspect", help="Pair records to PDFs and triage them.")
    ins.add_argument("export", help="BibTeX / RIS / CSL-JSON export file.")
    ins.add_argument(
        "pdf_root",
        nargs="?",
        default=None,
        help="Directory holding the PDFs. Optional — without it every "
        "record is triaged as far as its metadata allows.",
    )
    ins.add_argument("--limit", type=int, default=0, help="Assess at most N records.")
    ins.add_argument(
        "--run-dir", default=None, help="Where to write the run directory."
    )
    ins.add_argument("--json", dest="as_json", action="store_true")

    ap = sub.add_parser("apply", help="Copy a wave into inbox/ and ingest it.")
    ap.add_argument(
        "--run",
        required=True,
        help="Run directory from `inspect`. Required: a bare `apply` "
        "silently choosing among several runs is a footgun.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Import at most N papers this wave. Re-run for the next N.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the argv per paper; copy nothing, spend nothing.",
    )
    ap.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Concurrent ingest subprocesses (default: 4, or 1 when a phase "
             "this command reaches uses chat-relay)",
    )

    ver = sub.add_parser("verify", help="Did the import land, and what's left?")
    ver.add_argument("--run", required=True, help="Run directory from `inspect`.")
    ver.add_argument("--json", dest="as_json", action="store_true")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        return _run_preflight(args)
    if args.phase == "inspect":
        return _run_inspect(args)
    if args.phase == "apply":
        return _run_apply(args)
    if args.phase == "verify":
        return _run_verify(args)
    return 1
