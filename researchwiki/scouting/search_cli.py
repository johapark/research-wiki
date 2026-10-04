"""CLI for `researchwiki scout search` and `scout search fetch`.

Exit codes (search):
  0 — every requested source answered (zero leads is still 0)
  1 — bad arguments, or a source rejected the query (HTTP 400 / arXiv error)
  2 — a requested source was unreachable; the others are still printed first
Exit codes (fetch):
  0 — every key was downloaded or was already present
  1 — some key is not open access, unresolvable, or needs a manual download
  2 — a transport failure stopped the run; what landed before it is reported
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from . import recent as R
from . import search as S
from . import search_fetch as F

PROG = "researchwiki scout search"


# ---------- rendering ----------

def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _lead_lines(row: dict, today: str, abstracts: bool) -> list[str]:
    new = "NEW " if row.get("first_seen") == today else "    "
    date = row.get("publication_date") or f"{row.get('year') or '?'} (undated)"
    fit = f"fit {row['fit']:.2f}" if row.get("fit") is not None else "fit  —  "
    flag = "  **RETRACTED**" if row.get("retracted") else ""
    srcs = "+".join(row.get("sources") or [])
    lines = [f"- {new}{date}  {fit}  `{row['key']}`  {row.get('venue') or '—'}  [{srcs}]{flag}",
             f"    {row.get('title') or '(no title)'}"]
    authors = row.get("authors") or []
    if authors:
        lines.append(f"    {_clip(', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''), 120)}")
    trial = row.get("trial")
    if trial:
        phases = "/".join(trial.get("phases") or []) or "n/a"
        lines.append(f"    {trial.get('overall_status') or '?'} · {phases} · "
                     f"{trial.get('lead_sponsor') or '?'} · n={trial.get('enrollment') or '?'}"
                     + (" · results posted" if trial.get("has_results") else ""))
        pmids = [r["pmid"] for r in trial.get("references") or []]
        if pmids:
            lines.append(f"    linked PMIDs: {', '.join(pmids[:6])}"
                         + (" …" if len(pmids) > 6 else ""))
        docs = [d["label"] for d in trial.get("documents") or [] if d.get("label")]
        if docs:
            lines.append(f"    documents: {', '.join(docs)}")
    elif row.get("fetch_key") and row["fetch_key"] != row["key"]:
        lines.append(f"    fetch as `{row['fetch_key']}`")
    if row.get("nearest"):
        top = row["nearest"][0]
        lines.append(f"    nearest: [[{top['key']}]] ({top['score']:.2f})")
    for t in row.get("triggers") or []:
        lines.append(f"    ↳ on-topic for [[{t['page']}]]; closest trigger "
                     f"(z {t['z']:.1f}): {_clip(t['text'], 160)}")
    if abstracts and row.get("abstract"):
        label = "summary" if row.get("kind") == "trial" else "abstract"
        lines.append(f"    {label} ({srcs}, verbatim): {row['abstract']}")
    return lines


def _source_line(name: str, st: dict) -> str:
    if st["status"] == "ok":
        return f"{name}: {st['returned']}"
    return f"{name}: {st['status'].upper()} ({st['error']})"


def render(snapshot: dict, *, sort: str, abstracts: bool = False) -> str:
    today = snapshot["generated_at"][:10]
    n = snapshot["counts"]
    out = [f"# Literature search: {snapshot['query']}", ""]
    out.append("_" + "; ".join(_source_line(k, v) for k, v in snapshot["sources"].items())
               + f". {n['merged']} distinct after merging; {n['in_wiki']} already in the wiki, "
               f"{n['declined']} declined, {n['outside_window']} outside the window._")
    out.append("")
    matches, papers = snapshot["page_matches"], snapshot["leads"]
    if matches:
        out.append(f"## Near a page's open questions ({len(matches)})")
        out.extend(line for r in matches for line in _lead_lines(r, today, abstracts))
        out.append("")
    if sort == "source":
        for src in S.SOURCES[:-1]:
            rows = [r for r in papers if r["sources"][0] == src]
            if rows:
                out.append(f"## {src} ({len(rows)})")
                out.extend(line for r in rows for line in _lead_lines(r, today, abstracts))
                out.append("")
    else:
        key = (lambda r: -(r["fit"] if r.get("fit") is not None else -1.0)) if sort == "fit" \
            else (lambda r: r.get("publication_date") or "")
        rows = sorted(papers, key=key, reverse=(sort == "date"))
        out.append(f"## Papers ({len(rows)})")
        out.extend(line for r in rows for line in _lead_lines(r, today, abstracts))
        out.append("")
    if snapshot["trials"]:
        out.append(f"## ClinicalTrials.gov ({len(snapshot['trials'])})")
        out.extend(line for r in snapshot["trials"] for line in _lead_lines(r, today, abstracts))
        out.append("")
    if not (matches or papers or snapshot["trials"]):
        out.append("_(no new leads)_")
        out.append("")
    out.append("_Leads only: nothing here is evidence until its PDF is ingested. "
               "Download open-access PDFs into `inbox/` with "
               "`researchwiki scout search fetch <key> ...`; reject one for good with "
               "`researchwiki scout search --decline <key> --reason \"…\"`._")
    return "\n".join(out)


def render_fetch(report: dict) -> str:
    verb = "would fetch" if report["dry_run"] else "fetched"
    out = []
    for e in report["fetched"]:
        lic = f" ({e['license']})" if e.get("license") else ""
        caveat = (" — Europe PMC usually refuses scripted downloads; expect a manual one"
                  if report["dry_run"] and e.get("best_effort") else "")
        out.append(f"- {verb} `{e['key']}` → {e['path']}{lic}{caveat}")
    for e in report["already_present"]:
        out.append(f"- already in inbox: `{e['key']}` → {e['path']}")
    for e in report["skipped"]:
        out.append(f"- skipped `{e['key']}`: {e['reason']}")
    for e in report["manual"]:
        out.append(f"- download manually `{e['key']}` ({e['reason']}): {e['url']}")
    if report["stopped_on"]:
        out.append(f"- STOPPED at `{report['stopped_on']}`: {report['error']}")
    ready = report["fetched"] + report["already_present"]
    if ready and not report["dry_run"]:
        out.append("")
        out.append("Next — ingest each with its DOI override (batch mode refuses `--doi`), "
                   "or re-run with `--ingest` for one checkpointed batch:")
        out.extend(f"  {e['ingest_command']}" for e in ready)
    return "\n".join(out)


# ---------- fetch ----------

def _ingest(report: dict) -> int:
    """Hand everything fetched to one crash-safe batch, with per-file `--doi`."""
    from ..paths import wiki_root
    from ..tasks import _ingest_batch

    ready = report["fetched"] + report["already_present"]
    if not ready:
        return 0
    per_input: dict[str, list[str]] = {}
    paths = []
    for e in ready:
        path = (wiki_root() / e["path"]).resolve()
        paths.append(str(path))
        per_input[str(path)] = ["--doi", e["ingest_doi"]] if e.get("ingest_doi") else []
    return _ingest_batch.new_batch(
        paths, ["agent", "ingest"], [], _ingest_batch.resolve_batch_workers(None),
        per_input_args=per_input, workers_explicit=False,
    )


def fetch_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=f"{PROG} fetch",
        description="Download open-access PDFs for scout-search leads into inbox/. "
                    "Accepts DOIs, arxiv:<id>, pmid:<n>, pmcid:PMC<n>.",
    )
    ap.add_argument("keys", nargs="+", metavar="KEY")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve each key and print where it would come from; download nothing.")
    ap.add_argument("--ingest", action="store_true",
                    help="Afterwards, run one checkpointed `agent ingest` batch over the "
                         "fetched PDFs with per-file --doi overrides (spends model calls).")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit {fetched, already_present, skipped, manual, stopped_on, "
                         "error, dry_run}.")
    args = ap.parse_args(argv)
    if args.ingest and args.dry_run:
        print(f"{PROG} fetch: --ingest and --dry-run are exclusive", file=sys.stderr)
        return 1
    report = F.fetch(args.keys, dry_run=args.dry_run)
    print(json.dumps(report, indent=2) if args.as_json else render_fetch(report))
    if report["stopped_on"]:
        # Re-raise through the funnel so the code is 2 by the one documented route.
        from ..providers._http import StructuredProviderUnavailable
        raise StructuredProviderUnavailable(report["error"] or "download stopped")
    if args.ingest:
        code = _ingest(report)
        if code:
            return code
    return 1 if (report["skipped"] or report["manual"]) else 0


# ---------- search ----------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Keyword search across PubMed, arXiv, bioRxiv/medRxiv (via Europe PMC) "
                    "and ClinicalTrials.gov. Leads are ranked against the wiki and are "
                    "discovery-only. `scout search fetch KEY...` downloads open-access PDFs.",
    )
    ap.add_argument("query", nargs="*", help="Search terms (source query syntax passes through).")
    ap.add_argument("--query", dest="query_opt", metavar="TEXT",
                    help="The query, as an option (needed for a query that is literally 'fetch').")
    ap.add_argument("--source", action="append", choices=S.SOURCES, default=[],
                    help="Repeatable. Default: pubmed, arxiv, biorxiv, medrxiv "
                         "(clinicaltrials is opt-in).")
    ap.add_argument("--limit", type=int, default=S.DEFAULT_LIMIT,
                    help=f"Results requested per source (default {S.DEFAULT_LIMIT}, "
                         f"max {S.MAX_LIMIT}).")
    window = ap.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, help="Only leads published in the last N days.")
    window.add_argument("--since", help="Only leads published on or after YYYY-MM-DD.")
    ap.add_argument("--sort", choices=("source", "fit", "date"), default="source",
                    help="Group by source (default), or one list by wiki fit or by date.")
    ap.add_argument("--abstracts", action="store_true",
                    help="Also print each lead's verbatim abstract (trials: brief summary) "
                         "in the terminal view. --json always carries them.")
    ap.add_argument("--max-age-days", type=float, default=1,
                    help="Re-query a source when its cached answer is older than N days; "
                         "0 always re-queries (default 1).")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit the snapshot as JSON.")
    manage = ap.add_argument_group("declines (shared with `scout recent`)")
    manage.add_argument("--decline", metavar="KEY", help="Never show this lead again.")
    manage.add_argument("--reason", help="Why (required with --decline).")
    manage.add_argument("--undecline", metavar="KEY")
    manage.add_argument("--list-declined", action="store_true")
    return ap


def _declines(args) -> int | None:
    if not (args.decline or args.undecline or args.list_declined):
        return None
    try:
        if args.decline:
            if not (args.reason or "").strip():
                print(f"{PROG}: --decline needs --reason", file=sys.stderr)
                return 1
            print(f"declined {R.add_decline(args.decline, args.reason.strip(), source='search')}")
            return 0
        if args.undecline:
            key = R._stored_key(args.undecline, R.load_declines())
            if R.remove_decline(key):
                print(f"undeclined {key}")
                return 0
            print(f"{PROG}: {key} was not declined", file=sys.stderr)
            return 1
    except R.DeclineKeyError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    declines = R.load_declines()
    if args.as_json:
        print(json.dumps(declines, indent=2))
    else:
        for key, entry in sorted(declines.items()):
            print(f"{key}  ({entry.get('declined_at', '?')}): {entry.get('reason', '')}")
    return 0


def _window(args) -> tuple[_dt.date | None, str | None]:
    if args.since:
        since = R._parse_date(args.since)
        return (since, None) if since else (None, f"invalid --since {args.since!r}; expected YYYY-MM-DD")
    if args.days is not None:
        if args.days <= 0:
            return None, "--days must be positive"
        return _dt.date.today() - _dt.timedelta(days=args.days), None
    return None, None


def main(argv: list[str]) -> int:
    if argv and argv[0] == "fetch":
        return fetch_main(argv[1:])
    args = _parser().parse_args(argv)
    handled = _declines(args)
    if handled is not None:
        return handled
    query = (args.query_opt or " ".join(args.query)).strip()
    if not query:
        print(f"{PROG}: give a query (or --decline / --list-declined)", file=sys.stderr)
        return 1
    if not 1 <= args.limit <= S.MAX_LIMIT:
        print(f"{PROG}: --limit must be between 1 and {S.MAX_LIMIT}", file=sys.stderr)
        return 1
    if args.max_age_days < 0:
        print(f"{PROG}: --max-age-days must be non-negative", file=sys.stderr)
        return 1
    since, problem = _window(args)
    if problem:
        print(f"{PROG}: {problem}", file=sys.stderr)
        return 1
    sources = list(dict.fromkeys(args.source)) or list(S.DEFAULT_SOURCES)
    snapshot = S.run(query, sources=sources, limit=args.limit, since=since,
                     max_age_days=args.max_age_days)
    if args.as_json:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    else:
        print(render(snapshot, sort=args.sort, abstracts=args.abstracts))
    states = {v["status"] for v in snapshot["sources"].values()}
    if "unavailable" in states:
        from ..providers._http import StructuredProviderUnavailable
        failed = [k for k, v in snapshot["sources"].items() if v["status"] == "unavailable"]
        raise StructuredProviderUnavailable(
            f"{', '.join(failed)} unavailable; results above are from the other sources"
        )
    return 1 if "rejected" in states else 0
