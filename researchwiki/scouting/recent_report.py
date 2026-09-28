"""One current view of every `scout recent` run: the merge step under a dashboard.

Each `scout recent` invocation writes its own snapshot under `.s2-cache/recent/`,
keyed by its seed set, and a category run and a page run overlap heavily. Reading
them one at a time answers "what did this seed set return", not "what should I
look at now" — and a paper ingested or declined since a snapshot was written is
still sitting in it.

This module merges them: newest snapshot per seed set, papers re-checked against
the wiki and the decline list *at report time*, one row per paper carrying every
page it matched across runs, and the earliest `first_seen` of any snapshot that
saw it. `--json` is the aggregate a dashboard reads; the text form is the same
data for a terminal.

**Rendering lives downstream of this.** `visualize` sets the pattern — build a
data object, expose it with `--json`, substitute it into a self-contained
template — so an HTML dashboard is a renderer of this contract, not a reader of
prose. Nothing here writes to `wiki/`: these are unverified leads, and a page
under `wiki/` would be claiming otherwise (CLAUDE.md Rule 1).

Zero tokens, no network, no model: this only reads files `scout recent` already
wrote.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from ..fsatomic import read_json
from ..log import log
from ..wiki import read_pages
from . import recent as R

LOG_TAG = "scout-recent-report"
#: Bump on a breaking change to the `--json` shape: a dashboard parses it, so
#: adding a key is safe and renaming or removing one is not.
REPORT_SCHEMA_VERSION = 1
SEEN_FILENAME = "seen.json"


def load_snapshots(directory: Path | None = None) -> tuple[list[dict], int]:
    """Every readable snapshot, newest first, plus the count skipped.

    A snapshot from a schema version this build does not know is skipped rather
    than guessed at; an unreadable one is skipped the same way (`read_json`
    treats a truncated file as absent, which is also how the cache it lives in
    behaves).
    """
    directory = directory or R._recent_dir()
    snapshots: list[dict] = []
    skipped = 0
    if not directory.is_dir():
        return snapshots, skipped
    for path in sorted(directory.glob("*.json")):
        if path.name == SEEN_FILENAME:
            continue
        data = read_json(path)
        if not isinstance(data, dict) or data.get("schema_version") != R.SCHEMA_VERSION:
            skipped += 1
            continue
        snapshots.append({**data, "snapshot_path": str(path)})
    snapshots.sort(key=lambda s: str(s.get("generated_at") or ""), reverse=True)
    return snapshots, skipped


def _seed_identity(snapshot: dict) -> str:
    """What makes two snapshots the same run. The label is the seed set's name
    (`single-cell`, `synthesis/…`, `N paper(s)`), so a re-run of one seed set
    supersedes its predecessor while different seed sets both stay."""
    return str((snapshot.get("seeds") or {}).get("label") or snapshot["snapshot_path"])


def aggregate(
    *,
    since: _dt.date | None = None,
    days: int | None = None,
    today: _dt.date | None = None,
    directory: Path | None = None,
) -> dict:
    """Merge the snapshots into one current view.

    `days`/`since` re-apply a publication window at report time, so an old
    snapshot does not keep surfacing papers that have aged out of it. Omitted,
    every snapshot's own window stands.
    """
    today = today or _dt.date.today()
    cutoff = since or (today - _dt.timedelta(days=days) if days else None)
    snapshots, skipped_version = load_snapshots(directory)

    # Newest snapshot per seed set; older ones are superseded, not merged, or a
    # paper dropped by a re-run would be resurrected by its own predecessor.
    runs: list[dict] = []
    seen_identity: set[str] = set()
    superseded = 0
    for snapshot in snapshots:
        identity = _seed_identity(snapshot)
        if identity in seen_identity:
            superseded += 1
            continue
        seen_identity.add(identity)
        runs.append(snapshot)

    pages = read_pages()
    papers = R._paper_pages(pages)
    wiki_dois = {d for p in papers if (d := R._doi(p))}
    wiki_titles = {R._norm_title(str(p.fm.get("title") or "")) for p in papers} - {""}
    page_types = {p.key: p.page_type for p in pages}
    declines = R.load_declines()

    merged: dict[str, dict] = {}
    counts = {"ingested": 0, "declined": 0, "outside_window": 0, "page_gone": 0}
    for snapshot in runs:
        label = _seed_identity(snapshot)
        rows = list(snapshot.get("page_matches") or []) + list(snapshot.get("candidates") or [])
        for row in rows:
            key = row.get("key")
            if not isinstance(key, str) or not key:
                continue
            # `Candidate.key` is the DOI when there is one, so a key that is not
            # an `s2:` id is itself the DOI: fall back to it rather than trusting
            # one field of a snapshot written by another build.
            doi = (row.get("doi") or "").lower() or None
            if doi is None and not key.startswith("s2:"):
                doi = key.lower()
            if (doi and doi in wiki_dois) or R._norm_title(row.get("title") or "") in wiki_titles:
                counts["ingested"] += 1
                continue
            # An `s2:` decline still applies once S2 has given the paper a DOI,
            # the same asymmetry `filter_candidates` handles.
            paper_id = row.get("paper_id")
            s2_key = f"s2:{str(paper_id).lower()}" if paper_id else None
            if key in declines or (doi and doi in declines) or (s2_key and s2_key in declines):
                counts["declined"] += 1
                continue
            if cutoff is not None:
                published = R._parse_date(row.get("publication_date"))
                year = row.get("year")
                in_window = published >= cutoff if published else bool(year and year >= cutoff.year)
                if not in_window:
                    counts["outside_window"] += 1
                    continue
            entry = merged.get(key)
            if entry is None:
                entry = {k: row.get(k) for k in (
                    "key", "doi", "paper_id", "title", "venue", "publication_date",
                    "year", "citation_count", "has_abstract", "fit", "first_seen", "nearest",
                )}
                entry["matches"] = {}
                entry["runs"] = []
                merged[key] = entry
            if label not in entry["runs"]:
                entry["runs"].append(label)
            # Prefer the strongest evidence any run found for this paper.
            if (row.get("fit") or -1) > (entry.get("fit") or -1):
                entry["fit"] = row.get("fit")
                entry["nearest"] = row.get("nearest")
            first = [f for f in (entry.get("first_seen"), row.get("first_seen")) if f]
            entry["first_seen"] = min(first) if first else None
            for match in row.get("triggers") or []:
                page = match.get("page")
                if not isinstance(page, str):
                    continue
                if page not in page_types:
                    counts["page_gone"] += 1
                    continue
                prior = entry["matches"].get(page)
                if prior is None or (match.get("z") or -99) > (prior.get("z") or -99):
                    entry["matches"][page] = {**match, "page_type": page_types[page]}

    out_papers = []
    for entry in merged.values():
        matches = sorted(entry.pop("matches").values(), key=lambda m: -(m.get("z") or 0))
        ident = entry.get("doi") or (
            f"https://www.semanticscholar.org/paper/{entry['paper_id']}" if entry.get("paper_id") else ""
        )
        out_papers.append({
            **entry,
            "matches": matches,
            "best_z": matches[0].get("z") if matches else None,
            "decline_command": (
                f'researchwiki scout recent --decline {ident} --reason "…"' if ident else None
            ),
        })
    # Papers a page is waiting on first, then the closest to the corpus.
    out_papers.sort(key=lambda p: (
        0 if p["matches"] else 1, -(p["best_z"] or 0), -(p["fit"] or 0), p["title"] or ""))

    by_page: dict[str, dict] = {}
    for paper in out_papers:
        for match in paper["matches"]:
            bucket = by_page.setdefault(match["page"], {
                "page": match["page"], "page_type": match["page_type"], "papers": []})
            bucket["papers"].append({"key": paper["key"], "z": match.get("z"),
                                     "title": paper["title"], "trigger": match.get("text")})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "window": {"since": cutoff.isoformat() if cutoff else None, "days": days},
        "counts": {
            "snapshots": len(snapshots),
            "runs_used": len(runs),
            "runs_superseded": superseded,
            "snapshots_skipped_version": skipped_version,
            "papers": len(out_papers),
            "papers_with_matches": sum(1 for p in out_papers if p["matches"]),
            "pages_with_papers": len(by_page),
            "dropped": counts,
        },
        "runs": [{
            "label": _seed_identity(s),
            "generated_at": s.get("generated_at"),
            "since": s.get("since"),
            "scored": s.get("scored"),
            "snapshot_path": s.get("snapshot_path"),
            "page_matches": len(s.get("page_matches") or []),
            "candidates": len(s.get("candidates") or []),
        } for s in runs],
        "trigger_gates": (runs[0].get("trigger_gates") if runs else None),
        "papers": out_papers,
        "by_page": sorted(by_page.values(), key=lambda b: (-len(b["papers"]), b["page"])),
    }


# ---------- rendering ----------

def _paper_line(paper: dict, today: str) -> list[str]:
    new = "NEW " if paper.get("first_seen") == today else "    "
    date = paper.get("publication_date") or f"{paper.get('year') or '?'} (undated)"
    ident = paper.get("doi") or (
        f"https://www.semanticscholar.org/paper/{paper['paper_id']}" if paper.get("paper_id") else "(no id)")
    fit = f"fit {paper['fit']:.2f}" if paper.get("fit") is not None else "fit  —  "
    lines = [f"- {new}{date}  {fit}  `{ident}`  {paper.get('venue') or '—'}",
             f"    {paper.get('title') or '(no title)'}"]
    for match in paper["matches"]:
        text = match.get("text") or ""
        if len(text) > 150:
            text = text[:147] + "…"
        lines.append(f"    ↳ on-topic for [[{match['page']}]] (z {match.get('z', 0):.1f}): {text}")
    return lines


def render(report: dict, limit: int) -> str:
    today = report["generated_at"][:10]
    c = report["counts"]
    out = ["# Papers worth a look", ""]
    window = f", published since {report['window']['since']}" if report["window"]["since"] else ""
    out.append(
        f"_Merged from {c['runs_used']} seed set(s){window}: {c['papers']} paper(s) not in the "
        f"wiki, {c['papers_with_matches']} near an open question on {c['pages_with_papers']} "
        f"page(s). Dropped since the runs: {c['dropped']['ingested']} ingested, "
        f"{c['dropped']['declined']} declined, {c['dropped']['outside_window']} outside the window._"
    )
    out.append("")
    matched = [p for p in report["papers"] if p["matches"]]
    rest = [p for p in report["papers"] if not p["matches"]]
    shown = matched[:limit]
    out.append(f"## Near a page's open questions ({len(shown)} of {len(matched)})")
    out.extend(line for p in shown for line in _paper_line(p, today))
    if not matched:
        out.append("_(none)_")
    out.append("")
    rest_shown = rest[:max(0, limit - len(shown))]
    out.append(f"## Closest to the corpus ({len(rest_shown)} of {len(rest)})")
    out.extend(line for p in rest_shown for line in _paper_line(p, today))
    out.append("")
    if report["by_page"]:
        out.append("## By page")
        for bucket in report["by_page"]:
            out.append(f"- [[{bucket['page']}]] — {len(bucket['papers'])} paper(s)")
    out.append("")
    out.append("_Leads only: nothing here is evidence until its PDF is in `inbox/` and ingested. "
               "Re-run `researchwiki scout recent` to refresh a seed set._")
    return "\n".join(out)


# ---------- CLI ----------

def main(argv: list[str], *, prog: str = "researchwiki scout recent report") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Merge every `scout recent` snapshot into one current view: papers "
                    "not yet in the wiki, each with the pages it matched. Reads only "
                    "local files — no network, no model calls.",
    )
    window = ap.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, default=None,
                        help="Re-apply a publication window of N days at report time.")
    window.add_argument("--since", help="Re-apply a publication window from YYYY-MM-DD.")
    ap.add_argument("--limit", type=int, default=25,
                    help="Rows shown per section in text mode (default 25).")
    ap.add_argument("--out", metavar="PATH",
                    help="Write the rendering to a file instead of stdout.")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit the aggregate as JSON — the contract a dashboard reads.")
    args = ap.parse_args(argv)

    since = None
    if args.since:
        since = R._parse_date(args.since)
        if since is None:
            print(f"{prog}: invalid --since {args.since!r}; expected YYYY-MM-DD", file=sys.stderr)
            return 1
    if args.days is not None and args.days <= 0:
        print(f"{prog}: --days must be positive", file=sys.stderr)
        return 1

    report = aggregate(since=since, days=args.days)
    if report["counts"]["snapshots"] == 0:
        log("no snapshots under .s2-cache/recent/ — run `researchwiki scout recent "
            "--category <c>` first", tag=LOG_TAG)
        return 1
    text = json.dumps(report, indent=2, ensure_ascii=False) if args.as_json \
        else render(report, args.limit)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        log(f"wrote {path}", tag=LOG_TAG)
    else:
        print(text)
    return 0
