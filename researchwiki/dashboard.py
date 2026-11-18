"""Read the Markdown library into a portable, offline browser dashboard.

No model calls, index builds, database writes, or wiki mutations. Publication
dates and added dates remain separate; missing stamps never become file mtimes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import Counter
from importlib import resources
from pathlib import Path
from urllib.parse import quote, urlencode

from .paths import inbox_dir, papers_dir, wiki_dir, wiki_root
from .wiki import Page, read_pages

REFERENCE_TYPES = frozenset({"guidance", "protocol", "whitepaper", "book"})
BOOKKEEPING_TYPES = frozenset({"meta", "dashboard"})
_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def _stamp(value: object) -> str:
    """Normalize only real frontmatter dates, with explicit missing values."""
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass
    return ""


def _url(path: Path, out: Path | None, live: bool) -> str:
    if live:
        rel = path.relative_to(wiki_root()).as_posix()
        return "/files/" + quote(rel, safe="/")
    parent = (out or wiki_root() / "output/dashboard.html").absolute().parent
    return quote(os.path.relpath(path.absolute(), parent), safe="/")


def _page_record(page: Page, out: Path | None, live: bool, vault: str) -> dict:
    kind = page.page_type if isinstance(page.page_type, str) else "unknown"
    date_field = {"proposal": "created_at", "synthesis": "generated_at",
                  "idea": "generated_at", "concept": "generated_at"}.get(kind, "ingested_at")
    added = _stamp(page.fm.get(date_field))
    pdf = papers_dir() / f"{page.stem}.pdf"
    has_pdf = pdf.is_file()
    flags = []
    if not page.fm.get("type"):
        flags.append("Missing page type")
    if not added:
        flags.append("Missing added date")
    if not page.str_field("hook") and kind != "proposal":
        flags.append("Missing catalog description")
    if kind in {"paper", "commentary"} | REFERENCE_TYPES and not has_pdf:
        flags.append("Missing source PDF")
    if page.str_field("pdf_extraction_note"):
        flags.append("PDF extraction note")
    # Body text is copied verbatim; the dashboard never authors research prose.
    return {
        "id": page.key, "stem": page.stem, "title": page.str_field("title") or page.stem,
        "short_name": page.str_field("short_name"), "type": kind,
        "category": page.category, "topics": page.list_field("category"),
        "year": page.year_int(), "venue": page.str_field("venue"),
        "authors": page.str_field("authors") or page.str_field("issuer"),
        "hook": page.str_field("hook"), "keywords": page.list_field("keywords"),
        "tags": page.list_field("tags"), "topic_seed": page.str_field("topic_seed"),
        "status": page.str_field("status"), "verdict": page.str_field("verdict"),
        "direction": page.str_field("direction"), "proposed_page_type": page.str_field("proposed_page_type"),
        "doi": page.str_field("doi"), "author_model": page.str_field("author_model"),
        "added": added, "date_field": date_field, "body": page.body,
        "markdown_url": _url(page.path, out, live),
        "pdf_url": _url(pdf, out, live) if has_pdf else "",
        "obsidian_url": "obsidian://open?" + urlencode({"vault": vault, "file": "wiki/" + page.key}),
        "flags": flags, "links": [], "backlinks": [],
    }


def build_dashboard(*, out: Path | None = None, live: bool = False,
                    vault: str | None = None) -> dict:
    """Build from canonical pages, including undated documents in the library."""
    root = wiki_dir()
    pages = [p for p in read_pages() if p.path.parent != root
             and p.stem != "suggested-additions"
             and (not isinstance(p.page_type, str) or p.page_type not in BOOKKEEPING_TYPES)]
    vault = vault or wiki_root().name
    rows = [_page_record(p, out, live, vault) for p in pages]
    by_id = {row["id"]: row for row in rows}
    by_stem: dict[str, list[str]] = {}
    for row in rows:
        by_stem.setdefault(row["stem"], []).append(row["id"])
    for row in rows:
        targets = set()
        for match in _LINK.finditer(row["body"]):
            key = match.group(1).strip()
            if key not in by_id:
                candidates = by_stem.get(key, [])
                key = candidates[0] if len(candidates) == 1 else ""
            if key in by_id and key != row["id"]:
                targets.add(key)
        row["links"] = sorted(targets)
        for key in targets:
            by_id[key]["backlinks"].append(row["id"])
    for row in rows:
        row["backlinks"].sort()
    inbox = [{"name": p.name, "url": _url(p, out, live), "bytes": p.stat().st_size}
             for p in sorted(inbox_dir().glob("*.pdf")) if p.is_file()]
    categories = Counter(row["category"] for row in rows
                         if row["type"] in {"paper", "commentary"})
    types = Counter(row["type"] for row in rows)
    return {
        "meta": {"name": "Research Wiki", "repository": wiki_root().name,
                 "generated_at": dt.datetime.now().astimezone().isoformat(),
                 "live": live, "categories": dict(sorted(categories.items())),
                 "types": dict(sorted(types.items())), "n_pages": len(rows),
                 "n_pdfs": sum(bool(r["pdf_url"]) for r in rows)},
        "pages": rows, "inbox": inbox,
    }


def render_html(data: dict) -> str:
    assets = resources.files("researchwiki").joinpath("templates")
    template = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
    engine = assets.joinpath("vendor/minisearch-7.2.0.js").read_text(encoding="utf-8")
    search = assets.joinpath("dashboard-search.js").read_text(encoding="utf-8")
    # JSON lives in an inert script element. Escape '<' so corpus text cannot
    # close that element or inject HTML, including </script> in a quoted title.
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__MINISEARCH_SOURCE__", engine).replace("__DASHBOARD_SEARCH__", search).replace("__DASHBOARD_DATA__", payload)
