"""Cross-reference between `wiki/synthesis/suggested-additions.md` (the
human-curated citation-scout gap list) and its latest cached JSON.

`p2_entries_with_anchor_hits` surfaces DOIs listed under `## Priority 2`
of suggested-additions.md that now appear in the latest cached scout's
`shared_citation_anchors` — a signal the entry's count or category
profile has shifted and might warrant a status change. Descriptive
only; the LLM decides whether to move, re-annotate, or leave entries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...paths import s2_cache_dir, wiki_dir


_P2_DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s)\]]+)", re.IGNORECASE)


def find_p2_anchor_hits(pages: list[Path]) -> list[dict]:
    """Surface DOIs listed under `## Priority 2` in suggested-additions.md
    that now appear in the latest cached scout's shared_citation_anchors.

    Silent no-op (returns []) if no audit cache is present, no
    suggested-additions.md exists, or the cache file is malformed.
    """
    sa_path = wiki_dir() / "synthesis" / "suggested-additions.md"
    if not sa_path.exists():
        return []
    cache_dir = s2_cache_dir()
    if not cache_dir.exists():
        return []
    snapshots = sorted(cache_dir.glob("audit-*.json"))
    if not snapshots:
        return []
    try:
        audit_data = json.loads(snapshots[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    anchors_by_doi: dict[str, dict] = {
        (a.get("doi") or "").lower(): a
        for a in audit_data.get("shared_citation_anchors", [])
        if a.get("doi")
    }
    if not anchors_by_doi:
        return []

    text = sa_path.read_text(encoding="utf-8")
    # Extract just the Priority 2 section: from `## Priority 2` to the
    # next `## ` or EOF.
    m = re.search(r"^##\s+Priority\s+2\b[^\n]*\n", text, re.MULTILINE)
    if not m:
        return []
    p2_start = m.end()
    m2 = re.search(r"^##\s+", text[p2_start:], re.MULTILINE)
    p2_text = text[p2_start:p2_start + m2.start()] if m2 else text[p2_start:]

    hits: list[dict] = []
    seen: set[str] = set()
    for line in p2_text.splitlines():
        # Skip lines already marked as resolved.
        if "✅" in line or "INGESTED" in line or "~~" in line:
            continue
        for doi_match in _P2_DOI_RE.finditer(line):
            doi = doi_match.group(1).rstrip(".,;").lower()
            if doi in seen:
                continue
            seen.add(doi)
            anchor = anchors_by_doi.get(doi)
            if not anchor:
                continue
            hits.append({
                "doi": doi,
                "title": anchor.get("title", ""),
                "current_count": anchor.get("multi_paper_count", 0),
                "categories": anchor.get("categories", []),
            })
    return hits
