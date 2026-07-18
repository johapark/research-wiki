"""Hub-level refresh + spoke-citation upgrade utilities.

  - `refresh_concept(slug)` — regenerate a hub's "Cross-domain connections"
                              section from claim-graph edges. Read-only
                              draft; the caller decides whether to apply.
  - `upgrade_spokes()`      — backfill `[[stem#slug]]` anchors on existing
                              hubs whose spokes still use bare `[[stem]]`.
                              Idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..log import append_log_md, log
from ..paths import wiki_dir
from ..fsatomic import write_text_atomic
from ..wiki import commit_page, read_page, read_pages
from .term_claims import _best_claim_slug

# Edge statuses that must not resurface into a refreshed hub: a human-rejected
# edge, and one whose endpoint slug no longer resolves.
_TERMINAL_STATUSES = frozenset({"rejected", "stale"})

def refresh_concept(slug: str, *, dry_run: bool = False) -> dict:
    """Draft a `## Cross-domain connections` block for a concept hub, sourced
    from typed edges (corroborates / refines / builds_on / measures_same)
    among the hub's `instantiates`-linked claims.

    The draft is written to `.ingest/{slug}-concept-refresh/draft.md` for
    human review — never auto-applied. Author reviews, optionally edits,
    and copies bullets into the hub's Cross-domain connections section.
    Deleting the draft file rejects the refresh.

    Returns a stats dict with counts + the draft path. When no typed edges
    span ≥2 categories, returns with `n_bridges_found=0` and writes no file.
    """
    from math import inf
    from collections import defaultdict

    cdir = wiki_dir() / "concepts"
    hub_path = cdir / f"{slug}.md"
    if not hub_path.exists():
        raise ValueError(f"concept hub not found: {hub_path}")

    from ..claim_graph import open_edges_db, query
    from ..paths import ingest_dir

    edges_conn = open_edges_db()
    try:
        inst = query(edges_conn, relation="instantiates",
                     tgt_stem="concepts", tgt_slug=slug)
    finally:
        edges_conn.close()

    # Exclude terminal statuses: `rejected` (a human said no) and `stale`
    # (endpoint slug no longer resolves) must not resurface as members/bridges.
    inst = [e for e in inst if e.status not in _TERMINAL_STATUSES]

    # Member claims of this hub: sources of `instantiates` edges pointing at it.
    member_claims: set[tuple[str, str]] = {(e.src_stem, e.src_slug) for e in inst}

    if not member_claims:
        return {
            "slug": slug, "n_member_claims": 0, "n_bridges_found": 0,
            "draft_path": None, "dry_run": dry_run,
        }

    # Collect typed edges among the member claims. Load each paper's category
    # so we can identify cross-category bridges.
    typed_relations = ("corroborates", "measures_same", "refines", "builds_on")
    edges_conn = open_edges_db()
    try:
        all_typed = []
        for rel in typed_relations:
            all_typed.extend(
                e for e in query(edges_conn, relation=rel)
                if e.status not in _TERMINAL_STATUSES
            )
    finally:
        edges_conn.close()

    # Filter to edges where BOTH endpoints are member claims of this hub.
    inter_member: list = []
    for e in all_typed:
        if (e.src_stem, e.src_slug) in member_claims \
                and (e.tgt_stem, e.tgt_slug) in member_claims:
            inter_member.append(e)

    # Resolve stem → category via state.db.
    stems = {s for s, _ in member_claims}
    cat_by_stem = _resolve_stem_categories(stems)

    # Cross-category bridges: edges where src.category != tgt.category.
    bridges = []
    for e in inter_member:
        src_cat = cat_by_stem.get(e.src_stem, "")
        tgt_cat = cat_by_stem.get(e.tgt_stem, "")
        if src_cat and tgt_cat and src_cat != tgt_cat:
            bridges.append((e, src_cat, tgt_cat))

    if not bridges:
        return {
            "slug": slug, "n_member_claims": len(member_claims),
            "n_bridges_found": 0, "draft_path": None, "dry_run": dry_run,
        }

    # Group by (rel, category-pair) so identical bridges cluster.
    from collections import OrderedDict
    grouped: dict[tuple[str, tuple[str, str]], list] = OrderedDict()
    for e, sc, tc in bridges:
        key = (e.relation, tuple(sorted([sc, tc])))
        grouped.setdefault(key, []).append((e, sc, tc))

    # Draft the block.
    draft_lines = ["## Cross-domain connections", ""]
    for (rel, cats), items in grouped.items():
        cat_pair = " ↔ ".join(cats)
        draft_lines.append(f"### {rel} across {cat_pair}")
        draft_lines.append("")
        for e, sc, tc in items:
            src = f"[[{e.src_stem}#{e.src_slug}]]"
            tgt = f"[[{e.tgt_stem}#{e.tgt_slug}]]"
            rat = e.rationale or "(no rationale)"
            arrow = "→" if e.directed else "↔"
            draft_lines.append(f"- {src} {arrow} {tgt}: {rat}")
        draft_lines.append("")

    draft = "\n".join(draft_lines).rstrip() + "\n"

    out = {
        "slug": slug,
        "n_member_claims": len(member_claims),
        "n_bridges_found": len(bridges),
        "draft_path": None,
        "dry_run": dry_run,
    }
    if not dry_run:
        target_dir = ingest_dir() / f"{slug}-concept-refresh"
        target_dir.mkdir(parents=True, exist_ok=True)
        draft_path = target_dir / "draft.md"
        write_text_atomic(
            draft_path,
            "<!-- To apply: paste this block into the concept hub's "
            "## Cross-domain connections section, edit as needed, then remove "
            "this file. Delete this file to reject the refresh entirely. -->\n\n"
            + draft,
        )
        out["draft_path"] = str(draft_path)
    return out

def _resolve_stem_categories(stems: set[str]) -> dict[str, str]:
    """Batch-lookup stem → category from state.db. Silent {} on failure."""
    if not stems:
        return {}
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return {}
    try:
        placeholders = ",".join(["?"] * len(stems))
        rows = conn.execute(
            f"SELECT stem, category FROM papers WHERE stem IN ({placeholders})",
            list(stems),
        ).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()
    return {r["stem"]: r["category"] for r in rows}

def upgrade_spokes(*, dry_run: bool = False) -> dict:
    """Rewrite bare `[[stem]]` spokes in every wiki/concepts/*.md to
    `[[stem#claim_slug]]` where a matching contribution claim exists.

    Scans each concept page's `## How it appears across the corpus` section
    for spoke bullets. Bare-stem citations are candidates for upgrade;
    already-slug-cited spokes are left untouched. For each candidate,
    resolves the term (from the hub's `topic_seed`) against the paper's
    claims via _matching_claims; if a hit exists, rewrites `[[stem]]` →
    `[[stem#best_slug]]` in the bullet only (frontmatter `referenced_papers:`
    stays bare — that's whole-paper enumeration).

    Idempotent, safe to re-run. Returns:
      {hubs_scanned, hubs_updated, spokes_upgraded, spokes_skipped_no_claim}.
    """
    cdir = wiki_dir() / "concepts"
    stats = {
        "hubs_scanned": 0,
        "hubs_updated": 0,
        "spokes_upgraded": 0,
        "spokes_skipped_no_claim": 0,
        "per_hub": {},
    }
    if not cdir.exists():
        return stats

    # Matches a spoke bullet in the corpus-appearance section citing a bare
    # `[[category/stem]]` (or `[[stem]]`) target. Already-anchored bullets are
    # not matched — we specifically exclude `#` before the closing brackets.
    spoke_re = re.compile(r"(-\s*\[\[)([^\]\|#]+)(\]\])")

    for cp in sorted(cdir.glob("*.md")):
        stats["hubs_scanned"] += 1
        cpage = read_page(cp)
        if cpage is None:
            continue
        term = str(cpage.fm.get("topic_seed") or cpage.fm.get("title") or "").strip().strip('"').strip("'")
        if not term:
            continue
        text = cp.read_text()
        # Only operate within `## How it appears across the corpus` section.
        m = re.search(
            r"(## How it appears across the corpus\n)(.*?)(?=\n## |\Z)",
            text, re.S,
        )
        if not m:
            continue
        pre_len = m.end(1)
        section_body = m.group(2)
        upgraded = 0
        skipped = 0

        def _upgrade(match: re.Match) -> str:
            nonlocal upgraded, skipped
            prefix, target, suffix = match.group(1), match.group(2), match.group(3)
            # Bare stem — extract the paper stem (strip category prefix).
            target_stem = target.rsplit("/", 1)[-1]
            best_slug = _best_claim_slug(target_stem, term)
            if not best_slug:
                skipped += 1
                return match.group(0)  # unchanged
            upgraded += 1
            return f"{prefix}{target}#{best_slug}{suffix}"

        new_section = spoke_re.sub(_upgrade, section_body)
        if upgraded:
            new_text = text[: m.start(2)] + new_section + text[m.end(2):]
            if not dry_run:
                write_text_atomic(cp, new_text)
                commit_page(cp)
            stats["hubs_updated"] += 1
            stats["per_hub"][cp.stem] = {"upgraded": upgraded, "skipped": skipped}
        stats["spokes_upgraded"] += upgraded
        stats["spokes_skipped_no_claim"] += skipped

    return stats
