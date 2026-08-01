"""Promote `confirmed` claim edges into canonical markdown.

Mirrors the propose→review→apply loop used by `evolve`:

  1. `researchwiki claim-graph promote`  — draft one proposal file per
     confirmed edge under `.ingest/{synthesis-slug}-claim-edges/{edge-id}.md`.
     Nothing under `wiki/` changes. Author reviews.

  2. `researchwiki claim-graph promote --apply` — walk the proposal dir,
     insert each approved bullet into the target synthesis's `## Tensions /
     open questions` (for contradicts) or `## Evidence` (for corroborates)
     section, set the edge's status to `promoted`, refresh `generated_at:`,
     and remove the proposal file.

A confirmed edge with no synthesis page referencing both endpoints is
skipped with a WARN — there's no target to promote into. Author can
manually add both papers to an existing synthesis (or scaffold one) then
re-run promote.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..fsatomic import write_text_atomic
from ..paths import ingest_dir, wiki_dir
from .edges import Edge, open_edges_db, query, set_status


# Synthesis section headings we accept as promotion targets. Matched
# case-insensitively; the first form wins for a section that needs creation.
_CONTRADICTS_HEADINGS = (
    "## Tensions / open questions",
    "## Tensions",
    "## Open questions",
    "## Contradictions",
)
_CORROBORATES_HEADINGS = (
    "## Evidence",
    "## Corroborating evidence",
    "## Support",
)


@dataclass
class Proposal:
    """One (edge, synthesis) promotion proposal ready to write or apply."""
    edge_id: int
    relation: str
    src_stem: str
    src_slug: str
    src_text: str
    tgt_stem: str
    tgt_slug: str
    tgt_text: str
    rationale: str
    synthesis_path: Path
    target_heading: str          # first matching heading OR the default form
    bullet: str                  # the drafted markdown bullet


def _load_claim_texts(state_conn, pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Batch-lookup claim.text for a set of (stem, slug) pairs. Missing
    endpoints (stale / deleted rows) get an empty string so the caller can
    still emit a proposal that flags the mismatch."""
    if not pairs:
        return {}
    placeholders = ",".join(["(?, ?)"] * len(pairs))
    params: list[str] = []
    for s, slug in pairs:
        params.extend([s, slug])
    rows = state_conn.execute(
        f"SELECT paper_stem, claim_slug, text FROM claims "
        f" WHERE (paper_stem, claim_slug) IN (VALUES {placeholders})",
        params,
    ).fetchall()
    return {(r["paper_stem"], r["claim_slug"]): r["text"] for r in rows}


def _load_synthesis_index(state_conn) -> list[dict]:
    """Return synthesis/idea/concept pages with their referenced_papers stems.

    Reads raw_frontmatter from state.db (which mirrors YAML). We parse the
    referenced_papers list defensively — the field can be a real list (PyYAML)
    or the line parser's literal string form.
    """
    import json as _json

    rows = state_conn.execute(
        "SELECT stem, page_type, page_path, raw_frontmatter FROM papers "
        " WHERE page_type IN ('synthesis', 'idea', 'concept')"
    ).fetchall()
    # synthesis/idea pages no longer carry a `referenced_papers:` YAML field —
    # their citations live in the body (## References footnotes + inline
    # [[wikilink]]s). Derive referenced stems from the body wikilinks, unioned
    # with the YAML field where it still exists (concept pages).
    wikilink_re = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]*)?(?:\|[^\]]+)?\]\]")
    out: list[dict] = []
    for r in rows:
        try:
            fm = _json.loads(r["raw_frontmatter"])
        except (TypeError, ValueError):
            fm = {}
        refs = set(_parse_referenced_papers(fm.get("referenced_papers") or []))
        path = Path(r["page_path"])
        try:
            body = path.read_text(encoding="utf-8")
            refs |= {m.group(1).strip() for m in wikilink_re.finditer(body)}
        except OSError:
            pass
        out.append({
            "stem": r["stem"],
            "page_type": r["page_type"],
            "path": path,
            "referenced_stems": {s.rsplit("/", 1)[-1] for s in refs},
        })
    return out


def _parse_referenced_papers(raw) -> list[str]:
    """Normalize `referenced_papers:` to a flat list of paper keys.

    Accepts: list-of-strings (PyYAML), list-of-list, or the line parser's
    literal string form `"[[a]], [[b]]"`. Strips `[[...]]` wrappers.
    """
    out: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                out.append(entry.strip("[] "))
            elif isinstance(entry, list):
                out.extend(str(x).strip("[] ") for x in entry if x)
    elif isinstance(raw, str):
        for chunk in raw.split(","):
            out.append(chunk.strip().strip("[] "))
    return [x for x in out if x]


def _find_target_syntheses(
    synth_index: list[dict], src_stem: str, tgt_stem: str,
) -> list[dict]:
    """Synthesis pages that reference BOTH endpoint stems. Sorted by page_type
    so `synthesis` pages win over `idea`/`concept` when there's a tie."""
    order = {"synthesis": 0, "idea": 1, "concept": 2}
    matches = [
        s for s in synth_index
        if src_stem in s["referenced_stems"] and tgt_stem in s["referenced_stems"]
    ]
    matches.sort(key=lambda s: order.get(s["page_type"], 99))
    return matches


def _draft_bullet(edge: Edge, src_text: str, tgt_text: str) -> str:
    """Draft the markdown bullet the human will review.

    Contradicts: two-line bullet, one per side, both anchored via [[stem#slug]].
    Corroborates: one-line bullet + rationale.
    """
    src_cite = f"[[{edge.src_stem}#{edge.src_slug}]]"
    tgt_cite = f"[[{edge.tgt_stem}#{edge.tgt_slug}]]"
    if edge.relation == "contradicts":
        rat = edge.rationale.strip() if edge.rationale else "conflicting claims"
        return (
            f"- **{rat}**\n"
            f"    - {src_cite}: {src_text.strip()}\n"
            f"    - {tgt_cite}: {tgt_text.strip()}\n"
        )
    if edge.relation == "corroborates":
        rat = edge.rationale.strip() if edge.rationale else "affirming claims"
        return (
            f"- Both {src_cite} and {tgt_cite} affirm: {rat}\n"
            f"    - {src_cite}: {src_text.strip()}\n"
            f"    - {tgt_cite}: {tgt_text.strip()}\n"
        )
    # Fallback for future relation types.
    return f"- {src_cite} {edge.relation} {tgt_cite}: {edge.rationale}\n"


def _target_heading_for(relation: str) -> str:
    if relation == "contradicts":
        return _CONTRADICTS_HEADINGS[0]
    if relation == "corroborates":
        return _CORROBORATES_HEADINGS[0]
    return "## Notes"


def _accepted_headings_for(relation: str) -> tuple[str, ...]:
    if relation == "contradicts":
        return _CONTRADICTS_HEADINGS
    if relation == "corroborates":
        return _CORROBORATES_HEADINGS
    return ("## Notes",)


def draft_proposal(edge: Edge, synth: dict, claim_texts: dict) -> Proposal:
    """Assemble a Proposal from an edge + its target synthesis + preloaded claim text."""
    src_text = claim_texts.get((edge.src_stem, edge.src_slug), "")
    tgt_text = claim_texts.get((edge.tgt_stem, edge.tgt_slug), "")
    bullet = _draft_bullet(edge, src_text, tgt_text)
    return Proposal(
        edge_id=edge.id or 0,
        relation=edge.relation,
        src_stem=edge.src_stem,
        src_slug=edge.src_slug,
        src_text=src_text,
        tgt_stem=edge.tgt_stem,
        tgt_slug=edge.tgt_slug,
        tgt_text=tgt_text,
        rationale=edge.rationale or "",
        synthesis_path=synth["path"],
        target_heading=_target_heading_for(edge.relation),
        bullet=bullet,
    )


def render_proposal_md(p: Proposal) -> str:
    """One markdown file per (edge, synthesis) pair, written to
    `.ingest/{synthesis-stem}-claim-edges/{edge-id}.md`."""
    lines = [
        "---",
        f"edge_id: {p.edge_id}",
        f"relation: {p.relation}",
        f"src: {p.src_stem}#{p.src_slug}",
        f"tgt: {p.tgt_stem}#{p.tgt_slug}",
        f"synthesis: {p.synthesis_path.stem}",
        f"target_heading: {p.target_heading}",
        "---",
        "",
        f"# {p.relation.upper()} — {p.synthesis_path.stem}",
        "",
        f"**Rationale:** {p.rationale or '(no rationale recorded)'}",
        "",
        "## Draft bullet",
        "",
        p.bullet,
        "",
        "## To apply",
        "",
        f"Insert the bullet above into `wiki/synthesis/{p.synthesis_path.stem}.md` "
        f"under `{p.target_heading}` (creates the section if missing). ",
        "Then run `researchwiki claim-graph promote --apply` to apply this "
        "proposal atomically, or delete this file to reject.",
    ]
    return "\n".join(lines) + "\n"


def _proposal_dir_for(synthesis_path: Path) -> Path:
    """`.ingest/{synthesis-stem}-claim-edges/`. Groups per-synthesis so a
    single yes/no can cover all edges targeting one page."""
    return ingest_dir() / f"{synthesis_path.stem}-claim-edges"


@dataclass
class ProposeStats:
    edges_scanned: int = 0
    edges_promoted_already: int = 0
    edges_no_target: int = 0
    proposals_written: int = 0
    per_synthesis: dict[str, int] = field(default_factory=dict)


def propose_promotions(*, dry_run: bool = False) -> ProposeStats:
    """Draft one proposal per (confirmed edge × matching synthesis).

    Runs against state.db + .claim-graph/edges.db. Never auto-applies.
    Idempotent: re-running overwrites proposal files with the current draft
    (a re-judged rationale, say). Callers approve by leaving the file in
    place and running with --apply; reject by deleting the file.
    """
    stats = ProposeStats()

    try:
        from ..db.connection import get_connection
        state = get_connection()
    except Exception:
        return stats

    edges_conn = open_edges_db()
    try:
        candidates = query(edges_conn, status="confirmed")
        stats.edges_scanned = len(candidates)
        if not candidates:
            return stats

        # Prefetch: all endpoint claim texts + the synthesis index.
        pairs: set[tuple[str, str]] = set()
        for e in candidates:
            pairs.add((e.src_stem, e.src_slug))
            pairs.add((e.tgt_stem, e.tgt_slug))
        claim_texts = _load_claim_texts(state, pairs)
        synth_index = _load_synthesis_index(state)

        for edge in candidates:
            if edge.status == "promoted":
                stats.edges_promoted_already += 1
                continue
            matches = _find_target_syntheses(
                synth_index, edge.src_stem, edge.tgt_stem,
            )
            if not matches:
                stats.edges_no_target += 1
                continue
            # Pick the highest-priority match (synthesis > idea > concept).
            synth = matches[0]
            prop = draft_proposal(edge, synth, claim_texts)
            if not dry_run:
                target_dir = _proposal_dir_for(synth["path"])
                target_dir.mkdir(parents=True, exist_ok=True)
                write_text_atomic(target_dir / f"{edge.id}.md", render_proposal_md(prop))
            stats.proposals_written += 1
            key = synth["path"].stem
            stats.per_synthesis[key] = stats.per_synthesis.get(key, 0) + 1
    finally:
        edges_conn.close()
        state.close()
    return stats


# ---------- apply ----------


@dataclass
class ApplyStats:
    proposals_seen: int = 0
    applied: int = 0
    skipped_stale: int = 0
    skipped_missing_edge: int = 0
    errors: list[str] = field(default_factory=list)


def _insert_bullet_under_section(
    text: str, accepted_headings: tuple[str, ...],
    default_heading: str, bullet: str,
) -> str:
    """Insert `bullet` under the first accepted heading (case-insensitive).
    If none match, append the default heading at end and put the bullet under it.

    Preserves everything else. `bullet` should end with a newline.
    """
    # Try each accepted heading in order.
    for h in accepted_headings:
        pattern = re.compile(
            r"^(##[ \t]+" + re.escape(h.lstrip("# ").strip()) + r".*?)$"
            r"(.*?)(?=^##[ \t]+|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            head = m.group(1)
            body = m.group(2)
            new_section = head + "\n" + body.rstrip() + "\n" + bullet
            return text[: m.start()] + new_section + text[m.end():]
    # None matched — append.
    sep = "" if text.endswith("\n") else "\n"
    return text + sep + "\n" + default_heading + "\n\n" + bullet


def _refresh_generated_at(text: str) -> str:
    """Bump `generated_at:` in the frontmatter to today. No-op if missing."""
    today = date.today().isoformat()
    m = re.match(r"(?s)^(---\n)(.*?)(\n---\n)(.*)$", text)
    if not m:
        return text
    fm_pre, fm_body, fm_post, body = m.groups()
    new_fm = re.sub(
        r"^generated_at:\s*\S+", f"generated_at: {today}", fm_body,
        count=1, flags=re.MULTILINE,
    )
    return fm_pre + new_fm + fm_post + body


def _parse_proposal_frontmatter(md_path: Path) -> dict:
    """Extract the key/value pairs from a proposal file's frontmatter."""
    text = md_path.read_text()
    m = re.match(r"(?s)^---\n(.*?)\n---\n", text)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def apply_promotions(*, verbose: bool = False) -> ApplyStats:
    """Walk `.ingest/*-claim-edges/*.md` and apply each surviving proposal.

    "Surviving" == the human left the file in place. Deleted files ≡ rejected.
    A proposal whose edge is no longer `confirmed` (stale / rejected / already
    promoted) is skipped with a message. Applies:
      - insert the drafted bullet under the target section
      - refresh synthesis `generated_at:`
      - set edge status → `promoted`
      - remove the proposal file
    """
    stats = ApplyStats()
    base = ingest_dir()
    if not base.exists():
        return stats

    try:
        from ..db.connection import get_connection
        state = get_connection()  # noqa: F841 — reserved for future validation
    except Exception:
        state = None

    edges_conn = open_edges_db()
    try:
        for pdir in sorted(base.glob("*-claim-edges")):
            if not pdir.is_dir():
                continue
            for pf in sorted(pdir.glob("*.md")):
                stats.proposals_seen += 1
                fm = _parse_proposal_frontmatter(pf)
                try:
                    edge_id = int(fm.get("edge_id", "0"))
                except ValueError:
                    stats.errors.append(f"{pf}: bad edge_id")
                    continue
                # Look up the edge; skip if not confirmed anymore.
                row = edges_conn.execute(
                    "SELECT * FROM edges WHERE id = ?", (edge_id,),
                ).fetchone()
                if row is None:
                    stats.skipped_missing_edge += 1
                    continue
                if row["status"] != "confirmed":
                    stats.skipped_stale += 1
                    continue
                # Re-derive the bullet from the proposal file body — humans
                # may have edited it during review.
                text = pf.read_text()
                bm = re.search(r"## Draft bullet\n+((?:- .+\n?)+(?:    - .+\n?)*)",
                               text)
                if bm is None:
                    stats.errors.append(f"{pf}: no draft bullet found")
                    continue
                bullet = bm.group(1)
                if not bullet.endswith("\n"):
                    bullet += "\n"

                # Resolve target synthesis.
                synth_stem = fm.get("synthesis", "").strip()
                # Try both `synthesis/{stem}.md` and `ideas/`, `concepts/`.
                target_path: Path | None = None
                for subdir in ("synthesis", "ideas", "concepts"):
                    candidate = wiki_dir() / subdir / f"{synth_stem}.md"
                    if candidate.exists():
                        target_path = candidate
                        break
                if target_path is None:
                    stats.errors.append(f"{pf}: cannot find target page for stem={synth_stem}")
                    continue

                relation = fm.get("relation", "contradicts")
                accepted = _accepted_headings_for(relation)
                default = _target_heading_for(relation)
                orig = target_path.read_text()
                # Idempotent: a prior run may have crashed after the page write
                # but before pf.unlink() below, leaving the edge un-promoted and
                # the proposal on disk. Re-inserting would duplicate the bullet,
                # so skip the write when it's already present and just finalize.
                if bullet.strip() and bullet.strip() in orig:
                    pass
                else:
                    updated = _insert_bullet_under_section(orig, accepted, default, bullet)
                    updated = _refresh_generated_at(updated)
                    write_text_atomic(target_path, updated)
                # Set edge → promoted; remove proposal file.
                set_status(edges_conn, edge_id, "promoted")
                edges_conn.commit()
                pf.unlink()
                stats.applied += 1
                if verbose:
                    print(f"applied edge #{edge_id} → {target_path.name} ({relation})")
            # Remove empty proposal dirs to keep .ingest tidy.
            if not any(pdir.iterdir()):
                shutil.rmtree(pdir, ignore_errors=True)
    finally:
        edges_conn.close()
        if state is not None:
            state.close()
    return stats
