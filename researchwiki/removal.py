"""Retract a paper: find every trace of a stem, then remove the generated ones.

Deleting `wiki/{cat}/{stem}.md` by hand strands the PDF, the `index.md` bullet,
inbound back-links on every citing paper, `[[stem#slug]]` anchors mid-prose on
synthesis pages, `referenced_papers:` spokes on concept hubs, claim-graph edges,
and four tables' worth of rows. `lint` reports the wreckage; nothing cleaned it
up. This does.

**The load-bearing policy: generated text is removed, authored text is not.**

A back-link bullet and an `index.md` entry are *generated* — `promote` wrote
them, no human judgement is embedded in them, and removing them restores the
state the wiki would have had. They go automatically.

A sentence on a synthesis page citing `[[stem#slug]]` is *authored*, and it has
passed `check-grounding` and `grade synthesis`. There is no safe rewrite rule:
stripping the citation leaves a claim with no support, and deleting the sentence
can remove a conclusion several papers jointly carried. So those are **reported,
never edited** — the reviewer decides and re-runs the gates. Immediately after
`--apply`, `lint` will report `dangling_claim_anchors` on those pages. That is
the intended state: a visible to-do queue, not a defect.

`log.md` is append-only, so a removal appends an entry rather than editing
history out of it.

**Any page type can be the target.** `scan` resolves the argument by *filename
stem* across all of `wiki/`, and nothing branches on `type:` — a synthesis,
idea, concept, reference doc or commentary page is removed the same way a paper
is, with the paper-shaped machinery (PDF, supplementary dir, grade/figure
caches, `claims` rows) simply finding nothing. The paper-specific vocabulary in
this command's output is a naming artefact, not a restriction. Two consequences
worth stating:

- `AUTHORED_TYPES` protects *citing* pages, never the target. Removing a
  synthesis or idea page deletes hand-authored, twice-gated prose; dry-run-by-
  default is the only guard, and `wiki/` may be gitignored.
- The reciprocal `[[concepts/<slug>]]` bullets a concept hub puts on its member
  papers are generated back-links like any other, so removing the hub strips
  them. Its members are not otherwise touched — expect newly `orphans`-listed
  papers when the removed page was their only inbound link.

`index.md` and `log.md` are excluded from the page scan, so the wiki-root
bookkeeping pages can never be the target.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    figures_cache_dir,
    grade_cache_dir,
    index_path,
    ingest_dir,
    papers_dir,
    supp_dir,
    wiki_dir,
)

# Every wikilink form CLAUDE.md sanctions for referring to a paper:
#   [[category/stem]]  [[stem]]  [[stem#claim_slug]]  [[stem|alias]]
def _link_re(stem: str) -> re.Pattern[str]:
    return re.compile(
        rf"\[\[(?:[^\]|#]*/)?{re.escape(stem)}(?:[#|][^\]]*)?\]\]"
    )


# Page types whose prose this command must never rewrite.
AUTHORED_TYPES = {"synthesis", "idea", "concept"}


@dataclass
class ProseRef:
    """An authored citation. Reported for the reviewer; never edited."""
    path: Path
    page_type: str
    line_no: int
    line: str


@dataclass
class RemovalPlan:
    stem: str
    page_path: Path | None = None
    category: str | None = None
    page_type: str | None = None
    files: list[Path] = field(default_factory=list)
    backlink_pages: list[tuple[Path, int]] = field(default_factory=list)
    index_bullet: bool = False
    concept_hubs: list[Path] = field(default_factory=list)
    commentary_pages: list[Path] = field(default_factory=list)
    prose_refs: list[ProseRef] = field(default_factory=list)
    db_rows: dict[str, int] = field(default_factory=dict)
    edge_count: int = 0

    @property
    def exists(self) -> bool:
        return self.page_path is not None or bool(self.files) or bool(self.db_rows)

    @property
    def touched_paths(self) -> list[Path]:
        """Everything `apply` may write — the snapshot set for the journal.

        Prose-ref pages are deliberately absent: they are not written, so
        backing them up would claim an intent this command does not have.
        """
        paths: list[Path] = []
        if self.page_path:
            paths.append(self.page_path)
        paths += list(self.files)
        paths += [p for p, _ in self.backlink_pages]
        paths += list(self.concept_hubs)
        if self.index_bullet:
            paths.append(index_path())
        return paths


def _iter_pages() -> list[Path]:
    root = wiki_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.name not in {"index.md", "log.md"})


def _frontmatter_type(text: str) -> str:
    m = re.search(r"^type:\s*[\"']?([\w-]+)", text, re.MULTILINE)
    return m.group(1).lower() if m else ""


def scan(stem: str) -> RemovalPlan:
    """Enumerate every trace of `stem`. Read-only — writes nothing."""
    plan = RemovalPlan(stem=stem)
    link = _link_re(stem)

    # The page itself.
    for page in _iter_pages():
        if page.stem == stem:
            plan.page_path = page
            plan.category = page.parent.name
            plan.page_type = _frontmatter_type(page.read_text(encoding="utf-8")) or "paper"
            break

    # Files, all optional.
    for candidate in (
        papers_dir() / f"{stem}.pdf",
        supp_dir(stem),
        grade_cache_dir() / stem,
        figures_cache_dir() / stem,
        ingest_dir() / f"{stem}-evolution-proposals",
    ):
        if candidate.exists():
            plan.files.append(candidate)

    # Other pages that mention it.
    for page in _iter_pages():
        if page == plan.page_path:
            continue
        text = page.read_text(encoding="utf-8")
        if stem not in text:
            continue
        page_type = _frontmatter_type(text)

        if page_type in AUTHORED_TYPES:
            # Authored prose: report every citing line, edit none of them.
            for i, line in enumerate(text.splitlines(), start=1):
                if link.search(line):
                    plan.prose_refs.append(
                        ProseRef(path=page, page_type=page_type,
                                 line_no=i, line=line.strip())
                    )
            # A concept hub's `referenced_papers:` spoke registry *is*
            # generated bookkeeping, so it is cleaned even though the page's
            # prose is not.
            if page_type == "concept" and re.search(
                rf"^referenced_papers:.*", text, re.MULTILINE
            ):
                plan.concept_hubs.append(page)
            continue

        if page_type == "commentary":
            # The field's value plus any indented continuation lines (the YAML
            # list form). Scoped that way rather than `.*` with DOTALL, which
            # ran to end-of-file and flagged any commentary whose *body* merely
            # mentioned the stem.
            field_m = re.search(
                r"^primary_paper:[^\n]*(?:\n[ \t]+[^\n]*)*", text, re.MULTILINE
            )
            if field_m and stem in field_m.group(0):
                plan.commentary_pages.append(page)

        n_bullets = len([
            ln for ln in text.splitlines()
            if re.match(r"^\s*[-*]\s+", ln) and link.search(ln)
        ])
        if n_bullets:
            plan.backlink_pages.append((page, n_bullets))

    # index.md bullet. `_link_re` anchors the stem inside the wikilink, so a
    # stem that is a suffix of another (`lee-2025-…` vs `garcia-lee-2025-…`)
    # cannot match the other paper's bullet.
    idx = index_path()
    if idx.exists():
        plan.index_bullet = bool(re.search(
            rf"^\s*[-*]\s+{_link_re(stem).pattern}",
            idx.read_text(encoding="utf-8"), re.MULTILINE,
        ))

    plan.db_rows = _count_db_rows(stem)
    plan.edge_count = _count_edges(stem)
    return plan


def _count_db_rows(stem: str) -> dict[str, int]:
    try:
        from .db.connection import get_connection
    except Exception:
        return {}
    counts: dict[str, int] = {}
    try:
        conn = get_connection()
    except Exception:
        return {}
    try:
        for table, column in (
            ("papers", "stem"),
            ("claims", "paper_stem"),
            ("ingest_iterations", "paper_stem"),
            ("claim_overlap_runs", "paper_stem"),
            # Keyed on claim slugs, so a removed paper's pairs sit under either
            # endpoint column — both need clearing.
            ("cross_paper_judgements", "src_stem"),
            ("cross_paper_judgements", "tgt_stem"),
        ):
            try:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (stem,)
                ).fetchone()[0]
            except sqlite3.Error:
                continue
            if n:
                counts[table] = int(n)
    finally:
        conn.close()
    return counts


def _count_edges(stem: str) -> int:
    try:
        from .claim_graph.edges import edges_db_path, open_edges_db
    except Exception:
        return 0
    if not edges_db_path().exists():
        return 0
    try:
        conn = open_edges_db()
    except Exception:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_stem = ? OR tgt_stem = ?",
            (stem, stem),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# ---------- apply ----------

@dataclass
class RemovalResult:
    removed_files: list[Path] = field(default_factory=list)
    backlinks_removed: int = 0
    index_bullet_removed: bool = False
    concept_hubs_updated: list[Path] = field(default_factory=list)
    db_rows_deleted: dict[str, int] = field(default_factory=dict)
    edges_deleted: int = 0
    warnings: list[str] = field(default_factory=list)


def apply(plan: RemovalPlan, *, keep_pdf: bool = False) -> RemovalResult:
    """Execute `plan`. Wrapped in a mutation journal by the caller."""
    res = RemovalResult()

    for path in plan.files:
        if keep_pdf and path.suffix == ".pdf":
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        res.removed_files.append(path)

    if plan.page_path and plan.page_path.exists():
        plan.page_path.unlink()
        res.removed_files.append(plan.page_path)

    from .backlinks import remove_related_paper
    for page, _ in plan.backlink_pages:
        res.backlinks_removed += remove_related_paper(page, plan.stem)

    if plan.index_bullet:
        res.index_bullet_removed = _strip_index_bullet(plan.stem)

    for hub in plan.concept_hubs:
        if _strip_concept_spoke(hub, plan.stem):
            res.concept_hubs_updated.append(hub)

    res.db_rows_deleted = _delete_db_rows(plan.stem)
    res.edges_deleted = _delete_edges(plan.stem)
    return res


def _strip_index_bullet(stem: str) -> bool:
    from .fsatomic import update_locked

    # Same anchoring as `scan`'s probe: the wikilink must resolve to exactly
    # this stem, or a suffix-collision would delete another paper's bullet.
    pattern = re.compile(
        rf"^\s*[-*]\s+{_link_re(stem).pattern}.*$\n?", re.MULTILINE
    )

    def mutate(text: str) -> str:
        return pattern.sub("", text)

    return update_locked(index_path(), mutate, missing_ok=True)


def _strip_concept_spoke(hub: Path, stem: str) -> bool:
    """Drop the stem from a concept hub's `referenced_papers:` and its spoke
    bullet, and recompute `concept_span`.

    The spoke list is generated bookkeeping, unlike the hub's Definition prose,
    which this leaves alone.
    """
    from .fsatomic import write_text_atomic

    text = hub.read_text(encoding="utf-8")
    original = text
    link = _link_re(stem)

    # `referenced_papers: ["[[a/b]]", …]` (inline) or a YAML block list.
    def _drop_inline(m: re.Match) -> str:
        items = [
            it for it in re.findall(r'"([^"]+)"', m.group(1))
            if not link.search(it)
        ]
        return "referenced_papers: [" + ", ".join(f'"{i}"' for i in items) + "]"

    text = re.sub(r"^referenced_papers:\s*\[(.*?)\]", _drop_inline, text,
                  flags=re.MULTILINE | re.DOTALL)
    # `_link_re` anchors the stem inside the wikilink; the earlier substring
    # match (`[^\]]*stem[^\]]*`) also stripped spokes for any stem that merely
    # *contained* this one.
    text = re.sub(
        rf"^\s*-\s+\"?{_link_re(stem).pattern}\"?\s*$\n?",
        "", text, flags=re.MULTILINE,
    )
    # Spoke bullet in the body.
    text = re.sub(
        rf"^\s*[-*]\s+.*{_link_re(stem).pattern}.*$\n?",
        "", text, flags=re.MULTILINE,
    )

    # `concept_span` counts the distinct categories the spokes cover; with one
    # gone the count can only stay the same or fall, and a stale value would
    # misreport the hub's bridging value.
    remaining = set(re.findall(r'\[\[([^\]/#|]+)/[^\]]+\]\]', text))
    if remaining:
        text = re.sub(r"^concept_span:.*$", f"concept_span: {len(remaining)}",
                      text, flags=re.MULTILINE)

    if text == original:
        return False
    write_text_atomic(hub, text)
    return True


def _delete_db_rows(stem: str) -> dict[str, int]:
    from .db.connection import get_connection

    deleted: dict[str, int] = {}
    conn = get_connection()
    try:
        with conn:
            for table, column in (
                ("claim_overlap_runs", "paper_stem"),
                ("cross_paper_judgements", "src_stem"),
                ("cross_paper_judgements", "tgt_stem"),
                ("ingest_iterations", "paper_stem"),
                # `papers` last: claims CASCADE off it, so deleting it first
                # would make the claims count unobservable.
                ("claims", "paper_stem"),
                ("papers", "stem"),
            ):
                try:
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE {column} = ?", (stem,)
                    )
                except sqlite3.Error:
                    continue
                if cur.rowcount > 0:
                    deleted[table] = cur.rowcount
    finally:
        conn.close()
    return deleted


def _delete_edges(stem: str) -> int:
    from .claim_graph.edges import edges_db_path, open_edges_db

    if not edges_db_path().exists():
        return 0
    conn = open_edges_db()
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM edges WHERE src_stem = ? OR tgt_stem = ?", (stem, stem)
            )
            return cur.rowcount or 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
