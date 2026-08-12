"""Land assessed pages into `wiki/` + `papers/`. No LLM, no tokens.

Per page, in this order, each step journaled and individually idempotent:

    stage -> headings -> frontmatter -> relink -> pdf -> page -> commit

**Why the order is not negotiable.** `claim_slug` is content-addressed on
`(section, normalized text)` (`db/rebuild.py:212-217`), and `_upsert_claims` NULLs
*every* grader column when the text at a `(section, position)` changes
(`:266-288`). `last_graded_at` has exactly two writers and `db query` is
`PRAGMA query_only`, so there is no way to mark a claim graded without re-running
the grader. Therefore every byte of the body reaches its final form on the staged
copy, before the page is ever committed. A heading fixed after grading silently
discards that paper's grading work.

**Why `pdf` precedes `page`.** The reverse crash order leaves a wiki page with no
PDF, which makes `grade` raise `FileNotFoundError` and stores a null `pdf_path`
(`rebuild.py:115-121`). An orphan `papers/{stem}.pdf` is harmless — nothing scans
it — and re-running `apply --run <dir>` completes the page (the journal
makes each step idempotent; there is no `--resume` flag on this command, unlike
`agent ingest`).

**The source corpus is never mutated.** Every rewrite happens on a copy under
`run_dir/staged/`, and `_move_pdf` takes its `shutil.copy2` branch for external
sources. That's the primary rollback path, and it costs nothing.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..fsatomic import file_sha256, write_text_atomic
from ..paths import papers_dir, wiki_dir
from .manifest import RunDir
from .sections import rewrite_headings


@dataclass
class ApplyResult:
    src_key: str
    stem: str
    status: str            # landed | skipped | failed
    reason: str = ""
    target: str = ""
    pdf_status: str = ""


def stage_pdf(src_pdf: Path, stem: str, *, new_doi: str | None,
              already_done: bool) -> tuple[Path, str]:
    """Copy the source PDF to `papers/{stem}.pdf`. -> (target, status).

    Wraps `promote._move_pdf` because that function is **not idempotent for an
    external source**: on resume `src_pdf` still points outside the repo while
    `papers/{stem}.pdf` now exists, so it enters the collision branch, gets
    `unclear` from `classify_pdf_collision`, and raises — a page that succeeded
    once would fail on retry.

    status is `copied` | `already-present` | `conflict`. Only `copied` is
    reversible by `migrate rollback`; `already-present` means the file predates
    this run and must be left alone.
    """
    target = papers_dir() / f"{stem}.pdf"
    if already_done and target.exists():
        return target, "already-present"
    if target.exists():
        if file_sha256(target) == file_sha256(src_pdf):
            return target, "already-present"
        return target, "conflict"

    from ..agents.promote import _move_pdf
    _move_pdf(src_pdf, stem, new_doi=new_doi)
    return target, "copied"


def _relink(body: str, rename_map: dict[str, str], category_of: dict[str, str]) -> str:
    """Repoint `[[old-stem]]` → `[[category/new-stem]]` for renamed pages.

    The rename map is closed over the incoming batch: every rename is known at
    inspect time, and no page already in `wiki/` can link to a page that hasn't
    arrived yet. So this single pass over the staged copies is the whole job —
    nothing under `wiki/` needs rewriting.
    """
    if not rename_map:
        return body

    def _sub(m: re.Match) -> str:
        raw = m.group(1).strip()
        target, _, alias = raw.partition("|")
        target, _, anchor = target.partition("#")
        new = rename_map.get(target.strip())
        if new is None:
            return m.group(0)
        cat = category_of.get(new, "")
        full = f"{cat}/{new}" if cat else new
        rebuilt = full + (f"#{anchor}" if anchor else "") + (f"|{alias}" if alias else "")
        return f"[[{rebuilt}]]"

    return re.sub(r"\[\[([^\]]+)\]\]", _sub, body)


def apply_page(
    record: dict,
    run: RunDir,
    journal: dict,
    *,
    rename_map: dict[str, str],
    category_of: dict[str, str],
    migrated_at: str,
    dry_run: bool = False,
    accept_ambiguous: bool = False,
) -> ApplyResult:
    """Run the pipeline for one manifest record."""
    from .classify import _split_page
    from .frontmatter import map_keys, render_frontmatter

    src_key = record["src_page"]
    src = Path(src_key)
    stem = record["derived_stem"]
    category = record["target_category"]
    page_type = record.get("page_type") or "paper"
    done = RunDir.steps_done(journal, src_key)
    res = ApplyResult(src_key=src_key, stem=stem, status="landed")

    staged = run.staged_dir / f"{stem}.md"

    # 1. stage — the rollback anchor.
    if "stage" not in done or not staged.exists():
        try:
            shutil.copy2(src, staged)
        except Exception as e:
            return ApplyResult(src_key, stem, "failed", f"stage: {e}")
        run.record_step(journal, src_key, "stage", stem=stem, target=str(staged))

    text = staged.read_text(encoding="utf-8")
    fm, body = _split_page(text)

    # 2-4. body rewrites, all in memory, then one write.
    body, _plan = rewrite_headings(body, accept_ambiguous=accept_ambiguous)
    body = _relink(body, rename_map, category_of)
    fmp = map_keys(fm)
    block = render_frontmatter(fmp, stem=stem, category=category,
                              page_type=page_type, migrated_at=migrated_at)
    rebuilt = f"---\n{block}\n---\n\n{body.lstrip(chr(10))}"
    if not rebuilt.endswith("\n"):
        rebuilt += "\n"
    write_text_atomic(staged, rebuilt)
    for step in ("headings", "frontmatter", "relink"):
        run.record_step(journal, src_key, step)

    if dry_run:
        res.status = "skipped"
        res.reason = "dry run — staged only"
        res.target = str(staged)
        return res

    # 5. pdf, before the page.
    src_pdf = record.get("src_pdf")
    if src_pdf:
        try:
            _target, pdf_status = stage_pdf(
                Path(src_pdf), stem,
                new_doi=str(fmp.mapped.get("doi") or "") or None,
                already_done=done.get("pdf") in ("copied", "already-present"),
            )
        except Exception as e:
            return ApplyResult(src_key, stem, "failed", f"pdf: {e}")
        if pdf_status == "conflict":
            return ApplyResult(
                src_key, stem, "failed",
                f"papers/{stem}.pdf exists with different content — resolve by hand",
            )
        res.pdf_status = pdf_status
        run.record_step(journal, src_key, "pdf", pdf_status)

    # 6. page.
    target = wiki_dir() / category / f"{stem}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, staged.read_text(encoding="utf-8"))
    res.target = str(target)
    run.record_step(journal, src_key, "page", landed_sha256=file_sha256(target),
                    target=str(target))

    # 7. commit — DB row + claim extraction. Last, and no LLM.
    from ..wiki import commit_page
    commit_page(target)
    run.record_step(journal, src_key, "commit")
    return res


def build_rename_map(records: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """-> (old filename stem → new stem, new stem → category) for actionable items."""
    rename: dict[str, str] = {}
    category_of: dict[str, str] = {}
    for r in records:
        stem = r.get("derived_stem")
        if not stem:
            continue
        old = Path(r["src_page"]).stem
        if old != stem:
            rename[old] = stem
        category_of[stem] = r.get("target_category") or "other"
    return rename, category_of
