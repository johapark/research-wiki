"""Derived-cache and audit helpers for the agent commit phase."""

from __future__ import annotations

from ..log import log
from ..paths import index_path, log_path, wiki_dir


def promotion_decision_reason(result) -> str:
    """Stable audit string for both successful and failed promotion results."""
    reason = (
        f"category={result.category}; wiki={result.wiki_path}; "
        f"pdf={result.pdf_path}; backlinks_added={result.backlinks_added}; "
        f"index_updated={result.index_updated}; log_appended={result.log_appended}"
    )
    if result.warnings:
        reason += f"; warnings={result.warnings}"
    return reason


def update_indexes_after_promotion(result) -> None:
    """Upsert every parseable wiki page changed by a successful promotion.

    Indexes are derived caches, so failure is recorded as a recoverable warning
    rather than rolling back a valid page/PDF commit.
    """
    try:
        from ..index.incremental import update_page_indexes

        changed_pages = [result.wiki_path]
        changed_pages.extend(
            wiki_dir() / f"{key}.md" for key in result.backlinks_added
        )
        if result.index_updated:
            changed_pages.append(index_path())
        if result.log_appended:
            changed_pages.append(log_path())
        outcome = update_page_indexes(changed_pages)
        log(
            f"indexes  → {outcome['n_pages']} page(s) "
            f"(bm25={outcome['bm25_mode']}, semantic={outcome['semantic_mode']})",
            tag="agent",
        )
    except Exception as exc:
        warning = f"incremental index update failed: {exc}; run researchwiki reindex"
        result.warnings.append(warning)
        log(f"indexes  ⚠ {warning}", tag="agent")
