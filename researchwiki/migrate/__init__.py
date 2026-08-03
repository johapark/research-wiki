"""Bulk import of LLM-generated paper pages from an older or simpler wiki.

**Target**: one paper-derived markdown page per PDF, produced either by an older
release of this framework, or by a simpler "PDF in → summary page out" generator
(the Karpathy LLM-wiki pattern this project is built on, or a homegrown script).
The two standing assumptions are **one page per paper** and **the PDF is
available** — the PDF is what makes claims gradable and therefore what makes the
page citable at all.

Not for arbitrary note vaults. A page that isn't one-paper-shaped has no claims
to extract and no PDF to grade against, so it lands as an unciteable stub; those
belong in `references/` or a synthesis page, written by hand.

Migration adds this framework's *contract* to pages whose prose already exists.
It never re-authors from the PDF — that's `agent ingest`, and running it over an
imported corpus would overwrite the text you migrated in order to keep.

Modules:
  sections     — H2 heading alias table + rewriter (merge-on-collision)
  frontmatter  — YAML key aliasing, no value fabrication
  classify     — per-page assessment, incl. claims_before -> claims_after
  manifest     — run directory, manifest, journal
  apply        — staged rewrite -> land -> commit_page

Imports are lazy so `researchwiki migrate --help` doesn't pull the DB or the
embedding model. See `prompts/migration-backfill.md` for the workflow.
"""

from __future__ import annotations

__all__ = [
    "SECTION_ALIASES",
    "AMBIGUOUS_HEADINGS",
    "canonical_for",
    "plan_headings",
    "rewrite_headings",
    "FM_ALIASES",
    "map_keys",
]


def __getattr__(name: str):
    if name in ("SECTION_ALIASES", "AMBIGUOUS_HEADINGS", "canonical_for",
                "plan_headings", "rewrite_headings"):
        from . import sections
        return getattr(sections, name)
    if name in ("FM_ALIASES", "map_keys"):
        from . import frontmatter
        return getattr(frontmatter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
