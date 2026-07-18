"""Content categories — derived from the local `wiki/` tree, not predefined.

There is **no shipped domain taxonomy**. The valid content categories are
exactly the content subdirectories under `wiki/` — each exists because a
`wiki/<category>/` directory was explicitly created for it. Categories are
per-user and local (the whole `wiki/` tree is gitignored); a fresh clone
starts with none and derives its own from the user's papers.

Only the universal *scaffold* is fixed in code:

  - `PAGE_TYPE_DIRS` — `synthesis`, `ideas`, `references`. Structural page-type
    directories present in every wiki. They hold their own page types and are
    NEVER content categories: the research-paper classifier must not target
    them, and `is_valid()` rejects them.
  - `other` — the always-present content-category bucket and the classifier's
    abstention destination.

`DEFAULT_DIRS` (the page-type dirs plus `other`) are seeded on every clone via
committed `wiki/<dir>/.gitkeep` files; their contents stay gitignored.

Growth is deliberate, never implicit: a new content category appears only when
its `wiki/<category>/` directory is explicitly created (by the user or an LLM
agent). `is_valid()` rejects unknown categories so a typo or a classifier whim
can't silently spawn one — see the reject-unless-exists contract below.

Used by `researchwiki/search/__init__.py` to validate LLM-classifier output
(suggestions outside the content set abstain to `other`) and by `ingest`/
`suggest-splits` to validate explicit category arguments.

Also exposes `other_saturation_warning()` — the shared check used by both
`status` and the ingest paths to surface the "wiki/other/ is getting big,
run suggest-splits" nudge. Centralized here so the threshold + decay
constants don't drift across call sites.
"""

from __future__ import annotations

import time
from pathlib import Path

# Structural page-type directories — universal scaffold, NEVER content
# categories. They hold synthesis pages / idea pages / reference docs /
# concept hub notes.
PAGE_TYPE_DIRS: frozenset[str] = frozenset(
    {"synthesis", "ideas", "references", "concepts"}
)

# Default subdirs every wiki has regardless of domain: the page-type dirs plus
# `other` (the always-present content-category abstention bucket). Seeded on
# every clone via committed `wiki/<dir>/.gitkeep`.
DEFAULT_DIRS: frozenset[str] = PAGE_TYPE_DIRS | frozenset({"other"})


def content_categories() -> frozenset[str]:
    """The currently-valid content categories, derived from the local `wiki/`
    directory structure. A category is valid iff `wiki/<category>/` exists.
    Excludes the page-type scaffold dirs; always includes `other`. Reads the
    filesystem on each call (cheap; the set changes only when a dir is created
    or removed) so it never drifts from reality."""
    from .paths import wiki_dir
    cats = {"other"}
    wd = wiki_dir()
    if wd.exists():
        for p in wd.iterdir():
            name = p.name
            if p.is_dir() and name not in PAGE_TYPE_DIRS and not name.startswith("."):
                cats.add(name)
    return frozenset(cats)


def is_valid(category: str) -> bool:
    """Reject-unless-exists check for content categories (case-insensitive).

    Returns True only when `category` is an existing content category — i.e.
    `wiki/<category>/` is present and it isn't a page-type scaffold dir.
    Unknown categories are rejected: a new category must be created explicitly
    (`mkdir wiki/<category>/`, e.g. via `suggest-splits`) before papers can be
    ingested into it. Callers typically abstain to `other` or raise a
    "create it first" error on False."""
    return (category or "").strip().lower() in content_categories()


# `other`-saturation tunables. The warning fires when wiki/other/ has at
# least OTHER_SATURATION_THRESHOLD papers AND the stamp file is absent or
# older than SUGGEST_SPLITS_DECAY_DAYS. Both numbers chosen to balance
# "don't nag on every ingest" against "don't let the bucket grow forever."
OTHER_SATURATION_THRESHOLD = 10
SUGGEST_SPLITS_DECAY_DAYS = 7
SUGGEST_SPLITS_STAMP = ".suggest-splits-stamp"


def _stamp_path() -> Path:
    """Lazy import to avoid importing paths at module load (the wiki_root
    function reads CWD, which would freeze the resolution at import time)."""
    from .paths import wiki_root
    return wiki_root() / SUGGEST_SPLITS_STAMP


def write_stamp() -> None:
    """Touch the dismissal stamp. Called when the warning is surfaced (so it
    won't repeat for the decay window) and when `suggest-splits` runs."""
    _stamp_path().write_text(str(int(time.time())))


def stamp_age_days() -> float | None:
    """Days since the stamp was last written, or None if absent/unreadable."""
    p = _stamp_path()
    if not p.exists():
        return None
    try:
        ts = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    return (time.time() - ts) / 86400.0


def _count_category_pages(category: str) -> int:
    """Number of *.md files in wiki/<category>/. Returns 0 if the dir doesn't
    exist (cold-install case)."""
    from .paths import wiki_dir
    cat_dir = wiki_dir() / category
    if not cat_dir.exists():
        return 0
    return sum(1 for _ in cat_dir.glob("*.md"))


def _count_other_pages() -> int:
    """Number of *.md files in wiki/other/. Returns 0 if the dir doesn't
    exist (cold-install case)."""
    return _count_category_pages("other")


def other_saturation_warning(*, touch: bool = True) -> str | None:
    """Return a warning string if wiki/other/ is saturated and the stamp is
    absent or stale, else None. By default (touch=True) writes the stamp when
    returning non-None — the warning is shown to the user, so subsequent
    surfaces (status, next ingest) suppress for the decay window.

    Pass touch=False to peek without affecting decay state (useful for tests
    or for a UI that wants to compute the warning string but defer the
    "shown" semantics to a later moment).
    """
    n = _count_other_pages()
    if n < OTHER_SATURATION_THRESHOLD:
        return None
    age = stamp_age_days()
    if age is not None and age < SUGGEST_SPLITS_DECAY_DAYS:
        return None
    if touch:
        write_stamp()
    return (
        f"⚠ wiki/other/ has {n} paper(s) — taxonomy may be undersized.\n"
        f"  Run `researchwiki suggest-splits` to propose new categories or reassignments."
    )


# Within-category divergence tunables. Parallel to the `other`-saturation
# machinery above, but for the *opposite* signal: a populated content category
# that has grown a sub-cluster distinct enough to deserve its own sibling
# category. Detection (clustering) lives in
# `researchwiki/tasks/suggest_splits.py`; only the decay stamp lives here so the
# constant stays beside its sibling. Separate stamp file so the two warnings
# decay independently.
CATEGORY_DIVERGENCE_DECAY_DAYS = 7
CATEGORY_DIVERGENCE_STAMP = ".category-divergence-stamp"


def _divergence_stamp_path() -> Path:
    from .paths import wiki_root
    return wiki_root() / CATEGORY_DIVERGENCE_STAMP


def write_divergence_stamp() -> None:
    """Touch the divergence dismissal stamp — called when the status warning is
    surfaced and when `suggest-splits --category/--all` runs, so the nudge
    suppresses for the decay window."""
    _divergence_stamp_path().write_text(str(int(time.time())))


def divergence_stamp_age_days() -> float | None:
    """Days since the divergence stamp was last written, or None if absent."""
    p = _divergence_stamp_path()
    if not p.exists():
        return None
    try:
        ts = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    return (time.time() - ts) / 86400.0
