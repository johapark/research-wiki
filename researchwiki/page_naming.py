"""Names and content categories shared by synthesis and concept scaffolding."""

from collections import Counter

from .categories import PAGE_TYPE_DIRS
from .stems import slugify_phrase


def dominant_category(referenced: list[str]) -> str:
    """Content category for a synthesis page: the most common category prefix
    among its referenced papers (page-type dirs excluded).

    A synthesis lives in `wiki/synthesis/` (its page *type*, carried by
    `type:` and the directory), so its YAML `category:` should name the
    *content* field it surveys — classified the same way a paper page is,
    which for a cross-paper page is just the field its papers belong to.
    Falls back to `TODO` when no referenced paper carries a content-category
    prefix (the author fills it), mirroring the paper-page auto-suggest
    fallback. Ties break toward the first-referenced category (Counter keeps
    insertion order)."""
    cats: Counter[str] = Counter()
    for r in referenced:
        prefix = r.split("/", 1)[0] if "/" in r else ""
        if prefix and prefix not in PAGE_TYPE_DIRS:
            cats[prefix] += 1
    return cats.most_common(1)[0][0] if cats else "TODO"


def page_slug(title: str) -> str:
    """Page slug for a synthesis title. Thin alias over the shared helper —
    `concepts.candidates._term_slug` must produce byte-identical output for the
    same input, or a scaffolded hub's filename stops matching the edge that
    points at it. Sharing one implementation is what enforces that."""
    return slugify_phrase(title)
