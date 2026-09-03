"""Category selection shared by canonical and sandbox promotion paths."""

from __future__ import annotations

from ..errors import EnvironmentFailure


def suggest_category_for_page(
    title: str, summary: str, *, use_llm: bool = True,
) -> tuple[str | None, str]:
    """Return ``(category, strength)`` or ``(None, "none")`` on local misses.

    A typed provider failure is not an abstention: callers must stop before a
    canonical page is silently filed in ``other``.
    """
    try:
        from ..search import (
            SearchBackendUnavailable,
            get_default_backend,
            suggest_category,
            suggest_category_knn,
        )
    except ImportError:
        return None, "none"
    try:
        backend = get_default_backend()
    except SearchBackendUnavailable:
        return None, "none"
    try:
        suggestion = (suggest_category if use_llm else suggest_category_knn)(
            backend, title, summary,
        )
    except EnvironmentFailure:
        raise
    except Exception:
        return None, "none"
    if suggestion is None:
        return None, "none"
    return suggestion.category, suggestion.strength
