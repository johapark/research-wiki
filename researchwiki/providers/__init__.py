"""Scholarly-database providers.

Each provider implements `ScholarlyDatabaseProvider` and exposes a common DTO
(`ScholarlyArticle`). Only structural metadata + verbatim `abstract` / `tldr`
is allowed per CLAUDE.md Rule 1.
"""

from .base import ScholarlyArticle, ScholarlyDatabaseProvider
from .semantic_scholar import SemanticScholarProvider

__all__ = [
    "ScholarlyArticle",
    "ScholarlyDatabaseProvider",
    "SemanticScholarProvider",
    "get_default_provider",
]


def get_default_provider(
    log_tag: str = "ingest",
    *,
    force_refresh_days: int | None = None,
) -> ScholarlyDatabaseProvider:
    """Return the default provider instance.

    Today: Semantic Scholar (no API key required for modest usage).
    Future: selectable via config.yml.

    `force_refresh_days` is forwarded to the provider's cache reader. None
    (default) honors caches as written; 0 busts everything; N>0 bypasses
    entries older than N days. Used by `audit --refresh-cache`.
    """
    return SemanticScholarProvider(log_tag=log_tag, force_refresh_days=force_refresh_days)
