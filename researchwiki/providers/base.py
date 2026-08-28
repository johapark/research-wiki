"""Abstract base class for scholarly-database providers.

A provider maps external paper identifiers (DOI, title) to a common DTO
(`ScholarlyArticle`) and exposes reference / citation / recommendation lists
as the SAME DTO so downstream code is provider-agnostic.

Per CLAUDE.md Rule 1, only structural metadata (ids, refs, citations) plus the
verbatim `abstract` and AI-generated `tldr` fields are allowed to land in the
wiki — `tldr` must be verified against the PDF before it lands in prose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScholarlyArticle:
    """Common DTO for paper metadata returned by any provider.

    Fields mirror the subset that ingest and citation scouting actually consume.
    `raw` keeps the original provider response for debugging.
    """

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    abstract: str = ""
    tldr: str = ""  # AI-generated; must be verified per Rule 1
    external_ids: dict[str, str] = field(default_factory=dict)
    reference_count: int | None = None
    citation_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def doi_lower(self) -> str | None:
        return self.doi.lower() if self.doi else None


class ScholarlyDatabaseProvider(ABC):
    """Abstract provider interface.

    Implementations:
      * Required: `name`, `get_by_doi`, `search_by_title`.
      * Recommended: `get_references`, `get_citations`, `get_recommendations`
        (return `[]` if the provider does not support them).
      * Optional: `get_batch_metadata` (return `{}` if unsupported).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. "semantic-scholar")."""
        ...

    @abstractmethod
    def get_by_doi(self, doi: str) -> ScholarlyArticle | None:
        """Fetch a single paper by DOI. Return None if not found."""
        ...

    @abstractmethod
    def search_by_title(self, title: str) -> ScholarlyArticle | None:
        """Best-effort title match. Return None if no confident hit."""
        ...

    def get_references(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        """Return references cited by `article`. Default: unsupported → []."""
        return []

    def get_citations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        """Return papers citing `article`. Default: unsupported → []."""
        return []

    def get_recommendations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        """Return recommended related papers. Default: unsupported → []."""
        return []

    def get_batch_metadata(self, dois: list[str]) -> dict[str, ScholarlyArticle]:
        """Best-effort batch metadata lookup; default is unsupported."""
        return {}
