"""Search-backend abstraction.

The wiki needs three related capabilities: keyword search, See-Also (document
similarity), and category classification (nearest-neighbor vote). All three
can be served by one full-text index with BM25 scoring, so we define a single
`SearchBackend` ABC that a concrete backend (Tantivy, Whoosh, pure-python)
implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SearchBackendUnavailable(RuntimeError):
    """Raised when the index doesn't exist yet (e.g., pre-reindex first run)."""


@dataclass
class Document:
    """One indexable unit = one wiki page."""
    stem: str                    # e.g. "smith-2024-paper-title-slug"
    category: str                # e.g. "compbio"
    page_type: str               # "paper" | "synthesis" | "guidance" | "whitepaper" | ...
    title: str
    authors: str
    year: int | None
    summary: str                 # the `## Summary` section text (may be empty)
    body: str                    # full page body (post-frontmatter)
    keywords: str = ""           # YAML `keywords:` value verbatim

    @property
    def key(self) -> str:
        return f"{self.category}/{self.stem}"


@dataclass
class SearchHit:
    stem: str
    category: str
    page_type: str
    title: str
    score: float
    snippet: str                 # ~200-char highlight around the match

    @property
    def key(self) -> str:
        return f"{self.category}/{self.stem}"


class SearchBackend(ABC):
    """Abstract full-text / similarity index over wiki pages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def build(self, docs: list[Document]) -> None:
        """Rebuild the index from scratch with the given documents."""

    @abstractmethod
    def add(self, doc: Document) -> None:
        """Incrementally add one document to an existing index."""

    @abstractmethod
    def query(self, q: str, limit: int = 10) -> list[SearchHit]:
        """Ranked keyword search across title / summary / body / authors."""

    @abstractmethod
    def more_like(self, key: str, limit: int = 10) -> list[SearchHit]:
        """Find documents similar to the one identified by `category/stem`.

        The seed document itself is excluded from results.
        """

    @abstractmethod
    def more_like_text(
        self,
        text: str,
        limit: int = 10,
        page_type: str | None = None,
    ) -> list[SearchHit]:
        """Find documents similar to arbitrary text.

        Used by the ingest pipeline to classify a paper that is not yet in the
        index. If `page_type` is given, restrict results to that type.
        """

    @abstractmethod
    def exists(self) -> bool:
        """Return True iff the index has been built and is readable."""
