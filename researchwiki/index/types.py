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
from pathlib import Path

from ..errors import EnvironmentFailure


class SearchBackendUnavailable(EnvironmentFailure):
    """Raised when the index doesn't exist yet (e.g., pre-reindex first run).

    `EnvironmentFailure` (still a `RuntimeError`, so existing handlers are
    unaffected) is what makes this exit 2 rather than 3 when no caller catches
    it — the many callers that *do* catch it to degrade gracefully keep working.
    """


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
        """Incrementally upsert one document, replacing the same stem."""

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


def swap_in_directory(staged: Path, target: Path) -> None:
    """Publish `staged` at `target` so a crash leaves old, new, or nothing.

    `shutil.rmtree(target)` deletes entries in directory-listing order, so a
    crash part-way could keep `index_meta.json` and `.managed.json` while the
    segment files were gone — a directory that passed the reuse check and then
    failed every query. The old index is first *renamed* aside (one atomic
    step), so `target` is either the old complete index or absent, never a
    partial one; only then is the retired copy deleted. The two renames do
    leave a microsecond window where `target` is absent — a reader can see the
    old index, the new one, or briefly nothing, but never a half-built one.

    Lives here, in the dependency-free module both index builders already
    import, because `pdf_chunks` (which pioneered this) pulls in numpy and
    `pages_bm25` must stay light enough for the Windows CI job.
    """
    import os
    import secrets
    import shutil

    if target.is_symlink():
        # Renaming a symlink aside would replace the user's link with a real
        # directory and strand the link's own target. Publish *through* the
        # link instead, so a deliberately relocated index keeps its layout.
        target = Path(os.path.realpath(target))

    retired = None
    if target.exists():
        retired = target.with_name(f".{target.name}-retired-{secrets.token_hex(6)}")
        target.replace(retired)
    staged.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged.replace(target)
    except OSError:
        # The window between the two renames is the only moment `target` does
        # not exist. If publishing fails, put the old index back rather than
        # leaving the caller with nothing.
        if retired is not None and not target.exists():
            retired.replace(target)
            retired = None
        raise
    finally:
        if retired is not None:
            shutil.rmtree(retired, ignore_errors=True)
