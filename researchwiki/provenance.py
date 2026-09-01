"""Shared author-provenance vocabulary for page writers, lint, and migration."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping


LEGACY_AUTHOR_PROVENANCE = "legacy-unrecorded"
REFERENCE_PAGE_TYPES = frozenset({"guidance", "protocol", "whitepaper", "book"})
AUTHORED_PAGE_TYPES = frozenset(
    {"paper", "commentary", "synthesis", "concept", *REFERENCE_PAGE_TYPES}
)
AUTHOR_MODEL_PLACEHOLDERS = frozenset(
    {"", "todo", "tbd", "unknown", "none", "null", "exact-model-id"}
)


def normalized_author_model(value: Any) -> str:
    """Return a stripped model id, or ``""`` for placeholders."""
    model = str(value or "").strip().strip("\"'")
    return "" if model.lower() in AUTHOR_MODEL_PLACEHOLDERS else model


def has_usable_author_model(frontmatter: Mapping[str, Any]) -> bool:
    return bool(normalized_author_model(frontmatter.get("author_model")))


def authored_page_type(frontmatter: Mapping[str, Any]) -> bool:
    """Whether the page type participates in the author-provenance contract."""
    page_type = str(frontmatter.get("type") or "paper").strip().strip("\"'")
    return page_type in AUTHORED_PAGE_TYPES


def author_provenance_required(frontmatter: Mapping[str, Any]) -> bool:
    """Whether a missing model is actionable rather than pre-contract legacy.

    Paper and commentary pages became attributable when the ingest pipeline
    began stamping ``ingested_at``. Older pages remain outside the automatic
    finding unless a maintainer explicitly reviews them. The other authored
    page types have always been manual, so every one is in scope.
    """
    page_type = str(frontmatter.get("type") or "paper").strip().strip("\"'")
    if page_type in {"paper", "commentary"}:
        return bool(str(frontmatter.get("ingested_at") or "").strip())
    return page_type in {"synthesis", "concept", *REFERENCE_PAGE_TYPES}


def _valid_acknowledged_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value or "").strip())
    except ValueError:
        return False
    return True


def is_acknowledged_legacy(frontmatter: Mapping[str, Any]) -> bool:
    """Whether a page explicitly records genuinely unrecoverable authorship.

    The marker is deliberately narrow. A partial marker, a free-form value, or
    an invalid date does not silence lint. A real ``author_model`` supersedes
    the legacy state and therefore is not reported as acknowledged legacy.
    """
    return (
        not has_usable_author_model(frontmatter)
        and str(frontmatter.get("author_provenance") or "").strip()
        == LEGACY_AUTHOR_PROVENANCE
        and _valid_acknowledged_date(
            frontmatter.get("provenance_acknowledged_at")
        )
    )
