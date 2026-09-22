"""Shared author-provenance vocabulary for page writers, lint, and migration."""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Mapping


LEGACY_AUTHOR_PROVENANCE = "legacy-unrecorded"
REFERENCE_PAGE_TYPES = frozenset({"guidance", "protocol", "whitepaper", "book"})
AUTHORED_PAGE_TYPES = frozenset(
    {
        "paper",
        "commentary",
        "synthesis",
        "concept",
        "idea",
        *REFERENCE_PAGE_TYPES,
    }
)
AUTHOR_MODEL_PLACEHOLDERS = frozenset(
    {"", "todo", "tbd", "unknown", "none", "null", "exact-model-id"}
)
AUTHOR_MODEL_GENERIC_LABELS = frozenset(
    {"model", "llm", "openai", "anthropic", "google", "codex"}
)
# A bare family/version does not identify the tier or variant that authored the
# prose. Keep this intentionally shape-based: exact ids such as
# ``gpt-5.6-sol`` and ``claude-opus-4-7`` pass without maintaining a registry.
_GENERIC_MODEL_FAMILY = re.compile(
    r"(?<![\w.-])(?:gpt|claude|gemini|llama|qwen)[-_]?\d+(?:\.\d+)?(?![-\w.])",
    flags=re.IGNORECASE,
)


def normalized_author_model(value: Any) -> str:
    """Return a stripped model id, or ``""`` for placeholders."""
    model = str(value or "").strip().strip("\"'")
    return "" if model.lower() in AUTHOR_MODEL_PLACEHOLDERS else model


def specific_author_model(value: Any) -> str:
    """Return an exact model id, or ``""`` for generic/placeholder values."""
    model = normalized_author_model(value)
    if (
        not model
        or model.lower() in AUTHOR_MODEL_GENERIC_LABELS
        or _GENERIC_MODEL_FAMILY.search(model)
    ):
        return ""
    return model


def has_usable_author_model(frontmatter: Mapping[str, Any]) -> bool:
    return bool(specific_author_model(frontmatter.get("author_model")))


def authored_page_type(frontmatter: Mapping[str, Any]) -> bool:
    """Whether the page type participates in the author-provenance contract."""
    page_type = str(frontmatter.get("type") or "paper").strip().strip("\"'")
    return page_type in AUTHORED_PAGE_TYPES


def author_provenance_required(frontmatter: Mapping[str, Any]) -> bool:
    """Whether the document type requires an exact author model.

    Every content document participates, including ideas and paper/commentary
    pages that predate the ingest timestamp. Mechanically maintained ``meta``
    and ``dashboard`` pages are intentionally outside this contract.
    """
    return authored_page_type(frontmatter)


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


def author_model_requirement_satisfied(frontmatter: Mapping[str, Any]) -> bool:
    """Whether a page may pass an authored-document completion gate."""
    return (
        not author_provenance_required(frontmatter)
        or has_usable_author_model(frontmatter)
        or is_acknowledged_legacy(frontmatter)
    )
