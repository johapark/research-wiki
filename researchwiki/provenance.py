"""Shared author-provenance vocabulary for page writers, lint, and migration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


LEGACY_AUTHOR_PROVENANCE = "legacy-unrecorded"
REFERENCE_PAGE_TYPES = frozenset({"guidance", "protocol", "whitepaper", "book"})
AUTHORED_PAGE_TYPES = frozenset(
    {
        "paper",
        "commentary",
        "synthesis",
        "concept",
        "idea",
        "proposal",
        *REFERENCE_PAGE_TYPES,
    }
)
AUTHOR_MODEL_PLACEHOLDERS = frozenset(
    {"", "todo", "tbd", "unknown", "none", "null", "exact-model-id"}
)
AUTHOR_MODEL_GENERIC_LABELS = frozenset(
    {"model", "llm", "openai", "anthropic", "google", "codex"}
)
# A bare family/version token (``gpt-5.6``, ``claude-4``, ``qwen3``) may not
# identify the tier or variant that authored the prose: gpt-5.6 ships only as
# Sol/Terra/Luna. The shape alone cannot tell such an alias from a vendor's
# genuine bare-version id (``gpt-5.5``, ``gpt-4.1``), so a matching token counts
# as generic only when the pricing table does not list it as an exact model —
# see `_is_family_alias`. Suffixed ids (``gpt-5.6-sol``, ``claude-opus-4-7``)
# never match, and neither does an Ollama ``:tag`` (``qwen3:32b``), which names
# the variant.
_FAMILY_VERSION_TOKEN = re.compile(
    r"(?<![\w.-])(?:gpt|claude|gemini|llama|qwen)[-_]?\d+(?:\.\d+)?(?![-\w.:])",
    flags=re.IGNORECASE,
)


def normalized_author_model(value: Any) -> str:
    """Return a stripped model id, or ``""`` for placeholders."""
    model = str(value or "").strip().strip("\"'")
    return "" if model.lower() in AUTHOR_MODEL_PLACEHOLDERS else model


def _is_family_alias(model: str) -> bool:
    tokens = [m.group(0).lower() for m in _FAMILY_VERSION_TOKEN.finditer(model)]
    if not tokens:
        return False
    from .agents.model_config import priced_model_ids

    known = priced_model_ids()
    return any(token not in known for token in tokens)


def specific_author_model(value: Any) -> str:
    """Return an exact model id, or ``""`` for generic/placeholder values."""
    model = normalized_author_model(value)
    if (
        not model
        or model.lower() in AUTHOR_MODEL_GENERIC_LABELS
        or _is_family_alias(model)
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


def completion_gate_blocker(path: Path) -> tuple[str, str] | None:
    """``(finding, message)`` when a page cannot pass a completion gate.

    Parses the frontmatter strictly rather than through ``wiki.read_page``,
    which maps malformed YAML to ``{}`` so one typo never drops a page from
    search. For a gate that mapping is misleading: ``{}`` defaults to
    ``type: paper`` with no ``author_model``, so a page carrying a valid model
    beside an unrelated typo was told to record the model it already has, and
    fixing the named field changed nothing. The real defect is reported
    instead. ``None`` for text with no frontmatter block, which is not a wiki
    page and has nothing to check.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    try:
        frontmatter = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        # +2: PyYAML counts from 0 within the block; the file adds the fence.
        where = f" at line {mark.line + 2}" if mark is not None else ""
        detail = str(exc).split("\n")[0].strip()
        return (
            "invalid_frontmatter",
            f"{path}: frontmatter is not valid YAML{where} ({detail}); fix it "
            "before completing this page",
        )
    if not isinstance(frontmatter, dict):
        return (
            "invalid_frontmatter",
            f"{path}: frontmatter is not a YAML mapping; fix it before "
            "completing this page",
        )
    if author_model_requirement_satisfied(frontmatter):
        return None
    return (
        "missing_author_model",
        f"{path}: missing or underspecified `author_model`; record the exact "
        "model variant before completing this page",
    )
