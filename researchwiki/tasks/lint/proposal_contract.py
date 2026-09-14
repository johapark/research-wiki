"""Structural checks for canonical Markdown proposal records.

Proposal pages are the source of truth for proposal state and feedback; the
SQLite rows are disposable mirrors. These warn-only checks catch shapes that
would otherwise be skipped or incompletely reconstructed by ``db rebuild``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...proposals import DIRECTIONS, PROPOSED_PAGE_TYPES, STATUSES, parse_feedback


REQUIRED_FIELDS = (
    "proposal_id", "proposed_page_type", "direction", "status", "category",
    "topic_seed", "source_fingerprint", "author_model", "created_at", "updated_at",
)
REQUIRED_SECTIONS = (
    "Question", "Provisional thesis or hypothesis", "Why this connection matters",
    "Evidence and reasoning", "Distinction from existing pages",
    "Decisive uncertainty", "Suggested outline", "Feedback",
)
TRANSFER_SECTIONS = (
    "Source method", "Target problem", "Transfer mapping", "Mechanism",
    "Assumptions to test", "Necessary adaptations", "Baseline", "First experiment",
)
IDEA_TEST_SECTIONS = ("Mechanism", "Assumptions to test", "Baseline", "First experiment")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_EVIDENCE_LINK_RE = re.compile(r"\[\[[^\]#]+#[a-z]+-[a-f0-9]+(?:-\d+)?\]\]")
_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{24}$")
_PROPOSAL_ID_RE = re.compile(r"^prop-[a-f0-9]{12}$")


def _value(fm: dict, key: str) -> str:
    value = fm.get(key)
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip().strip("\"'")


def _violation(path: Path, kind: str, detail: str) -> dict:
    return {"page": path, "kind": kind, "detail": detail}


def check_page(path: Path, body: str, fm: dict) -> list[dict]:
    out: list[dict] = []
    for key in REQUIRED_FIELDS:
        if not _value(fm, key):
            out.append(_violation(path, "proposal_missing_field", f"no YAML `{key}:`"))

    page_type = _value(fm, "proposed_page_type")
    direction = _value(fm, "direction")
    status = _value(fm, "status")
    if page_type and page_type not in PROPOSED_PAGE_TYPES:
        out.append(_violation(path, "proposal_invalid_page_type", page_type))
    if direction and direction not in DIRECTIONS:
        out.append(_violation(path, "proposal_invalid_direction", direction))
    if status and status not in STATUSES:
        out.append(_violation(path, "proposal_invalid_status", status))
    fingerprint = _value(fm, "source_fingerprint")
    if fingerprint and not _FINGERPRINT_RE.fullmatch(fingerprint):
        out.append(_violation(path, "proposal_invalid_fingerprint",
                              "`source_fingerprint` must be 24 lowercase hex characters"))
    proposal_id = _value(fm, "proposal_id")
    if proposal_id and not _PROPOSAL_ID_RE.fullmatch(proposal_id):
        out.append(_violation(path, "proposal_invalid_id",
                              "`proposal_id` must match `prop-<12 lowercase hex>`"))

    h2s = [match.group(1).strip() for match in _H2_RE.finditer(body)]
    for heading in REQUIRED_SECTIONS:
        if heading not in h2s:
            out.append(_violation(path, "proposal_missing_section",
                                  f"no `## {heading}` H2 found"))
    if direction == "cross-category-application":
        h3s = [match.group(1).strip() for match in _H3_RE.finditer(body)]
        for heading in TRANSFER_SECTIONS:
            if heading not in h3s:
                out.append(_violation(path, "proposal_missing_transfer_field",
                                      f"no `### {heading}` block found"))
    if page_type == "idea":
        h3s = [match.group(1).strip() for match in _H3_RE.finditer(body)]
        for heading in IDEA_TEST_SECTIONS:
            if heading not in h3s:
                out.append(_violation(path, "proposal_missing_test_field",
                                      f"no `### {heading}` block found"))

    match = re.search(
        r"(?ms)^## Evidence and reasoning\s*$\n(.*?)(?=^##\s|\Z)", body,
    )
    if match and not _EVIDENCE_LINK_RE.search(match.group(1)):
        out.append(_violation(path, "proposal_missing_claim_evidence",
                              "evidence section has no `[[stem#claim_slug]]` citation"))

    feedback_headings = [h for h in _H3_RE.findall(body) if h.startswith("fb-")]
    feedback = parse_feedback(body)
    if len(feedback_headings) != len(feedback):
        out.append(_violation(path, "proposal_malformed_feedback",
                              "one or more `fb-*` entries cannot be parsed"))
    feedback_ids = [item.feedback_id for item in feedback]
    if len(feedback_ids) != len(set(feedback_ids)):
        out.append(_violation(path, "proposal_duplicate_feedback_id",
                              "feedback IDs must be unique within the page"))
    for item in feedback:
        if item.decision not in STATUSES:
            out.append(_violation(path, "proposal_invalid_feedback_status",
                                  f"{item.feedback_id} uses `{item.decision}`"))
    if feedback and status in STATUSES and feedback[-1].decision != status:
        out.append(_violation(path, "proposal_status_feedback_mismatch",
                              f"status is `{status}` but latest feedback is "
                              f"`{feedback[-1].decision}`"))
    return out


def find_proposal_contract_violations(
    pages: list[Path], pages_body: dict[Path, str], pages_fm: dict[Path, dict],
) -> list[dict]:
    out: list[dict] = []
    seen_ids: dict[str, Path] = {}
    for path in pages:
        fm = pages_fm.get(path, {}) or {}
        if path.parent.name != "proposals" or _value(fm, "type") != "proposal":
            continue
        out.extend(check_page(path, pages_body.get(path, ""), fm))
        proposal_id = _value(fm, "proposal_id")
        if proposal_id and proposal_id in seen_ids:
            out.append(_violation(path, "proposal_duplicate_id",
                                  f"also used by `{seen_ids[proposal_id].name}`"))
        elif proposal_id:
            seen_ids[proposal_id] = path
    return out
