"""Synced Markdown proposal records and their derived feedback history.

Proposal pages are canonical.  SQLite mirrors their structured fields for fast
filtering, but contains no proposal or feedback information that cannot be
recreated by ``researchwiki db rebuild``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .fsatomic import exclusive_lock, update_locked, write_text_atomic
from .paths import wiki_dir
from .wiki import Page, commit_page, extract_section, read_page, read_pages


DIRECTIONS = frozenset({
    "tension",
    "shared-mechanism",
    "complementary-limitations",
    "boundary-condition",
    "cross-category-application",
})
PROPOSED_PAGE_TYPES = frozenset({"synthesis", "idea"})
STATUSES = frozenset({
    "proposed", "shortlisted", "deferred", "rejected", "drafted", "published",
})

_FEEDBACK_RE = re.compile(
    r"^###\s+(fb-[a-f0-9]{12})\s+—\s+([a-z-]+)\s*$\n"
    r"- Created:\s*(.+?)\s*$\n"
    r"- Actor:\s*(.+?)\s*$\n\s*"
    r"(.*?)(?=^###\s+fb-[a-f0-9]{12}\s+—|\Z)",
    re.MULTILINE | re.DOTALL,
)
# A reason line that could open a ledger entry. Written with a leading
# backslash, which renders as the literal heading text in Markdown and is not
# an entry boundary to `_FEEDBACK_RE`; `parse_feedback` strips it back off.
_ENTRY_LIKE_LINE = re.compile(r"^(\\*)(###[ \t]+fb-)", re.MULTILINE)
_TOP_LEVEL_KEY = re.compile(r"^([A-Za-z_][\w-]*)[ \t]*:")


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _timestamp_key(value: str) -> float:
    """Sortable instant for an ISO timestamp; unparseable values sort first.

    Records sync between machines in different time zones, so the offset must
    be honoured: `09:00+09:00` precedes `20:00-07:00` the previous day, which a
    string comparison gets backwards.
    """
    try:
        moment = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return float("-inf")
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.timestamp()


def _escape_reason(reason: str) -> str:
    return _ENTRY_LIKE_LINE.sub(lambda m: "\\" + m.group(1) + m.group(2), reason)


def _unescape_reason(reason: str) -> str:
    return re.sub(r"^\\(\\*###[ \t]+fb-)", r"\1", reason, flags=re.MULTILINE)


def _set_frontmatter_fields(text: str, values: dict[str, str]) -> str:
    """Set top-level YAML keys inside the leading frontmatter only.

    Line-based rather than a regex over the whole file: a body line such as
    `status: …` is never touched, an emptied `key:` never absorbs the next
    line, and values are inserted verbatim (a regex replacement template would
    reinterpret the backslashes JSON escaping produces). A missing key is
    added before the closing fence. Each value must already be a YAML scalar.
    """
    if not text.startswith("---\n"):
        raise ValueError("proposal page has no YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("proposal page has unterminated YAML frontmatter")
    lines = text[4:end].split("\n")
    remaining = dict(values)
    out: list[str] = []
    skipping = False
    for line in lines:
        match = _TOP_LEVEL_KEY.match(line)
        if match:
            key = match.group(1)
            skipping = False
            if key in values:
                if key in remaining:
                    out.append(f"{key}: {remaining.pop(key)}")
                # A continuation of a replaced multi-line value is dropped
                # with it; a duplicate key is dropped so YAML reads one value.
                skipping = True
                continue
        elif skipping and (not line.strip() or line[:1] in " \t-"):
            continue
        else:
            skipping = False
        out.append(line)
    out.extend(f"{key}: {value}" for key, value in remaining.items())
    return "---\n" + "\n".join(out) + text[end:]


def _yaml_string(value: str) -> str:
    """JSON strings are valid YAML scalars and preserve punctuation safely."""
    return json.dumps(str(value), ensure_ascii=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64].rstrip("-") or "proposal"


def evidence_fingerprint(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        rows.append("\t".join([
            str(item.get("paper_stem") or ""),
            str(item.get("claim_slug") or ""),
            str(item.get("text") or "").strip(),
        ]))
    return hashlib.blake2s("\n".join(sorted(rows)).encode("utf-8"),
                           digest_size=12).hexdigest()


@dataclass(frozen=True)
class Feedback:
    feedback_id: str
    decision: str
    created_at: str
    actor: str
    reason: str


@dataclass
class ProposalRecord:
    proposal_id: str
    stem: str
    path: Path
    title: str
    proposed_page_type: str
    direction: str
    status: str
    question: str
    thesis: str
    topic_seed: str
    target_category: str
    source_fingerprint: str
    created_at: str
    updated_at: str
    author_model: str
    parent_proposal: str = ""
    resulting_page: str = ""
    feedback: list[Feedback] = field(default_factory=list)


def parse_feedback(body: str) -> list[Feedback]:
    # Feedback is the terminal, append-only ledger. Reasons may contain Markdown
    # headings, so the ordinary H2 section extractor would truncate a reason and
    # hide every subsequent event. Reading to EOF also recovers existing entries.
    heading = re.search(r"^##[ \t]+Feedback[ \t]*$", body, re.IGNORECASE | re.MULTILINE)
    if heading is None:
        return []
    section = body[heading.end():]
    out: list[Feedback] = []
    for match in _FEEDBACK_RE.finditer(section):
        out.append(Feedback(
            feedback_id=match.group(1),
            decision=match.group(2).strip(),
            created_at=match.group(3).strip(),
            actor=match.group(4).strip(),
            reason=_unescape_reason(match.group(5).strip()),
        ))
    return out


def parse_proposal(page: Page) -> ProposalRecord | None:
    if str(page.fm.get("type") or "").strip() != "proposal":
        return None
    proposal_id = str(page.fm.get("proposal_id") or "").strip()
    if not proposal_id:
        return None
    return ProposalRecord(
        proposal_id=proposal_id,
        stem=page.stem,
        path=page.path,
        title=page.str_field("title"),
        proposed_page_type=page.str_field("proposed_page_type"),
        direction=page.str_field("direction"),
        status=page.str_field("status", "proposed"),
        question=extract_section(page.body, "Question").strip(),
        thesis=extract_section(page.body, "Provisional thesis or hypothesis").strip(),
        topic_seed=page.str_field("topic_seed"),
        target_category=(page.list_field("category") or [""])[0],
        source_fingerprint=page.str_field("source_fingerprint"),
        created_at=page.str_field("created_at"),
        updated_at=page.str_field("updated_at"),
        author_model=page.str_field("author_model"),
        parent_proposal=page.str_field("parent_proposal"),
        resulting_page=page.str_field("resulting_page"),
        feedback=parse_feedback(page.body),
    )


def load_proposals(pages: list[Page] | None = None) -> list[ProposalRecord]:
    """Parsed proposal records, most recently updated first.

    Pass `pages` when the caller has already walked the wiki, so one command
    does not parse every page several times over.
    """
    records = []
    for page in pages if pages is not None else read_pages():
        record = parse_proposal(page)
        if record is not None:
            records.append(record)
    return sorted(
        records,
        key=lambda r: (_timestamp_key(r.updated_at or r.created_at), r.proposal_id),
        reverse=True,
    )


def find_proposal(identifier: str) -> ProposalRecord | None:
    wanted = identifier.strip()
    for record in _proposals_on_disk():
        if wanted in {record.proposal_id, record.stem}:
            return record
    return None


def _render_connections(connections: list[dict], evidence: dict[str, dict]) -> str:
    lines: list[str] = []
    for connection in connections:
        insight = str(connection.get("insight") or "").strip()
        ids = [str(v) for v in connection.get("evidence_ids") or []]
        citations = []
        for evidence_id in ids:
            item = evidence.get(evidence_id)
            if not item:
                continue
            stem = str(item.get("paper_stem") or "").strip()
            slug = str(item.get("claim_slug") or "").strip()
            citations.append(f"[[{stem}#{slug}]]" if slug else f"[[{stem}]]")
        if insight:
            suffix = " " + " ".join(citations) if citations else ""
            lines.append(f"- {insight}{suffix}")
    return "\n".join(lines) or "_(No evidence connection recorded.)_"


def render_proposal(
    *,
    proposal_id: str,
    stem: str,
    proposal: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    topic_seed: str,
    target_category: str,
    author_model: str,
    created_at: str,
    parent_proposal: str = "",
) -> str:
    direction = str(proposal["direction"])
    evidence = {str(i.get("id")): i for i in evidence_items}
    outline = "\n".join(
        f"- {str(item).strip()}" for item in proposal.get("outline") or []
        if str(item).strip()
    ) or "_(To be developed after selection.)_"

    transfer = ""
    if direction == "cross-category-application":
        fields = [
            ("Source method", "source_method"),
            ("Target problem", "target_problem"),
            ("Transfer mapping", "transfer_mapping"),
            ("Mechanism", "mechanism"),
            ("Assumptions to test", "assumptions_to_test"),
            ("Necessary adaptations", "necessary_adaptations"),
            ("Baseline", "baseline"),
            ("First experiment", "first_experiment"),
        ]
        blocks = []
        for label, key in fields:
            value = str(proposal.get(key) or "").strip()
            if value:
                blocks.append(f"### {label}\n\n{value}")
        transfer = "\n\n## Transfer mapping\n\n" + "\n\n".join(blocks)

    testable_design = ""
    if str(proposal["page_type"]) == "idea" and direction != "cross-category-application":
        fields = [
            ("Mechanism", "mechanism"),
            ("Assumptions to test", "assumptions_to_test"),
            ("Baseline", "baseline"),
            ("First experiment", "first_experiment"),
        ]
        blocks = [
            f"### {label}\n\n{str(proposal[key]).strip()}" for label, key in fields
        ]
        testable_design = "\n\n## Testable design\n\n" + "\n\n".join(blocks)

    title = str(proposal["title"]).strip()
    now = created_at
    fm = [
        "---",
        f"title: {_yaml_string(title)}",
        "type: proposal",
        f"proposal_id: {_yaml_string(proposal_id)}",
        f"proposed_page_type: {_yaml_string(str(proposal['page_type']))}",
        f"direction: {_yaml_string(direction)}",
        "status: proposed",
        f"category: [{target_category or 'other'}]",
        f"topic_seed: {_yaml_string(topic_seed)}",
        f"source_fingerprint: {_yaml_string(evidence_fingerprint(evidence_items))}",
        f"parent_proposal: {_yaml_string(parent_proposal)}",
        'resulting_page: ""',
        f"author_model: {_yaml_string(author_model)}",
        f"created_at: {_yaml_string(now)}",
        f"updated_at: {_yaml_string(now)}",
        f"tags: [proposal, {direction}]",
        "---",
    ]
    body = f"""

## Question

{str(proposal['question']).strip()}

## Provisional thesis or hypothesis

{str(proposal['thesis']).strip()}

## Why this connection matters

{str(proposal.get('why_it_matters') or '').strip()}

## Evidence and reasoning

{_render_connections(proposal.get('evidence_connections') or [], evidence)}

## Distinction from existing pages

{str(proposal.get('distinct_from_existing') or '').strip()}
{transfer}
{testable_design}

## Decisive uncertainty

{str(proposal.get('decisive_uncertainty') or '').strip()}

## Suggested outline

{outline}

## Feedback

<!-- Append feedback with `researchwiki proposals feedback`. -->
"""
    return "\n".join(fm) + body


def _proposals_on_disk() -> list[ProposalRecord]:
    """Proposal records from `wiki/proposals/` only — not a whole-wiki walk."""
    folder = wiki_dir() / "proposals"
    if not folder.is_dir():
        return []
    pages = [page for page in (read_page(md) for md in sorted(folder.glob("*.md")))
             if page is not None]
    return load_proposals(pages)


def create_proposal(
    proposal: dict[str, Any],
    *,
    evidence_items: list[dict[str, Any]],
    topic_seed: str,
    target_category: str,
    author_model: str,
    parent_proposal: str = "",
    acceptance_id: str = "",
    existing: dict[str, ProposalRecord] | None = None,
) -> ProposalRecord:
    """Write one proposal page, or return the one an earlier accept wrote.

    Runs under a lock on the proposals directory, so two concurrent accepts of
    one receipt cannot both miss the existing record and write it twice, and a
    page that already exists at the target path is never overwritten: that
    would discard feedback appended since. `existing` lets a caller accepting
    several entries scan the directory once rather than per entry.
    """
    validate_proposal(proposal, {str(i.get("id")) for i in evidence_items})
    short_id = acceptance_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[0-9a-f]{12}", short_id):
        raise ValueError("invalid acceptance id")
    proposal_id = f"prop-{short_id}"
    folder = wiki_dir() / "proposals"
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(folder / ".proposals"):
        known = existing if existing is not None else {
            record.proposal_id: record for record in _proposals_on_disk()
        }
        if proposal_id in known:
            commit_page(known[proposal_id].path)
            return known[proposal_id]
        stem = f"{_slugify(str(proposal['title']))}--{short_id[:8]}"
        out = folder / f"{stem}.md"
        if out.exists() or out.is_symlink():
            # Same title slug and id prefix as an unrelated record (or a record
            # the caller's snapshot predates). Never clobber a ledger page.
            stem = f"{_slugify(str(proposal['title']))}--{short_id}"
            out = folder / f"{stem}.md"
            if out.exists() or out.is_symlink():
                raise ValueError(f"refusing to overwrite existing proposal page: {out}")
        text = render_proposal(
            proposal_id=proposal_id,
            stem=stem,
            proposal=proposal,
            evidence_items=evidence_items,
            topic_seed=topic_seed,
            target_category=target_category,
            author_model=author_model,
            created_at=_now(),
            parent_proposal=parent_proposal,
        )
        write_text_atomic(out, text)
    commit_page(out)
    page = read_page(out)
    record = parse_proposal(page) if page else None
    if record is None:  # pragma: no cover - renderer and parser share contract
        raise ValueError(f"could not parse newly written proposal: {out}")
    if existing is not None:
        existing[record.proposal_id] = record
    return record


def append_feedback(
    identifier: str,
    *,
    decision: str,
    reason: str,
    actor: str = "user",
    resulting_page: str = "",
) -> Feedback:
    decision = decision.strip().lower()
    if decision not in STATUSES:
        raise ValueError(f"decision must be one of: {', '.join(sorted(STATUSES))}")
    record = find_proposal(identifier)
    if record is None:
        raise ValueError(f"proposal not found: {identifier}")
    if not reason.strip():
        raise ValueError("feedback reason must not be empty")
    feedback = Feedback(
        feedback_id=f"fb-{uuid.uuid4().hex[:12]}",
        decision=decision,
        created_at=_now(),
        actor=" ".join(actor.split()) or "user",
        reason=reason.strip(),
    )

    fields = {
        "status": decision,
        "updated_at": _yaml_string(feedback.created_at),
    }
    if resulting_page.strip():
        fields["resulting_page"] = _yaml_string(resulting_page.strip())

    def _mutate(text: str) -> str:
        updated = _set_frontmatter_fields(text, fields)
        # The actor sits on a single `- Actor:` line; a newline would break the
        # entry, so it is flattened. The reason may span lines and headings,
        # but a line shaped like an entry heading is escaped so it cannot
        # forge a second decision.
        entry = (
            f"\n### {feedback.feedback_id} — {feedback.decision}\n"
            f"- Created: {feedback.created_at}\n"
            f"- Actor: {' '.join(feedback.actor.split())}\n\n"
            f"{_escape_reason(feedback.reason)}\n"
        )
        return updated.rstrip() + "\n" + entry

    update_locked(record.path, _mutate, missing_ok=False)
    commit_page(record.path)
    return feedback


def validate_proposal(proposal: dict[str, Any], evidence_ids: set[str]) -> None:
    required = (
        "title", "page_type", "direction", "question", "thesis",
        "why_it_matters", "distinct_from_existing", "decisive_uncertainty",
    )
    missing = [key for key in required if not isinstance(proposal.get(key), str)
               or not proposal[key].strip()]
    if missing:
        raise ValueError(f"proposal missing required fields: {', '.join(missing)}")
    for key in ("source_method", "target_problem", "transfer_mapping", "mechanism",
                "assumptions_to_test", "necessary_adaptations", "baseline",
                "first_experiment", "parent_proposal"):
        if proposal.get(key) is not None and not isinstance(proposal[key], str):
            raise ValueError(f"{key} must be a string")
    if proposal["page_type"] not in PROPOSED_PAGE_TYPES:
        raise ValueError("page_type must be synthesis or idea")
    if proposal["direction"] not in DIRECTIONS:
        raise ValueError(f"unknown proposal direction: {proposal['direction']}")
    connections = proposal.get("evidence_connections") or []
    if not isinstance(connections, list) or not connections:
        raise ValueError("proposal needs at least one evidence connection")
    for connection in connections:
        if not isinstance(connection, dict):
            raise ValueError("evidence connection must be an object")
        if not isinstance(connection.get("insight"), str) or not connection["insight"].strip():
            raise ValueError("evidence connection needs an insight")
        if (not isinstance(connection.get("evidence_ids"), list)
                or not connection["evidence_ids"]
                or any(not isinstance(v, str) for v in connection["evidence_ids"])):
            raise ValueError("evidence connection needs at least one evidence id")
        unknown = set(str(v) for v in connection.get("evidence_ids") or []) - evidence_ids
        if unknown:
            raise ValueError(f"proposal references unknown evidence ids: {sorted(unknown)}")
    if not isinstance(proposal.get("outline", []), list) or any(
        not isinstance(item, str) for item in proposal.get("outline", [])
    ):
        raise ValueError("outline must be a list of strings")
    if proposal["direction"] == "cross-category-application":
        if proposal["page_type"] != "idea":
            raise ValueError("cross-category application must propose an idea page")
        transfer_fields = (
            "source_method", "target_problem", "transfer_mapping",
            "assumptions_to_test", "necessary_adaptations", "baseline",
            "first_experiment",
        )
        missing_transfer = [
            key for key in transfer_fields if not str(proposal.get(key) or "").strip()
        ]
        if missing_transfer:
            raise ValueError(
                "cross-category proposal missing: " + ", ".join(missing_transfer)
            )
    if proposal["page_type"] == "idea":
        idea_fields = ("mechanism", "assumptions_to_test", "baseline", "first_experiment")
        missing_idea = [
            key for key in idea_fields if not str(proposal.get(key) or "").strip()
        ]
        if missing_idea:
            raise ValueError("idea proposal missing: " + ", ".join(missing_idea))
