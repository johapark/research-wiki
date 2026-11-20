"""Locate one EVOLVE edit in a prior draft and splice back its validated body."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from ...grade.parser import SECTION_KEYS
from ..judge import parse_json_response

if TYPE_CHECKING:
    from .revise import CritiqueOutput

# Grade section keys are stable even when authors vary heading capitalization.
_GRADED_HEADINGS = {key: heading for heading, key in SECTION_KEYS}
_HEADING = re.compile(r"^ {0,3}(#{1,2})(?:[ \t]+([^\r\n]*))?[ \t]*\r?$", re.MULTILINE)
# The author contract uses ATX headings. In ambiguous Markdown, let the
# whole-page repair handle the structure instead of guessing byte boundaries.
_AMBIGUOUS = re.compile(r"<!--|^ {0,3}(?:`{3,}|~{3,}|(?:=+|-+)[ \t]*\r?$)", re.MULTILINE)

SECTION_SCHEMA = {
    "type": "object", "required": ["body"], "additionalProperties": False,
    "properties": {"body": {"type": "string"}},
}


@dataclass(frozen=True)
class SectionRepair:
    title: str
    start: int
    end: int

    def apply(self, draft: str, response: str) -> str | None:
        obj = parse_json_response(response)
        if not isinstance(obj, dict) or set(obj) != {"body"} or not isinstance(obj["body"], str):
            return None
        body = obj["body"].replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not body.strip() or _HEADING.search(body) or _AMBIGUOUS.search(body):
            return None
        newline = "\r\n" if "\r\n" in draft[:self.start] else "\n"
        body = body.replace("\n", newline)
        return draft[:self.start] + body + draft[self.end:]


def locate_section(draft: str, critique: CritiqueOutput) -> SectionRepair | None:
    wanted = {_GRADED_HEADINGS.get(claim.section) for claim in critique.weak_claims}
    if critique.coverage_gaps:
        # An anchor's location describes the source PDF. Only the critic can
        # tell us where the missing finding belongs in the wiki draft.
        wanted.add(critique.repair_section.casefold() if critique.repair_section else None)
    if len(wanted) != 1 or None in wanted or _AMBIGUOUS.search(draft):
        return None
    headings = list(_HEADING.finditer(draft))
    matches = [(i, h) for i, h in enumerate(headings)
               if h[1] == "##" and (h[2] or "").strip().casefold() in wanted]
    if len(matches) != 1:
        return None
    i, heading = matches[0]
    end = headings[i + 1].start() if i + 1 < len(headings) else len(draft)
    start = heading.end()
    body = draft[start:end]
    if not body.strip():
        return None
    start += len(body) - len(body.lstrip("\r\n"))
    end -= len(body) - len(body.rstrip("\r\n"))
    return SectionRepair(heading[2].strip(), start, end)
