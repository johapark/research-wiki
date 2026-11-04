"""Bounded caption inventory and conservative, document-scoped nomination."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re

from ..pdf.figures import FigureRef, locate_in_texts

REASONS = ('missing_detail', 'table_alignment', 'visual_evidence', 'conflict')
PRIORITIES = ('critical', 'high', 'normal')


@dataclass(frozen=True)
class VisualEntry:
    ref_id: str
    source: FigureRef
    main_result: bool


@dataclass(frozen=True)
class VisualRequest:
    ref_id: str
    priority: str
    reason: str
    question: str

    def to_dict(self):
        return asdict(self)


def build_inventory(pages: list[str], sections: dict) -> list[VisualEntry]:
    results = sections.get('results') or ''
    if not isinstance(results, str):
        results = str(results)
    entries = []
    for ref in locate_in_texts(pages):
        prefix = ('ed' if ref.extended else '') + ('table' if ref.kind == 'Table' else 'fig')
        label = rf"{'Tables?' if ref.kind == 'Table' else r'Fig(?:ure)?s?'}\.?\s*{ref.number}(?:[a-z])?\b"
        if ref.extended:
            label = r'Extended\s+Data\s+' + label
        # Exclude caption lines: a Results caption alone isn't a prose dependency.
        prose = '\n'.join(line for line in results.splitlines()
                          if not re.match(r'^\s*(?:Extended Data )?(?:Fig|Table)', line))
        entries.append(VisualEntry(f'{prefix}:{ref.number}', ref,
                                  bool(re.search(label, prose, re.I))))
    return sorted(entries, key=lambda e: (not e.main_result, e.source.extended,
                                         e.source.page, e.ref_id))


def inventory_prompt(entries: list[VisualEntry], limit: int = 2000) -> str:
    header = '\n\nVisual inventory (caption location; artwork may be elsewhere):\n'
    rows = []
    for entry in entries:
        ref = entry.source
        row = (f'{entry.ref_id} p{ref.page} repeats={list(ref.also_on)} '
               f'results={entry.main_result}: {ref.caption[:230]}\n')
        if len(header + ''.join(rows) + row) + 50 > limit:
            break
        rows.append(row)
    return header + ''.join(rows) + f'Omitted entries: {len(entries) - len(rows)}\n'


NOMINATION_SYSTEM = """
Also return visual_requests AFTER claims and keywords. Use only supplied visual
inventory IDs. Nominate up to 3 unresolved important questions (160 chars each):
missing_detail, table_alignment, visual_evidence, or conflict. priority is critical,
high, or normal. Questions are not claims; never guess answers, pages, or values.
Nominate main outcomes, controls or ablations whose evidence requires looking at
an image, or table associations lost in text extraction. Return [] if text resolves
the questions. Do not shorten or remove valid claims to fit nominations.
Shape: {"ref_id":"table:2","priority":"high","reason":"table_alignment",
"question":"Which condition corresponds to each column?"}.
"""


def nomination_schema(base: dict) -> dict:
    schema = deepcopy(base)
    schema['required'].append('visual_requests')
    schema['properties']['visual_requests'] = {
        'type': 'array', 'maxItems': 3, 'items': {
            'type': 'object', 'required': ['ref_id', 'priority', 'reason', 'question'],
            'properties': {
                'ref_id': {'type': 'string'},
                'priority': {'type': 'string', 'enum': list(PRIORITIES)},
                'reason': {'type': 'string', 'enum': list(REASONS)},
                'question': {'type': 'string', 'maxLength': 160},
            },
        },
    }
    return schema


def parse_requests(data: dict, entries: list[VisualEntry]) -> tuple[list[VisualRequest], str]:
    raw = data.get('visual_requests')
    if not isinstance(raw, list):
        return [], 'incomplete'
    valid_ids = {entry.ref_id for entry in entries}
    requests, seen, invalid = [], set(), False
    for item in raw:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str)
                for k in ('ref_id', 'priority', 'reason', 'question')):
            invalid = True
            continue
        ref, priority, reason, question = (item[k].strip() for k in
                                           ('ref_id', 'priority', 'reason', 'question'))
        if (ref not in valid_ids or priority not in PRIORITIES or reason not in REASONS
                or not question or len(question) > 160):
            invalid = True
            continue
        key = (ref, question.casefold())
        if key in seen:
            continue
        seen.add(key)
        requests.append(VisualRequest(ref, priority, reason, question))
    requests.sort(key=lambda req: PRIORITIES.index(req.priority))
    return requests[:3], 'incomplete' if invalid or len(requests) > 3 else 'complete'


def choose_requests(entries: list[VisualEntry], nominated: list[VisualRequest],
                    claims: list | None = None) -> list[VisualRequest]:
    """Main-results fallback occupies a slot even if nominations overlook it."""
    def relevance(entry):
        ref = entry.source
        label = rf"{'Tables?' if ref.kind == 'Table' else r'Fig(?:ure)?s?'}\.?\s*{ref.number}(?:[a-z])?\b"
        if ref.extended:
            label = r'Extended\s+Data\s+' + label
        weights = [PRIORITIES.index(claim.importance) for claim in claims or []
                   if claim.importance in PRIORITIES and
                   re.search(label, claim.location + ' ' + claim.content, re.I)]
        return min(weights) if weights else 3
    candidates = [entry for entry in entries if entry.main_result or relevance(entry) == 0]
    main = min(candidates, key=lambda entry: (relevance(entry), entry.source.page, entry.ref_id), default=None)
    requests = list(nominated)
    if main and not any(req.ref_id == main.ref_id for req in requests):
        requests.insert(0, VisualRequest(main.ref_id, 'critical', 'visual_evidence',
                         'What main result is visibly supported, including labels and qualifications?'))
    main_ids = {entry.ref_id for entry in entries if entry.main_result}
    grouped = {}
    for req in requests:
        priority = 'critical' if main and req.ref_id == main.ref_id else req.priority
        if req.ref_id in grouped:
            prior = grouped[req.ref_id]
            priority = min((priority, prior.priority), key=PRIORITIES.index)
            req = VisualRequest(req.ref_id, priority, prior.reason,
                                prior.question + '; ' + req.question)
        else:
            req = VisualRequest(req.ref_id, priority, req.reason, req.question)
        grouped[req.ref_id] = req
    return sorted(grouped.values(), key=lambda req: (
        bool(main and req.ref_id != main.ref_id), PRIORITIES.index(req.priority),
        req.ref_id not in main_ids, req.ref_id))


def missing_result_refs(sections: dict, inventory: list[VisualEntry]) -> list[str]:
    """Explicit Results references without located captions remain review gaps."""
    results = sections.get('results') or ''
    if not isinstance(results, str):
        return []
    known = {entry.ref_id for entry in inventory}
    mentioned = set()
    for match in re.finditer(r'(Extended\s+Data\s+|Supplementary\s+(?:Data\s+)?)?(Fig(?:ure)?s?|Tables?)\.?\s*(\d+)', results, re.I):
        if match[1] and match[1].lower().startswith('supplementary'):
            continue  # exhaustive supplementary inspection is outside this phase
        prefix = ('ed' if match[1] else '') + ('table' if match[2].lower().startswith('table') else 'fig')
        mentioned.add(f'{prefix}:{int(match[3])}')
    return sorted(mentioned - known)
