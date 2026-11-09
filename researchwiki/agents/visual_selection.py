"""Bounded caption inventory and conservative, document-scoped nomination."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re

from ..pdf.figures import FigureRef, is_caption_line, locate_in_texts

REASONS = ('missing_detail', 'table_alignment', 'visual_evidence', 'conflict')
PRIORITIES = ('critical', 'high', 'normal')


# A figure/table mention in prose. The leading word boundary keeps "notable 3"
# and "stable 24 h" from reading as Table 3 / Table 24; the series prefix is
# captured so a "Supplementary Fig. 2" or "Extended Data Fig. 2" mention is
# never mistaken for main Figure 2 (or vice versa). Target-claim locations also
# use the documented "ED Fig." shorthand. That shorthand is case-sensitive and
# takes no period: "...discharged from the ED. Table 2 lists..." ends a
# sentence on the emergency department and must stay main Table 2.
_SERIES = r'(?:(\b(?:Extended\s+Data|(?-i:ED)|Supplementary(?:\s+Data)?))\s+)?'
_MENTION_RE = re.compile(_SERIES + r'\b(Fig(?:ure)?s?|Tables?)\b\.?\s*(\d+)(?:[a-z])?\b', re.I)


def ref_id_for(kind: str, number: int, extended: bool) -> str:
    return ('ed' if extended else '') + ('table' if kind.lower().startswith('table') else 'fig') + f':{number}'


def mentioned_refs(text: str) -> set[str]:
    """Main and Extended Data ref IDs named in `text`; Supplementary is excluded."""
    found = set()
    for match in _MENTION_RE.finditer(text or ''):
        series = (match[1] or '').lower()
        if series.startswith('supplementary'):
            continue  # exhaustive supplementary inspection is outside this phase
        found.add(ref_id_for(match[2], int(match[3]), bool(series)))
    return found


def mentions(text: str, ref_id: str) -> bool:
    return ref_id in mentioned_refs(text)


def critical_claim_refs(claims: list | None) -> set[str]:
    """References whose critical priority survives missing caption locations."""
    return {ref for claim in claims or [] if claim.importance == 'critical'
            for ref in mentioned_refs(claim.location + ' ' + claim.content)}


# Where a paper's main narrative ends when it prints no Results heading. A
# Nature/Cell article runs Introduction-less body prose straight into
# "Discussion"; "Online content", "Methods" and the back matter follow.
_BODY_END_RE = re.compile(
    r'(?im)^\s*(?:\d+[.\s]\s*)?(?:discussion|conclusions?|online\s+content|methods|'
    r'materials\s+and\s+methods|references|bibliography|acknowledg(?:e)?ments?|'
    r'author\s+contributions|reporting\s+summary)\s*$')
_CONTRIBUTIONS_RE = re.compile(r'(?:\b[A-Z]\.(?:[A-Z]\.)*-?[A-Z]?\.?,\s*){4,}')


def visual_source_sections(full_text: str) -> tuple[dict, str]:
    """Uncapped sections for the visual phase, and where `results` came from.

    Section anchoring keys `results` on a heading. Nature-family articles
    print none, so their body prose (where every figure is cited) became no
    section at all; and a "Results" heading found *after* Methods is usually a
    false anchor, such as an author-contributions line "Experiments: N.H.K.,
    ...". Without a Results section no figure counts as Results-cited, and the
    missing-caption gate then rates every gap critical. When the heading is
    absent or sits after Methods, the body from the end of the abstract (or the
    introduction heading) to the first back-matter heading stands in for it.
    Returns `(sections, source)` with source "heading", "body" or "none".
    """
    from ..pdf.sections import anchor_sections, section_spans
    sections = anchor_sections(full_text, max_chars=len(full_text))
    spans = {name: (start, end) for start, end, name in section_spans(full_text)}
    results = spans.get('results')
    methods = spans.get('methods')
    heading_ok = (results is not None and (methods is None or results[0] < methods[0])
                  and not _CONTRIBUTIONS_RE.search(full_text[results[0]:results[0] + 600]))
    if heading_ok:
        return sections, 'heading'
    # The body starts where the abstract ends. An "Introduction"/"Background"
    # heading is not a safe start: figure-panel text can print "Background" as
    # an axis label mid-body, which would drop every figure cited before it.
    start = 0
    if 'abstract' in spans:
        start = spans['abstract'][0]
    end_match = _BODY_END_RE.search(full_text, start + 1)
    end = end_match.start() if end_match else len(full_text)
    body = full_text[start:end].strip()
    if len(body) < 2000:
        sections.pop('results', None)
        return sections, 'none'
    sections['results'] = body
    return sections, 'body'


def _prose_only(text: str) -> str:
    """Drop caption lines: a Results caption alone isn't a prose dependency."""
    return '\n'.join(line for line in text.splitlines() if not is_caption_line(line))


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
    cited = mentioned_refs(_prose_only(results))
    entries = []
    for ref in locate_in_texts(pages):
        ref_id = ref_id_for(ref.kind, ref.number, ref.extended)
        entries.append(VisualEntry(ref_id, ref, ref_id in cited))
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
    claim_refs = [(PRIORITIES.index(claim.importance),
                   mentioned_refs(claim.location + ' ' + claim.content))
                  for claim in claims or [] if claim.importance in PRIORITIES]

    def relevance(entry):
        weights = [weight for weight, refs in claim_refs if entry.ref_id in refs]
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
    return sorted(mentioned_refs(_prose_only(results)) - known)
