"""Mandatory support checks for all authored units, including short prose and headings."""
from dataclasses import replace

from ..grade.generated import strict_json
from ..grade.primitives import check_numerics
from ..operations._wire import WireModel
from dataclasses import dataclass
from typing import Literal
from .receipt import Review


def numeric_checks(draft, evidence):
    by_id = {e.evidence_id: e for e in evidence}
    units = []
    for unit in draft.units():
        selected = [by_id[eid] for eid in unit.evidence_ids]
        passages = ' '.join(e.passage for e in selected)
        drift = check_numerics(unit.text, passages, '')[1]
        anchored = any(e.claim_text for e in selected)
        if anchored:
            anchor_text = ' '.join(e.claim_text or e.passage for e in selected)
            drift += check_numerics(unit.text, anchor_text, '')[1]
        units.append(replace(unit, support='not_checked',
                             anchor_support='not_checked' if anchored else 'not_applicable',
                             numeric_drift=tuple(sorted(set(drift)))))
    return draft.with_units(units)


def judge_items(draft, evidence):
    by_id = {e.evidence_id: e for e in evidence}
    items, positions = [], []
    for i, unit in enumerate(draft.units()):
        selected = [by_id[eid] for eid in unit.evidence_ids]
        for field in ('support', 'anchor_support'):
            if field == 'anchor_support' and not any(e.claim_text for e in selected):
                continue
            text = '\n\n'.join(f'Paper: {e.stem}; citation: {e.citation}\n' + (
                e.claim_text or e.passage if field == 'anchor_support' else e.passage) for e in selected)
            positions.append((i, field))
            items.append({'id': len(items), 'claim': unit.text, 'source': text})
    return items, positions


@dataclass(frozen=True, kw_only=True)
class Verdict(WireModel):
    id: int
    verdict: Literal['supported', 'partial', 'unsupported']


def apply_judgment(text, draft, positions):
    data = strict_json(text)
    if not isinstance(data, dict) or set(data) != {'verdicts', 'review'} or not isinstance(data['verdicts'], list):
        raise ValueError('invalid synthesis judge fields')
    verdicts = [Verdict.from_dict(item) for item in data['verdicts']]
    by_id = {v.id: v.verdict for v in verdicts}
    if len(by_id) != len(verdicts) or set(by_id) != set(range(len(positions))):
        raise ValueError('missing, duplicate, or out-of-range synthesis verdicts')
    review = Review.from_dict(data['review'])
    units = list(draft.units())
    for j, (i, field) in enumerate(positions):
        units[i] = replace(units[i], **{field: by_id[j]})
    return draft.with_units(units), review
