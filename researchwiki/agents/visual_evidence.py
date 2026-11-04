"""Visual transcripts cannot certify themselves as readable PDF evidence."""
from __future__ import annotations

import json
import re


def parse_observations(raw: str, selected: list[dict], sections: dict) -> list[dict]:
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('observations'), list):
        raise ValueError('visual response must contain observations array')
    by_ref = {item['request']['ref_id']: item for item in selected}
    observations, seen = [], set()
    for observation in data['observations']:
        if not isinstance(observation, dict):
            raise ValueError('malformed visual observation')
        ref = observation.get('ref_id')
        selection = by_ref.get(ref) if isinstance(ref, str) else None
        if not selection or ref in seen or observation.get('page') != selection['page']:
            raise ValueError('visual observation has unknown, duplicate or wrong-page reference')
        fields = ('observation', 'visible_labels', 'units_conditions', 'uncertainty', 'source_quote')
        if any(not isinstance(observation.get(key), str) or len(observation[key]) > 1200
               for key in fields) or not isinstance(observation.get('answered'), bool):
            raise ValueError('malformed visual observation fields')
        seen.add(ref)
        clean = {key: observation[key].strip() for key in fields}
        clean.update(ref_id=ref, page=selection['page'], answered=observation['answered'])
        clean['corroborated'] = corroborate_observation(clean, selection, sections)
        observations.append(clean)
    return observations


def corroborate_observation(observation: dict, selection: dict, sections: dict) -> bool:
    """Accept only an exact, complete prose statement; no number-only matching.

    Ambiguous table association is always review-only in v1. The model cannot
    nominate a paraphrase or a bag of numbers as independent source evidence.
    """
    quote = observation['source_quote']
    if (not observation['answered'] or observation['uncertainty']
            or not observation['visible_labels'] or len(quote) < 40
            or quote != observation['observation']
            or selection['request']['reason'] == 'table_alignment'):
        return False
    ref_id = observation['ref_id']
    kind, number = ref_id.split(':', 1)
    label = rf"{'Tables?' if 'table' in kind else r'Fig(?:ure)?s?'}\.?\s*{number}(?:[a-z])?\b"
    if kind.startswith('ed'):
        label = r'Extended\s+Data\s+' + label
    # A sentence about some other result isn't evidence for this visual. The
    # source statement must explicitly associate its conditions with this ID.
    if not re.search(label, quote, re.I):
        return False
    normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
    quote = normalize(quote)
    # Only independently extracted, section-anchored prose, never caption/table
    # lines or the visual transcript. Whole sentence matching retains conditions.
    for key in ('abstract', 'results', 'discussion'):
        text = sections.get(key) or ''
        if not isinstance(text, str):
            continue
        lines = [line for line in text.splitlines() if not re.match(
            r'^\s*(?:Extended\s+Data\s+)?(?:Fig(?:ure)?\.?|Table)\s*\d+', line, re.I)]
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', normalize(' '.join(lines)))
        if quote in sentences and re.match(r'^[A-Z].*[.!?]$', quote):
            return True
    return False


def apply_evidence(targets, result) -> None:
    """Enrich a fixed target packet from independently readable prose only."""
    if targets is None:
        return
    from .phases.target_claims import TargetClaim
    for observation in result.observations:
        if not observation['corroborated']:
            continue
        quote = observation['source_quote']
        if quote not in targets.visual_evidence:
            targets.visual_evidence.append(quote)
        if len(targets.claims) < 35 and not any(claim.content == quote for claim in targets.claims):
            targets.claims.append(TargetClaim('headline', quote, 'high',
                f"Results/prose (corroborates {observation['ref_id']}, p{observation['page']})"))
