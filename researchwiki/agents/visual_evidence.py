"""Visual transcripts cannot certify themselves as readable PDF evidence."""
from __future__ import annotations

import json
import re

from .visual_selection import mentions

PROSE_SECTIONS = ('abstract', 'results', 'discussion')
# Values a model writes for "no uncertainty" when told to report uncertainty.
_NO_UNCERTAINTY = frozenset({'', 'none', 'n/a', 'na', 'no', 'nil', 'not applicable',
                             'none noted', 'none observed', 'no uncertainty', '-', '—'})
# Trust boundary: drop every line that opens on a figure/table label, caption
# or not. Over-exclusion only costs corroboration; under-exclusion would let a
# caption certify its own visual.
_LABEL_LINE_RE = re.compile(r'^\s*(?:Extended\s+Data\s+)?(?:Fig(?:ure)?\.?|Table)\s*\d+', re.I)


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def prose_sentences(sections: dict) -> list[str]:
    """Whole sentences of independently extracted prose, captions excluded.

    The single splitter used both to offer the model candidate quotes and to
    verify the quote it returns, so a quote copied from the offer matches.
    """
    out = []
    for key in PROSE_SECTIONS:
        text = sections.get(key) or ''
        if not isinstance(text, str):
            continue
        lines = [line for line in text.splitlines() if not _LABEL_LINE_RE.match(line)]
        out.extend(re.split(r'(?<=[.!?])\s+(?=[A-Z])', _normalize(' '.join(lines))))
    return [sentence for sentence in out if re.match(r'^[A-Z].*[.!?]$', sentence)]


def candidate_quotes(sections: dict, ref_id: str, *, limit: int = 6,
                     max_chars: int = 1800) -> list[str]:
    """Prose sentences that explicitly name `ref_id`, bounded for the prompt."""
    out, used = [], 0
    for sentence in prose_sentences(sections):
        if not 40 <= len(sentence) <= 1200 or not mentions(sentence, ref_id):
            continue
        if sentence in out or used + len(sentence) > max_chars or len(out) >= limit:
            continue
        out.append(sentence)
        used += len(sentence)
    return out


def _no_uncertainty(value: str) -> bool:
    return value.strip().rstrip('.').strip().casefold() in _NO_UNCERTAINTY


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
    """Accept only an exact, complete prose sentence; no number-only matching.

    The quote must be a whole sentence of the independently extracted
    abstract/results/discussion that explicitly names this figure/table. The
    observation must match that sentence after whitespace normalization:
    matching numbers cannot verify conditions, units, significance or meaning.
    Paraphrases and ambiguous table associations remain review-only.
    """
    quote = _normalize(observation['source_quote'])
    if (not observation['answered'] or not _no_uncertainty(observation['uncertainty'])
            or not observation['observation'] or not observation['visible_labels']
            or len(quote) < 40 or selection['request']['reason'] == 'table_alignment'):
        return False
    # A sentence about some other result isn't evidence for this visual. The
    # source statement must explicitly associate its conditions with this ID.
    if not mentions(quote, observation['ref_id']):
        return False
    if _normalize(observation['observation']) != quote:
        return False
    return quote in prose_sentences(sections)


def visual_evidence_block(quotes) -> str:
    """Suffix carrying corroborated prose into EVOLVE/DEBUG repair evidence."""
    if not quotes:
        return ''
    return '\n\nIndependent PDF prose corroborating inspected visuals:\n' + '\n'.join(quotes)


def apply_evidence(targets, result) -> None:
    """Enrich a fixed target packet from independently readable prose only."""
    if targets is None:
        return
    from .phases.target_claims import TargetClaim
    for observation in result.observations:
        if not observation['corroborated']:
            continue
        quote = _normalize(observation['source_quote'])
        if quote not in targets.visual_evidence:
            targets.visual_evidence.append(quote)
        if len(targets.claims) < 35 and not any(claim.content == quote for claim in targets.claims):
            targets.claims.append(TargetClaim('headline', quote, 'high',
                f"Results/prose (corroborates {observation['ref_id']}, p{observation['page']})"))
