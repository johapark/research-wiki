"""Deterministic source-derived responses for synthesis transport/gate tests."""
import json

from researchwiki.synthesis.document import HEADINGS

TEXT = 'Synthetic enzyme assays detect signals in controlled samples.'


def author_response(packet):
    stems = [s['stem'] for s in packet['sources']][:2]
    ids = [next(e['evidence_id'] for e in packet['evidence']
                if e['stem'] == stem and not e['claim_text']) for stem in stems]
    def clause(text=TEXT, selected=ids):
        return {'text': text, 'evidence_ids': selected}
    return {'draft': {
        'hook': clause(),
        'sections': [{'heading': heading, 'paragraphs': [clause()]} for heading in HEADINGS],
        'themes': [{'title': clause('Synthetic enzyme assays'), 'paragraphs': [clause()]}],
        'coverage': [{'stem': s['stem'], 'decision': 'cite' if s['stem'] in stems else 'exclude',
                      'reason': 'Direct assay evidence.' if s['stem'] in stems else 'Not used by this bounded comparison.'}
                     for s in packet['sources']],
    }}


def judge_response(packet, *, verdict='supported', quality=True):
    return {'verdicts': [{'id': item['id'], 'verdict': verdict} for item in packet['items']],
            'review': {'theme_organization': quality, 'cross_paper_reasoning': quality,
                       'coverage_rationale': quality, 'reason': 'Deterministic synthetic review fixture.'}}


def response(phase, prompt):
    packet = json.loads(prompt)
    return author_response(packet) if phase == 'synthesis_author' else judge_response(packet)
