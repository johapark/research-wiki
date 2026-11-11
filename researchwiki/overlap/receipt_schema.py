"""Validate receipt structure before any recovery or revert can replay it."""
from __future__ import annotations

import math
import re

from ..claim_graph.edges import RELATIONS


def _fields(value, fields, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    for key, expected in fields.items():
        kinds = expected if isinstance(expected, tuple) else (expected,)
        if type(value.get(key)) not in kinds:
            raise ValueError(f'{label}.{key} has a missing or invalid type')


def _optional(value, fields, label):
    _fields(value, {k: t for k, t in fields.items() if k in value}, label)


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')


def _links(links, status):
    for link in links:
        _fields(link, {'stems': list, 'insertions': list, 'citation_evidence': list}, 'link')
        if (len(link['stems']) != 2 or any(not isinstance(s, str) or not s for s in link['stems'])
                or len(link['insertions']) != 2
                or any(not isinstance(p, dict) for p in link['citation_evidence'])):
            raise ValueError('invalid link pair, insertions or citation evidence')
        _optional(link, {'active': bool, 'event_id': str, 'committed_pair': bool}, 'link')
        if status in {'committed', 'reverted'}:
            _fields(link, {'active': bool, 'event_id': str}, 'link')
            if not link['event_id']:
                raise ValueError('link event ID is empty')
        for insertion in link['insertions']:
            _fields(insertion, {'path': str, 'target_key': str, 'bullet': str}, 'insertion')
            _optional(insertion, {'owned': bool, 'reverted': bool,
                                  'before_text': str, 'after_sha256': (str, type(None))}, 'insertion')
            if status in {'committed', 'reverted'}:
                _fields(insertion, {'owned': bool}, 'insertion')


def validate(doc, version):
    _fields(doc, {'schema_version': int, 'run_id': str}, 'receipt')
    if doc['schema_version'] != version:
        raise ValueError('unsupported receipt schema version')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', doc['run_id']):
        raise ValueError('invalid receipt run ID')
    if doc.get('record_type') == 'run':
        return  # Backlog summaries are not replayable stem receipts.
    if doc.get('record_type') is not None:
        raise ValueError('unknown receipt record type')
    _fields(doc, {'stem': str, 'created_at': int, 'status': str, 'inputs': dict,
                  'verifier_version': str, 'result': dict, 'links': list, 'edges': list,
                  'coverage': (dict, type(None)), 'bookkeeping_pending': bool}, 'receipt')
    if not doc['stem'] or doc['created_at'] < 0 or doc['status'] not in {'staged', 'committed', 'reverted'}:
        raise ValueError('invalid receipt stem, creation time or status')
    _optional(doc, {'edges_pending': bool}, 'receipt')
    for expected in doc['inputs'].values():
        _fields(expected, {'path': str, 'page_claims': str, 'db_claims': str,
                           'pdf_sha256': (str, type(None)), 'identity': dict}, 'input')
    for edge in doc['edges']:
        _fields(edge, {'src': dict, 'tgt': dict, 'relation': str,
                       'rationale': str, 'cosine': (int, float)}, 'edge')
        for claim in (edge['src'], edge['tgt']):
            _fields(claim, {'paper_stem': str, 'section': str, 'text': str}, 'edge claim')
        if edge['relation'] not in RELATIONS:
            raise ValueError('unknown edge relation')
        _number(edge['cosine'], 'edge.cosine')
        _optional(edge, {'judge_model': str}, 'edge')
    _links(doc['links'], doc['status'])
    cov = doc['coverage']
    if cov is not None:
        _fields(cov, {'complete': bool, 'fingerprint': str, 'n_claims': int,
                      'n_candidates': int, 'n_judged': int, 'sim_threshold': (int, float)}, 'coverage')
        if any(cov[k] < 0 for k in ('n_claims', 'n_candidates', 'n_judged')):
            raise ValueError('negative coverage count')
        _number(cov['sim_threshold'], 'coverage.sim_threshold')
