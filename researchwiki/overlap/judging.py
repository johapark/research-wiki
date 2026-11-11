"""One canonical, direction-aware judgment; validated successes are reusable."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict

from ..claim_graph import compute_claim_slug
from ..claim_graph import edges

VERSION = 'overlap-pair-v1'
SYSTEM = """You classify two quoted research-paper claims for a research wiki.
A and B are ordered by durable identity, not chronology. Similar words, titles,
fields, or embedding similarity do not establish a relationship.
Pick ONE verdict:
corroborates: Independent papers affirm the SAME finding on the same entity or
benchmark, from comparable setups. Both findings strengthen each other.
measures_same: Same quantity or benchmark, but DIFFERENT cohorts, datasets or
conditions. Prefer this to corroborates when setups are not directly comparable.
refines: One claim sharpens, conditions or qualifies the other finding.
builds_on: One claim generalizes, extends or reuses the other's method/result.
none: Different entities, domains, datasets or contexts merely sound alike.
Be conservative. Prefer the weakest supported relation (measures_same <
corroborates < refines < builds_on). Unclear setups mean none.
Direction must be symmetric for corroborates/measures_same, a_to_b or b_to_a
for refines/builds_on (arrow from refining/building claim to its basis), and
none for none. Judge only the supplied claims; never infer engagement from titles.
Output JSON only with verdict, direction, and one short sentence rationale.
"""
CROSS_LINK_VERDICTS = frozenset({'corroborates', 'refines', 'builds_on'})
EDGE_ONLY_VERDICTS = frozenset({'measures_same'})
RELATION_VERDICTS = CROSS_LINK_VERDICTS | EDGE_ONLY_VERDICTS

DIRECTIONS = {
    'corroborates': {'symmetric'}, 'measures_same': {'symmetric'},
    'refines': {'a_to_b', 'b_to_a'}, 'builds_on': {'a_to_b', 'b_to_a'}, 'none': {'none'},
}
SCHEMA = {
    'type': 'object', 'required': ['verdict', 'direction', 'rationale'],
    'properties': {
        'verdict': {'type': 'string', 'enum': list(DIRECTIONS)},
        'direction': {'type': 'string', 'enum': ['symmetric', 'a_to_b', 'b_to_a', 'none']},
        'rationale': {'type': 'string'},
    },
}


def identity(claim):
    return (claim['paper_stem'], compute_claim_slug(claim['section'], claim['text']))


def canonical(a, b):
    return (a, b) if identity(a) <= identity(b) else (b, a)


def valid(value):
    return (isinstance(value, dict) and isinstance(value.get('verdict'), str)
            and value['verdict'] in DIRECTIONS
            and isinstance(value.get('direction'), str)
            and value['direction'] in DIRECTIONS[value['verdict']]
            and isinstance(value.get('rationale'), str))


def prompt(a, b):
    # Positions and retrieval cosine are not evidence and can differ on replay.
    return (f"Claim A — paper [[{a['paper_stem']}]] ({a['section']}):\n  {a['text']}\n\n"
            f"Claim B — paper [[{b['paper_stem']}]] ({b['section']}):\n  {b['text']}\n\n"
            'Output JSON only.')


def configuration():
    from ..agents import model_config
    config = asdict(model_config.for_phase('claim_overlap_judge'))
    config['provider'] = os.environ.get('RW_LLM_PROVIDER') or config['provider']
    # Endpoint affects identity, but credentials never enter receipts or keys.
    config['endpoint'] = os.environ.get('RW_LLM_BASE_URL') or model_config.base_url()
    if config['provider'] == 'anthropic':
        config['endpoint'] = os.environ.get('ANTHROPIC_BASE_URL') or config['endpoint']
    return config


def default_judge(text):
    from ..agents.judge import run_llm_judge
    return run_llm_judge(phase='claim_overlap_judge', system=SYSTEM, prompt=text,
                         schema=SCHEMA, include_metadata=True)


class JudgeSession:
    """Shared by every stem in a run. Injected judges never use production cache."""
    def __init__(self, *, dry_run=False, judge_fn=None, config=None):
        self.dry_run, self.judge_fn = dry_run, judge_fn
        self.config = config
        self.memo = {}
        self.calls = self.reused = 0

    def judge(self, a, b):
        a, b = canonical(a, b)
        if self.config is None:
            self.config = configuration() if self.judge_fn is None else {'injected': True}
        packet = {'version': VERSION, 'system': SYSTEM, 'schema': SCHEMA,
                  'phase': 'claim_overlap_judge', 'config': self.config,
                  'claims': [{'identity': identity(c), 'section': c['section'], 'text': c['text']}
                             for c in (a, b)]}
        key = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
        value = self.memo.get(key)
        if value is None and self.judge_fn is None:
            value = self._load(key)
        if value is not None and valid(value):
            self.reused += 1
            self.memo[key] = value
            return a, b, dict(value, reused=True, cache_key=key)
        self.calls += 1
        value = (self.judge_fn or default_judge)(prompt(a, b))
        if not valid(value):
            return a, b, None
        value = dict(value)
        self.memo[key] = value
        if not self.dry_run and self.judge_fn is None:
            self._save(key, packet, value)
        return a, b, dict(value, reused=False, cache_key=key)

    @staticmethod
    def _load(key):
        path = edges.claim_graph_dir() / 'edges.db'
        if not path.exists():
            return None
        with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as conn:
            try:
                row = conn.execute('SELECT result FROM overlap_judgments WHERE cache_key=?', (key,)).fetchone()
            except sqlite3.OperationalError as exc:
                if 'no such table' in str(exc):
                    return None
                raise
        return json.loads(row[0]) if row else None

    @staticmethod
    def _save(key, packet, value):
        conn = edges.open_edges_db()
        try:
            conn.execute('CREATE TABLE IF NOT EXISTS overlap_judgments '
                         '(cache_key TEXT PRIMARY KEY, packet TEXT NOT NULL, result TEXT NOT NULL)')
            conn.execute('INSERT OR REPLACE INTO overlap_judgments VALUES (?,?,?)',
                         (key, json.dumps(packet, sort_keys=True), json.dumps(value, sort_keys=True)))
            conn.commit()
        finally:
            conn.close()
