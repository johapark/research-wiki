"""Auditable maintenance of automatic overlap decisions; human rows stay intact."""
from __future__ import annotations

import json
import time

from ..claim_graph.edges import Edge, upsert_edge
from ..claim_graph.slug import SLUG_SCHEME_VERSION
from .judging import identity


def _key(row):
    return tuple(sorted(((row['src_stem'], row['src_slug']), (row['tgt_stem'], row['tgt_slug']))))


def duplicate_groups(conn):
    groups = {}
    for row in conn.execute("SELECT * FROM edges WHERE judge_phase='claim_overlap_judge'"):
        groups.setdefault(_key(row), []).append(dict(row))
    return [{'pair': key, 'rows': rows,
             'collapsible': all(r['status'] == 'candidate' and not r['directed'] for r in rows)
                            and len({r['relation'] for r in rows}) == 1}
            for key, rows in sorted(groups.items()) if len(rows) > 1]


def _archive(conn, row, *, reason, run_id, surviving_id=None):
    conn.execute('CREATE TABLE IF NOT EXISTS overlap_edge_archive '
                 '(archive_id INTEGER PRIMARY KEY, old_id INTEGER NOT NULL, surviving_id INTEGER, '
                 'run_id TEXT NOT NULL, reason TEXT NOT NULL, original TEXT NOT NULL, archived_at INTEGER NOT NULL)')
    conn.execute('INSERT INTO overlap_edge_archive '
                 '(old_id,surviving_id,run_id,reason,original,archived_at) VALUES (?,?,?,?,?,?)',
                 (row['id'], surviving_id, run_id, reason, json.dumps(row, sort_keys=True), int(time.time())))


def collapse_duplicates(conn, *, run_id, apply=False):
    report = duplicate_groups(conn)
    if not apply:
        return report
    with conn:
        for group in report:
            if not group['collapsible']:
                continue
            a, b = group['pair']
            rows = group['rows']
            keep = next((r for r in rows if (r['src_stem'], r['src_slug']) == a), rows[0])
            for row in rows:
                if row['id'] == keep['id']:
                    continue
                _archive(conn, row, run_id=run_id, surviving_id=keep['id'], reason='agreeing symmetric duplicate')
                conn.execute('DELETE FROM edges WHERE id=?', (row['id'],))
            if (keep['src_stem'], keep['src_slug']) != a:
                _archive(conn, keep, run_id=run_id, surviving_id=keep['id'], reason='canonical orientation')
                conn.execute('UPDATE edges SET src_stem=?,src_slug=?,tgt_stem=?,tgt_slug=? WHERE id=?',
                             (*a, *b, keep['id']))
            group['surviving_id'] = keep['id']
    return report


def replace_automatic_pair(conn, a, b, verdict, *, run_id, cosine=0.0):
    """Source-reviewed adjudication archives superseded candidates, including none."""
    from .judging import valid
    if not valid(verdict):
        raise ValueError('invalid adjudication')
    pair = tuple(sorted((identity(a), identity(b))))
    rows = [dict(row) for row in conn.execute("SELECT * FROM edges WHERE judge_phase='claim_overlap_judge'")
            if _key(row) == pair]
    if any(r['status'] in {'confirmed', 'promoted', 'rejected'} for r in rows):
        return {'status': 'human-decision-preserved', 'pair': pair}
    with conn:
        for row in rows:
            if row['status'] != 'candidate':
                continue
            _archive(conn, row, run_id=run_id, reason='source-reviewed adjudication')
            conn.execute('DELETE FROM edges WHERE id=?', (row['id'],))
        if verdict['verdict'] != 'none':
            src, tgt = (b, a) if verdict['direction'] == 'b_to_a' else (a, b)
            action, edge_id = upsert_edge(conn, Edge(
                *identity(src), *identity(tgt), verdict['verdict'], SLUG_SCHEME_VERSION,
                directed=verdict['direction'] in {'a_to_b', 'b_to_a'}, confidence=cosine,
                rationale=verdict['rationale'], judge_phase='claim_overlap_judge',
                judge_model=verdict.get('_metadata', {}).get('model') or ''))
        else:
            action, edge_id = 'removed', None
    return {'status': action, 'edge_id': edge_id, 'archived_ids': [r['id'] for r in rows if r['status'] == 'candidate']}
