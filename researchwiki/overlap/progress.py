"""Flushed JSON Lines progress; research evidence stays in receipts."""
import json
from pathlib import Path


def emit(path, *, run_id, stem=None, completed=0, total=1, result=None, retryable=None):
    if not path:
        return
    r = result or {}
    event = {'schema_version': 1, 'run_id': run_id,
             'event': 'finished' if stem else 'run_finished', 'stem': stem,
             'completed': completed, 'total': total, 'status': r.get('status', 'completed'),
             'candidates': r.get('n_candidates', 0), 'judged_pairs': r.get('judged_pairs', 0),
             'reused_judgments': r.get('reused_judgments', 0),
             'edge_only': len(r.get('edge_only', [])),
             'committed_link_pairs': r.get('committed_link_pairs', 0),
             'failures': len(r.get('judge_failed', [])), 'retryable_stems': retryable or []}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
        f.flush()
