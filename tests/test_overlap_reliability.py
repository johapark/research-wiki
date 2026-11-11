import pytest

from researchwiki.claim_graph import Edge, open_edges_db, upsert_edge


@pytest.mark.parametrize('status', ['confirmed', 'promoted', 'rejected'])
def test_automatic_judgment_preserves_entire_human_row(status):
    conn = open_edges_db()
    edge = Edge('a', 'kc-a', 'b', 'kc-b', 'refines', 1,
                directed=True, rationale='human evidence', judge_model='known', status=status)
    _, eid = upsert_edge(conn, edge)
    before = dict(conn.execute('SELECT * FROM edges WHERE id=?', (eid,)).fetchone())
    edge.rationale, edge.judge_model, edge.directed, edge.status = 'new', 'other', False, 'candidate'
    upsert_edge(conn, edge)
    after = dict(conn.execute('SELECT * FROM edges WHERE id=?', (eid,)).fetchone())
    assert after == before
    conn.close()


def _claim(stem, text='finding'):
    return {'paper_stem': stem, 'section': 'results', 'text': text, 'position': 0}


def test_pair_orientation_calls_once_and_preserves_direction():
    from researchwiki.overlap.judging import JudgeSession
    calls = []
    def judge(prompt):
        calls.append(prompt)
        return {'verdict': 'refines', 'direction': 'b_to_a', 'rationale': 'B qualifies A'}
    session = JudgeSession(judge_fn=judge)
    a, b = _claim('a'), _claim('b')
    first = session.judge(a, b)
    second = session.judge(b, a)
    assert first[:2] == second[:2] == (a, b)
    assert first[2]['direction'] == second[2]['direction'] == 'b_to_a'
    assert len(calls) == 1 and second[2]['reused']


@pytest.mark.parametrize('response', [
    None, [], {'verdict': 'garbled', 'direction': 'none', 'rationale': 'x'},
    {'verdict': 'none', 'direction': 'symmetric', 'rationale': 'x'},
    {'verdict': 'builds_on', 'direction': 'symmetric', 'rationale': 'x'},
    {'verdict': 'none', 'direction': [], 'rationale': 'x'},
    {'verdict': 'none', 'direction': 'none', 'rationale': 9},
    {'verdict': 'none', 'rationale': 'missing direction'},
])
def test_invalid_judgments_are_failures_and_never_memoized(response):
    from researchwiki.overlap.judging import JudgeSession
    session = JudgeSession(judge_fn=lambda _: response)
    for _ in range(2):
        assert session.judge(_claim('a'), _claim('b'))[2] is None
    assert session.calls == 2 and not session.memo


def test_persistent_none_cache_model_and_text_invalidation(monkeypatch):
    from researchwiki.overlap import judging
    calls = []
    def judge(prompt):
        calls.append(prompt)
        return {'verdict': 'none', 'direction': 'none', 'rationale': 'different endpoints',
                '_metadata': {'model': 'served-model', 'input_tokens': 12}}
    monkeypatch.setattr(judging, 'default_judge', judge)
    a, b = _claim('a'), _claim('b')
    config = {'model': 'requested', 'temperature': 0.2}
    cold = judging.JudgeSession(config=config)
    cold.judge(a, b)
    warm = judging.JudgeSession(config=config)
    _, _, value = warm.judge(b, a)
    assert value['reused'] and value['_metadata']['model'] == 'served-model'
    assert warm.calls == 0 and len(calls) == 1
    judging.JudgeSession(config=config).judge(a, _claim('b', 'changed finding'))
    judging.JudgeSession(config=config | {'model': 'different'}).judge(a, b)
    monkeypatch.setattr(judging, 'VERSION', 'next-prompt')
    judging.JudgeSession(config=config).judge(a, b)
    assert len(calls) == 4


def test_dry_run_does_not_populate_persistent_cache(monkeypatch):
    from researchwiki.overlap import judging
    monkeypatch.setattr(judging, 'default_judge', lambda _: {
        'verdict': 'none', 'direction': 'none', 'rationale': 'different'})
    a, b = _claim('a'), _claim('b')
    config = {'model': 'fixture'}
    judging.JudgeSession(config=config, dry_run=True).judge(a, b)
    fresh = judging.JudgeSession(config=config)
    fresh.judge(a, b)
    assert fresh.calls == 1


def test_symmetric_edge_refreshes_legacy_reversed_row_instead_of_duplicating():
    """Pre-canonical rows stored some symmetric pairs reversed; no second row."""
    from researchwiki.tasks.claim_overlap import _persist_typed_edge
    from researchwiki.claim_graph import compute_claim_slug
    a, b = _claim('a', 'alpha finding'), _claim('b', 'beta finding')
    sa, sb = (compute_claim_slug(c['section'], c['text']) for c in (a, b))
    conn = open_edges_db()
    upsert_edge(conn, Edge('b', sb, 'a', sa, 'corroborates', 1, judge_phase='claim_overlap_judge'))
    conn.commit()
    _persist_typed_edge('a', a, 'b', b, 'corroborates', 'same', 0.9)
    rows = conn.execute('SELECT src_stem, rationale FROM edges').fetchall()
    assert [(r['src_stem'], r['rationale']) for r in rows] == [('b', 'same')]
    conn.close()
