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


def test_duplicates_collapse_only_symmetric_candidates_and_archive():
    from researchwiki.overlap.maintenance import collapse_duplicates
    conn = open_edges_db()
    for src, dst in [('a', 'b'), ('b', 'a')]:
        upsert_edge(conn, Edge(src, 'kc-' + src, dst, 'kc-' + dst,
                             'corroborates', 1, judge_phase='claim_overlap_judge'))
        upsert_edge(conn, Edge(src, 'kc-' + src, dst, 'kc-' + dst,
                             'refines', 1, directed=True, judge_phase='claim_overlap_judge'))
    # Relation disagreement keeps the whole pair for source review.
    assert not collapse_duplicates(conn, run_id='test')[0]['collapsible']
    conn.execute("DELETE FROM edges WHERE relation='refines'")
    conn.commit()
    report = collapse_duplicates(conn, run_id='test', apply=True)
    assert report[0]['collapsible']
    rows = conn.execute('SELECT * FROM edges').fetchall()
    assert len(rows) == 1 and rows[0]['src_stem'] == 'a'
    assert rows[0]['judge_model'] == ''
    assert conn.execute('SELECT COUNT(*) FROM overlap_edge_archive').fetchone()[0] == 1
    conn.close()


def test_source_review_none_archives_candidates_and_preserves_human_rows():
    from researchwiki.overlap.maintenance import replace_automatic_pair
    from researchwiki.overlap.judging import identity
    conn = open_edges_db()
    a, b = _claim('a'), _claim('b')
    _, eid = upsert_edge(conn, Edge(*identity(a), *identity(b), 'refines', 1,
                                    directed=True, judge_phase='claim_overlap_judge'))
    verdict = {'verdict': 'none', 'direction': 'none', 'rationale': 'different endpoints'}
    report = replace_automatic_pair(conn, a, b, verdict, run_id='test')
    assert report['archived_ids'] == [eid]
    assert not conn.execute('SELECT * FROM edges').fetchall()
    assert conn.execute('SELECT old_id FROM overlap_edge_archive').fetchone()[0] == eid
    upsert_edge(conn, Edge(*identity(a), *identity(b), 'refines', 1,
                           status='confirmed', judge_phase='claim_overlap_judge'))
    before = [dict(r) for r in conn.execute('SELECT * FROM edges')]
    assert replace_automatic_pair(conn, a, b, verdict, run_id='test')['status'] == 'human-decision-preserved'
    assert [dict(r) for r in conn.execute('SELECT * FROM edges')] == before
    conn.close()
