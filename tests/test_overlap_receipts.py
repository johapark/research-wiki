from pathlib import Path

import pytest

from researchwiki.tasks import claim_overlap as co
from researchwiki.overlap import receipts
from researchwiki.grade.claim_overlap import OverlapCandidate
from researchwiki.db.connection import get_connection
from researchwiki.db.rebuild import upsert_page
from researchwiki.pdf.citation import CitationVerifier

_ORIGINAL_VERIFY = CitationVerifier.verify


@pytest.fixture
def pair(tmp_path, monkeypatch, verified_overlap_citations):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'wiki/ai'
    directory.mkdir(parents=True)
    paths = [directory / f'{stem}.md' for stem in ('new-2026-x', 'old-2020-y')]
    for path in paths:
        path.write_text('---\ntitle: Test paper\ntype: paper\ncategory: [ai]\n---\n\n'
                        '## Key Contributions\n\n- Same finding.\n\n## Related Papers\n\n(none)\n')
        upsert_page(path)
    c = get_connection()
    claims = [dict(c.execute('SELECT * FROM claims WHERE paper_stem=?', (p.stem,)).fetchone()) for p in paths]
    monkeypatch.setattr('researchwiki.grade.claim_overlap.find_claim_overlaps',
                        lambda stem, **kwargs: [OverlapCandidate(paths[1].stem, .9, claims[0], claims[1])])
    return c, paths, claims


def run(pair, **kwargs):
    c, paths, _ = pair
    return co.run(paths[0].stem, conn=c, judge_fn=lambda _: {
        'verdict': 'corroborates', 'direction': 'symmetric', 'rationale': 'Same finding'}, **kwargs)


def test_apply_revert_restores_original_hashes_and_counts(pair):
    c, paths, _ = pair
    before = [p.read_bytes() for p in paths]
    result = run(pair)
    assert result['status'] == 'committed' and result['committed_link_pairs'] == 1
    doc = receipts.load(Path(result['receipt']))
    assert all(i['owned'] for i in doc['links'][0]['insertions'])
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 1
    assert paths[0].stem not in co.find_backlog(c)
    preview = receipts.revert(result['run_id'], conn=c)
    assert len(preview['reverted']) == 2
    assert [p.read_bytes() for p in paths] != before
    receipts.revert(result['run_id'], apply=True, conn=c)
    assert [p.read_bytes() for p in paths] == before
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 0
    assert not receipts.revert(result['run_id'], apply=True, conn=c)['reverted']


def test_precommit_failure_rolls_back_both_pages_and_receipt(pair, monkeypatch):
    c, paths, _ = pair
    before = [p.read_bytes() for p in paths]
    original = receipts.append_related_paper
    calls = []
    def fail(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('second write failed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(receipts, 'append_related_paper', fail)
    result = run(pair)
    assert result['status'] == 'incomplete'
    assert [p.read_bytes() for p in paths] == before
    assert receipts.load(Path(result['receipt']))['status'] == 'staged'
    assert paths[0].stem in co.find_backlog(c)


def test_postcommit_failure_recovers_files_and_completion(pair, monkeypatch):
    c, paths, _ = pair
    import researchwiki.projections as projections
    original = projections.reconcile
    monkeypatch.setattr(projections, 'reconcile', lambda *_: (_ for _ in ()).throw(RuntimeError('projection unavailable')))
    result = run(pair)
    assert result['status'] == 'maintenance-pending'
    assert result['committed_link_pairs'] == 1
    assert all(result['applied'][0]['wrote'].values())
    assert 'auto-added; claim-overlap' in paths[0].read_text()
    assert paths[0].stem in co.find_backlog(c)
    monkeypatch.setattr(projections, 'reconcile', original)
    receipts.recover(conn=c)
    assert paths[0].stem not in co.find_backlog(c)
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 1
    receipts.recover(conn=c)
    assert paths[0].read_text().count('auto-added; claim-overlap') == 1


def test_completion_failure_after_generic_journal_cleared_recovers(pair, monkeypatch):
    c, paths, _ = pair
    original = co.record_run
    monkeypatch.setattr(co, 'record_run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('db unavailable')))
    result = run(pair)
    assert result['status'] == 'maintenance-pending'
    from researchwiki.mutation import pending_journals
    assert not pending_journals()
    monkeypatch.setattr(co, 'record_run', original)
    receipts.recover(conn=c)
    assert paths[0].stem not in co.find_backlog(c)


def test_revert_preserves_later_edits_and_edited_owned_bullet(pair):
    c, paths, _ = pair
    result = run(pair)
    paths[0].write_text(paths[0].read_text() + '\nLater manual note.\n')
    paths[1].write_text(paths[1].read_text().replace('claim-grounded match', 'Human revised relation'))
    receipts.revert(result['run_id'], apply=True, conn=c)
    assert 'Later manual note.' in paths[0].read_text()
    assert 'auto-added; claim-overlap' not in paths[0].read_text()
    assert 'Human revised relation' in paths[1].read_text()
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 1


def test_revert_never_overwrites_a_newer_completion(pair):
    c, paths, _ = pair
    old = run(pair)
    newer = run(pair)
    row = tuple(c.execute('SELECT * FROM claim_overlap_runs WHERE paper_stem=?', (paths[0].stem,)).fetchone())
    assert old['run_id'] != newer['run_id']
    receipts.revert(old['run_id'], apply=True, conn=c)
    assert tuple(c.execute('SELECT * FROM claim_overlap_runs WHERE paper_stem=?', (paths[0].stem,)).fetchone()) == row


def test_revert_cannot_close_changed_claim_set(pair):
    c, paths, _ = pair
    result = run(pair)
    paths[0].write_text(paths[0].read_text().replace('Same finding.', 'Changed finding.'))
    upsert_page(paths[0])
    assert paths[0].stem in co.find_backlog(c)
    receipts.revert(result['run_id'], apply=True, conn=c)
    assert paths[0].stem in co.find_backlog(c)


@pytest.mark.parametrize('changed', ['claim', 'pdf'])
def test_staged_review_rejects_changed_inputs(pair, changed):
    c, paths, _ = pair
    def judge(_):
        if changed == 'claim':
            paths[1].write_text(paths[1].read_text().replace('Same finding.', 'Changed finding.'))
        else:
            (Path('papers') / f'{paths[1].stem}.pdf').write_bytes(b'new PDF')
        return {'verdict': 'corroborates', 'direction': 'symmetric', 'rationale': 'same'}
    result = co.run(paths[0].stem, conn=c, judge_fn=judge)
    assert result['status'] == 'incomplete' and result['judge_failed']
    assert 'auto-added; claim-overlap' not in paths[0].read_text()


@pytest.mark.parametrize('changed_page', [0, 1])
@pytest.mark.parametrize('sync_db', [False, True])
def test_direct_coverage_rejects_claims_changed_during_none_judgment(pair, changed_page, sync_db):
    c, paths, _ = pair
    def judge(_):
        path = paths[changed_page]
        path.write_text(path.read_text().replace('Same finding.', 'Changed finding.'))
        if sync_db:
            upsert_page(path)
        return {'verdict': 'none', 'direction': 'none', 'rationale': 'different findings'}
    result = co.run(paths[0].stem, conn=c, judge_fn=judge)
    assert result['status'] == 'incomplete' and result['judge_failed']
    assert paths[0].stem in co.find_backlog(c)
    assert 'receipt' not in result


def test_direct_coverage_keeps_single_pdf_hash_pass(pair, monkeypatch):
    c, paths, _ = pair
    original = receipts.file_hash
    calls = []
    def hashed(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(receipts, 'file_hash', hashed)
    result = co.run(paths[0].stem, conn=c, judge_fn=lambda _: {
        'verdict': 'none', 'direction': 'none', 'rationale': 'different findings'})
    assert result['status'] == 'committed' and paths[0].stem not in co.find_backlog(c)
    assert len(calls) == 2 and 'receipt' not in result


def test_doi_in_methods_example_never_inserts_related_paper_bullets(pair, monkeypatch):
    c, paths, _ = pair
    paths[1].write_text(paths[1].read_text().replace('type: paper', 'type: paper\ndoi: 10.1234/exact'))
    upsert_page(paths[1])
    before = [p.read_bytes() for p in paths]
    for path in paths:
        (Path('papers') / f'{path.stem}.pdf').write_bytes(path.stem.encode())
    monkeypatch.setattr(CitationVerifier, 'verify', _ORIGINAL_VERIFY)
    monkeypatch.setattr('researchwiki.pdf.text.extract_pdf_page_texts', lambda path, **kw:
        ['References\nUnparseable reference text.\nMethods\nLiteral example DOI: 10.1234/exact']
        if path.stem == paths[0].stem else ['No bibliography here.'])
    result = run(pair)
    assert result['status'] == 'committed' and len(result['edge_only']) == 1
    assert not result['applied'] and result['committed_link_pairs'] == 0
    assert [p.read_bytes() for p in paths] == before


def test_revert_failure_after_commit_recovers_counts(pair, monkeypatch):
    c, paths, _ = pair
    result = run(pair)
    original = co.record_run
    monkeypatch.setattr(co, 'record_run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('count failure')))
    with pytest.raises(RuntimeError, match='count failure'):
        receipts.revert(result['run_id'], apply=True, conn=c)
    assert 'auto-added; claim-overlap' not in paths[0].read_text()
    monkeypatch.setattr(co, 'record_run', original)
    receipts.recover(conn=c)
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 0
    assert not receipts.revert(result['run_id'], apply=True, conn=c)['reverted']


def test_invalid_judgment_leaves_backlog_pending(pair):
    c, paths, _ = pair
    result = co.run(paths[0].stem, conn=c, judge_fn=lambda _: {'verdict': 'garbled'})
    assert result['judge_failed'] and not result['coincidence']
    assert paths[0].stem in co.find_backlog(c)
    # Nothing to commit: no receipt, no journal, and coverage stays open.
    assert 'receipt' not in result and result['status'] == 'incomplete'
    assert not receipts.root().exists() or not list(receipts.root().glob('*/*.json'))


def test_revert_preserves_a_preexisting_identical_bullet(pair):
    c, paths, _ = pair
    def judge(_):
        from researchwiki.backlinks import append_related_paper
        # Another writer adds the reciprocal bullet after nomination. Claim
        # fingerprints stay unchanged; this run must not claim ownership.
        append_related_paper(paths[1], 'ai/' + paths[0].stem, note=receipts.NOTE, sync_db=False)
        return {'verdict': 'corroborates', 'direction': 'symmetric', 'rationale': 'same'}
    result = co.run(paths[0].stem, conn=c, judge_fn=judge)
    doc = receipts.load(Path(result['receipt']))
    assert [i['owned'] for i in doc['links'][0]['insertions']] == [True, False]
    receipts.revert(result['run_id'], apply=True, conn=c)
    assert 'auto-added; claim-overlap' not in paths[0].read_text()
    assert 'auto-added; claim-overlap' in paths[1].read_text()


def test_revert_before_commit_failure_restores_pages_and_active_events(pair, monkeypatch):
    c, paths, _ = pair
    result = run(pair)
    before = [p.read_bytes() for p in paths]
    original = receipts.write_text_atomic
    calls = []
    def write(path, text):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('second removal failed')
        return original(path, text)
    monkeypatch.setattr(receipts, 'write_text_atomic', write)
    with pytest.raises(RuntimeError, match='second removal failed'):
        receipts.revert(result['run_id'], apply=True, conn=c)
    assert [p.read_bytes() for p in paths] == before
    assert receipts.load(Path(result['receipt']))['links'][0]['active']
    assert c.execute('SELECT n_confirmed FROM claim_overlap_runs').fetchone()[0] == 1


def test_five_pair_eight_page_cleanup_restores_all_bytes(tmp_path, monkeypatch, verified_overlap_citations):
    from researchwiki.pdf.citation import file_hash
    from researchwiki.overlap.judging import JudgeSession
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'wiki/ai'
    folder.mkdir(parents=True)
    stems = [f'fixture-{2020 + i}-paper' for i in range(8)]
    paths = [folder / f'{stem}.md' for stem in stems]
    for path in paths:
        path.write_text('---\ntitle: Test paper\ntype: paper\ncategory: [ai]\n---\n\n'
                        '## Key Contributions\n\n- Shared finding.\n\n## Related Papers\n\n(none)\n')
        (tmp_path / 'papers' / f'{path.stem}.pdf').write_bytes(b'synthetic pdf ' + path.stem.encode())
        upsert_page(path)
    before = {p: file_hash(p) for p in paths}
    c = get_connection()
    claims = {p.stem: dict(c.execute('SELECT * FROM claims WHERE paper_stem=?', (p.stem,)).fetchone()) for p in paths}
    pairing = dict(zip(stems[:5], [stems[5], stems[6], stems[7], stems[5], stems[6]]))
    def candidates(stem, **kwargs):
        other = pairing[stem]
        return [OverlapCandidate(other, .9, claims[stem], claims[other])]
    monkeypatch.setattr('researchwiki.grade.claim_overlap.find_claim_overlaps', candidates)
    run_id = receipts.new_run_id()
    session = JudgeSession(judge_fn=lambda _: {'verdict': 'corroborates', 'direction': 'symmetric', 'rationale': 'same'})
    for stem in stems[:5]:
        result = co.run(stem, conn=c, session=session, run_id=run_id)
        assert result['committed_link_pairs'] == 1
    assert sum(p.read_text().count('auto-added; claim-overlap') for p in paths) == 10
    assert c.execute('SELECT SUM(n_confirmed) FROM claim_overlap_runs').fetchone()[0] == 5
    receipts.revert(run_id, apply=True, conn=c)
    assert {p: file_hash(p) for p in paths} == before
    assert c.execute('SELECT SUM(n_confirmed) FROM claim_overlap_runs').fetchone()[0] == 0


def test_receipt_replay_cannot_bypass_failed_journal_recovery(pair, monkeypatch):
    from researchwiki.errors import EnvironmentFailure
    c, paths, _ = pair
    # A failed rollback may leave a committed-looking outcome file beside an
    # active journal. It cannot authorize completion until the journal drains.
    monkeypatch.setattr(receipts, 'recover_pending', lambda: ['rollback failed'])
    monkeypatch.setattr(receipts, 'pending_journals', lambda: [{'operation': 'claim-overlap', 'status': 'active'}])
    with pytest.raises(EnvironmentFailure, match='recovery incomplete'):
        receipts.recover(conn=c)
    assert paths[0].stem in co.find_backlog(c)


def test_one_pdf_extraction_per_shared_run_hash(pair, monkeypatch):
    from researchwiki.overlap.judging import JudgeSession
    c, paths, _ = pair
    calls = []
    monkeypatch.setattr(CitationVerifier, 'verify', _ORIGINAL_VERIFY)
    def extract(path, **kwargs):
        calls.append(path)
        return ['References\n1. Other A. An unrelated full paper title. (2020).']
    monkeypatch.setattr('researchwiki.pdf.text.extract_pdf_page_texts', extract)
    session = JudgeSession(dry_run=True, judge_fn=lambda _: {
        'verdict': 'corroborates', 'direction': 'symmetric', 'rationale': 'same finding'})
    for _ in range(2):
        result = co.run(paths[0].stem, conn=c, session=session, dry_run=True)
        assert not result['applied']
    # Fixture PDFs have identical bytes. Both orientations and both stem
    # evaluations therefore share one physical extraction of this SHA-256.
    assert len(calls) == 1
    assert session.calls == 1 and session.reused == 1


def test_restart_incomplete_stem_reuses_successes_only(pair, monkeypatch):
    from researchwiki.overlap import judging
    c, paths, claims = pair
    third = paths[0].parent / 'third-2021-paper.md'
    third.write_text(paths[1].read_text().replace('Same finding.', 'Another finding.'))
    upsert_page(third)
    other = dict(c.execute('SELECT * FROM claims WHERE paper_stem=?', (third.stem,)).fetchone())
    monkeypatch.setattr('researchwiki.grade.claim_overlap.find_claim_overlaps', lambda stem, **kwargs: [
        OverlapCandidate(paths[1].stem, .9, claims[0], claims[1]),
        OverlapCandidate(third.stem, .9, claims[0], other)])
    calls = []
    def judge(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            return None
        return {'verdict': 'none', 'direction': 'none', 'rationale': 'different endpoints',
                '_metadata': {'model': 'fixture-served'}}
    monkeypatch.setattr(judging, 'default_judge', judge)
    first = co.run(paths[0].stem, conn=c)
    assert first['judge_failed'] and first['judged_pairs'] == 2
    assert paths[0].stem in co.find_backlog(c)
    second = co.run(paths[0].stem, conn=c)
    assert second['judged_pairs'] == second['reused_judgments'] == 1
    assert not second['judge_failed'] and len(calls) == 3
    assert paths[0].stem not in co.find_backlog(c)


def test_missing_target_page_is_skipped_and_coverage_completes(pair, monkeypatch):
    from researchwiki.grade.claim_overlap import OverlapCandidate
    c, paths, claims = pair
    monkeypatch.setattr('researchwiki.grade.claim_overlap.find_claim_overlaps',
                        lambda stem, **kw: [OverlapCandidate('gone-2019-z', .9, claims[0], claims[1])])
    result = co.run(paths[0].stem, conn=c, judge_fn=lambda _: pytest.fail('nothing to judge'))
    assert not result['judge_failed']
    assert result['skipped'][0]['reason'] == 'missing wiki page'
    assert paths[0].stem not in co.find_backlog(c)


def test_unreadable_or_foreign_receipt_does_not_block_writes(pair):
    c, paths, _ = pair
    bad = receipts.root() / 'oldrun'
    bad.mkdir(parents=True)
    (bad / 'truncated.json').write_text('{"schema_version": 1, "bookkeeping_pending": true')
    (bad / 'future.json').write_text('{"schema_version": 99, "bookkeeping_pending": true}')
    assert run(pair)['status'] == 'committed'


@pytest.mark.parametrize('malformation', [
    'list', 'missing-edges', 'bad-created-at', 'bad-edge', 'bad-coverage',
    'bad-insertions', 'missing-bullet', 'bad-owned', 'missing-event', 'invalid-utf8',
])
def test_malformed_receipts_do_not_block_recovery_or_revert(pair, malformation):
    import copy
    import json
    c, paths, _ = pair
    result = run(pair)
    original = receipts.load(Path(result['receipt']))
    doc = copy.deepcopy(original)
    doc.update(bookkeeping_pending=True, edges_pending=True)
    if malformation == 'list':
        doc = [doc]
    elif malformation == 'missing-edges':
        del doc['edges']
    elif malformation == 'bad-created-at':
        doc['created_at'] = 'yesterday'
    elif malformation == 'bad-edge':
        doc['edges'][0]['src'] = {'paper_stem': paths[0].stem}
    elif malformation == 'bad-coverage':
        del doc['coverage']['n_claims']
    elif malformation == 'bad-insertions':
        doc['links'][0]['insertions'] = ['not an insertion']
    elif malformation == 'missing-bullet':
        del doc['links'][0]['insertions'][0]['bullet']
    elif malformation == 'bad-owned':
        doc['links'][0]['insertions'][0]['owned'] = 'yes'
    elif malformation == 'missing-event':
        del doc['links'][0]['event_id']
    bad = Path(result['receipt']).parent / 'bad.json'
    bad.write_bytes(b'{"bookkeeping_pending": true, "bad": "\xff"}'
                    if malformation == 'invalid-utf8' else json.dumps(doc).encode())
    receipts.recover(conn=c)
    with pytest.raises(ValueError):
        receipts.load(bad)
    assert receipts.load(Path(result['receipt'])) == original
    assert len(receipts.revert(result['run_id'], conn=c)['reverted']) == 2
    receipts.revert(result['run_id'], apply=True, conn=c)
    assert all('auto-added; claim-overlap' not in p.read_text() for p in paths)


def test_committed_unrelated_journal_does_not_block_overlap(pair, monkeypatch):
    c, paths, _ = pair
    monkeypatch.setattr(receipts, 'recover_pending', lambda: [])
    monkeypatch.setattr(receipts, 'pending_journals',
                        lambda: [{'operation': 'promote', 'status': 'committed'}])
    assert run(pair)['status'] == 'committed'


def test_backlog_recovers_once_per_invocation(pair, monkeypatch):
    from types import SimpleNamespace
    c, paths, _ = pair
    calls = []
    real = receipts.recover
    monkeypatch.setattr(receipts, 'recover', lambda **kw: calls.append(1) or real(**kw))
    monkeypatch.setattr(co, 'find_backlog', lambda *a, **k: [paths[0].stem, paths[1].stem])
    seen = []
    def fake_run(stem, **kw):
        seen.append(kw.get('recovered'))
        return {'stem': stem, 'n_candidates': 0, 'applied': [], 'edge_only': [],
                'coincidence': [], 'skipped': [], 'judge_failed': [], 'status': 'committed'}
    monkeypatch.setattr(co, 'run', fake_run)
    args = SimpleNamespace(dry_run=False, limit=None, json=True, progress_json=None, sim=0.83, top=10)
    co._run_backlog(args)
    assert len(calls) == 1 and seen == [True, True]
