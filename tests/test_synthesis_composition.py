"""No corpus or real models: full page gates plus adversarial composition/publication."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchwiki.agents import llm
from researchwiki.synthesis.sources import snapshot_scope
from researchwiki.db import upsert_page
from researchwiki.errors import EnvironmentFailure
from researchwiki.evidence import EvidenceScope
from researchwiki.fsatomic import file_sha256
from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.mutation import pending_journals, recover_pending
from researchwiki.search import build_documents_from_wiki
from researchwiki.synthesis import compose, SynthesisReceipt
from researchwiki.synthesis import pipeline, publication
from researchwiki.synthesis.gates import page_gates
from researchwiki.synthesis.publication import publish
from researchwiki.synthesis.receipt import Source
from tests.synthesis_fixture import TEXT, author_response, judge_response


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'wiki/cat').mkdir(parents=True)
    (tmp_path / 'papers').mkdir()
    spec = importlib.util.spec_from_file_location('acceptance', Path(__file__).parents[1] / '.github/scripts/fresh_install_acceptance.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    for stem in ('alpha', 'beta', 'outside'):
        text = TEXT if stem != 'outside' else 'Secretmarker unrelated enzyme experiment.'
        fixture.write_fixture_pdf(tmp_path / f'papers/{stem}.pdf', title=text, author='Fixture Author')
        path = tmp_path / f'wiki/cat/{stem}.md'
        path.write_text(f'---\ntype: paper\ntitle: {stem}\n---\n\n## Summary\n{text}\n\n## Key Contributions\n- {text}\n')
        upsert_page(path)
    TantivySearchBackend().build(build_documents_from_wiki())
    return snapshot_scope(EvidenceScope({'alpha', 'beta'}, library_root=str(tmp_path)))


def models(monkeypatch, *, author_edit=None, judge_edit=None, after=None, verdict='supported', quality=True):
    calls = []
    monkeypatch.setattr(llm, 'missing_provider_credentials', lambda **kw: [])
    def call(**kw):
        calls.append(kw)
        assert 'Secretmarker' not in kw['prompt'] and 'outside' not in kw['prompt']
        packet = json.loads(kw['prompt'])
        if kw['phase'] == 'synthesis_author':
            data = author_response(packet)
            if author_edit:
                data = author_edit(data)
        else:
            data = judge_response(packet, verdict=verdict, quality=quality)
            if judge_edit:
                data = judge_edit(data)
            if after:
                after()
        return llm.LLMResponse(json.dumps(data) if not isinstance(data, str) else data, 'fixture-synthesis-v1', 0, 20, 10)
    monkeypatch.setattr(llm, 'call', call)
    return calls


def make(corpus):
    return compose('Synthetic assay synthesis', 'synthetic enzyme assays', corpus)


def test_full_synthesis_passes_actual_page_gates_and_roundtrips(corpus, monkeypatch):
    calls = models(monkeypatch)
    result = make(corpus)
    assert result.status == 'verified', result.reason
    assert [c['phase'] for c in calls] == ['synthesis_author', 'synthesis_support']
    assert result.gates.grounding_total > 0 and result.gates.fidelity_graded > 0
    text = result.markdown()
    assert 'author_model: "fixture-synthesis-v1"' in text
    assert '## Findings\n\n### Synthetic enzyme assays' in text
    assert '[^alpha]: [[cat/alpha]]' in text and '[^beta]: [[cat/beta]]' in text
    assert '| [[alpha]] |' in text and 'TODO' not in text and 'outside' not in text
    assert page_gates(text, result.question, corpus).passed
    assert SynthesisReceipt.from_dict(result.to_dict()) == result
    assert SynthesisReceipt.from_dict(result.to_dict()).digest == result.digest
    assert not Path('wiki/synthesis').exists()


@pytest.mark.parametrize('verdict,quality', [('partial', True), ('unsupported', True), ('supported', False)])
def test_support_and_editorial_failure_reject_draft(corpus, monkeypatch, verdict, quality):
    models(monkeypatch, verdict=verdict, quality=quality)
    result = make(corpus)
    assert result.status == 'rejected'
    assert 'No verified synthesis' in result.markdown()
    with pytest.raises(ValueError, match='every prose'):
        replace(result, status='verified')


def test_claim_anchors_require_separate_support_and_render_durable_links(corpus, monkeypatch):
    def edit(data):
        packet = json.loads(calls[0]['prompt'])
        anchors = [next(e['evidence_id'] for e in packet['evidence']
                        if e['stem'] == stem and e['claim_text']) for stem in ('alpha', 'beta')]
        for unit in [data['draft']['hook'], *(s['paragraphs'][0] for s in data['draft']['sections']),
                     data['draft']['themes'][0]['title'], *data['draft']['themes'][0]['paragraphs']]:
            unit['evidence_ids'] = anchors
        return data
    calls = models(monkeypatch, author_edit=edit)
    result = make(corpus)
    assert result.status == 'verified', result.reason
    assert '[[alpha#kc-' in result.markdown() and '[[beta#kc-' in result.markdown()
    assert len(json.loads(calls[1]['prompt'])['items']) == len(result.draft.units()) * 2


def test_recategorization_during_generation_invalidates_paths(corpus, monkeypatch):
    def move():
        Path('wiki/other').mkdir()
        Path('wiki/cat/alpha.md').rename('wiki/other/alpha.md')
    models(monkeypatch, after=move)
    with pytest.raises(ValueError, match='citation paths changed'):
        make(corpus)


@pytest.mark.parametrize('kind', ['unknown', 'one-paper', 'missing-ledger', 'extra-section', 'numeric', 'check-fields', 'link'])
def test_invalid_author_prose_and_coverage_never_reach_judge(corpus, monkeypatch, kind):
    def edit(data):
        draft = data['draft']
        if kind == 'unknown':
            draft['hook']['evidence_ids'] = ['outside']
        elif kind == 'one-paper':
            for section in draft['sections']:
                for clause in section['paragraphs']:
                    clause['evidence_ids'] = [draft['hook']['evidence_ids'][0]]
            for theme in draft['themes']:
                for clause in theme['paragraphs']:
                    clause['evidence_ids'] = [draft['hook']['evidence_ids'][0]]
        elif kind == 'missing-ledger':
            draft['coverage'].pop()
        elif kind == 'extra-section':
            draft['sections'].append(draft['sections'][0])
        elif kind == 'numeric':
            draft['hook']['text'] = 'The assay detected 991 signals in controlled samples.'
        elif kind == 'check-fields':
            draft['hook']['support'] = 'supported'
        else:
            draft['hook']['text'] = 'See https://example.invalid for the real findings.'
        return data
    calls = models(monkeypatch, author_edit=edit)
    result = make(corpus)
    assert result.status == 'rejected' and len(calls) == 1
    assert not Path('wiki/synthesis').exists()


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'boolean-id', 'extra', 'malformed', 'missing-review'])
def test_judge_responses_require_complete_strict_verification(corpus, monkeypatch, kind):
    def edit(data):
        if kind == 'missing':
            data['verdicts'].pop()
        elif kind == 'duplicate':
            data['verdicts'].append(data['verdicts'][0])
        elif kind == 'boolean-id':
            data['verdicts'][0]['id'] = True
        elif kind == 'extra':
            data['verdicts'][0]['override'] = True
        elif kind == 'missing-review':
            del data['review']
        else:
            return 'not json'
        return data
    models(monkeypatch, judge_edit=edit)
    assert make(corpus).status == 'rejected'


def test_zero_graded_cannot_be_called_verified(corpus, monkeypatch):
    calls = models(monkeypatch)
    actual = pipeline.page_gates
    monkeypatch.setattr(pipeline, 'page_gates', lambda *a: replace(actual(*a), fidelity_graded=0))
    result = make(corpus)
    assert result.status == 'rejected' and len(calls) == 1


def test_source_edits_while_judging_refuse_receipt(corpus, monkeypatch):
    def edit():
        page = Path('wiki/cat/alpha.md')
        page.write_text(page.read_text() + '\nEdited source.\n')
    models(monkeypatch, after=edit)
    with pytest.raises(ValueError, match='stale'):
        make(corpus)


def test_empty_single_and_large_selections_spend_no_models(corpus, monkeypatch):
    calls = models(monkeypatch)
    assert make(EvidenceScope(set())).status == 'insufficient_evidence'
    assert make(EvidenceScope({'alpha'})).status == 'insufficient_evidence'
    with pytest.raises(ValueError, match='at most eight'):
        make(EvidenceScope({f'paper-{i}' for i in range(9)}))
    assert not calls


def test_author_abstention_and_duplicate_json_fail_closed(corpus, monkeypatch):
    calls = models(monkeypatch, author_edit=lambda _: {'draft': None})
    assert make(corpus).status == 'insufficient_evidence' and len(calls) == 1
    models(monkeypatch, author_edit=lambda _: '{"draft":null,"draft":null}')
    assert make(corpus).status == 'rejected'


def test_publish_is_explicit_idempotent_and_non_overwriting(corpus, monkeypatch):
    calls = models(monkeypatch)
    result = make(corpus)
    assert result.status == 'verified', result.reason
    before = len(calls)
    first = publish(result, corpus, 'synthetic-map')
    assert first['publication'] == 'published' and first['projections'] == 'current'
    text = Path(first['path']).read_text()
    assert text == result.markdown()
    assert '[[synthesis/synthetic-map]]' in Path('wiki/index.md').read_text()
    log = Path('wiki/log.md').read_text()
    assert publish(result, corpus, 'synthetic-map')['publication'] == 'already_published'
    assert Path('wiki/log.md').read_text() == log and len(calls) == before
    Path(first['path']).write_text(text + '\nA manual edit.\n')
    with pytest.raises(ValueError, match='different content'):
        publish(result, corpus, 'synthetic-map')
    assert Path(first['path']).read_text() == text + '\nA manual edit.\n'


def test_publication_guard_refuses_membership_race(corpus, monkeypatch):
    models(monkeypatch)
    result = make(corpus)
    guard = Path('application.json')
    guard.write_text('{"revision":1}')
    digest = file_sha256(guard)
    guard.write_text('{"revision":2}')
    with pytest.raises(ValueError, match='selection changed'):
        publish(result, corpus, 'guarded-map', guard_path=guard, guard_sha256=digest)
    assert not Path('wiki/synthesis/guarded-map.md').exists()


def test_failed_publication_rolls_back_canonical_files(corpus, monkeypatch):
    models(monkeypatch)
    result = make(corpus)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(publication, '_catalog', fail)
    with pytest.raises(EnvironmentFailure, match='cannot publish'):
        publish(result, corpus, 'failed-map')
    assert not Path('wiki/synthesis/failed-map.md').exists()
    assert not Path('wiki/index.md').exists() and not Path('wiki/log.md').exists()


def test_post_commit_projection_failure_is_recoverable(corpus, monkeypatch):
    import researchwiki.projections as projections
    models(monkeypatch)
    result = make(corpus)
    real = projections.reconcile
    def fail(*args, **kwargs):
        raise EnvironmentFailure('index unavailable')
    monkeypatch.setattr(projections, 'reconcile', fail)
    published = publish(result, corpus, 'recoverable-map')
    assert published['publication'] == 'published' and published['projections'] == 'maintenance_pending'
    assert Path(published['path']).exists() and pending_journals()
    before = Path('wiki/log.md').read_bytes()
    monkeypatch.setattr(projections, 'reconcile', real)
    recover_pending()
    assert not pending_journals() and Path('wiki/log.md').read_bytes() == before
    assert publish(result, corpus, 'recoverable-map')['publication'] == 'already_published'


def test_scoped_coverage_never_reports_outside_papers(corpus, monkeypatch, capsys):
    from researchwiki.tasks.check_coverage import main
    path = Path('draft.md')
    path.write_text('---\ntype: synthesis\ntopic_seed: enzyme\n---\n\n## Question\n\nWhich assay?\n')
    Path('scope.json').write_text(json.dumps(corpus.to_dict()))
    assert main([str(path), '--scope-file', 'scope.json', '--json']) == 1
    report = json.loads(capsys.readouterr().out)
    assert {h['stem'] for h in report['unreferenced']} == {'alpha', 'beta'}
    assert 'outside' not in json.dumps(report)


def test_publication_cli_invalid_receipt_is_a_user_error(corpus, monkeypatch, capsys):
    from researchwiki.__main__ import main
    Path('scope.json').write_text(json.dumps(corpus.to_dict()))
    Path('broken.json').write_text('{broken')
    assert main(['publish-synthesis', 'broken.json', '--slug', 'map', '--scope-file', 'scope.json', '--json']) == 1
    assert 'Traceback' not in capsys.readouterr().err
    assert not Path('wiki/synthesis/map.md').exists()


def test_paper_footnotes_cannot_mask_a_dangling_claim_anchor(corpus, monkeypatch):
    models(monkeypatch)
    result = make(corpus)
    text = result.markdown().replace('## Background', 'A sourced paragraph contains a bad specific claim anchor. [[alpha#kc-deadbeef]][^alpha]\n\n## Background')
    report = page_gates(text, result.question, corpus)
    assert report.dangling_anchors == 1 and not report.passed


@pytest.mark.parametrize('category', ['cell biology', '유전체', 'methods.v2'])
def test_source_keys_preserve_existing_category_names(category):
    source = Source(stem='alpha', key=f'{category}/alpha')
    assert Source.from_dict(source.to_dict()) == source


def test_spaced_category_can_be_composed_and_published(corpus, monkeypatch):
    Path('wiki/cell biology').mkdir()
    for stem in ('alpha', 'beta'):
        Path(f'wiki/cat/{stem}.md').rename(f'wiki/cell biology/{stem}.md')
    # The bare stems and bytes are stable; source paths are captured afresh.
    models(monkeypatch)
    result = make(corpus)
    assert result.status == 'verified', result.reason
    assert '[^alpha]: [[cell biology/alpha]]' in result.markdown()
    assert publish(result, corpus, 'spaced-category-map')['publication'] == 'published'
