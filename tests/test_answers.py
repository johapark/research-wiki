"""Fail-closed answer checks with disposable sources and deterministic model responses."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchwiki.agents import llm
from researchwiki.answers import AnswerReceipt, answer
from researchwiki.answers import pipeline, retrieval
from researchwiki.answers.contracts import ModelRun
from researchwiki.db import upsert_page
from researchwiki.evidence import EvidenceScope
from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.search import build_documents_from_wiki


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'wiki/cat').mkdir(parents=True)
    (tmp_path / 'papers').mkdir()
    spec = importlib.util.spec_from_file_location('acceptance', Path(__file__).parents[1] / '.github/scripts/fresh_install_acceptance.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    for stem, text in [('alpha', 'The enzyme assay detected 12 signals in the synthetic sample.'),
                       ('outside', 'Secretmarker experiment detected 99 signals in another sample.')]:
        fixture.write_fixture_pdf(tmp_path / f'papers/{stem}.pdf', title=text, author='Fixture Author')
        page = tmp_path / f'wiki/cat/{stem}.md'
        page.write_text(f'---\ntype: paper\ntitle: {stem}\n---\n\n## Summary\n{text}\n\n## Key Contributions\n- {text}\n')
        upsert_page(page)
    TantivySearchBackend().build(build_documents_from_wiki())
    return EvidenceScope({'alpha'}, library_root=str(tmp_path))


def model(monkeypatch, *, text='The enzyme assay detected 12 signals in the synthetic sample.', verdict='supported',
          author_raw=None, judge_raw=None, after=None, use_anchor=True):
    calls = []
    def call(**kw):
        calls.append(kw)
        if kw['phase'] == 'answer':
            packet = json.loads(kw['prompt'])
            assert 'Secretmarker' not in kw['prompt'] and 'outside' not in kw['prompt']
            evidence = next(e for e in packet['evidence'] if bool(e['claim_text']) == use_anchor)
            payload = author_raw or json.dumps({'statements': [{'text': text, 'evidence_id': evidence['evidence_id']}]})
        else:
            items = json.loads(kw['prompt'].split('\n', 1)[1])
            payload = judge_raw or json.dumps({'verdicts': [{'id': item['id'], 'verdict': verdict} for item in items]})
            if after:
                after()
        return llm.LLMResponse(payload, 'fixture-answer-v1', 0, 20, 10)
    monkeypatch.setattr(llm, 'call', call)
    monkeypatch.setattr(llm, 'missing_provider_credentials', lambda **kw: [])
    return calls


def test_verified_receipt_roundtrip_and_pdf_provenance(sources, monkeypatch):
    calls = model(monkeypatch)
    result = answer('enzyme signals', sources)
    assert result.status == 'verified'
    assert [c['phase'] for c in calls] == ['answer', 'answer_judge']
    assert len(json.loads(calls[1]['prompt'].split('\n', 1)[1])) == 2  # PDF AND cited claim
    assert '[[alpha#kc-' in result.markdown()
    assert 'outside' not in json.dumps(result.to_dict())
    assert result.scope.source_revisions and result.evidence[0].page_start == 1
    assert AnswerReceipt.from_dict(result.to_dict()) == result
    assert AnswerReceipt.from_dict(result.to_dict()).digest == result.digest


@pytest.mark.parametrize('verdict', ['partial', 'unsupported'])
def test_qualitative_failure_rejects_even_short_answers(sources, monkeypatch, verdict):
    model(monkeypatch, text='Enzyme works.', verdict=verdict)
    result = answer('enzyme signals', sources)
    assert result.status == 'rejected'
    assert 'Enzyme works' not in result.markdown()
    with pytest.raises(ValueError, match='failing checks'):
        replace(result, status='verified')


def test_numeric_drift_stops_before_judge(sources, monkeypatch):
    calls = model(monkeypatch, text='The assay detected 99 signals.')
    result = answer('enzyme signals', sources)
    assert result.status == 'rejected' and result.statements[0].numeric_drift
    assert len(calls) == 1


@pytest.mark.parametrize('raw', [
    '{"statements":[{"text":"Made up.","evidence_id":"outside"}]}',
    '{"statements":[{"text":"[[outside]]","evidence_id":"e0"}]}',
    '{"statements":[{"text":"See https://example.invalid","evidence_id":"e0"}]}',
    '{"statements":[{"text":"Unsupported","evidence_id":"e0","citation":"outside"}]}',
    '{"answer":"Uncited prose"}', '```json\n{}\n```',
    '{"statements":[],"statements":[]}',
])
def test_invalid_author_never_becomes_answer(sources, monkeypatch, raw):
    calls = model(monkeypatch, author_raw=raw)
    result = answer('enzyme signals', sources)
    assert result.status == 'rejected' and len(calls) == 1


@pytest.mark.parametrize('raw', [
    '{"verdicts":[]}', '{"verdicts":[{"id":true,"verdict":"supported"}]}',
    '{"verdicts":[{"id":0,"verdict":"supported"},{"id":0,"verdict":"supported"}]}',
    '{"verdicts":[{"id":0,"verdict":"supported"}]}', 'not json',
])
def test_missing_duplicate_or_malformed_judge_fails_closed(sources, monkeypatch, raw):
    model(monkeypatch, judge_raw=raw)
    assert answer('enzyme signals', sources).status == 'rejected'


def test_source_mutation_during_models_refuses_receipt(sources, monkeypatch):
    def edit():
        page = Path('wiki/cat/alpha.md')
        page.write_text(page.read_text() + '\nChanged source.')
    model(monkeypatch, after=edit)
    with pytest.raises(ValueError, match='stale'):
        answer('enzyme signals', sources)


def test_retrieved_text_must_exist_in_current_pdf_pages(sources, monkeypatch):
    monkeypatch.setattr(retrieval, 'pdf_section_search', lambda *a, **k: [
        dict(text='Fabricated cached passage.', page_start=1, page_end=1)])
    with pytest.raises(Exception, match='does not match current PDF'):
        answer('enzyme signals', sources)


def test_preflight_checks_both_phases_before_spending(sources, monkeypatch):
    calls = model(monkeypatch)
    def missing(*, phases):
        assert phases == ('answer', 'answer_judge')
        return ['judge credentials missing']
    monkeypatch.setattr(llm, 'missing_provider_credentials', missing)
    with pytest.raises(llm.ProviderUnavailable, match='judge credentials missing'):
        answer('enzyme signals', sources)
    assert not calls


def test_selected_phase_credentials_ignore_unused_provider(monkeypatch, tmp_path):
    from researchwiki.agents.model_config import ModelConfig
    monkeypatch.chdir(tmp_path)
    def config(name):
        provider = 'chat-relay' if name in ('answer', 'answer_judge') else 'anthropic'
        return ModelConfig(provider, 'fixture-model-v1', 0, 100)
    monkeypatch.setattr(llm.model_config, 'for_phase', config)
    assert llm.missing_provider_credentials(phases=('answer', 'answer_judge')) == []
    assert any('ANTHROPIC_API_KEY' in problem for problem in llm.missing_provider_credentials())


def test_empty_evidence_has_no_models_or_answer(sources, monkeypatch):
    def forbidden(**kw):
        pytest.fail('empty selection spent a model call')
    monkeypatch.setattr(llm, 'call', forbidden)
    result = answer('enzyme', EvidenceScope(set()))
    assert result.status == 'insufficient_evidence' and not result.models
    monkeypatch.setattr(pipeline, 'retrieve', lambda *a: ())
    assert answer('enzyme', sources).status == 'insufficient_evidence'


def test_raw_pdf_citation_and_author_abstention(sources, monkeypatch):
    model(monkeypatch, use_anchor=False)
    result = answer('enzyme signals', sources)
    assert result.status == 'verified' and '[[alpha]]' in result.markdown()
    model(monkeypatch, author_raw='{"statements":[]}')
    result = answer('enzyme signals', sources)
    assert result.status == 'insufficient_evidence' and len(result.models) == 1


@pytest.mark.parametrize('model_id', ['stub:fixture-v1', 'unknown', 'gpt-5.6', ''])
def test_placeholder_or_stub_models_cannot_verify(model_id):
    with pytest.raises(ValueError):
        ModelRun(phase='answer', model=model_id, input_tokens=0, output_tokens=0,
                 cache_read_tokens=0, cache_write_tokens=0)


def test_strict_receipt_rejects_unknown_schema_and_out_of_scope_evidence(sources, monkeypatch):
    model(monkeypatch)
    result = answer('enzyme signals', sources)
    for field, value in [('schema_version', True), ('status', 'published'), ('extra', 'ignored')]:
        with pytest.raises(ValueError):
            AnswerReceipt.from_dict({**result.to_dict(), field: value})
    data = result.to_dict()
    data['statements'][0]['support'] = 'not_checked'
    with pytest.raises(ValueError):
        AnswerReceipt.from_dict(data)


def test_cli_invalid_and_stale_requests_are_user_errors(sources, monkeypatch, capsys):
    from researchwiki.__main__ import main
    path = Path('selection.json')
    path.write_text(json.dumps(sources.to_dict()))
    assert main(['answer', '', '--scope-file', str(path), '--json']) == 1
    calls = model(monkeypatch)
    from researchwiki.tasks import answer as task
    def stale(*args):
        raise ValueError('answer source snapshot is stale')
    monkeypatch.setattr(task, 'answer', stale)
    assert main(['answer', 'enzyme', '--scope-file', str(path), '--json']) == 1
    assert not calls
    assert 'Traceback' not in capsys.readouterr().err
