"""Generic selection files work without any commercial package or project store."""
import importlib.util
import json
from types import SimpleNamespace

import pytest

from researchwiki import __main__ as cli
from researchwiki.db import upsert_page
from researchwiki.evidence import EvidenceScope
from researchwiki.fsatomic import file_sha256
from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.index.types import Document
from researchwiki.tasks import search, claims, pdf_search, synthesize, export, mcp_serve
from researchwiki.tasks._scope_file import read_scope_file


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'wiki/cat').mkdir(parents=True)
    (tmp_path / 'papers').mkdir()
    docs = []
    for stem in ('alpha', 'beta', 'excluded'):
        text = 'Enzyme signal observed.' if stem != 'excluded' else 'Secretmarker enzyme signal observed.'
        page = tmp_path / f'wiki/cat/{stem}.md'
        page.write_text(f'---\ntype: paper\ntitle: {stem}\nyear: 2026\nauthors: Example Author\n---\n\n## Summary\n{text}\n\n## Key Contributions\n- {text}\n')
        (tmp_path / f'papers/{stem}.pdf').write_bytes(b'%PDF-' + stem.encode())
        upsert_page(page)
        docs.append(Document(stem, 'cat', 'paper', stem, '', 2026, text, text))
    TantivySearchBackend().build(docs)
    sources = tuple((stem, file_sha256(tmp_path / f'wiki/cat/{stem}.md'),
                     file_sha256(tmp_path / f'papers/{stem}.pdf')) for stem in ('alpha', 'beta'))
    scope = EvidenceScope({'alpha', 'beta'}, library_root=str(tmp_path), source_revisions=sources)
    path = tmp_path / 'scope.json'
    path.write_text(json.dumps({**scope.to_dict(), 'fingerprint': scope.fingerprint}))
    return tmp_path, path, scope


@pytest.mark.parametrize('task,args,field', [
    (search, ['enzyme', '--mode', 'bm25', '--see-also'], 'stem'),
    (claims, ['enzyme', '--include-context'], 'paper_stem'),
    (claims, ['enzyme', '--mode', 'semantic'], 'paper_stem'),
    (claims, ['enzyme', '--mode', 'hybrid'], 'paper_stem'),
])
def test_scope_retrieval_and_fallback_do_not_widen(library, capsys, monkeypatch, task, args, field):
    from researchwiki.search import claims_hybrid
    monkeypatch.setattr(claims_hybrid, 'embed_texts', lambda *args: None)
    monkeypatch.setattr(claims_hybrid.claim_embeddings, 'get_claim_embeddings', lambda *args: None)
    _, path, _ = library
    assert task.main([*args, '--scope-file', str(path), '--json']) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data['scope']['stems'] == ['alpha', 'beta']
    assert 'Evidence scope: 2 papers' in captured.err
    assert {r[field] for r in data['results']} <= {'alpha', 'beta'}
    assert 'excluded' not in captured.out and 'Secretmarker' not in captured.out


def test_invalid_missing_and_stale_scope_fail_without_fallback(library, capsys):
    root, path, _ = library
    assert cli.main(['search', 'secretmarker', '--scope-file', 'missing', '--json']) == 2
    assert 'Secretmarker' not in capsys.readouterr().out
    (root / 'papers/alpha.pdf').write_bytes(b'%PDF-new')
    assert cli.main(['search', 'enzyme', '--scope-file', str(path), '--json']) == 1
    capsys.readouterr()
    path.write_text('{"schema_version": 99}')
    assert cli.main(['search', 'enzyme', '--scope-file', str(path), '--json']) == 1
    capsys.readouterr()


def test_scope_file_is_bound_to_library(library, monkeypatch, tmp_path):
    root, path, _ = library
    other = root / 'other'
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(ValueError, match='different library'):
        read_scope_file(path)


def test_direct_excluded_claim_and_pdf_never_read(library, monkeypatch, capsys):
    from researchwiki.search import tools
    _, path, _ = library
    monkeypatch.setattr(tools, 'build_pdf_index', lambda *args: pytest.fail('outside PDF opened'))
    for task, args in ((claims, ['--by-stem', 'excluded']), (pdf_search, ['excluded', 'secretmarker'])):
        assert task.main([*args, '--scope-file', str(path), '--json']) == 1
        assert json.loads(capsys.readouterr().out)['results'] == []


def test_model_reranking_context_stays_scoped(library, monkeypatch, capsys):
    from researchwiki.agents import llm
    _, path, _ = library
    prompts = []
    def call(**kwargs):
        prompts.append(kwargs['prompt'])
        return SimpleNamespace(text='{"order":["cat/beta","cat/alpha","cat/excluded"]}')
    monkeypatch.setattr(llm, 'call', call)
    monkeypatch.setattr(llm, 'response_usage', lambda result: {})
    assert search.main(['enzyme', '--mode', 'bm25', '--llm-rerank', '--scope-file', str(path), '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['results']['llm_rerank']['applied']
    assert [h['stem'] for h in data['results']['hits']] == ['beta', 'alpha']
    assert len(prompts) == 1 and 'Secretmarker' not in prompts[0] and 'excluded' not in prompts[0]


def test_synthesis_seed_and_explicit_selection_never_widen(library, capsys):
    root, path, scope = library
    assert synthesize.main(['--title', 'Scoped map', '--scope-file', str(path), '--topic-seed', 'enzyme']) == 0
    capsys.readouterr()
    evidence = (root / '.ingest/synthesis/scoped-map/evidence.md').read_text()
    assert '[[alpha#' in evidence and '[[beta#' in evidence and 'excluded' not in evidence
    snapshot = json.loads((root / '.ingest/synthesis/scoped-map/scope.json').read_text())
    assert snapshot['fingerprint'] == scope.fingerprint
    assert synthesize.main(['--title', 'Narrow map', '--scope-file', str(path), '--papers', 'alpha', '--topic-seed', 'enzyme']) == 0
    capsys.readouterr()
    evidence = (root / '.ingest/synthesis/narrow-map/evidence.md').read_text()
    assert '[[alpha#' in evidence and '[[beta#' not in evidence and 'excluded' not in evidence
    assert cli.main(['synthesize', '--title', 'Must fail', '--scope-file', str(path), '--papers', 'excluded']) == 1
    capsys.readouterr()
    assert not (root / 'wiki/synthesis/must-fail.md').exists()


def test_bibliography_empty_selection_and_mcp(library, capsys):
    root, path, _ = library
    assert export.main(['--scope-file', str(path), '--format', 'csl-json']) == 0
    assert {r['id'] for r in json.loads(capsys.readouterr().out)} == {'alpha', 'beta'}
    assert export.main(['--scope-file', str(path), '--stem', 'excluded', '--format', 'csl-json']) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert {r['stem'] for r in mcp_serve._do_search('enzyme', 10, 'bm25', str(path))} == {'alpha', 'beta'}
    assert mcp_serve._do_claims('secretmarker', None, 10, True, str(path)) == []
    empty = root / 'empty.json'
    empty.write_text(json.dumps(EvidenceScope(frozenset()).to_dict()))
    assert export.main(['--scope-file', str(empty), '--format', 'csl-json']) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert cli.main(['synthesize', '--scope-file', str(empty), '--title', 'No evidence']) == 1


def test_community_has_no_project_implementation():
    assert importlib.util.find_spec('researchwiki.projects') is None
    assert importlib.util.find_spec('researchwiki.tasks.projects') is None
