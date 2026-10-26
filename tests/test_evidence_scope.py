"""Adversarial scope checks use real rankers; excluded hits must not consume top-K."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from researchwiki.evidence import EvidenceScope
from researchwiki.index.types import Document
from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.index import pages_semantic
from researchwiki.search import claims_hybrid, tools
from researchwiki.search import hybrid
from researchwiki.db import get_connection, upsert_page


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'wiki/cat').mkdir(parents=True)
    (tmp_path / 'papers').mkdir()
    # More outside matches than either ranker's depth. Even k=1 must recover
    # the weaker included claim, so post-filtering a global top-K will fail.
    for stem in ['allowed', 'second'] + [f'outside-{i}' for i in range(65)]:
        text = 'Enzyme signal observed.' if stem in {'allowed', 'second'} else 'Enzyme secretmarker enzyme signal enzyme.'
        page = tmp_path / f'wiki/cat/{stem}.md'
        page.write_text(f'---\ntype: paper\ntitle: {stem}\n---\n\n## Summary\n{text}\n\n## Key Contributions\n- {text}\n')
        (tmp_path / f'papers/{stem}.pdf').write_bytes(b'%PDF-' + stem.encode())
        upsert_page(page)
    backend = TantivySearchBackend()
    docs = [Document(s, 'cat', 'paper', 'Enzyme', '', 2026, 'Enzyme', 'Enzyme')
            for s in ['allowed', 'second'] + [f'outside-{i}' for i in range(65)]]
    docs.append(Document('allowed', 'synthesis', 'synthesis', 'Secretmarker enzyme', '', 2026, 'Enzyme', 'Enzyme'))
    backend.build(docs)
    return tmp_path, backend, EvidenceScope(frozenset({'allowed'}))


def test_empty_scope_is_distinct_from_library_and_immutable(corpus, monkeypatch):
    root, backend, allowed = corpus
    assert backend.query('enzyme')
    empty = EvidenceScope(frozenset())
    assert backend.query('enzyme', scope=empty) == []
    assert tools.claim_lookup('enzyme', scope=empty) == []
    assert tools.claims_by_stem('allowed', scope=empty) == []
    def forbidden(*args, **kw):
        raise AssertionError('empty scope touched an index or model')
    monkeypatch.setattr(pages_semantic, '_load', forbidden)
    monkeypatch.setattr(claims_hybrid, '_rows', forbidden)
    monkeypatch.setattr(tools, 'build_pdf_index', forbidden)
    assert pages_semantic.query_text('enzyme', scope=empty) == []
    assert claims_hybrid.claim_query('enzyme', mode='hybrid', scope=empty) == []
    assert tools.pdf_section_search('allowed', 'enzyme', scope=empty) == []
    with pytest.raises(FrozenInstanceError):
        allowed.stems = frozenset()
    with pytest.raises(ValueError, match='different library'):
        tools.claim_lookup('enzyme', scope=EvidenceScope({'allowed'}, library_root=str(root / 'other')))


def test_bm25_filters_before_limit_and_excludes_crosspaper_pages(corpus):
    _, backend, scope = corpus
    hits = backend.query('enzyme', limit=1, scope=scope)
    assert [(h.stem, h.page_type) for h in hits] == [('allowed', 'paper')]
    assert backend.query('secretmarker', scope=scope) == []
    assert backend.more_like('cat/outside-1', scope=scope) == []
    assert backend.more_like('synthesis/allowed', scope=scope) == []
    both = EvidenceScope({'allowed', 'second'})
    assert [h.stem for h in backend.more_like('cat/allowed', limit=1, scope=both)] == ['second']
    assert [h.stem for h in backend.more_like_text('enzyme', limit=1, scope=scope)] == ['allowed']


def test_fts_and_like_filter_before_limit_and_no_outside_support(corpus):
    _, _, scope = corpus
    hits = tools.claim_lookup('enzyme', k=1, include_context=True, scope=scope)
    assert [h['paper_stem'] for h in hits] == ['allowed']
    assert tools.claim_lookup('secretmarker', scope=scope) == []
    assert tools.claims_by_stem('outside-1', include_context=True, scope=scope) == []
    conn = get_connection()
    import sqlite3
    conn.row_factory = sqlite3.Row
    rows = tools._claim_lookup_like(conn, ['enzyme'], 1, scope=scope)
    conn.close()
    assert [r['paper_stem'] for r in rows] == ['allowed']
    assert claims_hybrid._rows(include_context=True, scope=scope)[0]['paper_stem'] == 'allowed'


def test_semantic_claims_embed_only_scoped_rows_and_fallback_stays_scoped(corpus, monkeypatch):
    _, _, scope = corpus
    seen = []
    def vectors(rows):
        seen.extend(r['paper_stem'] for r in rows)
        return np.ones((len(rows), 2), dtype=np.float32) / np.sqrt(2)
    monkeypatch.setattr(claims_hybrid.claim_embeddings, 'get_claim_embeddings', vectors)
    monkeypatch.setattr(claims_hybrid, 'embed_texts', lambda texts: np.array([[1., 0.]], dtype=np.float32))
    for mode in ('semantic', 'hybrid'):
        hits = claims_hybrid.claim_query('enzyme', k=1, mode=mode, scope=scope)
        assert [h['paper_stem'] for h in hits] == ['allowed']
    assert set(seen) == {'allowed'}
    monkeypatch.setattr(claims_hybrid, 'embed_texts', lambda texts: None)
    for mode in ('semantic', 'hybrid'):
        assert [h['paper_stem'] for h in claims_hybrid.claim_query('enzyme', k=1, mode=mode, scope=scope)] == ['allowed']


def test_semantic_and_hybrid_pages_filter_matrix_before_ranking(corpus, monkeypatch):
    _, backend, scope = corpus
    rows = [dict(key=f'cat/{stem}', stem=stem, category='cat', page_type='paper', title=stem)
            for stem in ['allowed', 'second'] + [f'outside-{i}' for i in range(65)]]
    rows.append(dict(key='synthesis/allowed', stem='allowed', category='synthesis', page_type='synthesis', title='Secret'))
    arr = np.array([[.6, .8], [.5, .86]] + [[1., 0.]] * 66, dtype=np.float32)
    monkeypatch.setattr(pages_semantic, '_load', lambda: (arr, rows))
    monkeypatch.setattr(pages_semantic.semantic, 'embed_texts', lambda texts: np.array([[1., 0.]], dtype=np.float32))
    assert [h.stem for h in pages_semantic.query_text('enzyme', k=1, scope=scope)] == ['allowed']
    assert pages_semantic.query_stem('cat/outside-1', scope=scope) == []
    assert [h.stem for h in pages_semantic.query_stem('cat/allowed', k=1, scope=EvidenceScope({'allowed', 'second'}))] == ['second']
    monkeypatch.setattr(hybrid, 'TantivySearchBackend', lambda: backend)
    assert [h.stem for h in hybrid.hybrid_query('enzyme', limit=1, scope=scope)] == ['allowed']
    assert hybrid.hybrid_more_like('cat/outside-1', scope=scope) == []


def test_outside_pdf_is_never_opened(corpus, monkeypatch):
    _, _, scope = corpus
    def forbidden(*args, **kwargs):
        raise AssertionError('outside source was read')
    monkeypatch.setattr(tools, 'build_pdf_index', forbidden)
    assert tools.pdf_section_search('outside-1', 'secretmarker', scope=scope) == []


def test_scope_wire_roundtrip_and_malformed_snapshot_rejection():
    scope = EvidenceScope({'alpha'})
    data = {**scope.to_dict(), 'fingerprint': scope.fingerprint}
    assert EvidenceScope.from_dict(data) == scope
    for field, value in [('schema_version', 99), ('stems', 'alpha'),
                         ('source_revisions', [None]), ('selection_revision', True),
                         ('library_root', 1), ('fingerprint', 'changed')]:
        with pytest.raises(ValueError):
            EvidenceScope.from_dict({**data, field: value})
