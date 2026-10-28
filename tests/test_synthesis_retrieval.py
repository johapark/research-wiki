"""PDF evidence retrieval remains available to synthesis after answer removal."""
import importlib.util
from pathlib import Path

import pytest

from researchwiki.db import upsert_page
from researchwiki.evidence import EvidenceScope
from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.search import build_documents_from_wiki
from researchwiki.synthesis import retrieval


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



def test_pdf_fallback_keeps_separate_passages_for_a_multipart_question(sources, monkeypatch):
    monkeypatch.setattr(retrieval, 'claim_lookup', lambda *a, **k: [])
    passages = [dict(text='The enzyme assay detected 12 signals in the synthetic sample.', page_start=1, page_end=1),
                dict(text='A deterministic fixture validates recovery without a network.', page_start=2, page_end=2)]
    monkeypatch.setattr(retrieval, 'pdf_section_search', lambda *a, **kw: passages[:kw['k']])
    evidence = retrieval.retrieve('What did the assay detect and how was recovery validated?', sources)
    assert all(p['text'] in [e.passage for e in evidence] for p in passages)
    assert len(evidence) <= 3
    assert all(e.stem == 'alpha' and not e.claim_text for e in evidence)


def test_fallback_rejects_a_stale_second_passage_before_evidence_is_returned(sources, monkeypatch):
    monkeypatch.setattr(retrieval, 'claim_lookup', lambda *a, **k: [])
    monkeypatch.setattr(retrieval, 'pdf_section_search', lambda *a, **kw: [
        dict(text='The enzyme assay detected 12 signals in the synthetic sample.', page_start=1, page_end=1),
        dict(text='A fabricated second result must not enter the synthesis.', page_start=2, page_end=2)])
    with pytest.raises(Exception, match='does not match current PDF'):
        retrieval.retrieve('enzyme signals and recovery', sources)


def test_overview_is_verbatim_abstract_context_not_an_invented_summary(sources):
    pages = retrieval.extract_pdf_page_texts(Path('papers/alpha.pdf'))
    text, start, end = retrieval._overview(pages)
    assert start == end == 1
    assert 'Acceptance pipelines preserve durable research records.' in text
    assert retrieval._normalized(text) in retrieval._normalized(pages[0])
