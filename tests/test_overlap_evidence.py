import pytest

from researchwiki.pdf.citation import CitationVerifier, VERSION, file_hash
from researchwiki.pdf import text as pdf_text


def proof(tmp_path, monkeypatch, references, **identity):
    path = tmp_path / 'source.pdf'
    path.write_bytes(b'pdf fixture')
    monkeypatch.setattr(pdf_text, 'extract_pdf_page_texts', lambda *a, **k: ['Introduction.', references])
    expected = {'stem': 'target-2020-title', 'title': 'A full target paper title',
                'surname': 'target', 'year': 2020, 'doi': ''} | identity
    return CitationVerifier().verify(path, 'source', expected)


@pytest.mark.parametrize('references', [
    'References\n1. Other A. A full target paper title. Journal (2021).\n2. Target A. Different work. (2020).',
    'References\n1. Target A. A full target paper title. Journal (2021).',
    'References\n1. Target A. A full target paper title and more words. Journal (2020).',
    'Introduction\nTarget A. A full target paper title. (2020).\nReferences\n1. Other A. Different work. (2020).',
    'References\n1. Other A. A full target paper title. (2020).\n2. Target A. Different work. (2020).',
    'References\n1. Target A. A full target paper title. (2020).\n2. Target A. A full target paper title. (2020).',
])
def test_reference_identity_fails_closed(tmp_path, monkeypatch, references):
    result = proof(tmp_path, monkeypatch, references)
    assert not result['verified']
    assert result['reason']


@pytest.mark.parametrize('reference,identity,method', [
    ('1. Target A. A full target paper title. Journal (2020).', {}, 'title-author-year'),
    ('1. Other A. Different work. (2022). doi:10.1234/exact.', {'doi': '10.1234/exact'}, 'exact-doi'),
    ('1. Sirén J. Pangenomics enables genotyping of 5202 genomes. (2021).',
     {'title': 'Pangenomics enables genotyping 5,202 genomes', 'surname': 'siren', 'year': 2021}, 'title-author-year'),
])
def test_proof_coordinates_and_variants(tmp_path, monkeypatch, reference, identity, method):
    result = proof(tmp_path, monkeypatch, 'References\n' + reference, **identity)
    assert result['verified'] and result['method'] == method
    assert result['page'] == 2 and result['verifier_version'] == VERSION
    assert result['source_pdf_sha256'] == file_hash(tmp_path / 'source.pdf')
    text = 'Introduction.\n\nReferences\n' + reference
    start, end = result['match_span']
    assert text[start:end] and text[start:end] in result['excerpt']


def test_one_extraction_per_hash_and_input_invalidation(tmp_path, monkeypatch):
    path = tmp_path / 'source.pdf'
    path.write_bytes(b'first')
    calls = []
    def extract(*args, **kwargs):
        calls.append(args)
        return ['References\n1. Target A. A full target paper title. (2020).']
    monkeypatch.setattr(pdf_text, 'extract_pdf_page_texts', extract)
    v = CitationVerifier()
    expected = {'stem': 'target', 'title': 'A full target paper title', 'surname': 'target', 'year': 2020, 'doi': ''}
    assert v.verify(path, 'source', expected)['verified']
    assert not v.verify(path, 'source', expected | {'year': 2021})['verified']
    assert len(calls) == 1
    path.write_bytes(b'changed')
    assert v.verify(path, 'source', expected)['verified']
    assert len(calls) == 2


def test_figure_reference_label_does_not_license_main_text_title(tmp_path, monkeypatch):
    result = proof(tmp_path, monkeypatch,
                   'Figure\nReference\nLong-read RNA alignments\n'
                   'Target A. A full target paper title. (2020).\n'
                   'References\n1. Other A. Different work. (2020).')
    assert not result['verified']


def test_false_figure_heading_and_wrapped_page_range_preserve_real_entries(tmp_path, monkeypatch):
    result = proof(tmp_path, monkeypatch,
                   'Figure\nReference\nLong-read RNA alignments\n'
                   '99.9% measured accuracy\n'
                   'References\nOther A. Different work. Journal (2021), 199–\n205.\n'
                   'Target A. A full target paper title. Journal (2020).')
    assert result['verified']
    assert result['excerpt'].startswith('Target A.')
