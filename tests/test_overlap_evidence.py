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


# --- regressions: identity proofs the entry-scoped gate must not lose -------

@pytest.mark.parametrize('reference', [
    # Page range and a DOI-embedded date are not publication years.
    '1. Target A. A full target paper title. Cell 184, 1995–2008 (2020). doi:10.1016/j.cell.2019.12.015',
    # arXiv identifiers start with four digits too.
    '1. Target A. A full target paper title. arXiv:2012.01234 (2020).',
])
def test_years_inside_ranges_dois_and_arxiv_ids_are_not_entry_years(tmp_path, monkeypatch, reference):
    result = proof(tmp_path, monkeypatch, 'References\n' + reference)
    assert result['verified'] and result['method'] == 'title-author-year'


def test_preprint_and_journal_years_are_both_accepted(tmp_path, monkeypatch):
    for year in (2020, 2021):
        result = proof(tmp_path, monkeypatch,
                       f'References\n1. Target A. A full target paper title. Journal ({year}).',
                       year=2021, years=[2020, 2021])
        assert result['verified'], year


def test_two_distinct_publication_years_still_fail_closed(tmp_path, monkeypatch):
    result = proof(tmp_path, monkeypatch,
                   'References\n1. Target A. A full target paper title. (2020). Other B. (2022).')
    assert not result['verified']


def _nature_list(start, n, target_at=None):
    lines = []
    for i in range(start, start + n):
        who = 'Target, A.' if i == target_at else f'Other{i}, B.'
        what = 'A full target paper title' if i == target_at else f'Unrelated study number {i}'
        lines.append(f'{i}. {who} et al. {what}. Nature 1, 1–2 (2020).')
    return '\n'.join(lines)


def test_headingless_numbered_bibliography_is_segmented(tmp_path, monkeypatch):
    """Nature-family PDFs print the list with no References heading."""
    refs = ('…statements of data and code availability are available at '
            'https://doi.org/10.1038/own.\n' + _nature_list(1, 8, target_at=3))
    result = proof(tmp_path, monkeypatch, refs)
    assert result['verified'] and result['excerpt'].startswith('3. Target, A.')


def test_methods_reference_list_after_methods_heading_is_searched(tmp_path, monkeypatch):
    refs = ('References\n' + _nature_list(1, 6) + '\nMethods\nCells were cultured.\n'
            'Data availability\nSource data are provided.\n' + _nature_list(7, 2, target_at=8))
    result = proof(tmp_path, monkeypatch, refs)
    assert result['verified'] and result['excerpt'].startswith('8. Target, A.')


def test_numbered_prose_is_not_a_bibliography(tmp_path, monkeypatch):
    body = '\n'.join(f'{i}. Target A. A full target paper title. (2020).' for i in (1, 2))
    result = proof(tmp_path, monkeypatch, body)
    assert not result['verified']


def test_line_numbered_preprint_entry(tmp_path, monkeypatch):
    refs = 'References\n' + '\n'.join(
        f'{700 + i} {i}. {"Target, A." if i == 2 else f"Other{i}, B."} '
        f'{"A full target paper title" if i == 2 else f"Unrelated work {i}"}. J. 1, 1 (2020).'
        for i in range(1, 7))
    result = proof(tmp_path, monkeypatch, refs)
    assert result['verified'] and result['method'] == 'title-author-year'


def test_exact_doi_survives_unsegmentable_bibliography(tmp_path, monkeypatch):
    refs = ('References\nWe thank colleagues.\n'
            'Ainslie J and others: gqa training, 2023. https://doi.org/10.1234/exact')
    result = proof(tmp_path, monkeypatch, refs, doi='10.1234/exact')
    assert result['verified'] and result['method'] == 'exact-doi-unsegmented'
    # Title/author/year never verifies without an identified entry.
    assert not proof(tmp_path, monkeypatch, refs.replace('10.1234/exact', '10.1/other'))['verified']


@pytest.mark.parametrize('text', [
    'x' * 3100 + '\nLiteral example DOI for parsing tests: 10.1234/exact',
    'References\nUnparseable reference text.\nMethods\nLiteral example DOI: 10.1234/exact',
    'References\nUnparseable reference text.\nCode availability\nExample DOI: 10.1234/exact',
    'Figure\nReference\nLong-read alignments\nExample DOI: 10.1234/exact',
], ids=['headingless', 'methods', 'code-availability', 'figure-label'])
def test_unsegmented_doi_cannot_license_non_bibliography_text(tmp_path, monkeypatch, text):
    assert not proof(tmp_path, monkeypatch, text, doi='10.1234/exact')['verified']


def test_unsegmented_doi_searches_later_bibliography_and_bounds_excerpt(tmp_path, monkeypatch):
    text = ('Figure\nReference\nAlignments\nExample DOI: 10.1234/wrong\n'
            'References\nAinslie J and others: gqa training, 2023. doi:10.1234/exact\n'
            'Methods\nSamples were collected in 2020. doi:10.1234/wrong')
    result = proof(tmp_path, monkeypatch, text, doi='10.1234/exact')
    assert result['verified'] and result['method'] == 'exact-doi-unsegmented'
    assert 'Methods' not in result['excerpt'] and 'Alignments' not in result['excerpt']
    assert not proof(tmp_path, monkeypatch, text, doi='10.1234/wrong')['verified']


@pytest.mark.parametrize('entry_year,verified', [('', False), (' (2020).', True)])
def test_headingless_entry_stops_at_methods_before_continuing_list(
    tmp_path, monkeypatch, entry_year, verified,
):
    text = (_nature_list(1, 4) + '\n5. Target, A. A full target paper title.' + entry_year
            + '\nMethods\nSamples were collected in 2021.\n' + _nature_list(6, 2))
    result = proof(tmp_path, monkeypatch, text)
    assert result['verified'] is verified
    if verified:
        assert 'Methods' not in result['excerpt']
    else:
        # Borrowing the Methods year is also forbidden when it matches.
        assert not proof(tmp_path, monkeypatch, text.replace('2021', '2020'))['verified']


def test_headingless_methods_continuation_remains_citable(tmp_path, monkeypatch):
    text = (_nature_list(1, 5) + '\nMethods\nSamples were collected in 2021.\n'
            + _nature_list(6, 2, target_at=7))
    result = proof(tmp_path, monkeypatch, text)
    assert result['verified'] and result['excerpt'].startswith('7. Target, A.')


def test_extraction_memo_is_bounded_and_keeps_no_full_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_text, 'extract_pdf_page_texts',
                        lambda *a, **k: ['x' * 10_000, 'References\n1. Target A. A full target paper title. (2020).'])
    v = CitationVerifier(max_extractions=2)
    expected = {'stem': 't', 'title': 'A full target paper title', 'surname': 'target', 'year': 2020, 'doi': ''}
    for i in range(4):
        path = tmp_path / f'{i}.pdf'
        path.write_bytes(f'pdf {i}'.encode())
        assert v.verify(path, 's', expected)['verified']
    assert len(v.extractions) == 2
    entries, _loose, _offsets, _total = next(iter(v.extractions.values()))
    assert sum(len(text) for _, text in entries) < 200


# --- numbered body sections are not a bibliography --------------------------

_SECTIONS = ('1. Introduction\nWe study retention in large models across many settings.\n'
             '2. Related Work\nThe benchmark data (doi:10.1234/exact) were reused here.\n'
             '3. Approach\nWe propose a method and describe it in detail.\n'
             '4. Experiments\nResults follow in the tables below.\n'
             '5. Conclusion\nWe summarize the findings.\n6. Limitations\nSome remain.\n')


def test_numbered_body_sections_are_not_a_bibliography():
    from researchwiki.pdf.citation import bibliography_entries
    assert bibliography_entries(_SECTIONS) == []


def test_body_doi_under_numbered_sections_cannot_verify(tmp_path, monkeypatch):
    """Given-name-first references fail the entry parser; body text must not stand in."""
    refs = 'References\nIbrahim M Alabdulmohsin and Behnam Neyshabur. Revisiting scaling laws. NeurIPS, 2022.\n'
    result = proof(tmp_path, monkeypatch, _SECTIONS + refs, doi='10.1234/exact')
    assert not result['verified']


def test_unparsed_references_after_numbered_sections_reach_doi_fallback(tmp_path, monkeypatch):
    refs = ('References\nZeyuan Allen-Zhu and Yuanzhi Li. Physics of language models. '
            'CoRR, abs/2404.05405, 2024b. doi: 10.48550/ARXIV.2404.05405.\n')
    result = proof(tmp_path, monkeypatch, _SECTIONS.replace('(doi:10.1234/exact) ', '') + refs,
                   doi='10.48550/arxiv.2404.05405')
    assert result['verified'] and result['method'] == 'exact-doi-unsegmented'
    assert 'Related Work' not in result['excerpt']


def test_numbered_steps_with_dates_are_not_a_bibliography():
    """Year-free numbered prose stays out even when long enough to form a run."""
    from researchwiki.pdf.citation import bibliography_entries
    steps = '\n'.join(f'{i}. Retain structures with at most {i}024 residues.' for i in range(1, 8))
    assert bibliography_entries(steps) == []


def test_single_entry_continuation_after_page_break_is_kept(tmp_path, monkeypatch):
    """Entry 25 continues 1–24 under a repeated heading, then a gap splits the run.

    Mirrors marconato-2025: 26 is not an author-led entry, so 25 stands alone
    as a one-entry continuation and must still be reference-like.
    """
    refs = ('References\n' + _nature_list(1, 24) + '\nReferences\n'
            '25. Target, A. A full target paper title. bioRxiv (2020).\n'
            '26. anndataR: interoperability in R. GitHub.\n' + _nature_list(27, 6))
    result = proof(tmp_path, monkeypatch, refs)
    assert result['verified'] and result['excerpt'].startswith('25. Target, A.')


@pytest.mark.parametrize('after_main_references', [False, True])
def test_dated_procedure_steps_cannot_verify_a_literal_doi(tmp_path, monkeypatch, after_main_references):
    start = 6 if after_main_references else 1
    steps = '\n'.join(
        f'{i}. Retain samples collected in 2020. '
        + ('Literal example DOI: 10.1234/exact' if i == start + 1 else 'Store the samples.')
        for i in range(start, start + 5))
    text = 'References\n' + _nature_list(1, 5) + '\nMethods\n' + steps if after_main_references else steps
    result = proof(tmp_path, monkeypatch, text, doi='10.1234/exact')
    assert not result['verified']


@pytest.mark.parametrize('date', ['2020', 'July 2020'])
def test_named_sample_records_are_not_publication_references(tmp_path, monkeypatch, date):
    records = '\n'.join(
        f'{i}. Technician, A. Samples were collected in {date}. '
        + ('Literal example DOI: 10.1234/exact' if i == 2 else 'Samples were retained.')
        for i in range(1, 6))
    assert not proof(tmp_path, monkeypatch, records, doi='10.1234/exact')['verified']


@pytest.mark.parametrize('methods_start', [1, 6])
@pytest.mark.parametrize('doi', ['', '10.1234/main-target'])
def test_headingless_main_references_before_headed_methods_are_kept(tmp_path, monkeypatch, methods_start, doi):
    main = _nature_list(1, 5, target_at=3).splitlines()
    if doi:
        main[2] += ' doi:' + doi
    text = ('\n'.join(main) + '\nMethods\nCells were cultured.\nReferences\n'
            + _nature_list(methods_start, 2))
    result = proof(tmp_path, monkeypatch, text, doi=doi)
    assert result['verified']
    assert result['excerpt'].startswith('3. Target, A.') and 'Methods' not in result['excerpt']


def test_dated_body_steps_do_not_hide_unparsed_bibliography_doi(tmp_path, monkeypatch):
    steps = '\n'.join(f'{i}. Retain samples collected in 2020. Store the samples.' for i in range(1, 6))
    refs = ('References\nZeyuan Allen-Zhu and Yuanzhi Li. Physics of language models. '
            'CoRR, 2024. doi:10.1234/exact')
    result = proof(tmp_path, monkeypatch, steps + '\n' + refs, doi='10.1234/exact')
    assert result['verified'] and result['method'] == 'exact-doi-unsegmented'
    assert 'Retain samples' not in result['excerpt']


@pytest.mark.parametrize('author,publication', [
    ('Example AB, Other CD, et al.', 'Journal 2020;12:34-56.'),
    ('Alice Example and Bob Other.', 'Conference, 2020.'),
    ('Alice Example.', 'Conference, July 2020.'),
])
def test_headingless_citations_keep_initials_full_names_and_publication_styles(
    tmp_path, monkeypatch, author, publication,
):
    refs = '\n'.join(f'{i}. {author} A complete research paper title. {publication} doi:10.1234/ref-{i}'
                     for i in range(1, 6))
    result = proof(tmp_path, monkeypatch, refs, doi='10.1234/ref-3')
    assert result['verified'] and result['method'] == 'exact-doi'


def test_line_numbered_gutter_does_not_hide_wrapped_publication_years(tmp_path, monkeypatch):
    """denadel-2025: the year wraps onto a gutter-numbered line ("Cell,\\n227 2021.")."""
    refs = '\n'.join(f'{200 + 2 * i} {i}. Other{i}, B. Unrelated study number {i}. Journal,\n'
                     f'{201 + 2 * i} 2020. doi:10.1234/ref-{i}' for i in range(1, 8))
    result = proof(tmp_path, monkeypatch, refs, doi='10.1234/ref-3')
    assert result['verified'] and result['method'] == 'exact-doi'


def test_gutter_strip_keeps_a_wrapped_line_that_starts_with_a_year():
    from researchwiki.pdf.citation import _GUTTER
    assert _GUTTER.sub('', 'Cell,\n227 2021. doi') == 'Cell,\n2021. doi'
    assert _GUTTER.sub('', 'Proc.,\n2019 IEEE meeting') == 'Proc.,\n2019 IEEE meeting'
