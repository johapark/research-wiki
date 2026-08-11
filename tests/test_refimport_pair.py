"""Pairing export records to PDFs (`refimport.pair`).

PDFs are generated in-process with pypdfium2 so the tests exercise the real
extraction path — `pdf_shape`, `extract_pdf`, `detect_doi` — rather than a mock
of it. A pairing bug that only appears against real PDF text is exactly the kind
this module exists to prevent.
"""

from pathlib import Path

import pytest

from researchwiki.refimport.pair import (
    TITLE_ACCEPT,
    Pairing,
    PdfFacts,
    build_pdf_index,
    pair_items,
)
from researchwiki.refimport.parse import ExportItem

pypdfium2 = pytest.importorskip("pypdfium2")


def write_pdf(path: Path, lines: list[str], pages: int = 1) -> Path:
    """A PDF with real, extractable text. Content streams are written by hand:
    pypdfium2 is a renderer, not a writer, and a minimal PDF is small enough to
    build directly."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def page_stream(text_lines):
        body = "BT /F1 11 Tf 40 750 Td 14 TL\n"
        for ln in text_lines:
            esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += f"({esc}) Tj T*\n"
        return body + "ET"

    objs, kids = [], []
    for i in range(pages):
        stream = page_stream(lines if i == 0 else [f"page {i + 1} body text"] * 8)
        kids.append(4 + i * 2)
        objs.append((4 + i * 2, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                                f"/Contents {5 + i * 2} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"))
        objs.append((5 + i * 2, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))

    out = "%PDF-1.4\n"
    offsets = {}
    header = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {pages} >>"),
        (3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for num, body in header + objs:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{body}\nendobj\n"
    xref_at, n = len(out), max(offsets) + 1
    out += f"xref\n0 {n}\n0000000000 65535 f \n"
    for i in range(1, n):
        out += f"{offsets.get(i, 0):010d} 00000 n \n"
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))
    return path


@pytest.fixture
def pdf_root(tmp_path):
    return tmp_path / "pdfs"


def mk_item(**kw) -> ExportItem:
    base = dict(key="k", item_type="article", title="A draft synthetic pangenome reference",
                authors=["Ada Fixture"], year=2023, doi="10.1234/jtg.2023.0001")
    base.update(kw)
    return ExportItem(**base)


def run(items, pdf_root, export_dir=None):
    facts = build_pdf_index(pdf_root)
    return pair_items(items, facts, pdf_root=pdf_root,
                      export_dir=export_dir or pdf_root)


# ---------- the index ----------

def test_index_extracts_text_page_count_and_doi(pdf_root):
    write_pdf(pdf_root / "a.pdf",
              ["A draft synthetic pangenome reference",
               "doi:10.1234/jtg.2023.0001",
               "Ada Fixture, Brian Second"], pages=2)
    facts = build_pdf_index(pdf_root)
    assert len(facts) == 1
    assert facts[0].page_count == 2
    assert facts[0].doi == "10.1234/jtg.2023.0001"
    assert facts[0].chars_per_page > 0


def test_index_walks_subdirectories(pdf_root):
    """Reference managers nest attachments (`storage/<key>/file.pdf`), and the
    layout differs per tool — so recurse rather than encode a convention."""
    write_pdf(pdf_root / "storage" / "ABCD1234" / "paper.pdf", ["Some title here"])
    assert len(build_pdf_index(pdf_root)) == 1


def test_index_records_a_broken_pdf_instead_of_raising(pdf_root):
    """One corrupt file must not stop a 500-file import."""
    pdf_root.mkdir(parents=True)
    (pdf_root / "broken.pdf").write_bytes(b"not a pdf at all")
    facts = build_pdf_index(pdf_root)
    assert len(facts) == 1 and facts[0].page_count is None
    assert facts[0].chars_per_page is None


def test_index_is_empty_for_a_missing_root(tmp_path):
    assert build_pdf_index(tmp_path / "nope") == []


def test_chars_per_page_separates_a_scan_from_a_real_paper(pdf_root):
    write_pdf(pdf_root / "real.pdf", ["Real text " * 40] * 12)
    write_pdf(pdf_root / "scan.pdf", ["1"])
    facts = {f.path.name: f for f in build_pdf_index(pdf_root)}
    assert facts["real.pdf"].chars_per_page > 200
    assert facts["scan.pdf"].chars_per_page < 200


# ---------- rung 1: declared paths ----------

def test_declared_relative_path_pairs(pdf_root):
    write_pdf(pdf_root / "files" / "declared-paper.pdf", ["Anything"])
    item = mk_item(declared_files=["files/declared-paper.pdf"])
    pairings, _ = run([item], pdf_root)
    assert pairings[0].rung == "declared" and pairings[0].confidence == 1.0


def test_declared_path_resolves_against_the_export_directory(tmp_path):
    """Exports are commonly written beside their `files/` folder, not beside
    the PDF root the user points at."""
    export_dir = tmp_path / "export"
    write_pdf(export_dir / "files" / "p.pdf", ["Anything"])
    item = mk_item(declared_files=["files/p.pdf"])
    facts = build_pdf_index(export_dir)
    pairings, _ = pair_items([item], facts, pdf_root=tmp_path / "elsewhere",
                             export_dir=export_dir)
    assert pairings[0].rung == "declared"


def test_declared_path_falls_back_to_a_unique_basename(pdf_root):
    """Sync clients relocate trees; an export written before a move names a
    directory that no longer exists while the file itself is right there."""
    write_pdf(pdf_root / "moved" / "elsewhere" / "p.pdf", ["Anything"])
    item = mk_item(declared_files=["/old/gone/p.pdf"])
    assert run([item], pdf_root)[0][0].rung == "declared"


def test_ambiguous_basename_does_not_pair(pdf_root):
    """Two files with one name is not a match — it is a coin flip."""
    write_pdf(pdf_root / "a" / "p.pdf", ["Anything"])
    write_pdf(pdf_root / "b" / "p.pdf", ["Anything"])
    item = mk_item(declared_files=["/old/gone/p.pdf"], title="Unrelated words entirely")
    assert run([item], pdf_root)[0][0].primary is None


# ---------- rung 2: DOI ----------

def test_doi_in_the_pdf_pairs(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["Some Title", "https://doi.org/10.1234/jtg.2023.0001"])
    pairings, _ = run([mk_item(title="Nothing like the pdf text at all")], pdf_root)
    assert pairings[0].rung == "doi" and pairings[0].confidence == 0.9


def test_doi_pairing_is_case_insensitive(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["T", "doi:10.1234/JTG.2023.0001"])
    pairings, _ = run([mk_item(title="Nothing like the pdf text")], pdf_root)
    assert pairings[0].rung == "doi"


def test_doi_beats_title_for_the_same_file(pdf_root):
    """Rungs run as passes over all items, so a confident DOI match wins a file
    over a merely plausible title match on another record — whatever the order."""
    write_pdf(pdf_root / "x.pdf",
              ["A draft synthetic pangenome reference", "doi:10.1234/other.999"])
    title_only = mk_item(key="by-title", doi=None)
    doi_owner = mk_item(key="by-doi", title="Unrelated", doi="10.1234/other.999")
    pairings, _ = run([title_only, doi_owner], pdf_root)
    by_key = {p.item.key: p for p in pairings}
    assert by_key["by-doi"].rung == "doi"
    assert by_key["by-title"].primary is None


# ---------- rung 3: title ----------

def test_title_match_pairs_when_no_doi_is_available(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["A draft synthetic pangenome reference",
                                   "Ada Fixture, Brian Second", "Abstract"])
    pairings, _ = run([mk_item(doi=None)], pdf_root)
    assert pairings[0].rung == "title"
    assert pairings[0].confidence >= TITLE_ACCEPT


def test_unrelated_title_does_not_pair(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["Quantum chromodynamics on a lattice",
                                   "Entirely different subject matter"])
    pairings, unclaimed = run([mk_item(doi=None)], pdf_root)
    assert pairings[0].primary is None
    assert len(unclaimed) == 1


def test_unicode_dash_title_still_matches_its_ascii_pdf(pdf_root):
    """The comparison folds through `stems.strip_diacritics`, so the two
    spellings of one paper do not score as two papers."""
    write_pdf(pdf_root / "x.pdf", ["ATAC-seq: A Method for Assaying Chromatin Accessibility"])
    item = mk_item(title="ATAC‐seq: A Method for Assaying Chromatin Accessibility", doi=None)
    assert run([item], pdf_root)[0][0].rung == "title"


def test_best_title_match_wins_over_an_earlier_weaker_one(pdf_root):
    """Scored globally and sorted, so a strong match is never lost to a weaker
    match that merely came first."""
    write_pdf(pdf_root / "exact.pdf", ["Machine learning for protein structure prediction"])
    weak = mk_item(key="weak", title="Machine learning for something else entirely", doi=None)
    exact = mk_item(key="exact", title="Machine learning for protein structure prediction",
                    doi=None)
    pairings, _ = run([weak, exact], pdf_root)
    assert {p.item.key: p.primary for p in pairings}["exact"] is not None


# ---------- one file, one item ----------

def test_a_pdf_is_never_claimed_by_two_items(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["A draft synthetic pangenome reference"])
    a = mk_item(key="a", doi=None)
    b = mk_item(key="b", doi=None)
    pairings, _ = run([a, b], pdf_root)
    claimed = [p.primary for p in pairings if p.primary]
    assert len(claimed) == 1


def test_unclaimed_pdfs_are_reported(pdf_root):
    write_pdf(pdf_root / "orphan.pdf", ["Nobody references this document"])
    _, unclaimed = run([], pdf_root)
    assert [f.path.name for f in unclaimed] == ["orphan.pdf"]


# ---------- supplementary ----------

def test_supplementary_sibling_is_attached_not_paired_separately(pdf_root):
    """Where the export beats a flat folder: the manager knows two files belong
    to one item. Two paper pages for one paper is the failure to avoid."""
    write_pdf(pdf_root / "smith2023.pdf", ["A draft synthetic pangenome reference"])
    write_pdf(pdf_root / "smith2023-supplementary.pdf", ["Supplementary Figures S1-S9"])
    pairings, unclaimed = run([mk_item(doi=None)], pdf_root)
    assert pairings[0].primary.name == "smith2023.pdf"
    assert [p.name for p in pairings[0].supplementary] == ["smith2023-supplementary.pdf"]
    assert unclaimed == []


def test_an_unrelated_supplementary_file_is_not_attached(pdf_root):
    """Name-matching is required as well as the supplementary marker — an
    appendix from a different paper is not this paper's."""
    write_pdf(pdf_root / "smith2023.pdf", ["A draft synthetic pangenome reference"])
    write_pdf(pdf_root / "totally-other-supplement.pdf", ["Supplementary material"])
    pairings, unclaimed = run([mk_item(doi=None)], pdf_root)
    assert pairings[0].supplementary == []
    assert len(unclaimed) == 1


# ---------- shape ----------

def test_every_item_gets_a_pairing_even_with_no_pdfs(pdf_root):
    pdf_root.mkdir(parents=True)
    items = [mk_item(key="a"), mk_item(key="b", title="Second")]
    pairings, unclaimed = run(items, pdf_root)
    assert len(pairings) == 2
    assert all(isinstance(p, Pairing) and p.primary is None for p in pairings)
    assert unclaimed == []


def test_pairings_preserve_item_identity(pdf_root):
    pdf_root.mkdir(parents=True)
    items = [mk_item(key="a"), mk_item(key="b")]
    assert [p.item.key for p in run(items, pdf_root)[0]] == ["a", "b"]


# ---------- distinctiveness ----------

def test_rival_score_is_recorded_for_a_contested_pdf(pdf_root):
    """Two records scoring against one file must leave evidence of the contest,
    so triage can tell a confident match from a near-tie."""
    write_pdf(pdf_root / "x.pdf", ["Machine learning for protein structure prediction"])
    a = mk_item(key="a", title="Machine learning for protein structure prediction", doi=None)
    b = mk_item(key="b", title="Machine learning for protein structure modelling", doi=None)
    pairings, _ = run([a, b], pdf_root)
    winner = [p for p in pairings if p.primary][0]
    assert winner.rival > 0
    assert winner.margin < 0.5


def test_an_uncontested_match_has_no_rival(pdf_root):
    write_pdf(pdf_root / "x.pdf", ["A draft synthetic pangenome reference"])
    pairings, _ = run([mk_item(doi=None)], pdf_root)
    assert pairings[0].rival == 0.0
    assert pairings[0].margin == pairings[0].confidence
