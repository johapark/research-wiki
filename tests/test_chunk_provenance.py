"""Page + section provenance on PDF chunks.

Before this, a claim's `supporting_text` was an anonymous 250-word window:
nothing could say it came from §Results, p. 7. The chunk carries that now, and
`claims --include-context`, `pdf-search` and the fidelity grader display it.

Three things are load-bearing and pinned here:

  - **The join invariant.** `PAGE_SEPARATOR.join(extract_pdf_page_texts(...))`
    must equal `extract_pdf(...)[0]` byte for byte, because page offsets are
    measured against that string. Ligature repair is applied per page rather
    than to the join; the two agree only because the repair patterns are
    word-shaped and no word survives a page break.
  - **Chunk text is unchanged.** The chunker switched from `str.split()` to a
    regex so each window knows its character span. The emitted text must be
    identical to what the old implementation produced, or every cached
    embedding and every recorded `supporting_text` silently shifts.
  - **The cache version.** `build_pdf_index` returns early when the directory
    exists, so without a recorded version a schema change leaves every existing
    index in place, missing the new fields, with no error.

Hermetic apart from the fixture-backed tests at the bottom, which use the
bundled CC-BY PDFs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchwiki.index import pdf_chunks as pc
from researchwiki.pdf.sections import section_for_offset, section_spans
from researchwiki.pdf.text import (
    PAGE_SEPARATOR,
    extract_pdf,
    extract_pdf_page_texts,
    page_for_offset,
    page_offsets,
)

FIXTURES = Path(__file__).resolve().parent.parent / "benchmark-fixtures" / "pdfs"
STEM = "zhang-2026-mga-a-tool-for-haplotype-mixed"


# ---------- page offsets ----------

def test_page_offsets_are_cumulative():
    pages = ["abc", "de", "fghi"]
    assert page_offsets(pages) == [0, 5, 9]           # 3 + 2 sep, then 2 + 2 sep
    assert PAGE_SEPARATOR.join(pages)[9:] == "fghi"


def test_page_offsets_single_page_has_no_trailing_separator():
    assert page_offsets(["only"]) == [0]


def test_page_offsets_empty():
    assert page_offsets([]) == []


@pytest.mark.parametrize("offset,expected", [
    (0, 1), (2, 1), (4, 1),      # page 1 spans [0, 5)
    (5, 2), (6, 2),              # page 2 spans [5, 9)
    (9, 3), (12, 3),
])
def test_page_for_offset(offset, expected):
    assert page_for_offset([0, 5, 9], offset) == expected


def test_page_for_offset_rejects_negative():
    assert page_for_offset([0, 5, 9], -1) is None


def test_page_for_offset_empty_list():
    assert page_for_offset([], 0) is None


# ---------- section spans ----------

def _doc(*parts: str) -> str:
    return "\n".join(parts)


def test_section_spans_are_contiguous_and_ordered():
    text = _doc("Abstract", "a", "Introduction", "b", "Methods", "c", "Results", "d")
    spans = section_spans(text)
    names = [name for _, _, name in spans]
    assert names == ["abstract", "introduction", "methods", "results"]
    # Each span ends where the next begins; the last runs to end-of-text.
    for (_, end, _), (nxt_start, _, _) in zip(spans, spans[1:]):
        assert end == nxt_start
    assert spans[-1][1] == len(text)


def test_section_spans_keeps_first_occurrence_only():
    """A running header repeating 'Methods' on every page must not restart the
    section — long-standing `anchor_sections` behaviour, now shared."""
    text = _doc("Methods", "a", "Results", "b", "Methods", "c")
    spans = section_spans(text)
    assert [n for _, _, n in spans] == ["methods", "results"]


def test_section_spans_empty_when_no_headings():
    assert section_spans("just prose, no headings here") == []


def test_section_for_offset_before_first_heading_is_none():
    text = _doc("front matter", "Methods", "a")
    spans = section_spans(text)
    assert section_for_offset(spans, 0) is None
    assert section_for_offset(spans, text.index("Methods")) == "methods"


def test_anchor_sections_still_agrees_with_spans():
    """The refactor moved the segmentation into `section_spans`; the two must
    not drift, which is the whole reason there is one function."""
    from researchwiki.pdf.sections import anchor_sections
    text = _doc("Abstract", "a" * 50, "Methods", "b" * 50, "Results", "c" * 50)
    anchored = anchor_sections(text)
    span_names = {name for _, _, name in section_spans(text)}
    assert span_names <= set(anchored) | {"abstract"}
    assert "methods" in anchored and "results" in anchored


# ---------- chunking ----------

def test_chunk_text_is_unchanged_by_the_offset_rewrite():
    """The chunker switched from `str.split()` to `re.finditer`. If the emitted
    text moved, every cached embedding and stored `supporting_text` shifts."""
    text = " ".join(f"word{i}" for i in range(900))
    chunks = pc._chunk_text(text)
    expected = text.split()
    assert chunks[0].text == " ".join(expected[:pc.CHUNK_WORDS])
    step = pc.CHUNK_WORDS - pc.OVERLAP_WORDS
    assert chunks[1].text == " ".join(expected[step:step + pc.CHUNK_WORDS])


def test_chunks_without_provenance_inputs_are_unlabelled():
    chunks = pc._chunk_text(" ".join(f"w{i}" for i in range(400)))
    assert all(c.page_start is None and c.section is None for c in chunks)
    assert chunks[0].provenance() == ""


def test_chunk_carries_page_and_section():
    body = " ".join(f"w{i}" for i in range(120))
    pages = ["Methods\n" + body, "Results\n" + body]
    text = PAGE_SEPARATOR.join(pages)
    chunks = pc._chunk_text(
        text, chunk_words=40, overlap=0,
        page_starts=page_offsets(pages), spans=section_spans(text),
    )
    assert chunks[0].page_start == 1
    assert chunks[0].section == "methods"
    assert any(c.page_start == 2 and c.section == "results" for c in chunks)


def test_chunk_spanning_a_page_break_reports_a_range():
    pages = [" ".join(f"a{i}" for i in range(30)), " ".join(f"b{i}" for i in range(30))]
    text = PAGE_SEPARATOR.join(pages)
    chunks = pc._chunk_text(
        text, chunk_words=50, overlap=0, page_starts=page_offsets(pages),
    )
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


@pytest.mark.parametrize("kwargs,expected", [
    ({}, ""),
    ({"section": "results"}, "§results"),
    ({"page_start": 7, "page_end": 7}, "p. 7"),
    ({"section": "results", "page_start": 7, "page_end": 7}, "§results, p. 7"),
    ({"section": "methods", "page_start": 7, "page_end": 8}, "§methods, pp. 7-8"),
])
def test_provenance_display(kwargs, expected):
    assert pc.Chunk(chunk_id=0, text="x", **kwargs).provenance() == expected


def test_retrieved_chunk_shares_the_display_form():
    rc = pc.RetrievedChunk(chunk_id=0, score=1.0, text="x",
                           page_start=3, page_end=3, section="discussion")
    assert rc.provenance() == "§discussion, p. 3"


# ---------- cache versioning ----------

def test_missing_meta_reads_as_version_1(tmp_path):
    """An index written before versioning existed must be treated as stale,
    not as current — that is the whole point of the guard."""
    assert pc._read_index_version(tmp_path) == 1


def test_unparseable_meta_reads_as_version_1(tmp_path):
    (tmp_path / pc.INDEX_META_FILENAME).write_text("{not json", encoding="utf-8")
    assert pc._read_index_version(tmp_path) == 1


def test_round_trip_meta(tmp_path):
    pc._write_index_meta(tmp_path)
    assert pc._read_index_version(tmp_path) == pc.CACHE_VERSION
    assert json.loads((tmp_path / pc.INDEX_META_FILENAME).read_text())["cache_version"] \
        == pc.CACHE_VERSION


# ---------- fixture-backed ----------

pytestmark_fixtures = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="benchmark-fixtures/pdfs not present"
)


@pytestmark_fixtures
def test_page_texts_join_reproduces_extract_pdf():
    """The invariant every offset depends on."""
    pdf = FIXTURES / f"{STEM}.pdf"
    joined = PAGE_SEPARATOR.join(extract_pdf_page_texts(pdf, max_pages=30))
    assert joined == extract_pdf(pdf, max_pages=30)[0]


@pytestmark_fixtures
def test_every_page_offset_maps_back_to_its_own_page():
    pdf = FIXTURES / f"{STEM}.pdf"
    pages = extract_pdf_page_texts(pdf, max_pages=20)
    offsets = page_offsets(pages)
    for i, start in enumerate(offsets, start=1):
        assert page_for_offset(offsets, start) == i


@pytestmark_fixtures
def test_real_index_carries_provenance(tmp_path, monkeypatch):
    from researchwiki import paths
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(pc, "grade_cache_dir", lambda: tmp_path / ".grade-cache")

    idx = pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")
    assert pc._read_index_version(idx) == pc.CACHE_VERSION

    hits = pc.query_pdf(STEM, "running time and memory footprint", topk=3)
    assert hits
    labelled = [h for h in hits if h.provenance()]
    assert labelled, "a paper with clear headings should label its chunks"
    top = labelled[0]
    assert top.section in {"abstract", "introduction", "methods", "results",
                           "discussion", "references"}
    assert 1 <= top.page_start <= top.page_end


@pytestmark_fixtures
def test_stale_index_is_rebuilt_not_reused(tmp_path, monkeypatch):
    """The trap this guard exists for: `build_pdf_index` returns early when the
    directory exists, so a schema change would otherwise leave old indexes in
    place with the new fields absent and no error raised."""
    from researchwiki import paths
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(pc, "grade_cache_dir", lambda: tmp_path / ".grade-cache")

    idx = pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")
    (idx / pc.INDEX_META_FILENAME).write_text('{"cache_version": 1}', encoding="utf-8")

    rebuilt = pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")
    assert pc._read_index_version(rebuilt) == pc.CACHE_VERSION


@pytestmark_fixtures
def test_current_index_is_reused(tmp_path, monkeypatch):
    """Guard the other direction — the version check must not force a rebuild
    on every call."""
    from researchwiki import paths
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(pc, "grade_cache_dir", lambda: tmp_path / ".grade-cache")

    pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")
    calls = []
    monkeypatch.setattr(pc, "extract_pdf_page_texts",
                        lambda *a, **k: calls.append(1) or [""])
    pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")
    assert calls == [], "a current index must not be re-extracted"


# ---------- it reaches ClaimScore and the DB ----------

@pytestmark_fixtures
def test_claim_score_carries_provenance(tmp_path, monkeypatch):
    """The chunk's label has to survive into `ClaimScore`, since that is what
    `_persist_scores` writes and `claims --include-context` reads back."""
    from researchwiki import paths
    from researchwiki.grade.fidelity import paper as fid

    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(pc, "grade_cache_dir", lambda: tmp_path / ".grade-cache")
    monkeypatch.setattr(fid, "resolve_pdf", lambda stem: FIXTURES / f"{STEM}.pdf")
    pc.build_pdf_index(STEM, pdf_path=FIXTURES / f"{STEM}.pdf")

    class _Claim:
        section = "results"
        position = 0
        text = "MGA is slower than hifiasm because its runtime is dominated by LJA."
        is_cross_ref = False

    full_text = PAGE_SEPARATOR.join(
        extract_pdf_page_texts(FIXTURES / f"{STEM}.pdf", max_pages=pc.MAX_PDF_PAGES)
    )
    score = fid._score_claim(STEM, _Claim(), full_text, use_semantic=False)
    assert score.graded
    assert score.supporting_text
    assert score.supporting_provenance, "top-1 chunk should carry a label"
    assert score.supporting_provenance.startswith("§") or "p." in score.supporting_provenance


def test_ungraded_claim_has_empty_provenance_not_none():
    """`_persist_scores` writes NULL for an empty string; the dataclass field
    stays a str so `.strip()` at every display site is safe."""
    from researchwiki.grade.fidelity.paper import ClaimScore
    s = ClaimScore(
        section="results", position=0, text="x", is_cross_ref=True,
        top1_score=0.0, top3_mean=0.0, top1_chunk_id=-1,
        supporting_text="", supporting_provenance="",
        numeric_tokens=[], numeric_unmatched=[], semantic_score=None,
        negation_mismatch=False, graded=False,
    )
    assert s.supporting_provenance == ""
    assert "supporting_provenance" in s.to_dict()
