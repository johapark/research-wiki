"""`query_pdf` must treat claim prose as prose, not Tantivy query syntax.

Every caller passes a sentence from a wiki page. Tantivy's lenient parser reads
a leading `-` as MUST_NOT, so a Markdown bullet (`- The panel can…`) excluded
every chunk containing its first word and the query returned nothing. The
synthesis grader then marked correctly cited bullet claims `weak`: 24 of 492
bullet claims on the corpus's synthesis and idea pages lost all retrieval.
Hermetic: a hand-built index in tmp_path, no PDF.
"""

from __future__ import annotations

import pytest
import tantivy

from researchwiki.index import pdf_chunks as pc


@pytest.fixture
def index(tmp_path, monkeypatch):
    stem = "asri-2025-x"
    idx_dir = tmp_path / stem
    idx_dir.mkdir()
    index = tantivy.Index(pc._build_schema(), path=str(idx_dir))
    writer = index.writer(heap_size=15_000_000)
    for i, text in enumerate([
        "The panel can homogenize calls: frequent alleles induce false positives.",
        "Multi-allelic tandem repeats remain difficult for graph alignment.",
        "Unrelated chunk about sequencing depth and coverage.",
    ]):
        doc = tantivy.Document()
        doc.add_unsigned("chunk_id", i)
        doc.add_text("text", text)
        writer.add_document(doc)
    writer.commit()
    writer.wait_merging_threads()
    pc._write_index_meta(idx_dir)
    monkeypatch.setattr(pc, "_index_path_for", lambda s: tmp_path / s)
    return stem


def test_a_markdown_bullet_claim_still_retrieves(index):
    hits = pc.query_pdf(index, "- The panel can homogenize calls and induce false positives")
    assert hits and "homogenize" in hits[0].text


@pytest.mark.parametrize("claim", [
    "- **Multi-allelic** tandem repeats are hard",
    "+ tandem repeats remain difficult",
    "repeats - tandem - remain difficult",
])
def test_operator_shaped_prose_retrieves(index, claim):
    assert pc.query_pdf(index, claim)


@pytest.mark.parametrize("raw, cleaned", [
    ("- panel homogenize", " panel homogenize"),
    ("\n- item", "\n item"),
    ("+term", "term"),
    ("x - y", "x  y"),
])
def test_neutralize_drops_only_operator_position_signs(raw, cleaned):
    assert pc._neutralize_operators(raw) == cleaned


@pytest.mark.parametrize("text", [
    "multi-allelic vg-giraffe calls",
    "a 1-3 kb range",
    "Minigraph-Cactus and PGGB",
])
def test_interior_hyphens_are_kept(text):
    """Hyphenated terms are how the corpus spells method names."""
    assert pc._neutralize_operators(text) == text
