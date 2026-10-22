"""Entailment consumes complete, bounded source evidence, never UI previews."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from researchwiki.agents.phases import grade
from researchwiki.grade import support
from researchwiki.grade.fidelity import paper
from researchwiki.grade.parser import Claim
from researchwiki.index import pdf_chunks


def _hit(chunk_id, text):
    return pdf_chunks.RetrievedChunk(
        chunk_id=chunk_id, score=20.0, text=text,
        page_start=chunk_id + 1, page_end=chunk_id + 1, section="results",
    )


def _report(claims):
    return paper.PaperFidelityReport(
        stem="example-2026-paper", page_path="example-2026-paper.md",
        claims=claims, n_claims=len(claims),
        n_graded=sum(c.graded for c in claims),
        n_cross_refs=sum(c.is_cross_ref for c in claims), salience=None,
        mean_top1=20.0, median_top1=20.0,
        semantic_score=None, semantic_median=None,
        n_negation_mismatches=0, n_with_numeric_drift=0,
        weakest_claim=claims[0].text if claims else None,
        weakest_score=20.0, semantic_available=False,
    )


def _score(text="A supported result.", position=0, *, graded=True):
    return paper.ClaimScore(
        section="results", position=position, text=text,
        is_cross_ref=not graded, top1_score=20.0, top3_mean=20.0,
        top1_chunk_id=0, supporting_text="Capped display preview only.",
        supporting_provenance="p. 1", numeric_tokens=[], numeric_unmatched=[],
        semantic_score=None, negation_mismatch=False, graded=graded,
    )


def _grade(tmp_path, classifier=None):
    return grade.grade_draft(
        stem="example-2026-paper", draft_text="## Results\nA result.",
        metadata={}, sandbox_dir=tmp_path, use_semantic=False,
        support_classifier=classifier,
    )


def test_complete_late_evidence_reaches_llm_prompt(tmp_path, monkeypatch):
    """End-to-end through the optional grade phase and real prompt builder.

    Evidence beyond both old character caps and in a second chunk must reach
    the classifier; the persisted/display preview remains 500 characters.
    """
    from researchwiki.agents import llm

    first = "backgroundword " * 110 + "The enzyme retained activity after heating."
    second = "The cooled enzyme was assayed at room temperature."
    assert first.index("The enzyme") > 1200
    hits = [_hit(0, first), _hit(4, second)]
    query_calls = []

    def query(stem, text, topk, *, pdf_path=None):
        assert pdf_path is None
        query_calls.append((stem, text, topk))
        return hits

    monkeypatch.setattr(pdf_chunks, "query_pdf", query)
    monkeypatch.setattr(paper, "query_pdf", query)
    claim = Claim(
        "results", 0,
        "The enzyme retained activity after heating, assayed after cooling.",
        False,
    )
    score = paper._score_claim("example-2026-paper", claim, first + second, False)
    assert score.supporting_text == first[:500]
    assert "retained activity" not in score.to_dict()["supporting_text"]
    monkeypatch.setattr(grade, "grade_page", lambda **kwargs: _report([score]))

    def call(**kwargs):
        assert kwargs["phase"] == "claim_support"
        assert first in kwargs["prompt"]
        assert second in kwargs["prompt"]
        assert "<Chunk 4; §results, p. 5>" in kwargs["prompt"]
        return SimpleNamespace(text='{"verdicts": [{"id": 0, "verdict": "supported"}]}')

    monkeypatch.setattr(llm, "call", call)
    aggregate, _ = _grade(tmp_path, support.llm_support_classifier)
    assert aggregate["support_check_complete"] is True
    assert aggregate["n_support_checked"] == 1
    assert aggregate["n_unsupported"] == 0
    assert aggregate["checks"]["entailment"] == {
        "requested": True, "completed": True, "status": "passed",
    }
    assert query_calls[-1] == ("example-2026-paper", claim.text, 5)


def test_evidence_is_bounded_deduplicated_and_labelled(monkeypatch):
    hits = [_hit(i, f"Complete evidence {i}.") for i in range(6)]

    def query(stem, text, topk, *, pdf_path=None):
        assert pdf_path is None
        assert topk == 5
        # Even an over-returning backend cannot widen the packet beyond K.
        return hits

    monkeypatch.setattr(pdf_chunks, "query_pdf", query)
    evidence = support.retrieve_support_evidence("paper", "claim")
    for i in range(5):
        assert f"<Chunk {i}; §results, p. {i + 1}>\nComplete evidence {i}." in evidence
    assert "evidence 5" not in evidence
    hits[:] = [hits[0], hits[0], hits[1]]
    evidence = support.retrieve_support_evidence("paper", "claim")
    assert evidence.count("Complete evidence 0.") == 1
    assert "Complete evidence 1." in evidence


@pytest.mark.parametrize("hits", [[], [_hit(0, " \n ")]])
def test_no_evidence_is_empty_not_provenance_only(monkeypatch, hits):
    monkeypatch.setattr(pdf_chunks, "query_pdf", lambda *a, **kw: hits)
    assert support.retrieve_support_evidence("paper", "claim") == ""


def test_opt_out_does_not_retrieve_support_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "grade_page", lambda **kwargs: _report([_score()]))

    def unexpected(*args, **kwargs):
        pytest.fail("Opt-out must not retrieve entailment evidence")

    monkeypatch.setattr(grade, "retrieve_support_evidence", unexpected)
    aggregate, _ = _grade(tmp_path)
    assert "n_unsupported" not in aggregate
    assert "support_check_complete" not in aggregate
    assert aggregate["checks"]["entailment"]["status"] == "not_requested"


def test_missing_evidence_is_incomplete_and_unsupported_still_counts(tmp_path, monkeypatch):
    claims = [
        _score("An unsupported claim.", 0),
        _score("A claim with missing evidence.", 1),
        _score("A cross-reference not graded.", 2, graded=False),
    ]
    monkeypatch.setattr(grade, "grade_page", lambda **kwargs: _report(claims))
    seen_queries = []

    def query(stem, text, topk, *, pdf_path=None):
        assert pdf_path is None
        seen_queries.append(text)
        return [_hit(1, "The source reports a different result.")] if text == claims[0].text else []

    monkeypatch.setattr(pdf_chunks, "query_pdf", query)

    def classify(pairs):
        assert len(pairs) == 1
        assert pairs[0][0] == claims[0].text
        assert "Capped display preview" not in pairs[0][1]
        return ["unsupported"]

    aggregate, _ = _grade(tmp_path, classify)
    assert seen_queries == [c.text for c in claims[:2]]
    assert aggregate["support_check_complete"] is False
    assert aggregate["n_support_expected"] == 2
    assert aggregate["n_support_checked"] == 1
    assert aggregate["n_unsupported"] == 1
    assert aggregate["checks"]["entailment"] == {
        "requested": True, "completed": False, "status": "failed",
    }
    assert aggregate["unsupported_claims"] == [
        {"section": "results", "position": 0, "text": claims[0].text},
    ]


@pytest.fixture
def text_pdf_sources(tmp_path, monkeypatch):
    """Real indexes and source hashes, with only PDF extraction/model work stubbed."""
    monkeypatch.chdir(tmp_path)
    built_from = []

    def extract(path, **kwargs):
        built_from.append(Path(path).resolve())
        return [Path(path).read_text()]

    monkeypatch.setattr(pdf_chunks, "extract_pdf_page_texts", extract)
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *a: None)
    monkeypatch.setattr(paper, "extract_pdf", lambda p, **kw: (Path(p).read_text(), 1))
    monkeypatch.setattr(paper, "score_salience", lambda *a, **kw: None)
    (tmp_path / "papers").mkdir()
    (tmp_path / "inbox").mkdir()
    incoming = tmp_path / "inbox/journal.pdf"
    incoming.write_text(
        "Journal version reports zebrafish regeneration benchmark results "
        "with currentjournalmarker."
    )
    return SimpleNamespace(
        incoming=incoming,
        canonical=tmp_path / "papers/example-2026-paper.pdf",
        built_from=built_from,
    )


@pytest.mark.parametrize("canonical_exists", [False, True], ids=["new-paper", "replacement"])
@pytest.mark.parametrize("relative_string", [False, True], ids=["path", "relative-string"])
def test_support_check_keeps_incoming_pdf_source(
    tmp_path, text_pdf_sources, canonical_exists, relative_string,
):
    """The classifier must see the same incoming PDF as the real fidelity grader."""
    sources = text_pdf_sources
    old_text = (
        "Preprint version reports zebrafish regeneration benchmark results "
        "with obsoletepreprintmarker."
    )
    if canonical_exists:
        sources.canonical.write_text(old_text)
    pdf_path = (
        str(sources.incoming.relative_to(tmp_path)) if relative_string else sources.incoming
    )
    checked_pairs = []

    def classify(pairs):
        checked_pairs.extend(pairs)
        assert len(pairs) == 1
        assert "currentjournalmarker" in pairs[0][1]
        assert "obsoletepreprintmarker" not in pairs[0][1]
        return ["supported"]

    aggregate, _ = grade.grade_draft(
        stem="example-2026-paper",
        draft_text="## Key Contributions\n- Zebrafish regeneration benchmark results reported.",
        metadata={"type": "paper"}, sandbox_dir=tmp_path / "sandbox",
        pdf_path=pdf_path, use_semantic=False, support_classifier=classify,
    )

    assert len(checked_pairs) == 1
    assert aggregate["n_support_checked"] == 1
    assert aggregate["checks"]["entailment"] == {
        "requested": True, "completed": True, "status": "passed",
    }
    assert sources.built_from == [sources.incoming], "the index must not switch sources"
    assert sources.canonical.exists() is canonical_exists
    if canonical_exists:
        assert sources.canonical.read_text() == old_text


def test_support_evidence_without_override_uses_canonical_pdf(text_pdf_sources):
    """Omitting pdf_path must still revalidate against the canonical source."""
    sources = text_pdf_sources
    sources.canonical.write_text(
        "Preprint reports zebrafish regeneration benchmark with obsoletepreprintmarker."
    )
    pdf_chunks.build_pdf_index("example-2026-paper", pdf_path=sources.incoming)

    evidence = support.retrieve_support_evidence("example-2026-paper", "zebrafish benchmark")

    assert "obsoletepreprintmarker" in evidence
    assert "currentjournalmarker" not in evidence
    assert sources.built_from == [sources.incoming, sources.canonical]
