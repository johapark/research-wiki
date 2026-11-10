"""Repairs use complete bounded passages from the same PDF as grading."""

from types import SimpleNamespace

import pytest

from researchwiki.agents.phases import revise, revision_evidence as re
from researchwiki.agents.phases.draft import Draft
from researchwiki.index import pdf_chunks as pc


def _hit(cid=1, text="Late periodic glycine steering affects untouched residues.", score=5):
    return pc.RetrievedChunk(cid, score, text, 17, 18, "results")


def _build(**kwargs):
    return re.build_repair_evidence(
        sections={"introduction": "BROAD_FALLBACK"},
        pdf_full_text="opening context " * 5000,
        stem="paper", pdf_path="inbox/incoming.pdf", **kwargs,
    )


def test_late_evidence_has_provenance_override_and_one_copy(monkeypatch):
    calls = []

    def query(stem, query, **kwargs):
        calls.append((stem, query, kwargs))
        return [_hit(), _hit()]

    monkeypatch.setattr(pc, "query_pdf", query)
    result = _build(issues=[("target-001", "periodic glycine steering"),
                            ("results#1", "glycine affects untouched residues")])
    assert result.metrics["evidence_route"] == "targeted"
    assert result.text.count(_hit().text) == 1
    assert "§results, pp. 17-18" in result.text
    assert "target-001" in result.text and "results#1" in result.text
    assert len(calls) == 2
    assert all(c[2] == {"topk": 2, "pdf_path": "inbox/incoming.pdf"} for c in calls)
    assert result.metrics["evidence_queries"] == 2
    assert result.metrics["evidence_chunks"] == 1
    assert result.metrics["evidence_packet_chars"] <= re.PACKET_CHARS
    assert result.metrics["evidence_context_chars"] <= re.CONTEXT_CHARS
    assert len(result.text) <= re.PACKET_CHARS + re.CONTEXT_CHARS


def test_packet_skips_over_budget_chunks_without_slicing(monkeypatch):
    large = _hit(1, "periodic glycine " + "x" * re.PACKET_CHARS)
    smaller = _hit(2, "periodic glycine " + "y" * 11_000 + " COMPLETE_CHUNK_END")
    monkeypatch.setattr(pc, "query_pdf", lambda *a, **kw: [large, smaller])
    result = _build(issues=[("target-001", "periodic glycine")])
    assert result.metrics["evidence_route"] == "targeted"
    assert smaller.text in result.text
    assert large.text[:300] not in result.text
    assert len(result.text) <= 16_000


@pytest.mark.parametrize("hits", [[], [_hit(text="irrelevant background only")],
                                       [_hit(score=0)], [_hit(text="")]])
def test_unusable_hits_take_broad_fallback(monkeypatch, hits):
    monkeypatch.setattr(pc, "query_pdf", lambda *a, **kw: hits)
    result = _build(issues=[("target-001", "periodic glycine")])
    assert result.metrics["evidence_route"] == "broad"
    assert "BROAD_FALLBACK" in result.text


def test_one_missing_issue_discards_the_whole_targeted_route(monkeypatch):
    monkeypatch.setattr(pc, "query_pdf", lambda stem, query, **kw:
                        [_hit()] if "glycine" in query else [])
    result = _build(issues=[("target-001", "periodic glycine"),
                            ("target-002", "unseen second issue")])
    assert result.metrics["evidence_route"] == "broad"
    assert _hit().text not in result.text
    assert "BROAD_FALLBACK" in result.text


def test_issue_limit_uses_broad_context_without_queries(monkeypatch):
    monkeypatch.setattr(pc, "query_pdf", lambda *a, **kw: pytest.fail("query cap"))
    result = _build(issues=[(str(i), "periodic glycine") for i in range(5)])
    assert result.metrics["evidence_route"] == "broad"
    assert result.metrics["evidence_queries"] == 0


def test_four_queries_are_allowed_and_packet_overflow_falls_back(monkeypatch):
    calls = []

    def query(stem, text, **kwargs):
        cid = len(calls)
        calls.append(text)
        return [_hit(cid, "periodic glycine " + "z" * 3500)]

    monkeypatch.setattr(pc, "query_pdf", query)
    result = _build(issues=[(str(i), "periodic glycine") for i in range(4)])
    assert len(calls) == 4
    assert result.metrics["evidence_queries"] == 4
    assert result.metrics["evidence_route"] == "broad"
    assert result.metrics["evidence_fallback_reason"] == "packet-limit"
    assert "z" * 3500 not in result.text


def test_retrieval_failure_is_logged_and_falls_back(monkeypatch, capsys):
    def query(*a, **kw):
        raise OSError("cannot open index")

    monkeypatch.setattr(pc, "query_pdf", query)
    result = _build(issues=[("target-001", "periodic glycine")])
    assert result.metrics["evidence_fallback_reason"] == "retrieval-error"
    assert "using broad PDF context" in capsys.readouterr().err


def test_critical_gaps_precede_high_gaps_and_weak_claims():
    weak = SimpleNamespace(section="results", position=2, text="weak claim")
    issues = re.repair_issues([
        {"id": "high", "importance": "high", "text": "high gap"},
        {"id": "critical", "importance": "critical", "text": "critical gap"},
    ], [weak])
    assert [text for _, text in issues] == ["critical gap", "high gap", "weak claim"]


def test_debug_expansion_keeps_broad_source_without_retrieval(monkeypatch):
    monkeypatch.setattr(pc, "query_pdf", lambda *a, **kw: pytest.fail("expansion has no specific query"))
    monkeypatch.setattr(revise.llm, "call", lambda **kw: SimpleNamespace(
        text=kw["prompt"], model="m", temperature=0, input_tokens=1, output_tokens=1,
    ))
    draft = Draft("page", "m", 0, 0, 0)
    result = revise.debug(draft=draft, metadata={}, sections={"results": "BROAD_SOURCE"},
                          issues=["n_kc"], gate_reasons=[], stem="paper", pdf_path="inbox/p.pdf")
    assert "BROAD_SOURCE" in result.text
    assert result.evidence_metrics["evidence_route"] == "broad"


def test_index_retrieves_late_incoming_pdf_not_old_canonical(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "grade_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(pc, "_build_embeddings_cache", lambda *a: None)
    canonical, incoming = tmp_path / "old.pdf", tmp_path / "incoming.pdf"
    canonical.write_bytes(b"old source")
    incoming.write_bytes(b"incoming source")
    opening = "Opening background " * 3000
    pages = [opening, "Late periodic glycine steering affects untouched residues. " * 60]
    monkeypatch.setattr(pc, "extract_pdf_page_texts", lambda path, **kw:
                        pages if path == incoming else ["outdated canonical content " * 300])
    monkeypatch.setattr(pc, "resolve_pdf", lambda stem: canonical)
    pc.build_pdf_index("paper", pdf_path=canonical)
    result = re.build_repair_evidence(
        stem="paper", pdf_path=incoming, sections={}, pdf_full_text=opening,
        issues=[("target-late", "periodic glycine steering untouched residues")],
    )
    assert result.metrics["evidence_route"] == "targeted"
    assert "Late periodic glycine steering affects untouched residues." in result.text
    assert "p. 2" in result.text
    assert "outdated canonical content" not in result.text
