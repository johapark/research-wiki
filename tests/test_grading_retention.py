"""Full grading observations with synthetic text and deterministic responders."""

from io import BytesIO
from dataclasses import replace
import json
from pathlib import Path
import shutil
import uuid
import base64

import numpy as np
import pytest

from researchwiki.agents import llm, model_config, relay
from researchwiki.agents.budget import BudgetTracker, IngestBudget, activate
from researchwiki.agents.context import Context
from researchwiki.agents.phases.grade import grade_draft
from researchwiki.grade.fidelity import paper
from researchwiki.grade.support import llm_support_classifier
from researchwiki.index import pdf_chunks as chunks, embeddings
from researchwiki.records import BlobRef, RecordCorrupt, RecordStore, RecordStoreUnavailable
from researchwiki.records.export import export_attempt
from researchwiki.records.grading import current_run
from researchwiki.records.lifecycle import capture_operation
from researchwiki.records.jsonio import decode_json
from tests.test_ingest_retention_lifecycle import library as library


@pytest.fixture
def grading(library, monkeypatch):
    root, request = library
    # Controlled extractor output stresses long passages and whitespace offsets.
    # The lifecycle tests separately exercise the real PDFium extraction path.
    pages = ["Abstract\nSynthetic source for grading.\n\nResults\n" +
             "\n\n".join(f"marker{i} " + ("synthetic detailed context " * 60) + f" result {i}." for i in range(12)),
             "Discussion\nA full-document fallback contains 42 specimens and p <0 0065 measured units."]
    monkeypatch.setattr(chunks, "extract_pdf_page_texts", lambda *args, **kwargs: pages)
    monkeypatch.setattr(paper, "extract_pdf", lambda *args, **kwargs: ("\n\n".join(pages), {}))
    monkeypatch.setattr(chunks, "_build_embeddings_cache", lambda *args: None)
    monkeypatch.setattr(embeddings, "_get_model", lambda *args: None)
    with capture_operation(None) as owner:
        ctx = Context(attempt_id=str(uuid.uuid4()), pdf_path=Path(request.source_pdf), pdf_filename="source.pdf")
        owner.bind(ctx)
        yield root, ctx, owner


def write_page(root, text):
    path = root / "wiki/other/grading-fixture.md"
    path.write_text("---\ntype: paper\ntitle: Synthetic\n---\n## Key Contributions\n- " + text + "\n")
    return path


def grade(grading, text="The synthetic result uses 42 specimens.", semantic=False):
    root, ctx, _ = grading
    return paper.grade_page("grading-fixture", page_path=write_page(root, text), semantic=semantic,
                            persist=False, pdf_path=ctx.processing_pdf_path, include_salience=False)


def payloads(owner, slot):
    store, attempt = owner.records.store, owner.records.spec.attempt_id
    manifest = store.read_attempt(attempt)
    ids = next(i.artifact_ids for i in manifest.inventory if i.slot == slot)
    return [decode_json(store.read_blob(a.payload)) for a in manifest.artifacts if a.artifact_id in ids]


def evidence(owner):
    return next(value for value in payloads(owner, "grading.evidence") if value["boundary"] == "core_evidence")


def test_complete_candidates_and_exact_source_spans_survive_preview_truncation(grading):
    _, _, owner = grading
    report = grade(grading)
    bundle = evidence(owner)
    candidate = bundle["chunks"][bundle["claims"][0]["bm25"][0]["chunk"]]
    assert candidate["role"] == "retrieval_candidate"
    assert len(candidate["retrieval_text"]) > 500
    assert report.claims[0].supporting_text == candidate["retrieval_text"][:500]
    assert candidate["span_status"] == "verified"
    assert " ".join(candidate["source_text"].split()) == candidate["retrieval_text"]
    assert "\n" in candidate["source_text"]
    source = next(v for v in payloads(owner, "grading.source") if v.get("transform") == "pdfium-page-repair-v1")
    start, end = candidate["source_span"]
    assert source["text"][start:end] == candidate["source_text"]
    assert len(bundle["claims"][0]["bm25"]) == 5
    assert [v["rank"] for v in bundle["claims"][0]["bm25"]] == [1, 2, 3, 4, 5]


def test_numeric_full_pdf_fallback_and_normalized_inputs_are_distinct(grading, monkeypatch):
    _, _, owner = grading
    query = paper.query_pdf
    monkeypatch.setattr(paper, "query_pdf", lambda *args, **kwargs: [h for h in query(*args, **kwargs) if "specimens" not in h.text])
    report = grade(grading, "The synthetic result measures 42 specimens and 0.0065 units.")
    claim = evidence(owner)["claims"][0]
    numeric = claim["numeric"]
    fallback = next(v for v in numeric["matches"] if v["token"] == "42")
    assert fallback["full_pdf"] and not fallback["retrieved"]
    assert not report.claims[0].numeric_unmatched
    assert numeric["role"] == "value_presence_only"
    normalized = owner.records.store.read_blob(BlobRef.from_dict(numeric["normalized_full_pdf"]))
    assert "0.0065" in decode_json(normalized)["text"]
    raw = next(v for v in payloads(owner, "grading.source") if v.get("role") == "numeric_full_pdf_input")
    assert "0 0065" in raw["text"]


def test_negation_retains_exact_checked_neighborhood_and_soft_role(grading):
    _, _, owner = grading
    grade(grading, "The synthetic result does not include an exclusion rule.")
    claim = evidence(owner)["claims"][0]
    negation = claim["negation"]
    texts = [evidence(owner)["chunks"][h["chunk"]]["retrieval_text"] for h in claim["bm25"][:3]]
    assert negation["evidence_text"] == " ".join(texts)
    assert negation["role"] == "advisory_parity" and negation["mismatch"] is True
    assert negation["claim_tokens"] and not negation["evidence_tokens"]


def test_no_hits_and_cross_references_never_claim_numeric_or_support_verification(grading, monkeypatch):
    _, _, owner = grading
    monkeypatch.setattr(paper, "query_pdf", lambda *args, **kwargs: [])
    report = grade(grading)
    claim = evidence(owner)["claims"][0]
    assert report.claims[0].top1_chunk_id == -1 and not claim["bm25"]
    assert claim["numeric"]["status"] == "not_checked" and claim["negation"]["status"] == "not_checked"
    report = grade(grading, "[[other-source]] cites this paper.")
    assert report.claims[0].is_cross_ref and not report.claims[0].graded


def test_index_cache_version_and_offsets_preserve_existing_retrieval_text(grading):
    _, ctx, _ = grading
    grade(grading)
    metadata, source, digest = chunks.grading_snapshot("grading-fixture")
    assert chunks.CACHE_VERSION == 3 and metadata["source_representation_sha256"] == digest
    hits = chunks.query_pdf("grading-fixture", "synthetic context", pdf_path=ctx.processing_pdf_path)
    assert hits
    for hit in hits:
        assert " ".join(source["text"][hit.char_start:hit.char_end].split()) == hit.text
        assert hit.representation_sha256 == digest


def test_semantic_order_scores_and_selected_index_are_retained(grading, monkeypatch):
    _, _, owner = grading
    monkeypatch.setattr(embeddings, "_get_model", lambda *args: object())
    monkeypatch.setattr(paper, "get_chunk_embeddings", lambda *args, **kwargs: None)
    def embed(texts):
        return np.array([[1.0, 0.0] if len(t) < 100 else [0.7, 0.3] for t in texts], dtype=np.float32)
    monkeypatch.setattr(embeddings, "embed_texts", embed)
    report = grade(grading, semantic=True)
    claim = evidence(owner)["claims"][0]
    assert len(claim["semantic"]) == 10
    assert claim["semantic_check"]["status"] == "completed"
    assert len(claim["semantic_check"]["candidate_scores"]) == 10
    assert claim["semantic_check"]["best_index"] == 0
    assert report.claims[0].semantic_score == pytest.approx(0.7)
    assert claim["semantic_check"]["model_identity_origin"] == "configured"
    bundle = evidence(owner)
    def vector(identity):
        data = bundle["vectors"][identity]
        return np.frombuffer(base64.b64decode(data["data_base64"]), dtype=np.dtype(data["dtype"])).reshape(data["shape"])
    query = vector(claim["semantic_check"]["claim_vector"])
    candidates = np.stack([vector(identity) for identity in claim["semantic_check"]["candidate_vectors"]])
    assert np.allclose(candidates @ query, claim["semantic_check"]["candidate_scores"])


def test_index_representation_corruption_blocks_capture_and_further_model_calls(grading, monkeypatch):
    _, ctx, owner = grading
    directory = chunks.build_pdf_index("grading-fixture", pdf_path=ctx.processing_pdf_path)
    path = directory / chunks.SOURCE_REPRESENTATION_FILENAME
    value = decode_json(path.read_bytes())
    value["text"] += " substituted text"
    from researchwiki.records.jsonio import canonical_bytes
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(RecordCorrupt, match="representation changed"):
        grade(grading)
    assert owner.records.failure is not None
    with pytest.raises(RecordStoreUnavailable):
        llm.call(phase="author", prompt="must not proceed", use_stub=True)


def run_support(grading, classifier):
    _, ctx, _ = grading
    return grade_draft(stem="grading-fixture", draft_text="## Key Contributions\n- The synthetic result uses 42 specimens.\n",
                       metadata={"title": "Synthetic", "paper_type": "research"}, sandbox_dir=ctx.pdf_path.parent / "sandbox",
                       pdf_path=ctx.processing_pdf_path, use_semantic=False, support_classifier=classifier)


@pytest.mark.parametrize("route", ["openai-compatible", "chat-relay"])
def test_support_call_inputs_responses_and_verdicts_have_grading_lineage(grading, monkeypatch, route):
    root, _, owner = grading
    calls = []
    verdict = {"verdicts": [{"id": 0, "verdict": "supported"}]}
    cfg = model_config.ModelConfig(provider=route, model="fixture-support-model", temperature=0.1, max_tokens=4096)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setenv("RW_LLM_BASE_URL", "https://fixture.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-retain-key")
    def urlopen(request, **kwargs):
        calls.append(json.loads(request.data))
        return BytesIO(json.dumps({"model": "reported-fixture-model", "choices": [{"message": {"content": json.dumps(verdict)}}],
                                  "usage": {"prompt_tokens": 17, "completion_tokens": 9}}).encode())
    def handoff(pending, completed, *args, **kwargs):
        calls.append(json.loads(pending.read_text()))
        reply = {"verdicts": [{"id": 0, "verdict": "invalid"}]} if len(calls) == 1 else verdict
        relay._write_atomic_json(completed, {"via": "test/fixture-support-1", "structured": reply})
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(relay, "wiki_root", lambda: root)
    monkeypatch.setattr(relay, "_emit_handoff_message", handoff)
    scores, _ = run_support(grading, llm_support_classifier)
    assert scores["support_check_complete"] and scores["n_unsupported"] == 0
    inputs = next(v for v in payloads(owner, "grading.input") if v["boundary"] == "support_classifier_inputs")
    assert len(inputs["claims"][0]["context"]) == 500
    assert len(calls) == (1 if route == "openai-compatible" else 2)
    replies = payloads(owner, "grading.response")
    assert all(v["run_id"] == inputs["run_id"] for v in replies)
    if route == "chat-relay":
        assert [v["validation"] for v in replies[:2]] == ["rejected", "accepted"]
        assert not list((root / ".llm-relay").rglob("*.response.json"))
    assert payloads(owner, "target.response") == []
    assert owner.records.grading_call_ids and not owner.records.response_ids
    assert b"do-not-retain-key" not in b"".join(p.read_bytes() for p in owner.records.store.root.rglob("*") if p.is_file())


def test_failed_classifier_is_retained_without_fabricated_verdict(grading):
    _, _, owner = grading
    with pytest.raises(ValueError):
        run_support(grading, lambda pairs: [])
    result = payloads(owner, "grading.result")[-1]
    assert result["status"] == "failed" and result["error_code"] == "ValueError"
    final = payloads(owner, "grading.evidence")[-1]
    assert any(v["boundary"] == "support_classifier_return" and v["returned_json"] == "[]" for v in final["additional_checks"])
    assert not any(v.get("boundary") == "support_classifier_result" and v.get("status") == "completed" for v in final["additional_checks"])


def test_archive_failure_before_judge_prevents_model_call(grading, monkeypatch):
    _, _, owner = grading
    original = owner.records.store.record_artifact
    def write(slot, item):
        if slot == "grading.evidence":
            raise RecordStoreUnavailable("evidence archive failed")
        return original(slot, item)
    monkeypatch.setattr(owner.records.store, "record_artifact", write)
    monkeypatch.setattr(llm, "call", lambda **kwargs: pytest.fail("must retain evidence before dispatch"))
    with pytest.raises(RecordStoreUnavailable):
        run_support(grading, llm_support_classifier)


def test_captured_grading_verifies_after_all_sources_and_caches_are_removed(grading, tmp_path):
    root, ctx, owner = grading
    grade(grading)
    output = tmp_path / "portable-grading"
    export_attempt(owner.records.store, ctx.attempt_id, output)
    before = RecordStore(output).read_attempt(ctx.attempt_id)
    shutil.rmtree(owner.records.store.root)
    shutil.rmtree(chunks.grade_cache_dir())
    ctx.pdf_path.unlink()
    shutil.rmtree(root / "wiki")
    (root / "state.db").unlink(missing_ok=True)
    store = RecordStore(output)
    assert store.verify(ctx.attempt_id).manifest == before
    assert any(a.kind == "representation" for a in before.artifacts)


def test_disabling_capture_preserves_grade_values_and_creates_no_archive(library, monkeypatch):
    root, request = library
    monkeypatch.setattr(chunks, "_build_embeddings_cache", lambda *args: None)
    path = write_page(root, "Acceptance pipelines preserve durable research records.")
    expected = paper.grade_page("grading-fixture", page_path=path, semantic=False, persist=False,
                                pdf_path=request.source_pdf, include_salience=False).to_dict()
    assert not (root / ".researchwiki").exists()
    with capture_operation(None) as owner:
        ctx = Context(attempt_id=str(uuid.uuid4()), pdf_path=Path(request.source_pdf), pdf_filename="source.pdf")
        owner.bind(ctx)
        actual = paper.grade_page("grading-fixture", page_path=path, semantic=False, persist=False,
                                  pdf_path=ctx.processing_pdf_path, include_salience=False).to_dict()
    assert actual == expected
    assert current_run() is None


def test_support_response_archive_failure_settles_reported_usage_once(grading, monkeypatch):
    _, _, owner = grading
    cfg = model_config.ModelConfig(provider="openai-compatible", model="fixture-support-model", temperature=0.1, max_tokens=4096)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setenv("RW_LLM_BASE_URL", "https://fixture.invalid/v1")
    monkeypatch.setattr(model_config, "estimate_usd", lambda model, inp, out, **kwargs: (inp + out) / 1000)
    calls = []
    def urlopen(request, **kwargs):
        calls.append(request)
        return BytesIO(json.dumps({"choices": [{"message": {"content": '{"verdicts":[{"id":0,"verdict":"supported"}]}'}}],
                                  "usage": {"prompt_tokens": 17, "completion_tokens": 9}}).encode())
    original = owner.records.store.record_artifact
    def write(slot, item):
        if slot == "grading.response":
            raise RecordStoreUnavailable("cannot retain support reply")
        return original(slot, item)
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(owner.records.store, "record_artifact", write)
    tracker = BudgetTracker(IngestBudget(max_model_calls=2))
    with activate(tracker), pytest.raises(RecordStoreUnavailable):
        run_support(grading, llm_support_classifier)
    assert len(calls) == tracker.calls == 1 and tracker.tokens == 26
    assert tracker.cost_usd == pytest.approx(0.026)
    assert tracker.reserved_tokens == tracker.reserved_cost_usd == 0


def test_missing_support_context_has_explicit_not_run_status(grading, monkeypatch):
    _, _, owner = grading
    monkeypatch.setattr(paper, "query_pdf", lambda *args, **kwargs: [])
    scores, _ = run_support(grading, lambda pairs: pytest.fail("no evidence was eligible"))
    assert scores["n_support_checked"] == 0 and not scores["support_check_complete"]
    final = payloads(owner, "grading.evidence")[-1]
    assert any(v.get("status") == "not_run" and v.get("reason") == "no_eligible_context" for v in final["additional_checks"])
    assert not payloads(owner, "grading.request")


def test_page_input_is_retained_when_index_preparation_fails(grading, monkeypatch):
    _, _, owner = grading
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic index failure")
    monkeypatch.setattr(paper, "build_pdf_index", fail)
    with pytest.raises(RuntimeError, match="index failure"):
        grade(grading)
    inputs = payloads(owner, "grading.input")
    assert any(v.get("boundary") == "page_claim_input" and "42 specimens" in v["page_body"] for v in inputs)
    assert payloads(owner, "grading.result")[-1]["status"] == "failed"


def test_old_cache_version_rebuilds_exact_offsets_without_changing_hits(grading):
    _, ctx, _ = grading
    grade(grading)
    before = chunks.query_pdf("grading-fixture", "synthetic", pdf_path=ctx.processing_pdf_path)
    directory = chunks.grade_cache_dir() / "grading-fixture"
    path = directory / chunks.INDEX_META_FILENAME
    metadata = decode_json(path.read_bytes())
    metadata["cache_version"] = 2
    path.write_text(json.dumps(metadata))
    (directory / chunks.SOURCE_REPRESENTATION_FILENAME).unlink()
    after = chunks.query_pdf("grading-fixture", "synthetic", pdf_path=ctx.processing_pdf_path)
    assert [(h.chunk_id, h.text, h.score) for h in before] == [(h.chunk_id, h.text, h.score) for h in after]
    assert decode_json(path.read_bytes())["cache_version"] == 3
    assert all(h.char_start is not None and h.representation_sha256 for h in after)


def test_retrieval_from_wrong_index_representation_is_rejected(grading, monkeypatch):
    _, _, owner = grading
    query = paper.query_pdf
    def swapped(*args, **kwargs):
        return [replace(h, representation_sha256="f" * 64) for h in query(*args, **kwargs)]
    monkeypatch.setattr(paper, "query_pdf", swapped)
    with pytest.raises(RecordCorrupt, match="different index representation"):
        grade(grading)
    assert owner.records.failure is not None


def test_grading_payloads_can_be_inspected_through_cli_after_export(grading, tmp_path, capsys):
    _, ctx, owner = grading
    grade(grading)
    output = tmp_path / "portable-cli"
    export_attempt(owner.records.store, ctx.attempt_id, output)
    from researchwiki.__main__ import main
    assert main(["records", "--root", str(output), "show", ctx.attempt_id, "--grading", "--json"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert any(item["slot"] == "grading.evidence" and item["payload"]["chunks"] for item in result["grading"])
    assert any(item["slot"] == "grading.input" and item["payload"].get("boundary") == "page_claim_input" for item in result["grading"])
