"""Disposable extraction retention through real adapters and synthetic responders."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import uuid

from fresh_install_acceptance import write_fixture_pdf


def exercise(root, route):
    from researchwiki.agents import llm, model_config, relay
    from researchwiki.agents.phases.extract import extract_sections
    from researchwiki.agents.phases.target_claims import extract_target_claims
    from researchwiki.records import (
        AttemptFinished, AttemptSpec, CapturePolicy, Producer, RecordStore, extraction_requirements,
    )
    from researchwiki.records.extraction import ExtractionRecorder
    from researchwiki.records.jsonio import decode_json

    root.mkdir()
    original = root / "input.pdf"
    write_fixture_pdf(original, title="Synthetic Extraction Acceptance", author="Ada Lovelace")
    store = RecordStore.create(root / "archive")
    spec = AttemptSpec(attempt_id=str(uuid.uuid4()), policy=CapturePolicy(mode="required"),
                       producer=Producer(name="synthetic-extraction-acceptance", version="1"),
                       requirements=extraction_requirements(), created_at="2026-10-23T12:00:00+00:00")
    store.start_attempt(spec)
    records = ExtractionRecorder(store, spec.attempt_id)
    source = records.snapshot_source(original)
    sections, full_text = extract_sections(source.processing_path, records=records)
    value = {"claims": [{"type": "headline", "content": f"Synthetic finding {i}",
                          "importance": "high", "location": "Results"} for i in range(38)],
             "keywords": ["Synthetic fixture", "benchmark", "precision recall", "variant calling", "AUC"]}
    response = {"model": "synthetic-reported-model", "choices": [{"message": {
        "content": "```json\n" + json.dumps(value) + "\n```"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 9}}
    api_calls, handoffs = [], []
    def urlopen(request, **kwargs):
        api_calls.append(json.loads(request.data))
        return BytesIO(json.dumps(response).encode())
    def handoff(pending, completed, *args, **kwargs):
        handoffs.append(json.loads(pending.read_text()))
        reply = {"claims": "invalid first response"} if len(handoffs) == 1 else value
        relay._write_atomic_json(completed, {"via": "fixture/synthetic-model-1", "structured": reply,
                                             "authorization": "synthetic-secret-not-retained"})
    cfg = model_config.ModelConfig(provider=route, model="synthetic-configured-model", temperature=0.1, max_tokens=4096)
    with patch.object(model_config, "for_phase", return_value=cfg), \
         patch.object(model_config, "target_claims_max_chars", return_value=120000), \
         patch.object(llm.urllib.request, "urlopen", side_effect=urlopen), \
         patch.object(relay, "wiki_root", return_value=root), \
         patch.object(relay, "_emit_handoff_message", side_effect=handoff), \
         patch.dict(os.environ, {"RW_LLM_BASE_URL": "https://fixture.invalid/v1", "RW_RELAY_FRESH": "",
                                 "OPENAI_API_KEY": "synthetic-secret-not-retained"}):
        out = extract_target_claims(metadata={"title": "Synthetic"}, sections=sections,
                                    pdf_full_text=full_text, records=records)
    assert len(out.claims) == 35
    manifest = store.finish_attempt(spec.attempt_id, AttemptFinished(
        execution="finished", outcome="draft_ready", publication="not_published", projections="not_applicable"))
    assert manifest.capture_state == "complete"
    traces = [decode_json(store.read_blob(a.payload)) for a in manifest.artifacts if a.kind == "parse_trace"]
    assert len(traces[-1]["candidates"]) == 38
    assert sum(item["selection_reason"] == "cap" for item in traces[-1]["candidates"]) == 3
    if route == "chat-relay":
        answers = [decode_json(store.read_blob(a.payload)) for a in manifest.artifacts if a.kind == "model_response"]
        assert [item["validation"] for item in answers[:2]] == ["rejected", "accepted"]
        assert len(handoffs) == 2 and not api_calls
        assert not list((root / ".llm-relay").rglob("*.response.json"))
        shutil.rmtree(root / ".llm-relay")
    else:
        assert len(api_calls) == 1 and not handoffs
    original.unlink()
    relocated = root / "relocated-archive"
    shutil.move(store.root, relocated)
    assert RecordStore(relocated).verify(spec.attempt_id).manifest == manifest
    assert not any(p.suffix == ".db" for p in root.rglob("*"))
    assert all(b"synthetic-secret-not-retained" not in p.read_bytes() for p in relocated.rglob("*") if p.is_file())
    return {"route": route, "checklist": len(out.claims), "retained_candidates": 38}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-installed", action="store_true")
    args = parser.parse_args()
    import researchwiki
    if args.expect_installed:
        assert Path(researchwiki.__file__).is_relative_to(Path(sys.prefix)), researchwiki.__file__
        assert "site-packages" in Path(researchwiki.__file__).parts
    with tempfile.TemporaryDirectory(prefix="researchwiki-extraction-acceptance-") as tmp:
        root = Path(tmp)
        results = [exercise(root / route, route) for route in ("openai-compatible", "chat-relay")]
    print(json.dumps({"status": "passed", "scope": "synthetic extraction adapters",
                      "external_provider_calls": 0, "routes": results}))


if __name__ == "__main__":
    main()
