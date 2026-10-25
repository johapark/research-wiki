"""Portable complete grading evidence using a real synthetic PDF and mock API."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import uuid

from fresh_install_acceptance import write_fixture_pdf


REPO = Path(__file__).resolve().parents[2]


def exercise(root):
    from researchwiki.agents import llm, model_config
    from researchwiki.agents.context import Context
    from researchwiki.agents.phases.grade import grade_draft
    from researchwiki.grade.support import llm_support_classifier
    from researchwiki.index import pdf_chunks
    from researchwiki.records import RecordStore
    from researchwiki.records.export import export_attempt
    from researchwiki.records.grading import read_grading
    from researchwiki.records.lifecycle import capture_operation
    from researchwiki.records.lineage import PHASES, note_phase_omission

    root.mkdir()
    shutil.copytree(REPO / "config", root / "config")
    shutil.copytree(REPO / "prompts", root / "prompts")
    for folder in ("wiki/other", "papers", "inbox"):
        (root / folder).mkdir(parents=True)
    config = root / "config/models.grading.yaml"
    config.write_text('ingest:\n  retention:\n    mode: required\n    root: .researchwiki/records\n')
    source = root / "input.pdf"
    write_fixture_pdf(source, title="Synthetic Grading Acceptance", author="Ada Lovelace",
                      extra_lines=[f"Detailed synthetic context {i} reports 42 samples in the acceptance run." for i in range(26)])
    cfg = model_config.ModelConfig(provider="openai-compatible", model="synthetic-grading-model", temperature=0.1, max_tokens=4096)
    calls = []
    def urlopen(request, **kwargs):
        calls.append(json.loads(request.data))
        return BytesIO(json.dumps({"model": "synthetic-reported-model", "choices": [{"message": {
            "content": '{"verdicts":[{"id":0,"verdict":"supported"}]}'}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 9}}).encode())
    original = Path.cwd()
    try:
        os.chdir(root)
        with patch.dict(os.environ, {"RW_MODELS_CONFIG": str(config), "RW_LLM_BASE_URL": "https://fixture.invalid/v1",
                                     "OPENAI_API_KEY": "synthetic-grading-secret", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}), \
             patch.object(model_config, "for_phase", return_value=cfg), \
             patch.object(pdf_chunks, "_build_embeddings_cache", return_value=None), \
             patch.object(llm.urllib.request, "urlopen", side_effect=urlopen), capture_operation(None) as owner:
            ctx = Context(attempt_id=str(uuid.uuid4()), pdf_path=source, pdf_filename=source.name, verify_claim_entailment=True)
            owner.bind(ctx)
            scores, _ = grade_draft(stem="synthetic-grading", draft_text="## Key Contributions\n- The acceptance run reports 42 samples.\n",
                metadata={"title": "Synthetic", "paper_type": "research"}, sandbox_dir=root / "sandbox",
                pdf_path=ctx.processing_pdf_path, use_semantic=False, support_classifier=llm_support_classifier)
            assert scores["support_check_complete"] and len(calls) == 1
            # This fixture calls the grading function directly, without executing
            # the ingest runner's phases. Declare that deliberate scope explicitly.
            for phase in PHASES:
                note_phase_omission(phase, "standalone_grading_fixture")
            owner.finish({"scope": "synthetic-grading-acceptance", "execution": "finished", "outcome": "draft_ready",
                          "publication": "not_published", "projections": "not_applicable"})
            manifest = owner.records.store.read_attempt(ctx.attempt_id)
            assert manifest.capture_state == "complete"
            output = root.parent / "portable-grading"
            export_attempt(owner.records.store, ctx.attempt_id, output)
        for directory in (root / ".researchwiki", root / ".grade-cache", root / "wiki", root / "sandbox"):
            shutil.rmtree(directory)
        source.unlink()
        (root / "state.db").unlink(missing_ok=True)
        store = RecordStore(output)
        assert store.verify(ctx.attempt_id).manifest == manifest
        payloads = read_grading(store, ctx.attempt_id)
        bundle = next(p["payload"] for p in payloads if p["payload"].get("boundary") == "core_evidence")
        assert any(len(c["retrieval_text"]) > 500 and c["span_status"] == "verified" for c in bundle["chunks"].values())
        inputs = next(p["payload"] for p in payloads if p["payload"].get("boundary") == "support_classifier_inputs")
        assert len(inputs["claims"][0]["context"]) == 500
        assert any(p["slot"] == "grading.response" for p in payloads)
        proc = subprocess.run([sys.executable, "-m", "researchwiki", "records", "--root", str(output), "show",
                               ctx.attempt_id, "--grading", "--json"], capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["grading"] == payloads
        assert b"synthetic-grading-secret" not in b"".join(p.read_bytes() for p in output.rglob("*") if p.is_file())
    finally:
        os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-installed", action="store_true")
    args = parser.parse_args()
    import researchwiki
    if args.expect_installed:
        assert Path(researchwiki.__file__).is_relative_to(Path(sys.prefix)), researchwiki.__file__
        assert "site-packages" in Path(researchwiki.__file__).parts
    with tempfile.TemporaryDirectory(prefix="researchwiki-grading-acceptance-") as tmp:
        exercise(Path(tmp) / "library")
    print(json.dumps({"status": "passed", "scope": "synthetic grading retention", "external_provider_calls": 0,
                      "portable_after_cache_and_source_deletion": True}))


if __name__ == "__main__":
    main()
