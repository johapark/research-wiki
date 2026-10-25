"""Disposable retention cost experiment; no network, models or library writes.

Run from the checkout root with --output /tmp/retention-cost.json. Each sample
uses a fresh subprocess (imports excluded from timers), an empty grading cache,
then a warm-cache repeat. This exercises real extraction, target/author/grade/
critic/evolve wrappers and deliberately rejects one revision. It is a bounded
phase fixture, not an end-to-end publication or semantic-model benchmark.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from itertools import count
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import uuid


REPO = Path(__file__).resolve().parents[2]
CASES = {"short": (2, 2), "typical": (12, 2), "long": (48, 4)}
DRAFT = """## Summary
Synthetic benchmarking measures retention overhead.
## Key Contributions
- The synthetic experiment measures 999 specimens.
- The synthetic pipeline preserves durable research records.
## Methodology and Architecture
Deterministic source text exercises extraction and retrieval.
## Results
- The synthetic experiment measures 999 specimens.
## Limitations
This fixture does not measure real scientific quality.
## Related Papers
None.
"""
TARGET = {"claims": [{"type": "headline", "content": "The experiment measures 42 specimens.",
                      "importance": "critical", "location": "Results"}],
          "keywords": ["synthetic experiment", "specimens", "durable records", "retrieval", "benchmark"]}


def write_pdf(path, pages):
    """Uncompressed, entirely synthetic text PDF with visible lines on every page."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for page in range(pages):
        number = len(objects) + 1
        kids.append(f"{number} 0 R")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                        f"/Resources << /Font << /F1 3 0 R >> >> /Contents {number + 1} 0 R >>").encode())
        lines = ["Abstract" if page == 0 else "Results",
                 "The synthetic experiment measures 42 specimens.",
                 "The synthetic pipeline preserves durable research records."]
        lines += [f"Marker {page}-{i}: deterministic source text exercises extraction and retrieval." for i in range(32)]
        content = ("BT /F1 10 Tf 50 750 Td " + " ".join(f"({line}) Tj 0 -18 Td" for line in lines) + " ET").encode()
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Count {pages} /Kids [{' '.join(kids)}] >>".encode()
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    data.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    data.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(data)


class Meter:
    """Counts archive payload hash work and event JSON reads, not OS disk I/O."""

    def __init__(self):
        self.stage = "setup"
        self.counts = defaultdict(Counter)
        self.seconds = {}

    @contextmanager
    def timed(self, stage):
        previous, self.stage = self.stage, stage
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] = time.perf_counter() - start
            self.stage = previous

    @contextmanager
    def instrument(self):
        from researchwiki.records import store as module
        from researchwiki.records.storage import Storage

        verify, read, put, blob, project = (Storage.verify_blob, Storage.read_json, Storage.put_stream,
                                           module.RecordStore.read_blob, module.project)
        def verify_count(storage, ref):
            self.counts[self.stage]["payload_bytes_hashed"] += ref.size_bytes
            self.counts[self.stage]["blob_verifications"] += 1
            return verify(storage, ref)
        def read_count(storage, path):
            if path.parent.name == "events":
                self.counts[self.stage]["event_json_reads"] += 1
            return read(storage, path)
        def put_count(storage, stream, **kwargs):
            ref = put(storage, stream, **kwargs)
            self.counts[self.stage]["payload_bytes_hashed"] += ref.size_bytes
            return ref
        def blob_count(store, ref):
            self.counts[self.stage]["payload_bytes_hashed"] += ref.size_bytes
            return blob(store, ref)
        def project_count(events):
            self.counts[self.stage]["projection_events"] += len(events)
            self.counts[self.stage]["projection_calls"] += 1
            return project(events)
        with patch.object(Storage, "verify_blob", verify_count), patch.object(Storage, "read_json", read_count), \
             patch.object(Storage, "put_stream", put_count), patch.object(module.RecordStore, "read_blob", blob_count), \
             patch.object(module, "project", project_count):
            yield


def archive_bytes(store, manifest):
    slots = {key: item.slot for item in manifest.inventory for key in item.artifact_ids}
    unique, referenced = {}, Counter()
    for artifact in manifest.artifacts:
        slot = slots[artifact.artifact_id]
        category = ("source" if artifact.kind == "source_snapshot" else
                    "model" if artifact.kind in ("model_request", "model_response") else
                    "text" if artifact.kind == "representation" else
                    "evidence" if slot.startswith("grading.") else "other_payload")
        for ref in artifact.blobs():
            # All original/effective PDF references belong to source bytes.
            group = "source" if ref.media_type == "application/pdf" else category
            referenced[group] += ref.size_bytes
            unique.setdefault(ref.sha256, (group, ref.size_bytes))
    sizes = Counter()
    for group, size in unique.values():
        sizes[group] += size
    base = store.root / "attempts" / manifest.spec.attempt_id
    sizes["events"] = sum(p.stat().st_size for p in (base / "events").glob("*.json"))
    sizes["checkpoint"] = (base / "manifest.json").stat().st_size
    return {"unique_bytes_by_category": dict(sizes), "referenced_payload_bytes_by_category": dict(referenced),
            "unique_payload_bytes": sum(size for _, size in unique.values()),
            "referenced_payload_bytes": sum(referenced.values()), "unique_blobs": len(unique),
            "archive_total_file_bytes": sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())}


def run_phases(root, pages, slots, route, mode, cache):
    from researchwiki.agents import llm, model_config, relay, runner, runner_support
    from researchwiki.agents.budget import BudgetTracker, IngestBudget, activate
    from researchwiki.agents.context import Context
    from researchwiki.index import pdf_chunks
    from researchwiki.records import RecordStore
    from researchwiki.records.coverage import audit_coverage
    from researchwiki.records.export import export_attempt
    from researchwiki.records.jsonio import canonical_bytes, digest
    from researchwiki.records.lifecycle import capture_operation
    from researchwiki.records.lineage import PHASES, note_phase_omission
    from researchwiki.records.settings import Settings

    meter, iterations, calls = Meter(), [], []
    before_bytes = sum(p.stat().st_size for p in (root / "archive").rglob("*") if p.is_file())
    tracker = BudgetTracker(IngestBudget(max_model_calls=100))
    ids = count(1)
    def iteration(**kwargs):
        iterations.append(kwargs)
        return next(ids)
    def answer(phase):
        calls.append(phase)
        if phase == "target_claims":
            return json.dumps(TARGET)
        if phase == "critic":
            return "Remove the unsupported specimen count."
        if phase in ("author", "evolve"):
            return DRAFT + "\nHANDLE: Synthetic\nHOOK: Synthetic fixture for retention cost."
        raise AssertionError(f"unexpected provider phase: {phase}")
    def api(request, **kwargs):
        text = answer(json.loads(request.data)["model"])
        return BytesIO(json.dumps({"model": "synthetic-benchmark-1", "choices": [{"message": {"content": text}}],
                                  "usage": {"prompt_tokens": 17, "completion_tokens": 9}}).encode())
    def handoff(pending, completed, phase, *args, **kwargs):
        text = answer(phase)
        relay._write_atomic_json(completed, {"via": "fixture/synthetic-benchmark-1", "response": text})
    def config(phase):
        return model_config.ModelConfig(provider=route, model=phase, temperature=0.2, max_tokens=4096)

    ctx = Context(attempt_id=uuid.uuid4().hex, pdf_path=root / "input.pdf", pdf_filename="input.pdf",
                  paper_stem="synthetic-cost", metadata={"title": "Synthetic Cost", "paper_type": "research"},
                  use_semantic=False, sandbox_dir=root / "sandbox")
    with patch("researchwiki.records.lifecycle.selected", return_value=Settings(mode, root=root / "archive")), \
         patch.object(model_config, "for_phase", side_effect=config), \
         patch.object(model_config, "target_claims_max_chars", return_value=120000), \
         patch.object(model_config, "rate_for", return_value=None), \
         patch.object(runner, "write_iteration", side_effect=iteration), \
         patch.object(runner_support, "write_iteration", side_effect=iteration), \
         patch.object(llm.urllib.request, "urlopen", side_effect=api), \
         patch.object(relay, "wiki_root", return_value=root), \
         patch.object(relay, "_emit_handoff_message", side_effect=handoff), \
         patch.object(pdf_chunks, "_build_embeddings_cache", return_value=None), \
         meter.instrument(), activate(tracker), capture_operation(None) as owner:
        with meter.timed("bind"):
            owner.bind(ctx)
        with meter.timed("work"):
            ctx.sections, ctx.pdf_full_text = runner._phase_extract(ctx, None)
            ctx.target_claims = runner._phase_target_claims(ctx, None)
            for slot in range(slots):
                draft = runner._phase_author(ctx, None, 0.2, slot=slot)
                ctx.drafts.append(draft)
                runner._phase_grade(ctx, None, draft)
            ctx.winner = runner._phase_tournament(ctx, None)
            critique = runner._phase_critic(ctx, None, ctx.winner)
            assert critique.weak_claims, "fixture must exercise a real critic call"
            revised = runner._phase_evolve(ctx, None, ctx.winner, critique)
            runner._phase_grade(ctx, None, revised)
            runner_support.record_revision_decision(ctx, None, revised, operator="evolve", accepted=False,
                                                    writer=iteration)
        with meter.timed("finalize"):
            for phase in PHASES:
                if phase not in ("author", "grade", "tournament", "critic", "evolve"):
                    note_phase_omission(phase, "synthetic_cost_fixture_not_requested")
            owner.finish({"scope": "synthetic-cost-phases", "execution": "finished", "outcome": "draft_ready",
                          "publication": "not_published", "projections": "not_applicable"})
            owner.check_coverage()
        result = {"pages": pages, "author_slots": slots, "route": route, "mode": mode, "cache": cache,
                  "source_bytes": ctx.pdf_path.stat().st_size, "source_text_chars": len(ctx.pdf_full_text)}
        if owner.records:
            store = owner.records.store
            with meter.timed("verify"):
                report = store.verify(ctx.attempt_id)
                assert report.manifest.capture_state == "complete"
                assert audit_coverage(store, report.manifest)["status"] == "complete"
            with meter.timed("export"):
                exported = export_attempt(store, ctx.attempt_id, root / f"export-{cache}")
                assert RecordStore(exported).verify(ctx.attempt_id).manifest == report.manifest
            result.update(archive_bytes(store, report.manifest), events=report.event_count)
            result["archive_added_file_bytes"] = result["archive_total_file_bytes"] - before_bytes
        else:
            assert not (root / "archive").exists()
        signature = {"source_text": ctx.pdf_full_text, "targets": [c.to_dict() for c in ctx.target_claims.claims],
                     "drafts": [{"text": d.text, "scores": d.scores} for d in ctx.drafts + [revised]],
                     "winner": ctx.drafts.index(ctx.winner), "calls": calls,
                     "usage": [{k: v for k, v in row.items() if k.startswith("cost_")} for row in iterations]}
        budget = tracker.snapshot()
        result.update(output_sha256=digest(canonical_bytes(signature)), model_calls=len(calls),
                      budget={key: value for key, value in budget.items() if key != "wall_seconds"},
                      seconds=meter.seconds, work_counts=dict(meter.counts))
    return result


def worker(case, route, mode):
    pages, slots = CASES[case]
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="researchwiki-retention-cost-") as tmp:
        root = Path(tmp)
        shutil.copytree(REPO / "prompts", root / "prompts")
        for folder in ("wiki/other", "papers", "inbox"):
            (root / folder).mkdir(parents=True)
        write_pdf(root / "input.pdf", pages)
        try:
            os.chdir(root)
            with patch.dict(os.environ, {"RW_MODELS_CONFIG": str(root / "no-config.yaml"),
                                         "RESEARCHWIKI_DB_PATH": str(root / "unused.db"),
                                         "RW_LLM_BASE_URL": "https://fixture.invalid/v1", "OPENAI_API_KEY": "fixture",
                                         "RW_RELAY_FRESH": "1", "RW_MUTATION_JOURNAL": "1",
                                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}), \
                 redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                results = [run_phases(root, pages, slots, route, mode, cache) for cache in ("cold", "warm")]
            assert not (root / "unused.db").exists()
        finally:
            os.chdir(original)
    # Process peak includes imports and both cache runs; not per-attempt allocation.
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        peak = None
    return {"case": case, "process_peak_rss_bytes": peak, "runs": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--routes", nargs="+", choices=("openai-compatible", "chat-relay"),
                        default=["openai-compatible", "chat-relay"])
    parser.add_argument("--worker", nargs=3, metavar=("CASE", "ROUTE", "MODE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return
    if args.output is None or args.repeats < 1:
        parser.error("--output and positive --repeats are required")
    samples = []
    for case in args.cases:
        for route in args.routes:
            for repeat in range(args.repeats):
                pair = []
                # Alternate order to reduce a systematic mode-order bias.
                for mode in (("off", "required") if repeat % 2 == 0 else ("required", "off")):
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", case, route, mode],
                                          cwd=REPO, capture_output=True, text=True, timeout=300)
                    if proc.returncode:
                        raise RuntimeError(f"benchmark worker failed ({case}, {route}, {mode}):\n{proc.stderr}")
                    sample = {"repeat": repeat, **json.loads(proc.stdout)}
                    pair.append(sample)
                    samples.append(sample)
                for cache_index in (0, 1):
                    before, after = [s["runs"][cache_index] for s in pair]
                    assert before["output_sha256"] == after["output_sha256"], "capture changed scientific outputs/usage"
                    assert before["model_calls"] == after["model_calls"] and before["budget"] == after["budget"]
                print(f"passed {case} {route} repeat {repeat + 1}", file=sys.stderr, flush=True)
    args.output.write_text(json.dumps({"schema_version": 1, "python": platform.python_version(),
                                      "platform": platform.platform(), "samples": samples}, indent=2) + "\n")


if __name__ == "__main__":
    main()
