"""Compare encoders on one corpus, recording quality and warm CPU timings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from ..errors import EnvironmentFailure
from ..log import log
from .fixture import find_fixtures, load_fixture, RetrievalFixture
from .retrieval import diff_retrieval_scores, summarize
from .retrieval_reports import score_retrieval
from .retrieval_preflight import validate_fixture_anchors
from . import retrieval_embeddings as encoders


def _fixtures(stems: list[str] | None):
    names = stems if stems is not None else [s for s in find_fixtures() if s.startswith("retrieval/")]
    available, skipped = [], []
    for name in dict.fromkeys(names):
        fixture = load_fixture(name)
        if not isinstance(fixture, RetrievalFixture):
            raise ValueError(f"{name} is not a retrieval fixture")
        status = validate_fixture_anchors(fixture)
        entry = {"stem": name, "fixture_id": fixture.fixture_id,
                 "inactive_negatives": status.inactive_negatives,
                 "fixture_sha256": hashlib.sha256(fixture.source_path.read_bytes()).hexdigest()}
        if not status.available:
            skipped.append({**entry, "missing_expected": status.missing_expected})
        else:
            available.append((fixture, entry))
    if not available:
        raise EnvironmentFailure("no retrieval fixtures have all expected anchors in this corpus")
    return available, skipped


def _performance(model_name: str, fixtures, *, trust_remote_code: bool, query_prefix: str,
                 doc_prefix: str, repeats: int) -> dict:
    queries = [fixture.query for fixture, _ in fixtures][:5]
    kwargs = {"trust_remote_code": trust_remote_code, "prefix": query_prefix, "role": "query"}
    encoders._embed_with([queries[0]], model_name, **kwargs)
    query_times = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter()
            encoders._embed_with([query], model_name, **kwargs)
            query_times.append((time.perf_counter() - start) * 1000)
    rows = encoders._all_db_claims()
    indices = np.linspace(0, len(rows) - 1, min(64, len(rows)), dtype=int) if rows else []
    texts = [rows[int(i)]["text"] for i in indices]
    doc_kwargs = {"trust_remote_code": trust_remote_code, "prefix": doc_prefix, "role": "document"}
    doc_ms = None
    if texts:
        encoders._embed_with(texts[:8], model_name, **doc_kwargs)
        start = time.perf_counter()
        encoders._embed_with(texts, model_name, **doc_kwargs)
        doc_ms = (time.perf_counter() - start) * 1000
    return {"query_samples": len(query_times),
            "query_encode_median_ms": float(np.median(query_times)),
            "query_encode_p95_ms": float(np.percentile(query_times, 95)),
            "document_sample_count": len(texts), "document_sample_encode_ms": doc_ms,
            "document_texts_per_second": len(texts) * 1000 / doc_ms if doc_ms else None}


def compare_retrieval(baseline: str, candidate: str, *, stems: list[str] | None = None,
                      backends: tuple[str, ...] = ("bm25", "semantic", "hybrid"),
                      baseline_options: dict | None = None, candidate_options: dict | None = None,
                      timing_repeats: int = 3) -> dict:
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be positive")
    if not backends or any(b not in ("bm25", "semantic", "hybrid") for b in backends):
        raise ValueError("backends must contain bm25, semantic, or hybrid")
    fixtures, skipped = _fixtures(stems)
    snapshot = encoders.corpus_fingerprint()
    event_start = len(encoders._CACHE_EVENTS)
    arms, scored_arms = {}, {}
    for label, model, options in (("baseline", baseline, baseline_options or {}),
                                  ("candidate", candidate, candidate_options or {})):
        log(f"benchmarking {label}: {encoders.resolve_model(model)}", tag="retrieval")
        details = encoders.model_details(model, trust_remote_code=options.get("trust_remote_code", False))
        scores, summaries, scored = {}, {}, {}
        for backend in dict.fromkeys(backends):
            items = []
            for fixture, _ in fixtures:
                start = time.perf_counter()
                score = score_retrieval(fixture, model, backend, **options)
                entry = score.to_dict()
                entry["elapsed_ms"] = (time.perf_counter() - start) * 1000
                scores.setdefault(fixture.fixture_id, {})[backend] = entry
                scored.setdefault(fixture.fixture_id, {})[backend] = score
                items.append(score)
            summary = asdict(summarize(items, encoders.resolve_model(model), backend))
            summary.pop("per_fixture")
            summaries[backend] = summary
        timing = _performance(model, fixtures, repeats=timing_repeats,
                               trust_remote_code=options.get("trust_remote_code", False),
                               query_prefix=options.get("query_prefix", ""),
                               doc_prefix=options.get("doc_prefix", ""))
        arms[label] = {"model": details, "encoding_options": options,
                       "performance": timing, "summary": summaries, "scores": scores}
        scored_arms[label] = scored
        if encoders.corpus_fingerprint() != snapshot:
            raise EnvironmentFailure("corpus changed during comparison; rerun against a stable corpus")
    comparisons = []
    for fixture, entry in fixtures:
        for backend in dict.fromkeys(backends):
            scores = [scored_arms[label][fixture.fixture_id][backend]
                      for label in ("baseline", "candidate")]
            comparisons.append({**entry, "backend": backend,
                                "diff": diff_retrieval_scores(*scores).to_dict()})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus": snapshot, "fixtures_scored": len(fixtures), "fixtures_skipped": skipped,
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            **{name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "sentence-transformers")}},
            **arms, "comparisons": comparisons, "cache_events": encoders._CACHE_EVENTS[event_start:]}


def format_comparison(report: dict) -> str:
    lines = [f"Retrieval comparison: {report['fixtures_scored']} fixtures scored; "
             f"{len(report['fixtures_skipped'])} unavailable fixtures skipped.", "",
             f"Baseline: {report['baseline']['model']['model']}",
             f"Candidate: {report['candidate']['model']['model']}", "",
             "| Backend | Baseline nDCG | Candidate nDCG | Baseline critical recall | Candidate critical recall |",
             "|---|---:|---:|---:|---:|"]
    for backend, base in report["baseline"]["summary"].items():
        cand = report["candidate"]["summary"][backend]
        lines.append(f"| {backend} | {base['mean_ndcg']:.3f} | {cand['mean_ndcg']:.3f} | "
                     f"{base['mean_recall_critical']:.1%} | {cand['mean_recall_critical']:.1%} |")
    lines += ["", "Warm single-query encoding (excludes model loading and corpus embedding):"]
    for label in ("baseline", "candidate"):
        timing = report[label]["performance"]
        lines.append(f"- {label}: median {timing['query_encode_median_ms']:.1f} ms; "
                     f"p95 {timing['query_encode_p95_ms']:.1f} ms "
                     f"({timing['query_samples']} samples).")
    lines += ["", "Corpus build timings distinguish new encoding from reused vectors; "
              "use the document sample throughput for a comparison independent of cache warmth."]
    return "\n".join(lines)
