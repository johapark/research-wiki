"""Scoped grading inputs and observations; retrieval candidates remain candidates."""

from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
import hashlib
import re
import uuid
from pathlib import Path
import base64

from .contracts import Requirement, SourceRef
from .errors import RecordCorrupt, RecordStoreUnavailable


_run = ContextVar("researchwiki_grading_run", default=None)
_claim = ContextVar("researchwiki_grading_claim", default=None)


def _json_value(value):
    """Explicit projections for local paths/tuples; no arbitrary object dumps."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def current_run():
    return _run.get()


def claim_active():
    return _claim.get() is not None


def grading_requirements(verify=False):
    return tuple(Requirement(slot=f"grading.{name}", kind=kind, required=required) for name, kind, required in (
        ("source", "representation", True), ("input", "parse_trace", True),
        ("evidence", "parse_trace", True), ("result", "parse_trace", True),
        ("request", "model_request", verify), ("response", "model_response", verify)))


class GradingRun:
    def __init__(self, records, role, iteration):
        self.records, self.role = records, role
        self.run_id = uuid.uuid4().hex
        from .lineage import current_scope
        scope = current_scope()
        self.parents = [scope.input_id] if scope is not None else []
        self.response_ids = []
        self.chunks, self.claims, self.extra, self.vectors = {}, [], [], {}
        self.index_text, self.index_digest = None, None
        self.support_input = None
        self.normalized_sources = {}
        self.started = self.emit("input", {"boundary": "run_started", "role": role,
                                          "iteration": iteration, "coverage_version": 1})
        self.parents.append(self.started)
        records.grading_runs.append(self.run_id)

    def emit(self, slot, value, parents=None):
        return self.records.artifact(f"grading.{slot}", "parse_trace",
                                     _json_value({"schema_version": 1, "run_id": self.run_id, **value}),
                                     parents=tuple(self.parents if parents is None else parents))

    def representation(self, value):
        records = self.records
        blob = records._write(records.store.put_json, value)
        ref = SourceRef(document=records.source.source.document, role="effective", representation=blob)
        item = records.artifact("grading.source", "representation", None, payload=blob,
                                sources=(ref,), parents=(self.started, records.source.artifact_id))
        self.parents.append(item)
        return blob

    def page_input(self, page, claims, options):
        self.parents.append(self.emit("input", {"boundary": "page_claim_input", "stem": page.stem,
                                               "page_body": page.body, "claims": [asdict(c) for c in claims],
                                               "options": options, "parser": "paper-claims-v1"}))

    def inputs(self, page, claims, pdf_digest, full_text, metadata, representation, representation_digest, semantic):
        if (pdf_digest != self.records.source.source.document.sha256 or metadata.get("source_pdf_sha256") != pdf_digest
                or representation.get("source_pdf_sha256") != pdf_digest):
            raise RecordCorrupt("grading source differs from the retained processing source")
        if metadata.get("source_representation_sha256") != representation_digest:
            raise RecordCorrupt("grading index representation changed after indexing")
        self.semantic = semantic
        self.index_text = representation["text"]
        self.index_digest = self.representation(representation).sha256
        if self.index_digest != representation_digest:
            raise RecordCorrupt("grading index representation identity mismatch")
        numeric = self.representation({"schema_version": 1, "role": "numeric_full_pdf_input", "text": full_text,
                                       "transform": "pdfium-joined-repair-v1", "page_limit": metadata.get("max_pdf_pages")})
        self.numeric_source = numeric.to_dict()
        value = {"boundary": "grade_inputs", "stem": page.stem, "page_body": page.body,
                 "parser": "paper-claims-v1", "claims": [asdict(c) for c in claims],
                 "numeric_source": self.numeric_source,
                 "semantic": semantic,
                 "retriever": {"name": "tantivy-bm25", "query_transform": "neutralize-operators-v1",
                               "numeric_topk": 5, "numeric_context_topk": 3, "semantic_topk": 10,
                               "index": {k: metadata.get(k) for k in (
                                   "cache_version", "source_pdf_sha256", "source_representation_sha256",
                                   "chunk_words", "overlap_words", "max_pdf_pages", "text_transform", "retriever_version")}}}
        from ..fsatomic import file_sha256
        from importlib import import_module
        value["implementation"] = {name: file_sha256(Path(import_module(name).__file__)) for name in (
            "researchwiki.grade.fidelity.paper", "researchwiki.grade.primitives", "researchwiki.grade.scorer",
            "researchwiki.grade.support", "researchwiki.index.pdf_chunks", "researchwiki.index.embeddings")}
        self.parents.append(self.emit("input", value))

    def hits(self, label, hits):
        state = _claim.get()
        if state is None:
            return
        ranked = []
        for rank, hit in enumerate(hits, 1):
            identity = getattr(hit, "representation_sha256", None)
            if identity is not None and identity != self.index_digest:
                raise RecordCorrupt("retrieved evidence belongs to a different index representation")
            start, end = getattr(hit, "char_start", None), getattr(hit, "char_end", None)
            raw, status = None, "unavailable"
            if start is not None and end is not None and identity is not None:
                if not (0 <= start <= end <= len(self.index_text)):
                    raise RecordCorrupt("retrieval source span is out of bounds")
                raw = self.index_text[start:end]
                if " ".join(re.findall(r"\S+", raw)) != hit.text:
                    raise RecordCorrupt("retrieval text does not match its source span")
                status = "verified"
            key = f"{identity or 'unknown'}:{hit.chunk_id}"
            item = {"chunk_id": hit.chunk_id, "role": "retrieval_candidate", "retrieval_text": hit.text,
                    "source_text": raw, "source_span": [start, end] if status == "verified" else None,
                    "span_status": status, "representation_sha256": identity, "offset_unit": "unicode_codepoint",
                    "page_start": getattr(hit, "page_start", None), "page_end": getattr(hit, "page_end", None),
                    "section": getattr(hit, "section", None), "transform": "whitespace-word-join-v1"}
            if key in self.chunks and self.chunks[key] != item:
                raise RecordCorrupt("retrieved chunk identity changed during grading")
            self.chunks[key] = item
            ranked.append({"rank": rank, "chunk": key, "retrieval_score": float(hit.score)})
        state[label] = ranked

    def numeric(self, value, full_normalized):
        key = hashlib.sha256(full_normalized.encode("utf-8")).hexdigest()
        if key not in self.normalized_sources:
            self.normalized_sources[key] = self.representation({"schema_version": 1, "role": "numeric_normalized_full_pdf",
                "text": full_normalized, "transform": "spaced-thousands-then-lost-decimals-v1"}).to_dict()
        value["normalized_full_pdf"] = self.normalized_sources[key]
        _claim.get()["numeric"] = value

    def evidence(self, boundary="core_evidence"):
        item = self.emit("evidence", {"boundary": boundary, "chunks": self.chunks, "claims": self.claims,
                                      "additional_checks": self.extra, "vectors": self.vectors})
        self.parents.append(item)

    def vector(self, array):
        from .jsonio import canonical_bytes, digest
        value = {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C",
                 "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii")}
        identity = digest(canonical_bytes(value))
        self.vectors[identity] = value
        return identity

    def finish(self, status, result=None, error=None):
        self.parents.extend(self.response_ids)
        self.evidence("final_evidence" if status == "completed" else "partial_evidence")
        value = {"boundary": "grading_finished", "status": status,
                 "error_code": type(error).__name__ if error else None}
        if result is not None:
            if hasattr(result, "to_dict"):
                value["report"] = result.to_dict()
            elif isinstance(result, tuple):
                value["aggregate"] = result[0]
        self.emit("result", value)


def observed_grading(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        if current_run() is not None:
            return fn(*args, **kwargs)
        from .lifecycle import current_owner
        owner = current_owner()
        if owner is None or owner.records is None:
            return fn(*args, **kwargs)
        run = GradingRun(owner.records, fn.__name__, getattr(owner.ctx, "iteration", None))
        token = _run.set(run)
        try:
            if "draft_text" in kwargs:
                run.parents.append(run.emit("input", {"boundary": "draft_input", "draft_text": kwargs["draft_text"],
                                                       "stem": kwargs.get("stem")}))
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if isinstance(exc, RecordStoreUnavailable):
                owner.records.failure = exc
            if owner.records.failure is None:
                run.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=exc)
            raise
        else:
            run.finish("completed", result)
            return result
        finally:
            _run.reset(token)
    return call


def observed_claim(fn):
    @wraps(fn)
    def call(stem, claim, *args, **kwargs):
        run = current_run()
        if run is None:
            return fn(stem, claim, *args, **kwargs)
        semantic = getattr(run, "semantic", {})
        semantic_status = "not_requested" if not semantic.get("requested") else "unavailable" if not semantic.get("available") else "not_checked"
        state = {"claim": asdict(claim), "queries": [], "bm25": [], "semantic": [], "numeric": {"status": "not_checked"},
                 "negation": {"status": "not_checked"}, "semantic_check": {"status": semantic_status}}
        token = _claim.set(state)
        try:
            result = fn(stem, claim, *args, **kwargs)
            state["result"] = result.to_dict()
            return result
        except BaseException as exc:
            state["error_code"] = type(exc).__name__
            raise
        finally:
            run.claims.append(state)
            _claim.reset(token)
    return call


def observe_numeric(value, full_normalized):
    if _claim.get() is not None:
        current_run().numeric(value, full_normalized)


def observe_semantic(scores, best_index, model, device, claim_vector=None, candidate_vectors=None, token_limit=None):
    if _claim.get() is not None:
        _claim.get()["semantic_check"].update({"status": "completed", "candidate_scores": scores,
                                           "best_index": best_index, "model": model, "device": device,
                                           "aggregation": "max_cosine", "model_identity_origin": "configured",
                                           "model_revision": None, "model_token_limit": token_limit})
        if claim_vector is not None and candidate_vectors is not None:
            run = current_run()
            _claim.get()["semantic_check"].update({"claim_vector": run.vector(claim_vector),
                "candidate_vectors": [run.vector(row) for row in candidate_vectors]})


def observe_negation(value):
    if claim_active():
        _claim.get()["negation"] = value


def semantic_status(**value):
    if claim_active():
        _claim.get()["semantic_check"].update(value)


def note(value):
    if current_run() is not None:
        current_run().extra.append(value)


def observe_query(value):
    if _claim.get() is not None:
        _claim.get()["queries"].append(value)


class GradingCallRecorder:
    """Reuse provider capture without assigning support calls to target extraction."""
    def __init__(self, run):
        self.run, self.records = run, run.records
        self.response_ids = run.response_ids

    def check(self):
        self.records.check()

    def dependencies(self):
        return (self.run.support_input,) if self.run.support_input else tuple(self.run.parents)

    @property
    def answer_unavailable(self):
        return self.records.grading_answer_unavailable

    @answer_unavailable.setter
    def answer_unavailable(self, value):
        self.records.grading_answer_unavailable = value

    def artifact(self, slot, kind, value, **kwargs):
        target = {"target.request": "grading.request", "target.response": "grading.response", "target.trace": "grading.evidence"}[slot]
        item = self.records.artifact(target, kind, {"run_id": self.run.run_id, "model_phase": "claim_support", **value}, **kwargs)
        if target == "grading.request":
            self.records.grading_call_ids.add(kwargs["call_id"])
        return item


def read_grading(store, attempt_id):
    from .jsonio import decode_json

    manifest = store.read_attempt(attempt_id)
    slots = {item_id: item.slot for item in manifest.inventory if item.slot.startswith("grading.") for item_id in item.artifact_ids}
    return [{"slot": slots[item.artifact_id], "artifact": item.to_dict(), "payload": decode_json(store.read_blob(item.payload))}
            for item in manifest.artifacts if item.artifact_id in slots]
