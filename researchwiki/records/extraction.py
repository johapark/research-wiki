"""Explicit extraction recorder; lifecycle owns policy, scope and publishing."""

from dataclasses import asdict, dataclass
from pathlib import Path
import uuid

from .contracts import Artifact, SourceRef
from .errors import RecordConflict, RecordStoreUnavailable
from .store import now
from ..operations._wire import ContractError


@dataclass(frozen=True)
class ProcessingSource:
    original_path: Path
    processing_path: Path
    artifact_id: str
    source: SourceRef


class ExtractionRecorder:
    """Own observations for one already-started, required-capture attempt.

    Construction is read-only. The caller owns lifecycle, terminal outcome and
    publication. The optional lifecycle owner selects and binds this recorder.
    A failed write latches; subsequent work cannot silently resume recording.
    """

    def __init__(self, store, attempt_id):
        self.store = store
        self.spec = store.read_attempt(attempt_id).spec
        if self.spec.policy.mode != "required":
            raise RecordConflict("extraction recorder requires explicit required capture")
        self.source = None
        self.representation_id = None
        self.response_ids = []
        self.failure = None
        self.sections = None
        self.full_text = None
        self.result_id = None
        self.answer_unavailable = False

    def check(self):
        if self.failure is not None:
            raise self.failure

    def _write(self, fn, *args, **kwargs):
        self.check()
        try:
            return fn(*args, **kwargs)
        except RecordStoreUnavailable as exc:
            self.failure = exc
            raise
        except (ContractError, RecordConflict) as exc:
            self.failure = RecordStoreUnavailable("capture could not persist a valid observation")
            raise self.failure from exc

    def artifact(self, slot, kind, value, *, parents=(), call_id=None,
                 dispatch_id=None, origin="observed", payload=None, sources=None):
        payload = payload or self._write(self.store.put_json, value)
        if sources is None:
            sources = (self.source.source,) if self.source else ()
        item = self._write(Artifact, artifact_id=uuid.uuid4().hex, attempt_id=self.spec.attempt_id,
                        kind=kind, payload=payload, producer=self.spec.producer,
                        created_at=now(), origin=origin, sources=tuple(sources),
                        parents=tuple(parents), phase=("target_claims" if slot.startswith("target.") else
                                                      "extract" if slot.startswith("source.") else None),
                        call_id=call_id, dispatch_id=dispatch_id)
        self._write(self.store.record_artifact, slot, item)
        return item.artifact_id

    def snapshot_source(self, path):
        self.check()
        if self.source is not None:
            raise RecordConflict("processing source already selected")
        original = Path(path).resolve()
        blob = self._write(self.store.put_file, original)
        ref = SourceRef(document=blob, role="original")
        item = self.artifact("source.original", "source_snapshot", None,
                             payload=blob, sources=(ref,))
        self.source = ProcessingSource(original, self.store.root / blob.relative_path, item, ref)
        return self.source

    def bind_context(self, ctx):
        """Select a stable read path while preserving promotion's original path."""
        if ctx.attempt_id != self.spec.attempt_id or self.source is None:
            raise RecordConflict("processing context needs this attempt's source")
        if Path(ctx.pdf_path).resolve() != self.source.original_path:
            raise RecordConflict("processing context owns a different original source")
        self.check()
        ctx.processing_pdf_path = self.source.processing_path
        ctx.extraction_records = self

    def extract(self, path):
        from ..pdf.representation import extract_representation
        from ..pdf.sections import anchor_sections

        if self.source is None or Path(path).resolve() != self.source.processing_path:
            raise RecordConflict("extraction must read the retained processing source")
        self._write(self.store.read_blob, self.source.source.document)
        result = extract_representation(self.source.processing_path, max_pages=80)
        # Detect a changed archive payload before publishing a representation.
        self._write(self.store.read_blob, self.source.source.document)
        blob = self._write(self.store.put_json, result.record)
        ref = SourceRef(document=self.source.source.document, role="effective", representation=blob)
        self.representation_id = self.artifact("source.representation", "representation", None,
                                             payload=blob, sources=(ref,), parents=(self.source.artifact_id,))
        self.source = ProcessingSource(self.source.original_path, self.source.processing_path,
                                       self.source.artifact_id, ref)
        self.sections = anchor_sections(result.text, max_chars=4000)
        self.full_text = result.text
        return dict(self.sections), result.text

    def validate_inputs(self, sections, full_text):
        self.check()
        self.dependencies()
        if sections != self.sections or (full_text is not None and full_text != self.full_text):
            raise RecordConflict("target inputs differ from the retained source representation")

    def dependencies(self):
        if self.source is None or self.representation_id is None:
            raise RecordConflict("target capture requires a retained source and representation")
        return (self.representation_id,)

    def result(self, output, trace, *, synthetic=False):
        self._resolve_unavailable_answer()
        parents = tuple(self.response_ids) or self.dependencies()
        origin = "synthetic" if synthetic else "observed"
        value = asdict(output)
        if trace.get("status") == "call_failed":
            value["error"] = trace.get("error_code", "call_failed")
        result_id = self.artifact("target.result", "extraction_result",
                                  {"schema_version": 1, "output": value}, parents=parents, origin=origin)
        self.result_id = result_id
        self.artifact("target.trace", "parse_trace", trace, parents=(result_id,), origin=origin)

    def no_call(self):
        for slot in ("target.request", "target.response"):
            self._write(self.store.resolve_slot, self.spec.attempt_id, slot, "not_run", "synthetic_phase_no_model_call")

    def phase_failed(self, error):
        self.check()
        self._resolve_unavailable_answer()
        self.artifact("target.trace", "parse_trace",
                      {"schema_version": 1, "status": "call_failed", "error_code": type(error).__name__},
                      parents=tuple(self.response_ids) or self.dependencies())

    def _resolve_unavailable_answer(self):
        if self.answer_unavailable:
            self._write(self.store.resolve_slot, self.spec.attempt_id, "target.response", "unavailable",
                        "visible_answer_unavailable_from_invalid_protocol")

    def author_exposure(self, text, max_normal):
        if self.result_id is None:
            raise RecordConflict("author exposure requires a retained extraction result")
        self.artifact("target.trace", "parse_trace",
                      {"schema_version": 1, "boundary": "author_exposure", "text": text,
                       "max_normal": max_normal}, parents=(self.result_id,))
