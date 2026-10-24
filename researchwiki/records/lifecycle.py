"""One retention owner through runner and operation post-work, with direct fallback."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps

from .contracts import AttemptFinished, AttemptSpec, CapturePolicy, Producer, Requirement, extraction_requirements
from .errors import RecordStoreUnavailable
from .extraction import ExtractionRecorder
from .settings import selected
from .store import RecordStore, now
from .grading import grading_requirements
from .lineage import LineageRecorder, requirements as lineage_requirements


_owner = ContextVar("researchwiki_ingest_capture_owner", default=None)


def current_owner():
    return _owner.get()


def check_capture():
    owner = current_owner()
    if owner is not None:
        owner.check()


class CaptureOwner:
    def __init__(self, request_id=None):
        self.request_id = request_id
        self.settings = None
        self.records = None
        self.ctx = None
        self.visited = set()
        self.prepared = False
        self.receipt = None

    def configure(self):
        if self.settings is None:
            self.settings = selected()
            if self.settings.mode == "required":
                from ..mutation import journalling_enabled
                if not journalling_enabled():
                    raise RecordStoreUnavailable("required capture needs mutation journalling")

    def check(self):
        if self.records is not None:
            self.records.check()

    def bind(self, ctx):
        self.configure()
        self.ctx = ctx
        if self.settings.mode == "off":
            return
        from .. import __version__, ingest_ledger
        from ..operations.observation import current_recorder

        events = current_recorder()
        if events is not None:
            events.bind_attempt(ctx.attempt_id)
        binding = ingest_ledger.current_binding()
        spec = AttemptSpec(attempt_id=ctx.attempt_id, policy=CapturePolicy(mode="required"),
                           producer=Producer(name="researchwiki", version=__version__), created_at=now(),
                           request_id=self.request_id, job_id=binding["job_id"] if binding else None,
                           generation=binding["generation"] if binding else None,
                           requirements=extraction_requirements() + (
                               Requirement(slot="source.displaced", kind="source_snapshot", required=False),
                               Requirement(slot="publication.receipt", kind="terminal_result"),
                               Requirement(slot="attempt.result", kind="terminal_result"))
                           + grading_requirements(ctx.verify_claim_entailment) + lineage_requirements())
        store = RecordStore.create(self.settings.root)
        store.start_attempt(spec)
        self.records = ExtractionRecorder(store, ctx.attempt_id)
        self.lineage = LineageRecorder(self.records)
        self.records.snapshot_source(ctx.pdf_path)
        self.records.bind_context(ctx)

    def receipt_outcome(self):
        if self.receipt is None:
            return "unknown" if self.prepared else "not_published"
        return "published" if self.receipt == "committed" else "not_published"

    def finish(self, values):
        if self.records is None:
            return
        records = self.records
        self.check()
        manifest = records._write(records.store.read_attempt, records.spec.attempt_id)
        self.lineage.close(manifest)
        if records.grading_answer_unavailable:
            records._write(records.store.resolve_slot, records.spec.attempt_id, "grading.response", "unavailable",
                           "visible_verification_answer_unavailable")
        for item in manifest.inventory:
            if item.slot.startswith("lineage.") or item.status != "pending" or item.slot in ("publication.receipt", "attempt.result"):
                continue
            ran = "extract" if item.slot == "source.representation" else "target_claims"
            if item.slot.startswith("grading."):
                ran = "grade" if records.grading_runs else "grading_not_executed"
                if not records.grading_runs or (item.slot in ("grading.request", "grading.response") and not records.grading_call_ids):
                    records._write(records.store.resolve_slot, records.spec.attempt_id, item.slot, "not_run",
                                   "grading_not_executed_or_no_judge_call")
                    continue
                records._write(records.store.resolve_slot, records.spec.attempt_id, item.slot, "unknown", "grading_output_unavailable")
                continue
            if item.slot == "source.displaced" or ran not in self.visited:
                status, reason = "not_run", "phase_not_executed"
            else:
                status, reason = "unknown", "phase_did_not_retain_output"
            records._write(records.store.resolve_slot, records.spec.attempt_id, item.slot, status, reason)
        if self.prepared and self.receipt is None:
            raise RecordStoreUnavailable("publication handoff unresolved; reconcile the retained journal")
        if self.receipt is None:
            records.artifact("publication.receipt", "terminal_result",
                             {"schema_version": 1, "state": values["publication"], "boundary": "no_publication_transaction"})
        records.artifact("attempt.result", "terminal_result", {"schema_version": 1, **values})
        records._write(records.store.finish_attempt, records.spec.attempt_id, AttemptFinished(
            execution=values["execution"], outcome=values["outcome"], publication=values["publication"],
            projections=values["projections"]))

    def finish_operation(self, execution, facts):
        if self.settings is None or self.settings.mode == "off" or self.ctx is None:
            return execution.result
        result = execution.result
        publication = "published" if facts.publication or result.publication == "published" else self.receipt_outcome()
        result = replace(result, attempt_id=self.ctx.attempt_id, publication=publication,
                         projections=result.projections if publication != "not_published" else "not_applicable")
        try:
            self.finish({"scope": "operation", **result.to_dict()})
        except RecordStoreUnavailable as exc:
            from ..operations.results import error_from_exception
            execution.error = exc
            result = replace(result, error=error_from_exception(exc),
                             post_work="partial" if publication == "published" else result.post_work,
                             warnings=result.warnings + ("Capture incomplete; inspect records and reconcile publication before retry",))
        return result


@contextmanager
def capture_operation(request_id):
    owner = CaptureOwner(request_id)
    token = _owner.set(owner)
    try:
        yield owner
    finally:
        _owner.reset(token)


def bind_attempt(ctx):
    owner = current_owner()
    if owner is not None:
        owner.bind(ctx)


def record_ingestion(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        if current_owner() is not None:
            return fn(*args, **kwargs)
        from ..operations.lifecycle import IngestFacts, current_facts, observe_lifecycle

        facts = current_facts() or IngestFacts()
        from ..operations.observation import current_recorder
        events = kwargs.get("event_sink") or current_recorder()
        with capture_operation(getattr(events, "request_id", None)) as owner, observe_lifecycle(facts):
            owner.configure()
            error, ctx = None, None
            try:
                ctx = fn(*args, **kwargs)
                return ctx
            except BaseException as exc:
                error = exc
                raise
            finally:
                if owner.records is not None:
                    publication = "published" if facts.publication or (ctx and ctx.outcome == "already_present") else owner.receipt_outcome()
                    outcome = {"promoted": "published", "sandboxed": "draft_ready", "already_present": "already_present"}.get(
                        getattr(ctx, "outcome", None), "failed")
                    try:
                        owner.finish({"scope": "runner", "execution": "interrupted" if isinstance(error, KeyboardInterrupt) else "finished",
                                      "outcome": None if isinstance(error, KeyboardInterrupt) else outcome,
                                      "publication": publication, "projections": facts.projections if publication == "published" else "not_applicable"})
                    except RecordStoreUnavailable as exc:
                        exc.publication = publication
                        if error is None:
                            raise
    return run
