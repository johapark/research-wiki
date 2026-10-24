"""Draft and decision capture, independent of best-effort progress delivery."""

from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
from inspect import signature
from threading import RLock
import uuid

from .contracts import Requirement
from .grading import _json_value
from .observation import finite_setting


PHASES = ("reconcile", "crosslinks", "author", "grade", "tournament", "critic", "evolve", "debug", "commit")
MODEL_PHASES = {"reconcile": {"reconcile"}, "crosslinks": {"link_generation"},
                "author": {"author"}, "critic": {"critic"}, "evolve": {"evolve"},
                "debug": {"debug"}, "commit": {"short_name", "keywords", "classifier"}}
_scope = ContextVar("researchwiki_draft_lineage", default=None)


def current_scope():
    return _scope.get()


def requirements():
    return (Requirement(slot="lineage.drafts", kind="parse_trace", required=False),
            Requirement(slot="lineage.decisions", kind="parse_trace", required=False)) + tuple(
        Requirement(slot=f"lineage.{phase}.{slot}", kind=kind)
        for phase in PHASES for slot, kind in (
            ("input", "parse_trace"), ("output", "parse_trace"),
            ("request", "model_request"), ("response", "model_response")))


class LineageRecorder:
    def __init__(self, records):
        self.records = records
        self.lock = RLock()
        self.drafts = {}
        self.critiques = {}
        self.runs = set()
        self.calls = set()
        self.unavailable = set()

    def artifact(self, slot, value, *, parents=(), kind="parse_trace", **kwargs):
        return self.records.artifact(f"lineage.{slot}", kind, _json_value({"schema_version": 1, **value}),
                                     parents=tuple(dict.fromkeys(parents)), **kwargs)

    def draft(self, draft, parents=()):
        # Keep a strong reference so object IDs cannot be recycled within an attempt.
        # Object IDs never enter the portable record; artifact UUIDs are authoritative.
        with self.lock:
            previous = self.drafts.get(id(draft))
            value = {key: getattr(draft, key) for key in (
                "text", "model", "iteration_id", "stance", "handle", "hook",
                "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
            value["temperature"] = finite_setting(draft.temperature)
            if previous is not None and previous[1] == value:
                return previous[2]
            scope = current_scope()
            if scope is not None and scope.recorder is self:
                parents = (*parents, scope.input_id, *scope.responses)
            item = self.artifact("drafts", {"boundary": "draft", "draft": value},
                                 parents=tuple(parents) + ((previous[2],) if previous else ()))
            self.drafts[id(draft)] = (draft, value, item)
            return item

    def close(self, manifest):
        for item in manifest.inventory:
            if not item.slot.startswith("lineage."):
                continue
            parts = item.slot.split(".")
            phase = parts[1]
            if len(parts) == 3 and parts[2] == "response" and phase in self.unavailable:
                self.records._write(self.records.store.resolve_slot, self.records.spec.attempt_id,
                                    item.slot, "unavailable", "visible_phase_answer_unavailable")
            elif item.status == "pending":
                ran = phase in self.runs and not (parts[-1] in ("request", "response") and phase not in self.calls)
                self.records._write(self.records.store.resolve_slot, self.records.spec.attempt_id,
                                    item.slot, "unknown" if ran else "not_run",
                                    "phase_output_unavailable" if ran else "phase_or_model_call_not_executed")


class PhaseRun:
    def __init__(self, recorder, phase, ctx, arguments):
        self.recorder, self.phase, self.ctx = recorder, phase, ctx
        self.run_id = uuid.uuid4().hex
        self.responses = []
        self.parents = [recorder.records.source.artifact_id]
        self.subject = arguments.get("draft") or arguments.get("prior_draft")
        if phase in ("debug", "commit"):
            self.subject = ctx.winner
        if self.subject is not None:
            self.parents.append(recorder.draft(self.subject))
        if phase == "tournament":
            self.parents.extend(recorder.draft(d) for d in ctx.drafts)
        values = {"iteration": ctx.iteration, "slot": arguments.get("slot"),
                  "temperature": finite_setting(arguments.get("temperature")),
                  "synthetic": ctx.use_stub, "paper_stem": ctx.paper_stem}
        if self.subject is not None:
            values.update(scores=self.subject.scores, claim_details=[asdict(c) for c in self.subject.claim_details])
        if phase == "tournament":
            from ..agents.fitness import tournament_key
            values["candidates"] = [{"draft": recorder.draft(d), "scores": d.scores,
                                      "key": list(tournament_key(d))} for d in ctx.drafts]
            values["rule"] = "stable-max-tournament-key-v1"
        if phase == "evolve":
            critique_id = recorder.critiques.get(id(arguments["critique"]))
            if critique_id is not None:
                self.parents.append(critique_id[1])
            values["critique"] = asdict(arguments["critique"])
        if phase == "debug":
            values.update({key: arguments[key] for key in ("cleaned_text", "n_kc", "issues")})
            values.update(gate=asdict(arguments["gate"]), verification=asdict(arguments["verification"]))
        if phase == "reconcile":
            source = recorder.records.source.source
            values.update({key: arguments.get(key) for key in (
                "doi_override", "title_override", "year_override", "authors_override",
                "use_llm_reconcile")})
            values["source_pdf_sha256"] = source.document.sha256
        if phase == "crosslinks":
            values.update(metadata=ctx.metadata, sections=ctx.sections,
                          use_stub=ctx.use_stub, paper_stem=ctx.paper_stem)
        if phase == "commit":
            values.update(metadata=ctx.metadata, paper_stem=ctx.paper_stem,
                          promotion_mode=ctx.promote_mode, gate_reasons=ctx.gate_reasons,
                          verified_candidates=[asdict(candidate) for candidate in ctx.crosslink_candidates])
        self.input_id = self.emit("input", {"boundary": "phase_input", **values}, parents=self.parents)
        recorder.runs.add(phase)

    def emit(self, slot, value, *, parents=()):
        return self.recorder.artifact(f"{self.phase}.{slot}",
            {"run_id": self.run_id, "phase": self.phase, **value}, parents=parents)

    def finish(self, result=None, error=None):
        parents = [self.input_id, *self.responses]
        value = {"boundary": "phase_output", "status": "completed" if error is None else
                 "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                 "error_code": type(error).__name__ if error else None}
        if error is None:
            if self.phase in ("author", "evolve"):
                value["draft"] = self.recorder.draft(result, parents)
                parents.append(value["draft"])
            elif self.phase == "critic":
                value["critique"] = asdict(result)
            elif self.phase == "grade":
                value.update(draft=self.recorder.draft(self.subject), scores=self.subject.scores,
                             claim_details=[asdict(c) for c in self.subject.claim_details])
            elif self.phase == "tournament":
                value.update(selected=self.recorder.draft(result), decision_origin="deterministic_selection")
            elif self.phase == "debug":
                value.update(cleaned_text=result[0], n_kc=result[1], gate=asdict(result[2]),
                             verification=asdict(result[3]), winner=self.recorder.draft(self.ctx.winner))
            elif self.phase == "reconcile":
                value.update(metadata=result[0], iteration_id=result[1])
            elif self.phase == "crosslinks":
                value["candidates"] = [asdict(candidate) for candidate in result]
            elif self.phase == "commit":
                value.update(result_name=result.name, outcome=self.ctx.outcome,
                             paper_stem=self.ctx.paper_stem)
        item = self.emit("output", value, parents=parents)
        if error is None and self.phase == "critic":
            self.recorder.critiques[id(result)] = (result, item)


def record_phase(name, fn):
    """Wrap content owners, rather than consuming or trusting progress events."""
    if name not in PHASES:
        return fn
    sig = signature(fn)

    @wraps(fn)
    def call(ctx, *args, **kwargs):
        from .lifecycle import current_owner
        owner = current_owner()
        if owner is None or owner.records is None:
            return fn(ctx, *args, **kwargs)
        owner.check()
        bound = sig.bind(ctx, *args, **kwargs)
        bound.apply_defaults()
        run = PhaseRun(owner.lineage, name, ctx, bound.arguments)
        token = _scope.set(run)
        try:
            result = fn(ctx, *args, **kwargs)
        except BaseException as exc:
            if owner.records.failure is None:
                run.finish(error=exc)
            raise
        else:
            run.finish(result)
            return result
        finally:
            _scope.reset(token)
    return call


class PhaseCallRecorder:
    """Adapt the shared API/relay capture without target-extraction ownership."""
    def __init__(self, run):
        self.run, self.records = run, run.recorder.records
        self.response_ids = run.responses

    def check(self):
        self.records.check()

    def dependencies(self):
        return (self.run.input_id,)

    @property
    def answer_unavailable(self):
        return self.run.phase in self.run.recorder.unavailable

    @answer_unavailable.setter
    def answer_unavailable(self, value):
        if value:
            self.run.recorder.unavailable.add(self.run.phase)

    def artifact(self, slot, kind, value, **kwargs):
        suffix = {"target.request": "request", "target.response": "response", "target.trace": "output"}[slot]
        if suffix == "request":
            self.run.recorder.calls.add(self.run.phase)
        return self.run.recorder.artifact(f"{self.run.phase}.{suffix}",
            {"run_id": self.run.run_id, "model_phase": self.run.phase, **value}, kind=kind, **kwargs)


def revision_decision(ctx, draft, operator, accepted):
    from .lifecycle import current_owner
    owner = current_owner()
    if owner is None or owner.records is None:
        return
    records = owner.lineage
    candidate = records.draft(draft)
    prior = records.draft(ctx.winner) if ctx.winner else None
    records.artifact("decisions", {"boundary": "revision_selection", "operator": operator,
        "iteration": ctx.iteration, "candidate": candidate, "prior": prior, "accepted": accepted,
        "candidate_scores": draft.scores, "prior_scores": ctx.winner.scores if ctx.winner else None,
        "decision_origin": "fitness_rule"}, parents=[item for item in (candidate, prior) if item])


def commit_input(ctx, cleaned_text, verification, gate):
    run = current_scope()
    if run is not None and run.phase == "commit":
        run.input_id = run.emit("input", {"boundary": "verified_page_before_publication",
            "cleaned_text": cleaned_text, "verification": asdict(verification),
            "gate": asdict(gate), "gate_reasons": list(gate.reasons),
            "winner": run.recorder.draft(ctx.winner)}, parents=(run.input_id,))
        return run.input_id
    return None


def tournament_decision(winner, rationale):
    run = current_scope()
    if run is not None and run.phase == "tournament":
        selected = run.recorder.draft(winner)
        run.emit("output", {"boundary": "selection_decision", "selected": selected,
                           "rationale": rationale, "decision_origin": "deterministic_selection"},
                 parents=(run.input_id, selected))


def read_lineage(store, attempt_id):
    from .jsonio import decode_json
    manifest = store.read_attempt(attempt_id)
    slots = {key: item.slot for item in manifest.inventory if item.slot.startswith("lineage.")
             for key in item.artifact_ids}
    return [{"slot": slots[item.artifact_id], "artifact": item.to_dict(),
             "payload": decode_json(store.read_blob(item.payload))}
            for item in manifest.artifacts if item.artifact_id in slots]
