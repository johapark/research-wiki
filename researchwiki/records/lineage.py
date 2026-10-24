"""Draft and decision capture, independent of best-effort progress delivery."""

from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
from inspect import signature
from threading import RLock
import uuid

from .contracts import Requirement, SourceRef
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
    result = [Requirement(slot="lineage.drafts", kind="parse_trace", required=False),
              Requirement(slot="lineage.decisions", kind="parse_trace", required=False),
              Requirement(slot="lineage.omissions", kind="parse_trace", required=False),
              Requirement(slot="lineage.page", kind="representation", required=False)]
    for phase in PHASES:
        result.extend((Requirement(slot=f"lineage.{phase}.input", kind="parse_trace"),
                       Requirement(slot=f"lineage.{phase}.output", kind="parse_trace"),
                       Requirement(slot=f"lineage.{phase}.request", kind="model_request"),
                       Requirement(slot=f"lineage.{phase}.response", kind="model_response")))
    return tuple(result)


class LineageRecorder:
    def __init__(self, records):
        self.records = records
        self.lock = RLock()
        self.drafts = {}
        self.critiques = {}
        self.runs = set()
        self.calls = set()
        self.unavailable = set()
        self.rejected_calls = set()
        self.call_omissions = {}
        self.phase_omissions = {}
        self.phase_states = {}
        self.last_phase_output = None
        self.page_artifact_id = None

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

    def rendered_page(self, path):
        """Retain exact frontmatter/body bytes before the journal commit point."""
        records = self.records
        blob = records._write(records.store.put_file, path)
        scope = current_scope()
        parents = []
        if scope is not None and scope.recorder is self and scope.phase == "commit":
            parents.append(scope.input_id)
            if scope.ctx.winner is not None:
                parents.append(self.draft(scope.ctx.winner))
        source = records.source.source
        item = records.artifact("lineage.page", "representation", None, payload=blob,
            sources=(SourceRef(document=source.document, role="effective", representation=blob),),
            parents=tuple(parents))
        self.page_artifact_id = item
        return item, blob

    def omission(self, target, phase, reason, *, call=None, context=None, run=None):
        """Retain one explicit conditional omission with a stable reason code."""
        slot = "lineage.omissions"
        if not any(item.slot == slot for item in self.records.spec.requirements):
            return None
        parents = (run.input_id,) if run is not None else (
            (self.last_phase_output,) if self.last_phase_output else ())
        value = {"boundary": "conditional_omission", "target": target,
                 "phase": phase, "call": call, "reason": reason,
                 "context": context or {}}
        if run is not None:
            value["run_id"] = run.run_id
        return self.artifact("omissions", value, parents=parents)

    def close(self, manifest, terminal=None):
        terminal = terminal or {}
        error = terminal.get("error") or {}
        error_code = terminal.get("error_code") or (
            error.get("code") if isinstance(error, dict) else None)
        if not error_code and terminal.get("execution") == "interrupted":
            error_code = "interrupted"
        outcome = terminal.get("outcome")

        def stopped_reason():
            if outcome == "already_present":
                return "duplicate_source_short_circuit", {"outcome": outcome}
            if error_code:
                reason = "pipeline_interrupted" if error_code == "interrupted" else "pipeline_stopped_after_error"
                return reason, {"error_code": error_code}
            return None, {}

        stop_reason, stop_context = stopped_reason()
        for phase in PHASES:
            if phase in self.runs or phase in self.phase_omissions:
                continue
            if stop_reason:
                self.phase_omissions.setdefault(phase, []).append(stop_reason)
                self.omission("phase", phase, stop_reason, context=stop_context)

        # Omission artifacts and their terminal causes update the manifest.
        # Refresh it before resolving pending inventory so slot states are current.
        manifest = self.records._write(self.records.store.read_attempt, self.records.spec.attempt_id)
        for item in manifest.inventory:
            if not item.slot.startswith("lineage."):
                continue
            parts = item.slot.split(".")
            if len(parts) == 2:
                if item.status != "pending":
                    continue
                reason = {"drafts": "no_draft_created",
                          "decisions": "no_revision_decision_created",
                          "omissions": "no_conditional_omissions_observed",
                          "page": "canonical_page_not_captured"}.get(parts[1], "optional_lineage_not_observed")
                if parts[1] == "page":
                    if outcome == "draft_ready":
                        reason = "sandbox_page_not_published"
                    elif outcome == "already_present":
                        reason = "duplicate_source_no_new_page"
                    elif stop_reason:
                        reason = "publication_not_reached_after_failure"
                self.records._write(self.records.store.resolve_slot, self.records.spec.attempt_id,
                                    item.slot, "not_run", reason)
                continue
            phase = parts[1]
            phase_reason = next(iter(reversed(self.phase_omissions.get(phase, ()))), None) or stop_reason
            if len(parts) == 3 and parts[2] == "response" and phase in self.unavailable:
                self.records._write(self.records.store.resolve_slot, self.records.spec.attempt_id,
                                    item.slot, "unavailable", "one_or_more_visible_answers_unavailable")
            elif item.status == "pending":
                slot = parts[2]
                if phase not in self.runs:
                    status, reason = ("not_run", phase_reason) if phase_reason else (
                        "unknown", "phase_skip_reason_not_observed")
                elif slot in ("input", "output"):
                    status, reason = "unknown", "phase_observation_not_retained"
                elif slot in ("request", "response"):
                    names = MODEL_PHASES.get(phase, set())
                    if not names:
                        status, reason = "not_run", "phase_has_no_model_call"
                    elif phase not in self.calls:
                        omitted = [self.call_omissions.get((phase, name), ()) for name in sorted(names)]
                        if all(omitted):
                            status = "not_run"
                            reasons = sorted({code for values in omitted for code in values})
                            reason = reasons[0] if len(reasons) == 1 else "all_conditional_model_calls_omitted"
                        else:
                            status, reason = "unknown", "call_omission_reason_not_observed"
                    elif slot == "response":
                        failed = next((state for state in reversed(self.phase_states.get(phase, ()))
                                       if state.get("status") in ("failed", "interrupted")), None)
                        rejected = any(call_phase == phase for call_phase, _ in self.rejected_calls)
                        if rejected:
                            status, reason = "failed", "model_transport_rejected"
                        elif failed:
                            status, reason = "failed", "phase_failed_before_response_retention"
                        else:
                            status, reason = "unknown", "visible_response_not_retained"
                    else:
                        status, reason = "unknown", "model_request_not_retained"
                else:
                    status, reason = "unknown", "phase_output_not_retained"
                self.records._write(self.records.store.resolve_slot, self.records.spec.attempt_id,
                                    item.slot, status, reason)


def note_call_omission(model_phase, reason, context=None):
    """Record why a declared model call did not happen in the active phase."""
    run = current_scope()
    if run is None or model_phase not in MODEL_PHASES.get(run.phase, set()):
        return None
    key = (run.phase, model_phase)
    run.recorder.call_omissions.setdefault(key, []).append(reason)
    return run.recorder.omission("model_call", run.phase, reason, call=model_phase,
                                 context=context, run=run)


def note_phase_omission(phase, reason, context=None):
    """Record why a conditional ingest phase was not entered."""
    run = current_scope()
    recorder = run.recorder if run is not None else None
    if recorder is None:
        from .lifecycle import current_owner
        owner = current_owner()
        recorder = getattr(owner, "lineage", None) if owner is not None else None
    if recorder is None:
        return None
    recorder.phase_omissions.setdefault(phase, []).append(reason)
    return recorder.omission("phase", phase, reason, context=context,
                              run=run)


class PhaseRun:
    def __init__(self, recorder, phase, ctx, arguments):
        self.recorder, self.phase, self.ctx = recorder, phase, ctx
        self.run_id = uuid.uuid4().hex
        self.responses = []
        self.call_names = set()
        self.response_call_names = set()
        self.rejected_call_names = set()
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
        for call in sorted(self.call_names - self.response_call_names):
            if call in self.rejected_call_names:
                reason = "transport_rejected_without_visible_answer"
            elif error is not None:
                reason = "phase_failed_before_visible_answer"
            else:
                reason = "visible_answer_not_observed"
            self.recorder.omission("model_response", self.phase, reason, call=call,
                context={"error_code": type(error).__name__ if error else None}, run=self)
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
                if self.recorder.page_artifact_id:
                    value["rendered_page"] = self.recorder.page_artifact_id
                    parents.append(self.recorder.page_artifact_id)
        item = self.emit("output", value, parents=parents)
        self.recorder.phase_states.setdefault(self.phase, []).append({
            "status": value["status"], "error_code": value["error_code"],
            "artifact_id": item,
        })
        self.recorder.last_phase_output = item
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
    def __init__(self, run, model_phase):
        self.run, self.records, self.model_phase = run, run.recorder.records, model_phase
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
            self.run.recorder.omission("model_response", self.run.phase,
                "visible_answer_unavailable", call=self.model_phase,
                context={"observation": "answer_none"}, run=self.run)

    def artifact(self, slot, kind, value, **kwargs):
        suffix = {"target.request": "request", "target.response": "response", "target.trace": "output"}[slot]
        if suffix == "request":
            self.run.recorder.calls.add(self.run.phase)
            self.run.call_names.add(self.model_phase)
        elif suffix == "response":
            self.run.response_call_names.add(self.model_phase)
        elif suffix == "output" and value.get("boundary") == "transport_rejection":
            self.run.recorder.rejected_calls.add((self.run.phase, self.model_phase))
            self.run.rejected_call_names.add(self.model_phase)
        return self.run.recorder.artifact(f"{self.run.phase}.{suffix}",
            {"run_id": self.run.run_id, "model_phase": self.run.phase,
             "call_phase": self.model_phase, **value}, kind=kind, **kwargs)


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
    result = []
    for item in manifest.artifacts:
        if item.artifact_id not in slots:
            continue
        raw = store.read_blob(item.payload)
        payload = {"text": raw.decode("utf-8")} if slots[item.artifact_id] == "lineage.page" else decode_json(raw)
        result.append({"slot": slots[item.artifact_id], "artifact": item.to_dict(), "payload": payload})
    return result
