"""Audit invocation/call observations independently of aggregate slot presence.

This only understands the engine's capture metadata. It does not grade prose or
verify source offsets. Historical producer reports remain immutable.
"""

from collections import defaultdict

from ..operations._wire import ContractError
from .contracts import Requirement
from .jsonio import decode_json


SLOT = 'attempt.coverage'


def coverage_requirement():
    return Requirement(slot=SLOT, kind='parse_trace')


def audit_coverage(store, manifest):
    return _Audit(store, manifest).report()


def retain_coverage(records):
    """Publish the audit before terminal closure; gaps make capture incomplete."""
    if not any(req.slot == SLOT for req in records.spec.requirements):
        return
    manifest = records._write(records.store.read_attempt, records.spec.attempt_id)
    report = records._write(audit_coverage, records.store, manifest)
    records.artifact(SLOT, 'parse_trace', report)
    if report['status'] != 'complete':
        records._write(records.store.resolve_slot, records.spec.attempt_id, SLOT,
                       'unknown', 'invocation_coverage_incomplete')
    return report


class _Audit:
    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        self.strict = any(req.slot == SLOT for req in manifest.spec.requirements)
        self.issues, self.limitations, self.entries = [], set(), []
        self.calls = defaultdict(list)
        self.phase_runs, self.grade_runs = {}, {}
        slots = {key: item.slot for item in manifest.inventory for key in item.artifact_ids}
        for artifact in manifest.artifacts:
            slot = slots[artifact.artifact_id]
            if (slot.startswith(('lineage.', 'grading.')) and slot not in (
                'lineage.page', 'lineage.drafts', 'lineage.decisions', 'grading.source'
            )) or artifact.kind in ('model_request', 'model_response') or artifact.call_id is not None:
                raw = store.read_blob(artifact.payload)
                try:
                    value = decode_json(raw)
                    if not isinstance(value, dict):
                        raise ContractError('capture metadata must be an object')
                except ContractError:
                    if self.strict:
                        self.issue('unreadable_capture_metadata', artifact=artifact.artifact_id)
                    else:
                        self.limitations.add('capture_payload_format_not_auditable')
                    continue
                self.entries.append((slot, artifact, value))
                if artifact.kind in ('model_request', 'model_response') and artifact.call_id is None:
                    if self.strict:
                        self.issue('model_call_identity_missing', artifact=artifact.artifact_id)
                    else:
                        self.limitations.add('model_call_identity_not_declared')
                if artifact.call_id is not None:
                    self.calls[artifact.call_id].append((slot, artifact, value))

    def issue(self, reason, *, state='unknown', **identity):
        self.issues.append({'reason': reason, 'state': state, **identity})

    def report(self):
        if not self.strict:
            self.limitations.add('producer_did_not_declare_invocation_audit')
        for req, item in zip(self.manifest.spec.requirements, self.manifest.inventory):
            if req.slot != SLOT and req.required and item.status not in ('retained', 'not_run'):
                self.issue('required_inventory_unresolved', slot=item.slot, state=item.status)
        self._phases()
        self._grading()
        for call_id, entries in self.calls.items():
            self._call(call_id, entries)
        return {
            'schema_version': 1, 'audit_version': 1, 'boundary': 'invocation_coverage',
            'status': 'incomplete' if self.issues else 'limited' if self.limitations else 'complete',
            'audited_through_sequence': self.manifest.last_sequence,
            'audited_through_sha256': self.manifest.last_event_sha256,
            'phase_runs': len(self.phase_runs), 'grading_runs': len(self.grade_runs),
            'logical_calls': len(self.calls),
            'dispatches': sum(value.get('boundary') == 'client_dispatch'
                              for entries in self.calls.values() for _, _, value in entries),
            'issues': self.issues, 'limitations': sorted(self.limitations),
        }

    def _runs(self, prefix, start_boundary, end_boundary):
        runs = defaultdict(list)
        for slot, artifact, value in self.entries:
            if slot.startswith(prefix) and isinstance(value.get('run_id'), str):
                runs[value['run_id']].append((slot, artifact, value))
        starts = {}
        for run_id, entries in runs.items():
            inputs = [(a, v) for slot, a, v in entries
                      if slot.endswith('.input') and v.get('boundary') == start_boundary]
            outputs = [v for slot, _, v in entries
                       if slot.endswith(('.output', '.result')) and v.get('boundary') == end_boundary]
            if len(inputs) != 1:
                self.issue('invocation_input_missing_or_duplicated', run_id=run_id)
                continue
            starts[run_id] = inputs[0]
            if len(outputs) != 1:
                self.issue('invocation_terminal_missing_or_duplicated', run_id=run_id)
            elif outputs[0].get('status') not in ('completed', 'failed', 'interrupted'):
                self.issue('invocation_terminal_status_invalid', run_id=run_id)
            if inputs[0][1].get('coverage_version') != 1:
                self.limitations.add('invocation_expectations_not_declared')
                if self.strict:
                    self.issue('invocation_expectations_not_declared', run_id=run_id)
        return starts, runs

    def _phases(self):
        self.phase_runs, runs = self._runs('lineage.', 'phase_input', 'phase_output')
        for run_id, (_, start) in self.phase_runs.items():
            expected = start.get('expected_calls')
            if not isinstance(expected, list) or any(not isinstance(name, str) for name in expected):
                if self.strict:
                    self.issue('conditional_call_expectations_missing', run_id=run_id)
                continue  # An older producer's model-call obligations are unknown.
            entries = runs[run_id]
            called = {v.get('call_phase') for _, _, v in entries if v.get('boundary') == 'logical'}
            omitted = {v.get('call') for _, _, v in entries
                       if v.get('boundary') == 'conditional_omission'
                       and v.get('target') == 'model_call' and v.get('reason')}
            for name in expected:
                if name not in called and name not in omitted:
                    self.issue('conditional_call_unaccounted', run_id=run_id,
                               phase=start.get('phase'), model_phase=name)

    def _grading(self):
        self.grade_runs, runs = self._runs('grading.', 'run_started', 'grading_finished')
        for run_id, (start_artifact, _) in self.grade_runs.items():
            entries = runs[run_id]
            completed = any(v.get('boundary') == 'grading_finished' and v.get('status') == 'completed'
                            for _, _, v in entries)
            if completed:
                boundaries = {v.get('boundary') for _, _, v in entries}
                for boundary in ('page_claim_input', 'grade_inputs', 'final_evidence'):
                    if boundary not in boundaries:
                        self.issue('grading_observation_missing', run_id=run_id, boundary=boundary)
                if not any(start_artifact.artifact_id in a.parents for a in self.manifest.artifacts
                           if a.kind == 'representation'):
                    self.issue('grading_source_missing', run_id=run_id)
            for _, artifact, value in entries:
                if value.get('boundary') != 'support_classifier_inputs':
                    continue
                expected = value.get('model_call_expected')
                if expected is None:
                    self.limitations.add('support_call_expectation_not_declared')
                if expected is True and not any(
                    v.get('boundary') == 'logical' and artifact.artifact_id in a.parents
                    for _, a, v in entries
                ):
                    self.issue('support_call_unaccounted', run_id=run_id,
                               input_artifact=artifact.artifact_id)

    def _call(self, call_id, entries):
        logical = [(a, v) for _, a, v in entries if v.get('boundary') == 'logical']
        if len(logical) != 1:
            self.issue('logical_request_missing_or_duplicated', call_id=call_id)
        scopes = {(slot.split('.')[0], value.get('run_id')) for slot, _, value in entries}
        if len(scopes) != 1:
            self.issue('call_crosses_invocations', call_id=call_id)
        dispatches = [a.dispatch_id for _, a, v in entries if v.get('boundary') == 'client_dispatch']
        requests = set(dispatches)
        if len(dispatches) != len(requests):
            self.issue('dispatch_identity_reused', call_id=call_id)
        if None in requests:
            self.issue('dispatch_identity_missing', call_id=call_id)
        visible = [(a, v) for _, a, v in entries if v.get('boundary') == 'visible_answer']
        rejected = {a.dispatch_id for _, a, v in entries if v.get('boundary') == 'transport_rejection'}
        for dispatch_id in requests:
            answers = [v for a, v in visible if a.dispatch_id == dispatch_id]
            if not answers and dispatch_id not in rejected:
                self.issue('dispatch_outcome_unobserved', call_id=call_id, dispatch_id=dispatch_id)
        if not visible:
            self.issue('visible_answer_unobserved', call_id=call_id,
                       state='failed' if rejected else 'unknown')
        for _, artifact, value in entries:
            if artifact.kind == 'model_response' and artifact.dispatch_id is not None and artifact.dispatch_id not in requests:
                self.issue('answer_dispatch_missing', call_id=call_id, dispatch_id=artifact.dispatch_id)
        for artifact, value in visible:
            if value.get('answer') is None:
                self.issue('visible_answer_unavailable', call_id=call_id,
                           dispatch_id=artifact.dispatch_id, state='unavailable')
