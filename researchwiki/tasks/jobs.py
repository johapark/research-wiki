"""Submit, run, inspect, and reconcile durable local ingestion jobs."""

import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import traceback

from ..errors import EnvironmentFailure
from ..jobs import JobConflict, JobSupervisor
from ..operations import ContractError
from ._jobs_args import build_parser, library_root, make_request


def _envelope(command, *, error=None, **data):
    return dict(schema_version=1, command=command, **data, error=error)


def _error(code, category, message):
    return dict(code=code, category=category, message=message)


def _lookup(supervisor, job_id):
    path = supervisor.store.path(job_id)  # validates syntax before touching disk
    try:
        path.stat()
    except FileNotFoundError as exc:
        raise JobConflict("unknown job") from exc
    return supervisor.get_job(job_id)


def _stop_worker(handle):
    """Interrupt our foreground worker; this is never queued cancellation."""
    process = handle.process
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        process.terminate()
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        process.kill()
        process.wait()


def _run(supervisor, job_id):
    handle = supervisor.start_job(job_id, stdout=subprocess.DEVNULL)
    try:
        record = supervisor.wait(handle)
    except KeyboardInterrupt:
        _stop_worker(handle)
        record = supervisor.refresh_job(job_id)
        return record, 1, _error("interrupted", "validation", "Worker interrupted; reconcile publication before retry")
    error = (_error("worker_not_finished", "environment", "Worker did not produce a terminal result; inspect or reconcile this job")
             if record.result is None else None)
    return record, _execution_exit(record), error


def _execution_exit(record):
    result = record.result
    if result is None or record.execution == "interrupted":
        return 2
    if result.publication == "published":
        return 0  # published content needs repair, never another paid add
    if result.outcome == "draft_ready":
        return 0 if result.promotion == "never" else 1
    if result.error is not None:
        return {"validation": 1, "gate": 1, "budget": 1, "environment": 2, "internal": 3}[result.error.category]
    return 2 if result.publication == "unknown" else 1


def _dispatch(args, supervisor, root, env_file):
    command = args.command
    if env_file and command != "submit":
        raise ContractError("global --env-file applies to jobs submit; existing jobs retain their stored profile")
    if command == "submit":
        record = supervisor.submit_ingest(make_request(args, root, env_file))
    elif command == "list":
        records = supervisor.list_jobs()
        if args.state:
            records = tuple(record for record in records if record.execution == args.state)
        return _envelope(command, jobs=[record.to_dict() for record in records]), 0
    else:
        record = _lookup(supervisor, args.job_id)
        if command == "events":
            events = supervisor.read_events(args.job_id, after_sequence=args.after_sequence)
            return _envelope(command, job_id=args.job_id,
                             events=[event.to_dict() for event in events],
                             next_sequence=events[-1].sequence if events else args.after_sequence,
                             pending_relays=[handoff.to_dict() for handoff in supervisor.pending_relays(args.job_id)]), 0
        if command == "run":
            record, code, error = _run(supervisor, args.job_id)
            return _envelope(command, job=record.to_dict(), error=error), code
        if command == "reconcile":
            record = supervisor.reconcile_job(args.job_id)
            if record.result and (record.result.publication == "unknown" or record.result.projections == "maintenance_pending"):
                return _envelope(command, job=record.to_dict(), error=_error(
                    "reconciliation_pending", "environment", "Recovery remains unresolved or maintenance is pending; inspect this job's diagnostics")), 2
        if command == "retry":
            record = supervisor.retry_job(args.job_id)
        if command == "cancel":
            record = supervisor.cancel_queued_job(args.job_id)
    return _envelope(command, job=record.to_dict()), 0


def _next_command(action, record, supervisor):
    command = ["researchwiki", "jobs", action, record["job_id"], "--library-root", str(supervisor.store.library_root)]
    if supervisor.store.base is not None:
        command.extend(["--store-root", str(supervisor.store.base)])
    return shlex.join(command)


def _print_record(record, supervisor):
    print(f"Job: {record['job_id']}  ({record['execution']})")
    result = record["result"]
    if result is None:
        if record["execution"] == "queued":
            print("Run: " + _next_command("run", record, supervisor))
        else:
            print("Check owner / recover: " + _next_command("reconcile", record, supervisor))
        return
    print(f"Outcome: {result['outcome'] or 'unresolved'}; publication: {result['publication']}; "
          f"projections: {result['projections']}; post-work: {result['post_work']}")
    for field in ("page_path", "pdf_path", "draft_path"):
        if result[field]:
            print(f"  {field}: {result[field]}")
    for reason in result["gate_reasons"]:
        print("  Gate: " + reason)
    for warning in result["warnings"]:
        print("  Warning: " + warning)
    if result["error"]:
        print(f"  Error: {result['error']['code']}; retry: {result['error']['retry']}")
    if result["publication"] == "unknown" or result["projections"] == "maintenance_pending":
        print("Recover: " + _next_command("reconcile", record, supervisor))


def _render(payload, as_json, supervisor=None):
    if as_json:
        print(json.dumps(payload, allow_nan=False, indent=2))
        return
    if "job" in payload:
        _print_record(payload["job"], supervisor)
    elif "jobs" in payload:
        if not payload["jobs"]:
            print("No jobs.")
        for job in payload["jobs"]:
            result = job["result"] or {}
            print(f"{job['job_id']}  {job['execution']:<11} {result.get('outcome') or '-':<15} "
                  f"{job['request']['request_id']}")
    elif "events" in payload:
        for event in payload["events"]:
            print(f"{event['sequence']:>5} {event['kind']:<26} {event['phase'] or '-'}")
        for handoff in payload["pending_relays"]:
            print(f"Relay {handoff['op_id']}: prompt={handoff['prompt_path']} response={handoff['response_path']}")
        print(f"Next sequence: {payload['next_sequence']}")
    if payload["error"]:
        print("researchwiki jobs: " + payload["error"]["message"], file=sys.stderr)


def main(argv: list[str], *, env_file=None) -> int:
    args = None
    supervisor = None
    as_json = "--json" in argv
    try:
        args = build_parser().parse_args(argv)
        as_json = bool(args.as_json)
        root = library_root(args.library_root)
        supervisor = JobSupervisor(root, store_root=Path(args.store_root).expanduser().resolve() if args.store_root else None)
        payload, code = _dispatch(args, supervisor, root, env_file)
    except (ContractError, ValueError) as exc:
        code = 1
        payload = _envelope(args.command if args else None, error=_error("invalid_request", "validation", str(exc)))
    except (EnvironmentFailure, OSError) as exc:
        code = 2
        payload = _envelope(args.command if args else None, error=_error("environment_unavailable", "environment", str(exc)))
    except KeyboardInterrupt:
        code = 1
        payload = _envelope(args.command if args else None, error=_error("interrupted", "validation", "Command interrupted; inspect job state before repeating the action"))
    except Exception:
        traceback.print_exc()
        code = 3
        payload = _envelope(args.command if args else None, error=_error("internal_error", "internal", "Internal error; see stderr diagnostics"))
    _render(payload, as_json, supervisor)
    return code
