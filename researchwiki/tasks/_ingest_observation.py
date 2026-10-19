"""Opt-in single-PDF CLI artifacts; never replace ingestion with a test script."""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from ..operations import (
    ContractError, EventRecorder, IngestOptions, IngestRequest, IngestResult,
    OperationError, error_from_exception, observe_models,
)


def request_from_args(args) -> IngestRequest:
    from ..env_profiles import ACTIVE_ENV_FILE_VAR

    values = {name: getattr(args, name) for name in (
        "n_drafts", "max_evolve", "verify_claim_entailment", "doi", "title", "year",
        "allow_rename", "no_cross_link", "claim_overlap", "contradiction_alert", "memory_evolve",
        "max_model_calls", "max_tokens", "max_cost_usd", "max_wall_seconds",
    )}
    values.update(
        use_stub=args.stub, use_semantic=not args.no_semantic,
        use_llm_reconcile=args.llm_reconcile,
        promotion="always" if args.auto_promote else "never" if args.force_sandbox else "auto",
        authors=tuple(a.strip() for a in args.authors.split(";") if a.strip()) if args.authors else None,
        supplementary=tuple(str(Path(p).expanduser().resolve()) for p in (args.supplementary or ())),
        author_prompt_file=str(Path(args.author_prompt_file).expanduser().resolve()) if args.author_prompt_file else None,
    )
    return IngestRequest(
        request_id=uuid.uuid4().hex, library_root=str(Path.cwd().resolve()),
        source_pdf=str(Path(args.pdfs[0]).expanduser().resolve()),
        provider_profile=os.environ.get(ACTIVE_ENV_FILE_VAR), options=IngestOptions(**values),
    )


class CliObservation:
    def __init__(self, args):
        self.request = request_from_args(args)
        self.event_path = self._output(getattr(args, "events_file", None))
        self.result_path = self._output(getattr(args, "result_file", None))
        if self.event_path is not None and self.event_path == self.result_path:
            raise ContractError("event and result files must be different")
        self.stream = None
        self.execution = None
        self.error = None
        self.started = False
        self.recorder = EventRecorder(self.request.request_id, self._event)

    def _output(self, value):
        if value is None:
            return None
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ContractError("report path cannot be an existing symlink")
        path = path.resolve()
        if path.exists():
            raise ContractError(f"report already exists: {path}")
        root = Path(self.request.library_root)
        for directory in ("wiki", "papers", "inbox"):
            content = (root / directory).resolve()
            if path == content or content in path.parents:
                raise ContractError("report path must be outside wiki/, papers/, and inbox/")
        return path

    def open(self):
        created = []
        try:
            for path in (self.event_path, self.result_path):
                if path is None:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # Reserve with exclusive creation; never clobber a source or a
                # previously generated report, even after validation races.
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                created.append(path)
                stream = os.fdopen(fd, "w", encoding="utf-8")
                if path == self.event_path:
                    self.stream = stream
                else:
                    stream.close()
        except OSError:
            self.close()
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def _event(self, event):
        if self.stream is not None:
            self.stream.write(json.dumps(event.to_dict(), allow_nan=False) + "\n")
            self.stream.flush()

    def finish(self, exit_code):
        if self.result_path is None:
            return
        if self.execution is not None:
            result = self.execution.result
        else:
            error = error_from_exception(self.error) if self.error is not None else OperationError(
                code="command_rejected", category="validation",
                message=f"CLI exited {exit_code} before returning an ingest result", retry="after_correction",
            )
            result = IngestResult(
                request_id=self.request.request_id, attempt_id=self.recorder.attempt_id,
                execution="interrupted" if isinstance(self.error, KeyboardInterrupt) else "finished",
                outcome=None if isinstance(self.error, KeyboardInterrupt) else "failed",
                publication="unknown" if self.started else "not_published",
                projections="unknown" if self.started else "not_applicable", error=error,
            )
        if self.recorder.delivery_errors and self.execution is None:
            result = replace(result, warnings=result.warnings + (
                f"{self.recorder.delivery_errors} progress event delivery failure(s)",
            ))
        from ..fsatomic import write_json_atomic
        write_json_atomic(self.result_path, result.to_dict())

    def close(self):
        if self.stream is not None:
            stream = self.stream
            self.stream = None
            try:
                stream.close()
            except OSError:
                print("[operations] progress stream close failed; inspect the CLI trace", file=sys.stderr)


def run_with_reports(args, command, *, prog):
    if not (getattr(args, "events_file", None) or getattr(args, "result_file", None)):
        return command(args)
    if len(args.pdfs) != 1 or args.workers is not None or args.resume:
        print(f"{prog}: observation reports require one PDF without --workers/--resume", file=sys.stderr)
        return 1
    if args.auto_promote and args.force_sandbox:
        print(f"{prog}: --auto-promote and --force-sandbox are mutually exclusive", file=sys.stderr)
        return 1
    try:
        observer = CliObservation(args)
        observer.open()
    except (ContractError, OSError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 1 if isinstance(exc, ContractError) else 2
    args._observation = observer
    code = 3
    try:
        with observe_models(observer.recorder):
            code = command(args)
        return code
    except BaseException as exc:
        observer.error = exc
        raise
    finally:
        try:
            observer.finish(code)
        except Exception as exc:
            print(f"{prog}: result report unavailable ({type(exc).__name__}); inspect the CLI trace", file=sys.stderr)
        finally:
            observer.close()
            del args._observation
