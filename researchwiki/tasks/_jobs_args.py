"""Job CLI syntax and typed request construction; no execution or profile loads."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ..operations import ContractError, IngestOptions, IngestRequest
from ..paths import missing_checkout_assets


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ContractError(message)


def _nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _common(parser, *, child=False):
    default = argparse.SUPPRESS if child else None
    parser.add_argument("--library-root", default=default, help="Selected initialized checkout (required)")
    parser.add_argument("--store-root", default=default, help="Machine-local job store base override")
    parser.add_argument("--json", dest="as_json", action="store_true", default=default,
                        help="Emit a version 1 JSON envelope; diagnostics stay on stderr")


def build_parser():
    parser = Parser(prog="researchwiki jobs", description="Submit, run, inspect, and recover local ingestion jobs.")
    _common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("submit", "Queue one PDF or a versioned IngestRequest JSON file; does not start work"),
        ("run", "Run a queued job in the foreground with visible relay handoffs"),
        ("list", "List durable job records without modifying them"),
        ("show", "Inspect a durable job record"),
        ("events", "Read progress after a sequence cursor and expose active relay handoffs"),
        ("reconcile", "Resolve publication evidence and repair local projections; no provider replay"),
        ("retry", "Queue a fresh retry of known-unpublished work; does not start it"),
        ("cancel", "Cancel queued work only; running work cannot be cancelled"),
    ):
        sub = commands.add_parser(command, help=help_text, description=help_text)
        _common(sub, child=True)
        if command == "submit":
            sub.add_argument("pdf", nargs="?", help="PDF path, relative to the selected library if not absolute")
            sub.add_argument("--request-file", help="Version 1 IngestRequest JSON (absolute execution paths)")
            sub.add_argument("--request-id", help="Stable caller ID, required when submitting a PDF")
            sub.add_argument("--options-file", help="IngestOptions JSON; auxiliary paths must be absolute")
            sub.add_argument("--provider-profile", help="Profile path relative to the selected library")
        elif command == "list":
            sub.add_argument("--state", choices=("queued", "running", "finished", "interrupted", "cancelled"))
        else:
            sub.add_argument("job_id")
            if command == "events":
                sub.add_argument("--after-sequence", type=_nonnegative, default=0)
    return parser


def library_root(value):
    if value is None:
        raise ContractError("--library-root is required")
    root = Path(value).expanduser().resolve()
    if missing_checkout_assets(root) or not (root / "wiki").is_dir():
        raise ContractError("--library-root must name an initialized checkout with runtime assets")
    return root


def _read_json(path):
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("JSON input file does not exist") from exc
    except (ValueError, UnicodeError) as exc:
        raise ContractError("JSON input file is not valid UTF-8 JSON") from exc


def _at_root(root, value):
    path = Path(value).expanduser()
    return str((root / path).resolve())


def make_request(args, root, env_file):
    if bool(args.pdf) == bool(args.request_file):
        raise ContractError("submit requires either a PDF or --request-file")
    if args.request_file:
        if any((args.request_id, args.options_file, args.provider_profile)):
            raise ContractError("--request-file cannot be combined with PDF request fields")
        request = IngestRequest.from_dict(_read_json(args.request_file))
        if Path(request.library_root).resolve() != root:
            raise ContractError("request file belongs to a different --library-root")
    else:
        if not args.request_id:
            raise ContractError("--request-id is required when submitting a PDF")
        options = IngestOptions.from_dict(_read_json(args.options_file)) if args.options_file else IngestOptions()
        request = IngestRequest(request_id=args.request_id, library_root=str(root),
                                source_pdf=_at_root(root, args.pdf), options=options,
                                provider_profile=_at_root(root, args.provider_profile) if args.provider_profile else None)
    if env_file:
        if request.provider_profile:
            raise ContractError("choose a stored provider profile or global --env-file, not both")
        request = replace(request, provider_profile=str(Path(env_file).expanduser().resolve()))
    return request
