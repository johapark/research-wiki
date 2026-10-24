"""Inspect retained ingestion attempts and manage historical ingest records."""

import argparse
import json
from pathlib import Path

from ..errors import EnvironmentFailure
from ..records import RecordConflict, RecordStore
from ..records.export import export_attempt
from ..records.publication import publication_receipts


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="archive path; otherwise use the selected profile")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "show", "verify", "export", "recover", "inventory", "backup"):
        sub = subs.add_parser(name)
        sub.add_argument("--json", action="store_true")
        if name not in ("list", "recover", "inventory", "backup"):
            sub.add_argument("attempt_id")
        if name == "verify":
            sub.add_argument("--repair-checkpoint", action="store_true")
        if name == "show":
            sub.add_argument("--lineage", action="store_true", help="include full draft and selection lineage payloads")
            sub.add_argument("--grading", action="store_true", help="include full grading input/evidence/result payloads")
        if name == "export":
            sub.add_argument("--output", type=Path, required=True)
        if name == "inventory":
            sub.add_argument("--legacy", action="store_true",
                             help="inventory historical SQLite ingest records")
            sub.add_argument("--db-path", type=Path,
                             help="read this SQLite file instead of the selected and legacy DB paths")
        if name == "backup":
            sub.add_argument("--legacy", action="store_true",
                             help="back up a historical SQLite ingest database")
            sub.add_argument("--db-path", type=Path, required=True,
                             help="source SQLite database path")
            sub.add_argument("--output", type=Path, required=True,
                             help="new directory for state.db and its manifest")
    return parser


def _dispatch(args, store):
    if args.command == "backup":
        if not args.legacy:
            raise ValueError("records backup currently requires --legacy")
        from ..records.legacy_backup import backup_legacy_database
        return backup_legacy_database(args.db_path, args.output)
    if args.command == "inventory":
        if not args.legacy:
            raise ValueError("records inventory currently requires --legacy")
        from ..records.legacy import inventory_legacy
        return inventory_legacy(db_override=args.db_path)
    if args.command == "recover":
        return _recover(store)
    if args.command == "list":
        return {"attempts": [{"attempt_id": m.spec.attempt_id, "capture_state": m.capture_state,
                               "terminal": m.terminal.to_dict() if m.terminal else None}
                              for m in store.list_attempts()]}
    if args.command == "show":
        result = {"manifest": store.read_attempt(args.attempt_id).to_dict(),
                  "publication_receipts": publication_receipts(store, args.attempt_id)}
        if args.lineage:
            from ..records.lineage import read_lineage
            result["lineage"] = read_lineage(store, args.attempt_id)
        if args.grading:
            from ..records.grading import read_grading
            result["grading"] = read_grading(store, args.attempt_id)
        return result
    if args.command == "verify":
        if args.repair_checkpoint:
            store.repair_checkpoint(args.attempt_id)
        report = store.verify(args.attempt_id)
        return {"attempt_id": args.attempt_id, "checkpoint": report.checkpoint,
                "event_count": report.event_count, "blob_count": report.blob_count,
                "capture_state": report.manifest.capture_state}
    output = export_attempt(store, args.attempt_id, args.output)
    return {"attempt_id": args.attempt_id, "output": str(output)}


def _recover(store):
    from ..mutation import corpus_lock, pending_journals, recover_pending

    archive_id = store.metadata().archive_id
    if not (Path.cwd() / "wiki").is_dir():
        raise RecordConflict("publication recovery must run in its initialized library")
    with corpus_lock():
        for journal in pending_journals():
            details = journal.get("details")
            binding = details.get("records_binding") if isinstance(details, dict) else None
            if not isinstance(binding, dict):
                raise RecordConflict("pending journals lack this archive's binding; review them before recovery")
            if binding.get("archive_id") != archive_id or binding.get("root") != str(store.root):
                raise RecordConflict("pending journals include other archives; review them before recovery")
        notes = recover_pending()
        if pending_journals():
            from ..records.errors import RecordStoreUnavailable
            raise RecordStoreUnavailable("publication journals remain unresolved; inspect archive and journal integrity")
    return {"recovery_notes": notes}


def main(argv, *, env_file=None):
    args = _parser().parse_args(argv)
    try:
        if args.command in ("inventory", "backup"):
            result = {"schema_version": 1, "command": args.command,
                      **_dispatch(args, None), "error": None}
        else:
            if args.root:
                root = Path(args.root).expanduser().resolve()
            else:
                from ..records.settings import selected
                root = selected().root
                if root is None:
                    raise RecordConflict("select an archive with --root or an enabled ingest.retention profile")
            result = {"schema_version": 1, "command": args.command,
                      **_dispatch(args, RecordStore(root)), "error": None}
        code = 0
    except (RecordConflict, ValueError) as exc:
        result, code = {"schema_version": 1, "command": args.command, "error": str(exc)}, 1
    except (EnvironmentFailure, OSError) as exc:
        result, code = {"schema_version": 1, "command": args.command, "error": str(exc)}, 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif result["error"]:
        print(result["error"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return code
