"""Consistent, non-destructive SQLite backup for historical ingest state."""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..fsatomic import file_sha256, write_private_json
from .errors import RecordConflict, RecordStoreUnavailable
from .legacy import _readonly_connection


def backup_legacy_database(source: Path, output: Path) -> dict:
    """Write a verified SQLite online-backup bundle into a new directory.

    The source is opened read-only and copied with SQLite's backup API, which
    reads a transactionally consistent database snapshot, including committed
    WAL content visible to SQLite. The destination is never overwritten.
    """
    source = Path(source).expanduser().resolve(strict=False)
    output_input = Path(output).expanduser()
    if output_input.exists() or output_input.is_symlink():
        raise RecordConflict(f"backup destination already exists: {output_input}")
    output = output_input.resolve(strict=False)
    if not source.is_file():
        raise RecordConflict(f"source database is not a regular file: {source}")
    if not output.parent.is_dir():
        raise RecordConflict(f"backup parent directory does not exist: {output.parent}")
    if output.exists() or output.is_symlink():
        raise RecordConflict(f"backup destination already exists: {output}")

    try:
        source_info = source.stat()
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RecordConflict(f"backup destination already exists: {output}") from exc
    except OSError as exc:
        raise RecordStoreUnavailable(f"cannot prepare backup destination {output}: {exc}") from exc

    db_path = output / "state.db"
    source_conn = None
    target_conn = None
    started = _now()
    try:
        fd = os.open(db_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
        os.close(fd)
        source_conn = _readonly_connection(source)
        target_conn = sqlite3.connect(str(db_path))
        source_conn.backup(target_conn, pages=256, sleep=0.1)

        integrity = target_conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RecordStoreUnavailable(f"backup integrity check failed: {integrity}")
        tables = sorted(row[0] for row in target_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ))
        history_rows = None
        if "ingest_iterations" in tables:
            history_rows = target_conn.execute("SELECT COUNT(*) FROM ingest_iterations").fetchone()[0]
        target_conn.close()
        target_conn = None
        source_conn.close()
        source_conn = None

        completed = _now()
        info = db_path.stat()
        manifest = {
            "schema": "researchwiki.legacy_sqlite_backup",
            "schema_version": 1,
            "backup_id": str(uuid.uuid4()),
            "created_at": completed,
            "source": {
                "path": str(source),
                "main_file_size_bytes_at_start": source_info.st_size,
                "main_file_mtime_ns_at_start": source_info.st_mtime_ns,
            },
            "snapshot": {
                "method": "sqlite_online_backup_api",
                "started_at": started,
                "completed_at": completed,
                "consistency": "consistent SQLite snapshot; committed WAL content visible to SQLite is included",
            },
            "database": {
                "filename": "state.db",
                "size_bytes": info.st_size,
                "sha256": file_sha256(db_path),
                "sqlite_version": sqlite3.sqlite_version,
            },
            "validation": {
                "integrity_check": integrity,
                "tables": tables,
                "ingest_iterations_rows": history_rows,
            },
            "limitations": [
                "This bundle contains the selected SQLite database only.",
                "Machine-local jobs, relay mailboxes, mutation journals, PDFs, and wiki files are not included.",
                "A consistent database snapshot does not reconstruct missing historical source hashes or model inputs.",
            ],
        }
        write_private_json(output / "manifest.json", manifest)
        return {
            "backup_id": manifest["backup_id"],
            "output": str(output),
            "database": str(db_path),
            "manifest": str(output / "manifest.json"),
            "database_sha256": manifest["database"]["sha256"],
            "ingest_iterations_rows": history_rows,
        }
    except BaseException as exc:
        if target_conn is not None:
            try:
                target_conn.close()
            except sqlite3.Error:
                pass
        if source_conn is not None:
            try:
                source_conn.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(output, ignore_errors=True)
        if isinstance(exc, (RecordConflict, RecordStoreUnavailable)):
            raise
        if isinstance(exc, (sqlite3.Error, OSError)):
            raise RecordStoreUnavailable(f"could not create consistent SQLite backup: {exc}") from exc
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
