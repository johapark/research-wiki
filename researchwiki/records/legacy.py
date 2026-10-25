"""Read-only inventory of historical SQLite ingest records.

This is an audit view, not an importer. It deliberately avoids the normal DB
connection helper because that helper can create, migrate, or seed a database.
"""

from __future__ import annotations

import json
import sqlite3
import stat
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..db.connection import legacy_db_path


_ITERATION_COLUMNS = {
    "id", "attempt_id", "paper_stem", "pdf_filename", "iteration", "role",
    "decision", "model_used", "gate_metrics", "created_at",
}
_REQUIRED_ITERATION_COLUMNS = {"id", "attempt_id", "iteration", "role", "created_at"}
_TEXT_FLAGS = {
    "draft_text": "has_draft",
    "grader_scores": "has_grader_scores",
    "critic_notes": "has_critic_notes",
}


def inventory_legacy(*, db_override: Path | None = None,
                     library_root: Path | None = None) -> dict:
    """Inventory SQLite history and adjacent local evidence without writes."""
    candidates = _database_candidates(db_override)
    databases = [_inventory_database(path, roles) for path, roles in candidates]
    attempt_ids = {
        attempt["attempt_id"]
        for database in databases if database["status"] == "available"
        for attempt in database["attempts"] if isinstance(attempt["attempt_id"], str)
    }
    from .related_inventory import inventory_related_state

    return {
        "inventory_version": 1,
        "read_only": True,
        "scope": "legacy_sqlite_ingest_history",
        "databases": databases,
        "related_evidence": inventory_related_state(
            Path.cwd() if library_root is None else library_root, attempt_ids
        ),
        "limitations": [
            "This inventory does not import or rewrite SQLite rows or adjacent files.",
            "Legacy iteration history does not contain the original PDF hash or full target-claim list.",
            "Claims-table source hashes describe rebuildable claim rows and are not bound to historical attempts.",
            "Consumed relay mailbox payloads are deleted and cannot be inventoried retrospectively.",
            "Adjacent jobs, journals, drafts, and caches are summarized as metadata; their content is not exported.",
        ],
    }


def _database_candidates(db_override: Path | None) -> list[tuple[Path, list[str]]]:
    if db_override is not None:
        return [(db_override.expanduser().resolve(strict=False), ["explicit"])]

    from ..db.connection import db_path as selected_db_path

    raw = [
        (selected_db_path(prepare=False), "selected"),
        (legacy_db_path(), "legacy_global"),
    ]
    merged: dict[Path, list[str]] = {}
    for path, role in raw:
        resolved = Path(path).expanduser().resolve(strict=False)
        merged.setdefault(resolved, []).append(role)
    return [(path, roles) for path, roles in merged.items()]


def _inventory_database(path: Path, source_roles: list[str]) -> dict:
    result = {
        "path": str(path),
        "source_roles": source_roles,
        "status": "not_found",
        "size_bytes": None,
        "iteration_rows": 0,
        "attempt_count": 0,
        "attempts": [],
        "claims_source_hashes": {
            "status": "unavailable",
            "binding_scope": "current_claim_rows_not_historical_attempts",
            "paper_stems": [],
        },
    }
    try:
        info = path.stat()
    except FileNotFoundError:
        return result
    except OSError as exc:
        result["status"] = "unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not stat.S_ISREG(info.st_mode):
        result["status"] = "not_a_regular_file"
        return result
    try:
        result["size_bytes"] = info.st_size
        with closing(_readonly_connection(path)) as conn:
            conn.execute("BEGIN")
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            if "ingest_iterations" not in tables:
                result["status"] = "no_ingest_history"
                result["claims_source_hashes"] = _claims_hash_inventory(conn, tables)
                return result

            columns = {row[1] for row in conn.execute("PRAGMA table_info(ingest_iterations)")}
            missing = sorted(_REQUIRED_ITERATION_COLUMNS - columns)
            if missing:
                result["status"] = "unsupported_schema"
                result["missing_required_columns"] = missing
                result["claims_source_hashes"] = _claims_hash_inventory(conn, tables)
                return result

            selected = [f'"{name}"' for name in sorted(_ITERATION_COLUMNS & columns)]
            selected.extend(
                f'CASE WHEN "{column}" IS NOT NULL AND length("{column}") > 0 '
                f'THEN 1 ELSE 0 END AS "{alias}"'
                for column, alias in _TEXT_FLAGS.items() if column in columns
            )
            if "gate_metrics" in columns:
                selected.append(
                    'CASE WHEN "role" = \'target_claims\' THEN "gate_metrics" '
                    'ELSE NULL END AS "target_gate_metrics"'
                )
            rows = conn.execute(
                "SELECT " + ", ".join(selected)
                + ' FROM ingest_iterations ORDER BY "id"'
            ).fetchall()
            result["status"] = "available"
            result["iteration_rows"] = len(rows)
            result["attempts"] = _summarize_attempts(rows, columns)
            result["attempt_count"] = len(result["attempts"])
            result["claims_source_hashes"] = _claims_hash_inventory(conn, tables)
    except (sqlite3.Error, OSError) as exc:
        result["status"] = "unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _readonly_connection(path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite URI and reject SQL writes at the connection."""
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")

    def deny_mutation(action, *_):
        denied = {
            sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW,
            sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_VIEW,
            sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_INDEX, sqlite3.SQLITE_DROP_TEMP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TEMP_VIEW,
            sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW,
            sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_REINDEX,
            sqlite3.SQLITE_ANALYZE, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        }
        return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

    conn.set_authorizer(deny_mutation)
    return conn


def _summarize_attempts(rows: list[sqlite3.Row], columns: set[str]) -> list[dict]:
    grouped: dict[object, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["attempt_id"]].append(row)

    summaries = []
    for attempt_id, attempt_rows in grouped.items():
        roles = Counter(str(row["role"]) for row in attempt_rows)
        stems = sorted({str(row["paper_stem"]) for row in attempt_rows
                        if "paper_stem" in columns and row["paper_stem"]})
        pdf_files = sorted({Path(str(row["pdf_filename"])).name for row in attempt_rows
                            if "pdf_filename" in columns and row["pdf_filename"]})
        model_counts = Counter(str(row["model_used"]) for row in attempt_rows
                               if "model_used" in columns and row["model_used"])
        timestamps = sorted(
            timestamp for row in attempt_rows
            if (timestamp := _epoch(row["created_at"])) is not None
        )
        ids = [int(row["id"]) for row in attempt_rows if row["id"] is not None]
        target_rows = [row for row in attempt_rows
                       if str(row["role"]) == "target_claims"]
        target_summary = _target_claim_summary(target_rows, columns)
        summaries.append({
            "attempt_id": attempt_id,
            "source_row_id_range": [min(ids), max(ids)] if ids else None,
            "iteration_rows": len(attempt_rows),
            "roles": dict(sorted(roles.items())),
            "paper_stems": stems,
            "source_pdf_filenames": pdf_files,
            "models_observed": dict(sorted(model_counts.items())),
            "draft_rows": sum(bool(row["has_draft"]) for row in attempt_rows)
                          if "draft_text" in columns else None,
            "grader_score_rows": sum(bool(row["has_grader_scores"]) for row in attempt_rows)
                                 if "grader_scores" in columns else None,
            "critic_note_rows": sum(bool(row["has_critic_notes"]) for row in attempt_rows)
                                if "critic_notes" in columns else None,
            "target_claim_history": target_summary,
            "original_pdf_sha256": "not_recorded_in_iteration_history",
            "model_request_response_artifacts": "not_recorded_in_iteration_history",
            "prompt_parser_identity": "not_recorded_in_iteration_history",
            "first_observed_at": _utc(timestamps[0]) if timestamps else None,
            "last_observed_at": _utc(timestamps[-1]) if timestamps else None,
        })
    return sorted(summaries, key=lambda item: str(item["attempt_id"]))


def _target_claim_summary(rows: list[sqlite3.Row], columns: set[str]) -> dict:
    decisions = Counter(str(row["decision"]) for row in rows
                        if "decision" in columns and row["decision"] is not None)
    claim_counts = []
    keyword_counts = []
    if "gate_metrics" in columns:
        for row in rows:
            raw = row["target_gate_metrics"]
            if not raw:
                continue
            try:
                metrics = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metrics, dict):
                continue
            claim_count = metrics.get("target_claims")
            keyword_count = metrics.get("keywords")
            if isinstance(claim_count, int) and not isinstance(claim_count, bool):
                claim_counts.append(claim_count)
            if isinstance(keyword_count, int) and not isinstance(keyword_count, bool):
                keyword_counts.append(keyword_count)
    return {
        "records": len(rows),
        "decisions": dict(sorted(decisions.items())),
        "claim_counts_recorded": len(claim_counts),
        "claim_count_total_when_recorded": sum(claim_counts) if claim_counts else None,
        "keyword_counts_recorded": len(keyword_counts),
        "keyword_count_total_when_recorded": sum(keyword_counts) if keyword_counts else None,
        "full_claim_lists_retained_here": False,
    }


def _claims_hash_inventory(conn: sqlite3.Connection, tables: set[str]) -> dict:
    empty = {
        "status": "not_available",
        "binding_scope": "current_claim_rows_not_historical_attempts",
        "paper_stems": [],
    }
    if "claims" not in tables:
        return empty
    columns = {row[1] for row in conn.execute("PRAGMA table_info(claims)")}
    if not {"paper_stem", "source_pdf_sha256"}.issubset(columns):
        return empty
    rows = conn.execute(
        "SELECT paper_stem, COUNT(*) AS claim_rows, "
        "SUM(CASE WHEN source_pdf_sha256 IS NOT NULL AND source_pdf_sha256 != '' "
        "THEN 1 ELSE 0 END) AS hash_bound_claim_rows, "
        "COUNT(DISTINCT CASE WHEN source_pdf_sha256 != '' THEN source_pdf_sha256 END) "
        "AS distinct_hashes FROM claims GROUP BY paper_stem ORDER BY paper_stem"
    ).fetchall()
    return {
        "status": "available",
        "binding_scope": "current_claim_rows_not_historical_attempts",
        "paper_stems": [
            {
                "paper_stem": row["paper_stem"],
                "claim_rows": row["claim_rows"],
                "hash_bound_claim_rows": row["hash_bound_claim_rows"],
                "distinct_source_pdf_hashes": row["distinct_hashes"],
            }
            for row in rows
        ],
    }


def _epoch(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _utc(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None
