"""Regression tests for the FTS-install / schema-migration concurrency fixes.

Two batch-ingest workers each open their own connection and run
`init_schema` → `_migrate` → `_install_claims_fts` against the same DB file.
Before the fixes, the first post-upgrade batch could crash a worker two ways:

- `DROP TRIGGER claims_au` (no IF EXISTS): both workers read the old-form
  trigger via an unlocked sqlite_master read, both take the drop branch, and
  the loser hits "no such trigger".
- `ALTER TABLE claims ADD COLUMN` (no IF NOT EXISTS): both observe the column
  absent via an unlocked PRAGMA read, both ALTER, and the loser hits
  "duplicate column name".
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from researchwiki.db.connection import (
    _install_claims_fts,
    _migrate,
    _safe_add_column,
)

_OLD_CLAIMS_AU = (
    "CREATE TRIGGER claims_au AFTER UPDATE ON claims BEGIN"
    "  INSERT INTO claims_fts(claims_fts, rowid, text) "
    "  VALUES('delete', old.id, old.text);"
    "  INSERT INTO claims_fts(rowid, text) VALUES (new.id, new.text);"
    "END"
)


def _fts5_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE _t USING fts5(x)")
        c.close()
        return True
    except sqlite3.OperationalError:
        return False


requires_fts5 = pytest.mark.skipif(not _fts5_available(), reason="SQLite build lacks FTS5")


def _minimal_db(path: Path) -> sqlite3.Connection:
    """A DB with just the tables the FTS install / migrate paths touch."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row  # get_connection sets this; _migrate reads row["name"]
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(
        "CREATE TABLE claims (id INTEGER PRIMARY KEY, paper_stem TEXT, text TEXT);"
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    conn.commit()
    return conn


# ---- #7: _safe_add_column tolerates a concurrent duplicate --------------

def test_safe_add_column_tolerates_duplicate():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, text TEXT, embed_model TEXT)")
    # Column already present (a peer added it) → must NOT raise.
    _safe_add_column(conn, "ALTER TABLE claims ADD COLUMN embed_model TEXT")
    # A genuinely new column still gets added.
    _safe_add_column(conn, "ALTER TABLE claims ADD COLUMN brand_new TEXT")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(claims)")}
    assert "brand_new" in cols
    # A non-duplicate error still propagates.
    with pytest.raises(sqlite3.OperationalError):
        _safe_add_column(conn, "ALTER TABLE nonexistent ADD COLUMN x TEXT")


def test_migrate_is_idempotent_on_existing_columns(tmp_path):
    """_migrate runs the ADD COLUMNs through _safe_add_column, so re-running
    after the columns exist (or a peer added them) doesn't raise."""
    if not _fts5_available():
        pytest.skip("SQLite build lacks FTS5")
    conn = _minimal_db(tmp_path / "state.db")
    _migrate(conn)  # first run adds embed_model / supporting_text / claim_slug
    cols = {r[1] for r in conn.execute("PRAGMA table_info(claims)")}
    assert {"embed_model", "supporting_text", "claim_slug"} <= cols
    # Second run must be a clean no-op (no "duplicate column name").
    _migrate(conn)
    conn.close()


# ---- #3: claims_au trigger migration + IF EXISTS ------------------------

@requires_fts5
def test_claims_au_migrated_to_update_of_text(tmp_path):
    conn = _minimal_db(tmp_path / "state.db")
    _install_claims_fts(conn)  # baseline: new-form trigger + fts table
    # Simulate a pre-upgrade DB: replace with the old broad trigger.
    conn.execute("DROP TRIGGER claims_au")
    conn.execute(_OLD_CLAIMS_AU)
    conn.commit()

    _install_claims_fts(conn)  # migration path: drop old, create narrow
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='claims_au'"
    ).fetchone()[0]
    assert "UPDATE OF text" in ddl

    # Idempotent: calling again with the new-form trigger present is a no-op.
    _install_claims_fts(conn)
    ddl2 = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='claims_au'"
    ).fetchone()[0]
    assert "UPDATE OF text" in ddl2
    conn.close()


@requires_fts5
def test_concurrent_install_fts_no_crash(tmp_path):
    """Two connections migrating the old-form trigger at once must not crash
    (DROP TRIGGER IF EXISTS + serialized DDL under busy_timeout)."""
    db = tmp_path / "state.db"
    conn = _minimal_db(db)
    _install_claims_fts(conn)          # create fts table + triggers
    conn.execute("DROP TRIGGER claims_au")
    conn.execute(_OLD_CLAIMS_AU)       # downgrade to old-form to force migration
    conn.commit()
    conn.close()

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker():
        try:
            c = sqlite3.connect(str(db))
            c.execute("PRAGMA busy_timeout = 5000")
            barrier.wait(timeout=5)
            _install_claims_fts(c)
            c.close()
        except BaseException as e:  # noqa: BLE001 - capture for assertion
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"concurrent _install_claims_fts raised: {errors!r}"
    # End state: exactly one claims_au, narrowed to UPDATE OF text.
    check = sqlite3.connect(str(db))
    ddl = check.execute(
        "SELECT sql FROM sqlite_master WHERE name='claims_au'"
    ).fetchone()[0]
    check.close()
    assert "UPDATE OF text" in ddl
