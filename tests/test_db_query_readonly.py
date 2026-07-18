"""Regression tests for the `researchwiki db query` read-only path.

`db query` opens the DB `mode=ro` and layers `PRAGMA query_only = ON` plus an
authorizer that denies `SQLITE_ATTACH`/`SQLITE_DETACH`. That deny-by-default
combination — not a first-verb keyword blacklist — is what actually enforces
read-only. These tests pin the properties that matter:

1. `ATTACH` (incl. interior-comment / abutting-token forms) and `VACUUM INTO`
   are refused before they can create or write a file. A verb blacklist missed
   `VACUUM INTO` (whole-DB exfiltration) and the `ATTACH/**/DATABASE` form.
2. `mode=ro` refuses to auto-create the DB file → a friendly "no state.db"
   message (exit 2, environment error) instead of a traceback.
3. Errors executing the user's SQL (write/attach/vacuum refusal, syntax error,
   multi-statement, no-such-table) render as a one-line stderr message with
   exit 1 (user-input error), never a Python traceback.
4. URI construction survives a `%` in the path and a *relative*
   RESEARCHWIKI_DB_PATH (`Path.as_uri()` raises on relative paths, so the code
   must `resolve()` first).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from researchwiki.tasks import db as db_task


def _empty_state_db(path: Path) -> None:
    """Create a minimal state.db skeleton (empty papers/claims tables)."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE papers (stem TEXT PRIMARY KEY, title TEXT);
        CREATE TABLE claims (id INTEGER PRIMARY KEY, text TEXT);
        INSERT INTO papers VALUES ('demo-2026-x', 'demo');
        INSERT INTO claims VALUES (1, 'sensitive claim body');
        """
    )
    conn.commit()
    conn.close()


class _Args:
    def __init__(self, sql: str, *, file=None, as_json: bool = False):
        self.sql = sql
        self.file = file
        self.json = as_json


def test_attach_is_refused(tmp_path, monkeypatch, capsys):
    """ATTACH slips past `mode=ro` (which only guards the main DB) — the
    authorizer must deny it before it can open/write an attached file."""
    main_db = tmp_path / "state.db"
    aux_db = tmp_path / "aux.db"
    _empty_state_db(main_db)

    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args(
        f"ATTACH DATABASE 'file:{aux_db}?mode=rwc' AS aux; "
        f"CREATE TABLE aux.stolen AS SELECT * FROM claims;"
    ))
    err = capsys.readouterr().err
    assert rc == 1
    assert "attach" in err.lower()
    # The aux file must not have been created — the authorizer denied the
    # ATTACH at prepare time, before the file was opened.
    assert not aux_db.exists()


def test_attach_with_leading_comment_is_refused(tmp_path, monkeypatch, capsys):
    """A `/* ... */ ATTACH ...` payload is refused — the authorizer denies the
    ATTACH action regardless of surrounding comments/whitespace."""
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args(
        "/* just a benign SELECT, honest */\n"
        f"ATTACH DATABASE 'file:{tmp_path / 'x.db'}?mode=rwc' AS aux;"
    ))
    err = capsys.readouterr().err
    assert rc == 1
    assert "attach" in err.lower()


def test_attach_interior_comment_is_refused(tmp_path, monkeypatch, capsys):
    """`ATTACH/**/DATABASE ...` defeated the old first-verb split (it tokenized
    to `attach/**/database`). The authorizer denies it regardless of tokenizing."""
    main_db = tmp_path / "state.db"
    aux_db = tmp_path / "x.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args(
        f"ATTACH/**/DATABASE 'file:{aux_db}?mode=rwc' AS aux"
    ))
    err = capsys.readouterr().err
    assert rc == 1
    assert "attach" in err.lower()
    assert not aux_db.exists()


def test_vacuum_into_is_refused(tmp_path, monkeypatch, capsys):
    """`VACUUM INTO 'path'` copies the whole DB (incl. claim bodies) to an
    arbitrary path and is NOT blocked by mode=ro. VACUUM INTO attaches its
    target internally, so the ATTACH authorizer denies it."""
    main_db = tmp_path / "state.db"
    leak_db = tmp_path / "leak.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args(f"VACUUM INTO '{leak_db}'"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "read-only" in err.lower() or "vacuum" in err.lower()
    # The exfiltration target must not have been created.
    assert not leak_db.exists()


def test_detach_is_refused(tmp_path, monkeypatch, capsys):
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args("DETACH DATABASE aux;"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "detach" in err.lower()


def test_missing_db_gives_friendly_message(tmp_path, monkeypatch, capsys):
    """First-run: `mode=ro` refuses to create the file. The CLI should print
    a helpful `db rebuild` hint instead of a raw OperationalError traceback.
    This is an environment error (nothing to query), so exit 2."""
    missing = tmp_path / "nonexistent.db"
    monkeypatch.setattr(db_task, "db_path", lambda: missing)

    rc = db_task._cmd_query(_Args("SELECT 1"))
    err = capsys.readouterr().err
    assert rc == 2
    assert "no state.db" in err
    assert "db rebuild" in err


def test_write_statement_is_refused(tmp_path, monkeypatch, capsys):
    """INSERT/UPDATE/DELETE/CTE-writes are refused via the engine-level
    mode=ro guard. A rejected write is a user-input error → exit 1."""
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    for sql in [
        "INSERT INTO papers VALUES ('x', 'y')",
        "UPDATE papers SET title='y' WHERE stem='demo-2026-x'",
        "DELETE FROM papers",
        "WITH x AS (SELECT 1) UPDATE papers SET title='z' WHERE stem='demo-2026-x'",
    ]:
        rc = db_task._cmd_query(_Args(sql))
        err = capsys.readouterr().err
        assert rc == 1, f"expected refuse for {sql!r}"
        assert "read-only" in err.lower() or "readonly" in err.lower(), \
            f"stderr for {sql!r} was: {err!r}"


def test_non_write_error_is_friendly(tmp_path, monkeypatch, capsys):
    """A missing-table error renders as a one-line stderr message, not a raw
    Python traceback. A bad table name is user input → exit 1."""
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args("SELECT * FROM nonexistent_table"))
    err = capsys.readouterr().err
    assert rc == 1
    # Should be a one-liner mentioning the underlying cause, without a
    # 'Traceback (most recent call last)' banner.
    assert "no such table" in err.lower()
    assert "Traceback" not in err


def test_multi_statement_is_friendly(tmp_path, monkeypatch, capsys):
    """A benign multi-statement paste raises sqlite3.ProgrammingError (not
    OperationalError). The broadened except must render it as a one-liner
    (exit 1), not let it escape as a traceback."""
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args("SELECT 1; SELECT 2"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "researchwiki db query:" in err
    assert "Traceback" not in err


def test_path_with_percent_opens(tmp_path, monkeypatch, capsys):
    """The URI construction must survive a `%` in the path — a naive f-string
    misinterpreted `%25` as a percent-encoded byte."""
    db = tmp_path / "has%pct.db"
    _empty_state_db(db)
    monkeypatch.setattr(db_task, "db_path", lambda: db)

    rc = db_task._cmd_query(_Args("SELECT count(*) FROM papers"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "1" in out  # the one demo row


def test_relative_db_path_opens(tmp_path, monkeypatch, capsys):
    """A relative RESEARCHWIKI_DB_PATH must still open — `path.resolve()`
    absolutizes it before `as_uri()` (which raises on relative paths)."""
    monkeypatch.chdir(tmp_path)
    _empty_state_db(tmp_path / "state.db")
    monkeypatch.setattr(db_task, "db_path", lambda: Path("state.db"))  # relative

    rc = db_task._cmd_query(_Args("SELECT count(*) FROM papers"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "1" in out


def test_select_succeeds(tmp_path, monkeypatch, capsys):
    """Baseline: a plain SELECT still returns rows and exits 0 (the authorizer
    and query_only don't interfere with reads)."""
    main_db = tmp_path / "state.db"
    _empty_state_db(main_db)
    monkeypatch.setattr(db_task, "db_path", lambda: main_db)

    rc = db_task._cmd_query(_Args("SELECT stem FROM papers"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo-2026-x" in out
