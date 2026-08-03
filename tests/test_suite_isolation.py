"""The suite must not write to the developer's real state.db.

`db_path()` resolves from the current working directory (`_repo_key()` over
`wiki_root()`), so any test reaching a DB write path lands in the real per-repo
DB unless something redirects it. That actually happened:
`backlinks.append_related_paper` calls `commit_page` → `db.upsert_page` whenever
it appends (`backlinks.py:81`), so `test_appends_when_absent` inserted a `papers`
row keyed on a pytest tmp dir into the real DB on every run. It then surfaced in
`researchwiki lint` as `db_drift.extra` and as a phantom zero-claim paper — the
suite manufacturing corpus defects.

The autouse `_isolate_state_db` fixture in `conftest.py` is the fix. These tests
pin it, because the failure is silent: everything passes, and the damage only
shows up later in a lint report.
"""

from __future__ import annotations

import os
from pathlib import Path

from researchwiki.db import connection


def test_db_path_env_override_is_set():
    assert "RESEARCHWIKI_DB_PATH" in os.environ, (
        "conftest's _isolate_state_db fixture is not active — the suite will "
        "write to the real per-repo state.db"
    )


def test_db_path_resolves_under_the_pytest_tmp_root(tmp_path):
    """Same `tmp_path` root the fixture used, so the DB is per-test."""
    resolved = connection.db_path()
    # tmp_path is .../pytest-of-<user>/pytest-<n>/<test-name><i>; the fixture's
    # path shares the pytest-<n> parent.
    assert tmp_path.parent in resolved.parents, (
        f"db_path() = {resolved} is outside the pytest tmp root {tmp_path.parent}"
    )


def test_writing_through_the_connection_does_not_touch_the_repo_db():
    """A real write lands in the isolated file, and that file is not the repo's."""
    conn = connection.get_connection()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS _probe (x INTEGER)")
        conn.execute("INSERT INTO _probe (x) VALUES (1)")
        conn.commit()
    finally:
        conn.close()

    isolated = Path(os.environ["RESEARCHWIKI_DB_PATH"])
    assert isolated.exists()
    # The repo's own DB lives under the XDG data base, never under a tmp dir.
    assert connection._XDG_BASE not in isolated.parents
