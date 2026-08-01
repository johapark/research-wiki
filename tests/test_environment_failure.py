"""EnvironmentFailure is what actually decides exit code 2 vs 3.

The exit-code sweep in a prior session aligned every literal `return 1/2/3`
with the contract, but left the 2-vs-3 *decision* unclassifiable at the point
it's made: 17 of 33 task `main()`s wrap nothing, so an unreachable state.db
inside them escaped as Python's own uncaught-exception path — code 3,
"internal bug" — when the truth was 2, "check your machine". The other 16
used `except Exception: return 2`, which fails the opposite way: a genuine
bug in the grader (KeyError, AttributeError, whatever) gets reported as an
environment error and its traceback thrown away.

`researchwiki.errors.EnvironmentFailure` fixes this by moving the
classification to where the failure actually happens (the DB/index boundary)
and carrying it up in the exception type. The CLI funnel catches that one
type and maps it to 2; everything else — including a real bug from inside a
`try` block that used to catch it — reaches the generic handler and gets 3
plus a traceback.

Hermetic throughout: sqlite3.connect and provider calls are monkeypatched,
no real DB or network touched.
"""

from __future__ import annotations

import sqlite3

import pytest

from researchwiki import __main__ as cli
from researchwiki.errors import EnvironmentFailure


# ---------- errors.py itself ----------

def test_environment_failure_is_a_runtime_error():
    # Must stay a RuntimeError so pre-existing `except RuntimeError` /
    # `except Exception` degrade-gracefully handlers are unaffected — this
    # type only changes the outcome for exceptions nobody catches.
    assert issubclass(EnvironmentFailure, RuntimeError)


# ---------- db/connection.py: get_connection wraps sqlite3/OSError ----------

def test_get_connection_wraps_sqlite_error(monkeypatch, tmp_path):
    from researchwiki.db import connection as dbc

    def boom(_path):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(dbc.sqlite3, "connect", boom)
    with pytest.raises(dbc.StateDBUnavailable):
        dbc.get_connection(tmp_path / "state.db")


def test_state_db_unavailable_is_an_environment_failure():
    from researchwiki.db.connection import StateDBUnavailable
    assert issubclass(StateDBUnavailable, EnvironmentFailure)


def test_get_connection_wraps_a_corrupt_db_file(monkeypatch, tmp_path):
    """"file is not a database" arrives as a bare `sqlite3.DatabaseError`, not
    an OperationalError — corruption is the machine's problem, so it must still
    map to 2."""
    from researchwiki.db import connection as dbc

    def boom(_path):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(dbc.sqlite3, "connect", boom)
    with pytest.raises(dbc.StateDBUnavailable):
        dbc.get_connection(tmp_path / "state.db")


@pytest.mark.parametrize("exc_name", [
    "ProgrammingError",   # wrong SQL / wrong binding count
    "InterfaceError",     # sqlite3 API misuse
    "IntegrityError",     # constraint violation
    "DataError",
    "InternalError",
])
def test_get_connection_does_not_disguise_a_sql_bug_as_environment(
        monkeypatch, tmp_path, exc_name):
    """Regression: `get_connection` first caught `sqlite3.Error`, the root of
    the whole hierarchy. That swept in ProgrammingError/InterfaceError/
    IntegrityError — all of which mean *our* SQL or API use is wrong — and
    reported them as StateDBUnavailable, i.e. exit 2 "check your machine" with
    no traceback. That's precisely the bug-as-environment-error inversion
    EnvironmentFailure was introduced to remove, so it must not reappear at the
    boundary that was supposed to fix it."""
    from researchwiki.db import connection as dbc

    exc_type = getattr(sqlite3, exc_name)

    def boom(_path):
        raise exc_type("simulated defect in our own SQL")

    monkeypatch.setattr(dbc.sqlite3, "connect", boom)
    with pytest.raises(exc_type):
        dbc.get_connection(tmp_path / "state.db")


def test_get_connection_classifies_a_data_dir_failure(monkeypatch):
    """Regression: `db_path()` used to be evaluated *before* the guard, so the
    two most likely failures on a fresh machine — `mkdir` denied under
    ~/.local/share, or the disk filling during the legacy-DB `copy2` — escaped
    as a bare OSError and reported exit 3, "internal bug"."""
    from researchwiki.db import connection as dbc

    def boom():
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(dbc, "db_path", boom)
    with pytest.raises(dbc.StateDBUnavailable):
        dbc.get_connection()


class _RecordingConn:
    """Minimal stand-in for a sqlite3.Connection that records `close()`.

    A real connection can't be used here — `sqlite3.Connection.close` is a
    read-only C attribute, so it can't be wrapped to observe the call. This
    fake implements only the surface `get_connection` touches before
    `init_schema`: a settable `row_factory` and `execute` for the three PRAGMAs.
    """

    def __init__(self):
        self.row_factory = None
        self.closed = False

    def execute(self, *_a, **_kw):
        return self

    def close(self):
        self.closed = True


@pytest.mark.parametrize("raised,expected", [
    # The wrap arm (environment) and the re-raise arm (our bug) both need the
    # same cleanup — a leak on the bug path would be just as real.
    (sqlite3.OperationalError("disk I/O error"), "StateDBUnavailable"),
    (sqlite3.ProgrammingError("bad binding count"), "ProgrammingError"),
])
def test_get_connection_closes_the_connection_on_a_failed_bootstrap(
        monkeypatch, tmp_path, raised, expected):
    """The connection is opened before `init_schema` runs, so a bootstrap
    failure used to leave it open holding a WAL lock that the sibling workers
    of a 4-way `agent ingest` batch then contend on."""
    from researchwiki.db import connection as dbc

    fake = _RecordingConn()
    monkeypatch.setattr(dbc.sqlite3, "connect", lambda *a, **kw: fake)

    def boom(_conn):
        raise raised
    monkeypatch.setattr(dbc, "init_schema", boom)

    expected_type = (dbc.StateDBUnavailable if expected == "StateDBUnavailable"
                     else sqlite3.ProgrammingError)
    with pytest.raises(expected_type):
        dbc.get_connection(tmp_path / "state.db")
    assert fake.closed, (
        f"connection was leaked when init_schema raised {type(raised).__name__}"
    )


# ---------- grade/grounding.py + index/types.py: existing types re-based ----------

def test_claim_db_unavailable_is_an_environment_failure():
    from researchwiki.grade.grounding import ClaimDBUnavailable
    assert issubclass(ClaimDBUnavailable, EnvironmentFailure)


def test_search_backend_unavailable_is_an_environment_failure():
    from researchwiki.index.types import SearchBackendUnavailable
    assert issubclass(SearchBackendUnavailable, EnvironmentFailure)


# ---------- the funnel: EnvironmentFailure -> 2, uncaught elsewhere -> 3 ----------

@pytest.fixture
def fake_task(monkeypatch):
    import sys
    import types

    def install(fn):
        mod = types.ModuleType("researchwiki.tasks.faketask")
        mod.__doc__ = "Synthetic task for environment-failure tests."
        mod.main = fn
        monkeypatch.setitem(sys.modules, "researchwiki.tasks.faketask", mod)
        monkeypatch.setattr(cli, "_discover_tasks", lambda: {"faketask": "faketask"})
        return mod
    return install


def test_uncaught_environment_failure_returns_2(fake_task, capsys):
    def task_main(argv):
        raise EnvironmentFailure("state.db is locked")
    fake_task(task_main)

    assert cli.main(["faketask"]) == 2
    err = capsys.readouterr().err
    assert "state.db is locked" in err
    # No traceback — this is a diagnosable condition, not a bug report.
    assert "Traceback" not in err


def test_uncaught_environment_failure_subclass_returns_2(fake_task, capsys):
    # A concrete subclass (StateDBUnavailable, ClaimDBUnavailable,
    # SearchBackendUnavailable, or a future one) must be caught by the same
    # `except EnvironmentFailure` — this is the whole point of the hierarchy.
    class _CustomEnvFailure(EnvironmentFailure):
        pass

    def task_main(argv):
        raise _CustomEnvFailure("index not built")
    fake_task(task_main)

    assert cli.main(["faketask"]) == 2


def test_generic_runtimeerror_still_returns_3_not_2(fake_task, capsys):
    # A plain RuntimeError (not EnvironmentFailure) must NOT be swept into 2 —
    # otherwise the whole point of typing the exception is lost.
    def task_main(argv):
        raise RuntimeError("unrelated bug")
    fake_task(task_main)

    assert cli.main(["faketask"]) == 3
    assert "Traceback" in capsys.readouterr().err


# ---------- task modules no longer catch-all into 2 ----------

def test_grade_synthesis_lets_a_real_bug_propagate(tmp_path, monkeypatch):
    """Regression: `_grade_synthesis.main` used to wrap `grade_synthesis` in
    `except Exception: return 2`, reporting a bug in the grader as an
    environment error. It must now propagate so the CLI funnel can tell a
    ValueError (bug, code 3) apart from an EnvironmentFailure (code 2)."""
    from researchwiki.tasks import _grade_synthesis as mod

    page = tmp_path / "page.md"
    page.write_text("---\ntitle: x\n---\nbody\n")

    def boom(*a, **kw):
        raise ValueError("not an environment failure")
    monkeypatch.setattr(mod, "grade_synthesis", boom)

    with pytest.raises(ValueError):
        mod.main([str(page)])


def test_claim_overlap_lets_a_real_bug_propagate(monkeypatch):
    """Regression: `claim_overlap.main` used to wrap `run` in
    `except Exception as e: return 2`, which also swallowed the traceback
    for genuine bugs (only LookupError — "no such stem" — is real user
    error and stays a returned 1)."""
    from researchwiki.tasks import claim_overlap as mod

    def boom(*a, **kw):
        raise ValueError("not an environment failure")
    monkeypatch.setattr(mod, "run", boom)

    with pytest.raises(ValueError):
        mod.main(["some-stem"])


def test_claim_overlap_still_returns_1_for_unknown_stem(monkeypatch):
    # The one exception kept as a return: LookupError is genuinely bad input.
    from researchwiki.tasks import claim_overlap as mod

    def boom(*a, **kw):
        raise LookupError("no wiki page for stem")
    monkeypatch.setattr(mod, "run", boom)

    assert mod.main(["no-such-stem"]) == 1


def test_grade_paper_lets_a_real_bug_propagate(monkeypatch):
    from researchwiki.tasks import grade as mod

    def boom(*a, **kw):
        raise ValueError("not an environment failure")
    monkeypatch.setattr(mod, "grade_page", boom)

    with pytest.raises(ValueError):
        mod.main(["paper", "some-stem"])


def test_check_coverage_lets_a_probe_bug_propagate(tmp_path, monkeypatch):
    from researchwiki.tasks import check_coverage as mod
    import researchwiki.search as search_mod

    page = tmp_path / "page.md"
    page.write_text('---\ntopic_seed: "x"\n---\nbody\n')

    class _FakeBackend:
        def query(self, *a, **kw):
            raise ValueError("not an environment failure")

    monkeypatch.setattr(search_mod, "get_default_backend", lambda: _FakeBackend())

    with pytest.raises(ValueError):
        mod.main([str(page)])


def test_check_coverage_still_returns_2_for_unbuilt_index(tmp_path, monkeypatch):
    from researchwiki.tasks import check_coverage as mod
    import researchwiki.search as search_mod
    from researchwiki.search import SearchBackendUnavailable

    page = tmp_path / "page.md"
    page.write_text('---\ntopic_seed: "x"\n---\nbody\n')

    class _FakeBackend:
        def query(self, *a, **kw):
            raise SearchBackendUnavailable("index not built")

    monkeypatch.setattr(search_mod, "get_default_backend", lambda: _FakeBackend())

    assert mod.main([str(page)]) == 2


def test_claim_graph_reconcile_import_guard_only_covers_the_import(monkeypatch):
    """Regression: the old message said "state.db unreachable" but the guard
    only ever wrapped the *import* statement, not the `get_connection()` call
    a few lines below — so the message described a failure mode the code
    couldn't actually detect. A bug in `get_connection()` itself must now
    propagate (StateDBUnavailable, raised there, is caught by the funnel)."""
    from researchwiki.tasks import claim_graph as mod
    from researchwiki.db.connection import StateDBUnavailable

    class _FakeEdges:
        def close(self):
            pass

    monkeypatch.setattr(mod, "open_edges_db", lambda: _FakeEdges())

    def boom():
        raise StateDBUnavailable("simulated: cannot open state.db")

    import researchwiki.db.connection as dbc
    monkeypatch.setattr(dbc, "get_connection", boom)

    with pytest.raises(StateDBUnavailable):
        mod._run_reconcile(as_json=False)


def test_claim_graph_reconcile_closes_edges_when_the_db_fails(monkeypatch):
    """The `try/finally` that closes the edges DB used to start one line *below*
    `get_connection()`, so a DB failure left the edges DB open on the way out.
    Making StateDBUnavailable a clean, reachable failure made that path more
    likely, not less."""
    from researchwiki.tasks import claim_graph as mod
    from researchwiki.db.connection import StateDBUnavailable

    closed: list[bool] = []

    class _FakeEdges:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(mod, "open_edges_db", lambda: _FakeEdges())

    def boom():
        raise StateDBUnavailable("simulated: cannot open state.db")

    import researchwiki.db.connection as dbc
    monkeypatch.setattr(dbc, "get_connection", boom)

    with pytest.raises(StateDBUnavailable):
        mod._run_reconcile(as_json=False)
    assert closed, "edges DB was leaked when get_connection failed"
