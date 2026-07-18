"""Per-repo state-DB path resolution + one-time legacy migration.

The DB is a rebuildable cache but the grounding layer (`researchwiki claims`,
check-grounding) reads it, so two wikis on one machine must not share one
`papers`/`claims` table. These pin: env override wins; the default path is
keyed per repo; and the legacy global DB is copy-seeded into an *established*
repo (has pages) but not a fresh clone.
"""

from researchwiki.db import connection


def _point_xdg(monkeypatch, base):
    """Repoint the module-level XDG constants at a tmp base (they're computed
    at import from the real HOME)."""
    monkeypatch.setattr(connection, "_XDG_BASE", base)
    monkeypatch.setattr(connection, "_LEGACY_DB", base / "state.db")


def _make_repo(root, *, established: bool):
    (root / "wiki").mkdir(parents=True, exist_ok=True)
    if established:
        (root / "wiki" / "ai").mkdir(parents=True, exist_ok=True)
        (root / "wiki" / "ai" / "x.md").write_text("---\ntype: paper\n---\n")


def test_env_override_wins(tmp_path, monkeypatch):
    custom = tmp_path / "somewhere" / "custom.db"
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(custom))
    assert connection.db_path() == custom
    assert custom.parent.is_dir()  # parent created


def test_default_path_is_per_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHWIKI_DB_PATH", raising=False)
    base = tmp_path / "xdg"
    _point_xdg(monkeypatch, base)
    repo = tmp_path / "myrepo"
    _make_repo(repo, established=False)
    monkeypatch.chdir(repo)

    path = connection.db_path()
    assert path.parent.parent == base / "repos"       # …/repos/<key>/state.db
    assert path.name == "state.db"
    assert path.parent.name.startswith("myrepo-")     # basename + hash


def test_distinct_repos_get_distinct_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHWIKI_DB_PATH", raising=False)
    _point_xdg(monkeypatch, tmp_path / "xdg")
    a, b = tmp_path / "repo-a", tmp_path / "repo-b"
    _make_repo(a, established=False)
    _make_repo(b, established=False)

    monkeypatch.chdir(a)
    pa = connection.db_path()
    monkeypatch.chdir(b)
    pb = connection.db_path()
    assert pa != pb


def test_legacy_migrated_into_established_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHWIKI_DB_PATH", raising=False)
    base = tmp_path / "xdg"
    _point_xdg(monkeypatch, base)
    base.mkdir(parents=True)
    (base / "state.db").write_bytes(b"LEGACYDATA")
    (base / "state.db-wal").write_bytes(b"WAL")

    repo = tmp_path / "established"
    _make_repo(repo, established=True)
    monkeypatch.chdir(repo)

    path = connection.db_path()
    assert path.read_bytes() == b"LEGACYDATA"                 # seeded
    assert path.with_name("state.db-wal").read_bytes() == b"WAL"  # sidecar too
    assert (base / "state.db").exists()                       # legacy left in place


def test_fresh_clone_does_not_inherit_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHWIKI_DB_PATH", raising=False)
    base = tmp_path / "xdg"
    _point_xdg(monkeypatch, base)
    base.mkdir(parents=True)
    (base / "state.db").write_bytes(b"LEGACYDATA")

    repo = tmp_path / "freshclone"
    _make_repo(repo, established=False)                        # no pages
    monkeypatch.chdir(repo)

    path = connection.db_path()
    assert not path.exists()  # not seeded — starts clean


def test_connection_uses_wal_and_busy_timeout(tmp_path):
    """New connections open the DB in WAL mode with a 5s busy timeout.

    Parallel `agent ingest` writes many rows concurrently (papers, claims,
    ingest_iterations). Under the default DELETE journal mode, any writer
    holds an exclusive lock that serializes readers — the "convoy effect"
    that hermes-state.db explicitly designs around. WAL lets readers and
    one writer run in parallel; busy_timeout gives contending writers a
    5s window to retry before raising SQLITE_BUSY. Sister DB
    `.claim-graph/edges.db` runs WAL for the same reason.
    """
    conn = connection.get_connection(tmp_path / "state.db")
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert journal_mode == "wal"
    assert busy_timeout == 5000


def test_existing_per_repo_db_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHWIKI_DB_PATH", raising=False)
    base = tmp_path / "xdg"
    _point_xdg(monkeypatch, base)
    base.mkdir(parents=True)
    (base / "state.db").write_bytes(b"LEGACYDATA")

    repo = tmp_path / "established"
    _make_repo(repo, established=True)
    monkeypatch.chdir(repo)

    # Pre-create this repo's DB — migration must not clobber it.
    first = connection.db_path()
    first.write_bytes(b"MINE")
    again = connection.db_path()
    assert again.read_bytes() == b"MINE"
