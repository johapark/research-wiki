"""SQLite connection management + schema bootstrap.

DB lives outside the wiki repo so OneDrive/iCloud sync never sees the SQLite
WAL files (the EPERM issue we hit during the grader spike). Resolution order:

    $RESEARCHWIKI_DB_PATH                                  (env override — wins)
    ~/.local/share/researchwiki/repos/<repo>-<hash>/state.db   (per-repo default)

The DB is **per-repo**, keyed by the wiki root, so multiple wikis on one
machine (or a fresh clone beside an established repo) don't share one `papers`/
`claims` table — which would make `researchwiki claims` ground against a
foreign repo's papers. The DB is a derived cache (rebuildable from markdown via
`db rebuild`); only `ingest_iterations` cost history is non-reconstructable, so
the first run in an **established** repo copy-migrates the pre-existing global
`state.db` into that repo's per-repo path (see `_maybe_migrate_legacy`).

The schema is created on first use; subsequent connections are no-ops.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from pathlib import Path

from ..errors import EnvironmentFailure


class StateDBUnavailable(EnvironmentFailure):
    """state.db couldn't be opened, or its schema bootstrap failed.

    `get_connection` has ~70 call sites and almost none of them guard it, so
    before this existed a locked DB, a full disk, or an unreadable XDG data dir
    surfaced as a bare `sqlite3.OperationalError` — which the CLI funnel could
    only classify as exit 3, "internal bug". This makes it 2, "look at the
    machine", which is what it always was.

    Scope is deliberately narrow: only the sqlite3 errors that describe the
    *machine* map here. `sqlite3.Error` is the root of the whole hierarchy and
    includes `ProgrammingError` (wrong SQL / wrong binding count),
    `InterfaceError` (API misuse), `IntegrityError` and `DataError` — all of
    which mean our code is wrong, not the disk. Catching those here would
    report a bug as an environment error, which is the exact inversion
    `EnvironmentFailure` exists to remove; `_BUG_SHAPED_SQLITE_ERRORS` keeps
    them out.
    """


# sqlite3 exceptions that indicate a defect in *our* SQL or API use rather than
# a problem with the machine. Re-raised unwrapped by `get_connection` so they
# reach the CLI funnel's generic handler (exit 3 + traceback), which is what
# actually helps diagnose them. Note these are all `sqlite3.DatabaseError`
# subclasses except InterfaceError, so they must be caught *before* the broader
# DatabaseError arm.
_BUG_SHAPED_SQLITE_ERRORS = (
    sqlite3.ProgrammingError,
    sqlite3.InterfaceError,
    sqlite3.IntegrityError,
    sqlite3.DataError,
    sqlite3.InternalError,
)


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_XDG_BASE = Path.home() / ".local" / "share" / "researchwiki"
# Pre-per-repo location; migrated from on first run in an established repo.
_LEGACY_DB = _XDG_BASE / "state.db"
# Sidecar files SQLite keeps alongside the main DB in WAL mode.
_DB_SIDECARS = ("-wal", "-shm")


def _repo_key() -> str:
    """Stable per-repo directory name: the wiki root's basename plus a short
    hash of its absolute path (disambiguates same-named repos in different
    locations)."""
    from ..paths import wiki_root
    root = wiki_root().resolve()
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    return f"{root.name}-{digest}"


def db_path() -> Path:
    """Return the resolved DB path, creating the parent dir if needed.

    `$RESEARCHWIKI_DB_PATH` overrides everything (single explicit DB). Otherwise
    the path is scoped to the current wiki repo, migrating the legacy global DB
    on first use for an established repo."""
    override = os.environ.get("RESEARCHWIKI_DB_PATH")
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path = _XDG_BASE / "repos" / _repo_key() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_migrate_legacy(path)
    return path


def _maybe_migrate_legacy(target: Path) -> None:
    """One-time, non-destructive seed of a new per-repo DB from the pre-per-repo
    global `state.db`.

    Runs only when the per-repo DB doesn't exist yet, the legacy global DB does,
    and the current wiki is **established** (has ≥1 page) — so a fresh/throwaway
    clone starts clean instead of inheriting another repo's papers. Copies (not
    moves) the DB + its WAL sidecars, leaving the legacy file in place; it can
    be deleted once the migration is confirmed."""
    if target.exists() or not _LEGACY_DB.exists():
        return
    try:
        from ..paths import wiki_dir
        established = any(wiki_dir().glob("*/*.md"))
    except Exception:
        established = False
    if not established:
        return
    for suffix in ("", *_DB_SIDECARS):
        src = _LEGACY_DB.with_name(_LEGACY_DB.name + suffix)
        if src.exists():
            shutil.copy2(src, target.with_name(target.name + suffix))
    try:
        from ..log import log
        log(f"seeded per-repo state.db from legacy {_LEGACY_DB} → {target} "
            f"(legacy left in place; safe to delete once verified)", tag="db")
    except Exception:
        pass


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection to the state DB. Initializes the schema on first call.

    WAL is enabled so parallel `agent ingest` (the default when `inbox/` has
    ≥2 PDFs) doesn't serialize on the DELETE-journal writer-exclusive lock.
    Persists in the DB file header, so setting it on every connect is
    idempotent — SQLite no-ops when the file is already WAL. Sister DB
    `.claim-graph/edges.db` runs WAL for the same reason
    (researchwiki/claim_graph/edges.py:14-16). Sidecar files (`-wal`,
    `-shm`) are why `db_path()` lives outside the wiki repo — see the
    module docstring on the OneDrive EPERM issue.

    `busy_timeout` matches Python's default `sqlite3.connect(timeout=…)`;
    set it explicitly here so the intent survives if a caller ever opens
    the DB with an explicit `timeout=0`.
    """
    # `db_path()` is inside the guard, not before it: it does `mkdir(parents=True)`
    # under ~/.local/share and may `shutil.copy2` the legacy DB, so "can't create
    # the data dir" and "disk filled during migration" originate *here*. Those are
    # the most likely failures on a fresh machine and the most clearly
    # environmental things in this module — leaving them outside meant the guard
    # covered the easy cases and missed the motivating ones.
    target: Path | None = None
    conn: sqlite3.Connection | None = None
    try:
        target = Path(path) if path is not None else db_path()
        conn = sqlite3.connect(str(target))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
    except _BUG_SHAPED_SQLITE_ERRORS:
        # Our SQL or our API use is wrong. Let it reach the funnel as exit 3
        # with a traceback rather than mislabelling it an environment error.
        _close_quietly(conn)
        raise
    except (sqlite3.DatabaseError, OSError) as e:
        # DatabaseError covers OperationalError (locked / unopenable / disk full
        # / readonly) and a bare DatabaseError ("file is not a database" —
        # corruption); OSError covers the db_path() filesystem work above.
        # Bug-shaped DatabaseError subclasses were already re-raised.
        _close_quietly(conn)
        where = f" at {target}" if target is not None else ""
        raise StateDBUnavailable(f"cannot open state.db{where}: {e}") from e
    return conn


def _close_quietly(conn: sqlite3.Connection | None) -> None:
    """Best-effort close for a half-initialized connection.

    `get_connection` opens the connection before running the schema bootstrap,
    so a failure in `init_schema` used to leave it open. CPython's refcounting
    collects it soon after, but "soon" is long enough to hold a WAL lock that
    the sibling workers of a 4-way `agent ingest` batch then contend on.
    Swallows its own errors — we're already unwinding a failure and the
    original exception is the one worth propagating.
    """
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def init_schema(conn: sqlite3.Connection) -> None:
    """Execute the schema DDL (idempotent — `IF NOT EXISTS` everywhere) and
    apply lightweight forward migrations.

    `IF NOT EXISTS` only protects the *table*, not its columns — once a table
    exists, ALTER is the only way to introduce a new column. We introspect
    `PRAGMA table_info` and apply each missing column once. Cheap and adds
    no per-call overhead on warm DBs.
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate(conn)
    conn.commit()


def _safe_add_column(conn: sqlite3.Connection, ddl: str) -> None:
    """Run an `ALTER TABLE … ADD COLUMN`, tolerating a concurrent add.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, and the `PRAGMA table_info`
    guard below is an unlocked read: two concurrent batch-ingest connections
    can both observe a column absent and both issue the ALTER, so the loser
    hits `duplicate column name`. That specific error means a peer already
    added it — swallow it; anything else propagates.
    """
    try:
        conn.execute(ddl)
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for tables created in earlier schema versions."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(claims)")}
    if "embed_model" not in cols:
        _safe_add_column(conn, "ALTER TABLE claims ADD COLUMN embed_model TEXT")
    if "supporting_text" not in cols:
        _safe_add_column(conn, "ALTER TABLE claims ADD COLUMN supporting_text TEXT")
    if "supporting_provenance" not in cols:
        _safe_add_column(
            conn, "ALTER TABLE claims ADD COLUMN supporting_provenance TEXT"
        )
    if "claim_slug" not in cols:
        # New rows get their slug computed at upsert time; existing rows are
        # backfilled on the next `db rebuild`.
        _safe_add_column(conn, "ALTER TABLE claims ADD COLUMN claim_slug TEXT")
    # Index creation runs unconditionally (IF NOT EXISTS is a no-op when it
    # already exists) — this is also the sole owner of idx_claims_slug, since
    # schema.sql can't create it (executescript runs before the ADD COLUMN).
    # Note: SQLite can't ALTER-ADD a UNIQUE constraint, so uniqueness on
    # (paper_stem, claim_slug) is enforced only by the CREATE TABLE path
    # (fresh DBs). For migrated DBs, `_upsert_claims` handles collisions
    # by appending -{position} — same code path either way.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_claims_slug ON claims(paper_stem, claim_slug)"
    )
    iter_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(ingest_iterations)")
    }
    # Direct `_migrate` callers in recovery/tests may expose only the claims
    # table. Normal bootstrap creates ingest_iterations via schema.sql first;
    # an empty PRAGMA result means the table is absent, not that every column
    # needs adding to a nonexistent table.
    if iter_cols:
        if "duration_ms" not in iter_cols:
            _safe_add_column(
                conn, "ALTER TABLE ingest_iterations ADD COLUMN duration_ms INTEGER"
            )
        if "gate_metrics" not in iter_cols:
            _safe_add_column(
                conn, "ALTER TABLE ingest_iterations ADD COLUMN gate_metrics TEXT"
            )
    _install_claims_fts(conn)


def _install_claims_fts(conn: sqlite3.Connection) -> None:
    """Create the `claims_fts` FTS5 virtual table and keep-in-sync triggers.

    External-content mode (`content=claims`, `content_rowid=id`) means FTS
    doesn't duplicate the `text` column; it stores only its inverted index
    keyed on the source rowid, and reads through to `claims` on query.
    Triggers propagate insert/delete/update so rebuild.py doesn't have to
    know FTS exists.

    Silent no-op only when the SQLite build lacks FTS5 (`OperationalError:
    no such module: fts5`) — any other OperationalError propagates so a
    disk-full or permission failure doesn't leave the DB in a half-installed
    state with no diagnostic. `claim_lookup` transparently falls back to
    LIKE when the table is genuinely absent.

    Backfill: after creating the virtual table on a migrated DB with
    existing rows, the FTS index is empty; the `('rebuild')` command
    populates it from the content table. External-content FTS5 forwards
    `SELECT COUNT(*) FROM claims_fts` to the underlying `claims` table, so
    row-count comparisons can't detect an empty *index* — we track state
    via a `claims_fts_backfilled` row in `schema_meta`, but also re-check
    that the FTS shadow tables (`claims_fts_data`, `claims_fts_idx`, …)
    still exist. If any went missing (external DROP, corruption), we
    re-issue the `('rebuild')` command.
    """
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5("
            "  text,"
            "  content=claims,"
            "  content_rowid=id,"
            "  tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
    except sqlite3.OperationalError as e:
        if "no such module" in str(e).lower() and "fts5" in str(e).lower():
            # SQLite build without FTS5. claim_lookup detects the missing
            # table at query time and uses the LIKE fallback.
            return
        raise
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN"
        "  INSERT INTO claims_fts(rowid, text) VALUES (new.id, new.text);"
        "END"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN"
        "  INSERT INTO claims_fts(claims_fts, rowid, text) "
        "  VALUES('delete', old.id, old.text);"
        "END"
    )
    # `AFTER UPDATE OF text` limits trigger firing to rows whose indexed
    # column actually changed. Without this, every `_upsert_claims` slug
    # backfill and every grader UPDATE re-tokenizes the row. A previous
    # release created this trigger without the `OF text` clause; drop the
    # broad version if present so the narrower one takes over.
    au_ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='claims_au'"
    ).fetchone()
    if au_ddl and "UPDATE OF text" not in (au_ddl[0] or ""):
        # IF EXISTS: two concurrent connections can both read the old-form DDL
        # (unlocked read) and both try to drop it; without IF EXISTS the loser
        # crashes with "no such trigger" on the first post-upgrade batch.
        conn.execute("DROP TRIGGER IF EXISTS claims_au")
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE OF text ON claims BEGIN"
        "  INSERT INTO claims_fts(claims_fts, rowid, text) "
        "  VALUES('delete', old.id, old.text);"
        "  INSERT INTO claims_fts(rowid, text) VALUES (new.id, new.text);"
        "END"
    )
    _maybe_backfill_claims_fts(conn)


# Shadow tables SQLite creates for an external-content FTS5 index. We inspect
# sqlite_master for these to detect the "index dropped externally" case that a
# naive sentinel-only check would miss. Only the option-invariant core is
# listed: `claims_fts_data`, `claims_fts_idx`, and `claims_fts_config` exist
# for every FTS5 external-content table. (`claims_fts_docsize` is dropped by
# `columnsize=0`, and `detail=none` drops it too — including it here would make
# the count never match under those options and force a full rebuild on every
# connect. If the CREATE VIRTUAL TABLE options in `_install_claims_fts` change,
# revisit this list deliberately.)
_FTS_SHADOW_TABLES = ("claims_fts_data", "claims_fts_idx", "claims_fts_config")


def _fts_ready(conn: sqlite3.Connection) -> bool:
    """True when the FTS index has been backfilled and its shadow tables are
    intact — the "skip the rebuild" predicate, used identically on both sides
    of the `BEGIN IMMEDIATE` in `_maybe_backfill_claims_fts`."""
    sentinel = conn.execute(
        "SELECT value FROM schema_meta WHERE key='claims_fts_backfilled'"
    ).fetchone()
    shadow_present = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
        f"({','.join('?' * len(_FTS_SHADOW_TABLES))})",
        _FTS_SHADOW_TABLES,
    ).fetchone()[0]
    return sentinel is not None and shadow_present == len(_FTS_SHADOW_TABLES)


def _maybe_backfill_claims_fts(conn: sqlite3.Connection) -> None:
    """Populate the FTS index from `claims` when needed.

    Wraps the check-and-rebuild in `BEGIN IMMEDIATE` so concurrent batch-
    ingest workers can't both observe the sentinel absent and both issue
    a full rebuild. If a peer already claimed the write lock, we back off
    (busy_timeout handles the wait); by the time we retry the sentinel is
    set and we no-op.
    """
    if _fts_ready(conn):
        return

    # Serialize the rebuild across concurrent connections. IMMEDIATE takes
    # the reserved lock, so a racing worker waits (busy_timeout=5s) rather
    # than duplicating the work.
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as e:
        # Only a busy lock means "a peer is doing the rebuild; skip". Their
        # sentinel write makes our next connect no-op. Anything else (IOERR,
        # CORRUPT) is a real fault — re-raise it rather than silently leaving
        # the FTS index unbackfilled with no diagnostic.
        if "locked" in str(e).lower() or getattr(e, "sqlite_errorcode", None) == getattr(
            sqlite3, "SQLITE_BUSY", -1
        ):
            return
        raise
    try:
        # Re-check under lock — the peer may have finished between our
        # readiness check and the BEGIN.
        if _fts_ready(conn):
            conn.execute("COMMIT")
            return
        n_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        if n_claims:
            conn.execute("INSERT INTO claims_fts(claims_fts) VALUES('rebuild')")
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) "
            "VALUES ('claims_fts_backfilled', '1')"
        )
        conn.execute("COMMIT")
    except Exception:
        # Don't let a failed ROLLBACK (e.g. the failure was in COMMIT and the
        # transaction already ended) mask the original exception.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
