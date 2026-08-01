"""SQLite-backed cache of typed claim-graph edges.

Backing file: `.claim-graph/edges.db` (gitignored; rebuildable from source
judges). Deliberately separate from `state.db` because edge verdicts are
LLM-authored — putting them in state.db would violate the "no LLM content"
invariant declared at researchwiki/db/schema.sql:1-18.

Edge identity is the pair `(src_stem+src_slug, tgt_stem+tgt_slug, relation)`.
Neither slug alone is a global identifier (see researchwiki/claim_graph/slug.py);
identity is always scoped by paper stem. For the `instantiates` relation,
`tgt_stem` is the literal string "concepts" and `tgt_slug` is the concept-page
slug (e.g. "prime-editing").

Open with `PRAGMA journal_mode = WAL` so parallel ingests can write
concurrently without corrupting the store.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import claim_graph_dir


_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS edges (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    src_stem            TEXT NOT NULL,
    src_slug            TEXT NOT NULL,
    tgt_stem            TEXT NOT NULL,
    tgt_slug            TEXT NOT NULL,
    relation            TEXT NOT NULL,   -- contradicts | corroborates | measures_same | refines | builds_on | instantiates
    directed            INTEGER NOT NULL DEFAULT 0,   -- 0/1
    confidence          REAL,
    rationale           TEXT,             -- one-line judge rationale (advisory)
    judge_phase         TEXT,
    judge_model         TEXT,
    slug_scheme_version INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'candidate',  -- candidate | confirmed | promoted | rejected | stale
    created_at          INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL,
    UNIQUE(src_stem, src_slug, tgt_stem, tgt_slug, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_src        ON edges(src_stem, src_slug);
CREATE INDEX IF NOT EXISTS idx_edges_tgt        ON edges(tgt_stem, tgt_slug);
CREATE INDEX IF NOT EXISTS idx_edges_relation   ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_status     ON edges(status);
"""


# Valid statuses in the lifecycle. Enforced at the Python layer (SQLite
# doesn't do CHECK-constraint-on-column enforcement well across ALTER).
STATUSES = {"candidate", "confirmed", "promoted", "rejected", "stale"}

# Valid relations. `instantiates` is written by the concepts detector with
# tgt_stem="concepts"; the others come from the judges.
RELATIONS = {
    "contradicts", "corroborates", "measures_same",
    "refines", "builds_on", "instantiates",
}


@dataclass
class Edge:
    src_stem: str
    src_slug: str
    tgt_stem: str
    tgt_slug: str
    relation: str
    slug_scheme_version: int
    directed: bool = False
    confidence: float | None = None
    rationale: str = ""
    judge_phase: str = ""
    judge_model: str = ""
    status: str = "candidate"
    id: int | None = None
    created_at: int | None = None
    updated_at: int | None = None

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "src_stem": self.src_stem,
            "src_slug": self.src_slug,
            "tgt_stem": self.tgt_stem,
            "tgt_slug": self.tgt_slug,
            "relation": self.relation,
            "directed": bool(self.directed),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "judge_phase": self.judge_phase,
            "judge_model": self.judge_model,
            "status": self.status,
            "slug_scheme_version": self.slug_scheme_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def edges_db_path() -> Path:
    """Resolve `.claim-graph/edges.db`, creating the parent dir if needed."""
    d = claim_graph_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "edges.db"


def open_edges_db(path: Path | None = None) -> sqlite3.Connection:
    """Open the edge cache. Initializes schema idempotently (WAL mode)."""
    target = path if path is not None else edges_db_path()
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _row_to_edge(r: sqlite3.Row) -> Edge:
    return Edge(
        id=r["id"],
        src_stem=r["src_stem"],
        src_slug=r["src_slug"],
        tgt_stem=r["tgt_stem"],
        tgt_slug=r["tgt_slug"],
        relation=r["relation"],
        directed=bool(r["directed"]),
        confidence=r["confidence"],
        rationale=r["rationale"] or "",
        judge_phase=r["judge_phase"] or "",
        judge_model=r["judge_model"] or "",
        slug_scheme_version=r["slug_scheme_version"],
        status=r["status"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def upsert_edge(
    conn: sqlite3.Connection, edge: Edge,
    *, skip_if_rejected: bool = True,
) -> tuple[str, int]:
    """Insert or refresh a single edge. Returns ("inserted"|"updated"|"skipped", id).

    `skip_if_rejected`: when True (default) and an existing row with the same
    identity is `rejected`, we DON'T overwrite it — that would defeat the
    "human said no; don't re-surface it" invariant from §3.3. To reopen a
    rejected edge, delete the row explicitly with `delete_edge()`.

    Identity is `(src_stem, src_slug, tgt_stem, tgt_slug, relation)`.
    """
    if edge.relation not in RELATIONS:
        raise ValueError(f"unknown relation: {edge.relation!r}")
    if edge.status not in STATUSES:
        raise ValueError(f"unknown status: {edge.status!r}")

    now = int(time.time())
    existing = conn.execute(
        "SELECT id, status FROM edges "
        " WHERE src_stem=? AND src_slug=? AND tgt_stem=? AND tgt_slug=? AND relation=?",
        (edge.src_stem, edge.src_slug, edge.tgt_stem, edge.tgt_slug, edge.relation),
    ).fetchone()

    if existing is not None:
        if skip_if_rejected and existing["status"] == "rejected":
            return "skipped", existing["id"]
        conn.execute(
            "UPDATE edges SET "
            "  directed=?, confidence=?, rationale=?, judge_phase=?, judge_model=?, "
            "  slug_scheme_version=?, status=?, updated_at=? "
            " WHERE id=?",
            (
                int(edge.directed), edge.confidence, edge.rationale,
                edge.judge_phase, edge.judge_model, edge.slug_scheme_version,
                edge.status, now, existing["id"],
            ),
        )
        return "updated", existing["id"]

    cur = conn.execute(
        "INSERT INTO edges ("
        "  src_stem, src_slug, tgt_stem, tgt_slug, relation, directed, "
        "  confidence, rationale, judge_phase, judge_model, slug_scheme_version, "
        "  status, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            edge.src_stem, edge.src_slug, edge.tgt_stem, edge.tgt_slug,
            edge.relation, int(edge.directed), edge.confidence, edge.rationale,
            edge.judge_phase, edge.judge_model, edge.slug_scheme_version,
            edge.status, now, now,
        ),
    )
    return "inserted", cur.lastrowid


def set_status(conn: sqlite3.Connection, edge_id: int, new_status: str) -> bool:
    """Transition an edge's status. Returns True iff a row was actually changed."""
    if new_status not in STATUSES:
        raise ValueError(f"unknown status: {new_status!r}")
    now = int(time.time())
    cur = conn.execute(
        "UPDATE edges SET status=?, updated_at=? WHERE id=?",
        (new_status, now, edge_id),
    )
    return cur.rowcount > 0


def delete_edge(conn: sqlite3.Connection, edge_id: int) -> bool:
    """Hard-delete an edge row (used to reopen a `rejected` verdict)."""
    cur = conn.execute("DELETE FROM edges WHERE id=?", (edge_id,))
    return cur.rowcount > 0


def query(
    conn: sqlite3.Connection,
    *,
    src_stem: str | None = None,
    src_slug: str | None = None,
    tgt_stem: str | None = None,
    tgt_slug: str | None = None,
    stem: str | None = None,   # matches either src_stem or tgt_stem
    relation: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[Edge]:
    """Filter edges by any subset of the query dims. Returns most-recent-first."""
    where: list[str] = []
    params: list = []
    if src_stem is not None:
        where.append("src_stem = ?"); params.append(src_stem)
    if src_slug is not None:
        where.append("src_slug = ?"); params.append(src_slug)
    if tgt_stem is not None:
        where.append("tgt_stem = ?"); params.append(tgt_stem)
    if tgt_slug is not None:
        where.append("tgt_slug = ?"); params.append(tgt_slug)
    if stem is not None:
        where.append("(src_stem = ? OR tgt_stem = ?)")
        params.extend([stem, stem])
    if relation is not None:
        where.append("relation = ?"); params.append(relation)
    if status is not None:
        where.append("status = ?"); params.append(status)

    sql = "SELECT * FROM edges"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC, id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [_row_to_edge(r) for r in conn.execute(sql, params)]


@dataclass
class ReconcileStats:
    scanned: int = 0
    marked_stale_missing: int = 0    # endpoint slug no longer resolves in state.db
    marked_stale_version: int = 0    # slug_scheme_version drift
    active: int = 0                  # rows in non-terminal statuses after reconcile
    by_relation: dict[str, int] = field(default_factory=dict)


def reconcile(edges_conn: sqlite3.Connection, state_db_conn) -> ReconcileStats:
    """Mark edges stale when an endpoint slug no longer resolves in state.db,
    or when the slug scheme version has drifted.

    `state_db_conn` is a live connection to state.db (from
    researchwiki.db.connection.get_connection). We don't own it here — caller
    passes it in so this module has no reverse dependency on the DB package.

    Idempotent — marking an already-stale row stale is a no-op.
    """
    stats = ReconcileStats()

    # Snapshot current slug scheme version from state.db.
    row = state_db_conn.execute(
        "SELECT value FROM schema_meta WHERE key='slug_scheme_version'"
    ).fetchone()
    current_version = int(row[0]) if row and row[0] is not None else None

    # Snapshot every live (stem, slug) pair. `instantiates` edges point to
    # concept-page slugs (tgt_stem="concepts"), which live in papers/, not
    # claims/ — so allow that endpoint to resolve against papers.stem too.
    live_claims = {
        (r["paper_stem"], r["claim_slug"])
        for r in state_db_conn.execute(
            "SELECT paper_stem, claim_slug FROM claims WHERE claim_slug IS NOT NULL"
        )
    }
    live_concepts = {
        r["stem"] for r in state_db_conn.execute(
            "SELECT stem FROM papers WHERE page_type = 'concept'"
        )
    }

    now = int(time.time())
    for row in edges_conn.execute(
        "SELECT id, src_stem, src_slug, tgt_stem, tgt_slug, relation, "
        "       status, slug_scheme_version FROM edges"
    ).fetchall():
        stats.scanned += 1
        stats.by_relation[row["relation"]] = stats.by_relation.get(row["relation"], 0) + 1

        if row["status"] == "stale":
            continue

        # Endpoint resolution.
        src_ok = (row["src_stem"], row["src_slug"]) in live_claims
        if row["tgt_stem"] == "concepts":
            tgt_ok = row["tgt_slug"] in live_concepts
        else:
            tgt_ok = (row["tgt_stem"], row["tgt_slug"]) in live_claims

        if not (src_ok and tgt_ok):
            edges_conn.execute(
                "UPDATE edges SET status='stale', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            stats.marked_stale_missing += 1
            continue

        if current_version is not None and row["slug_scheme_version"] != current_version:
            edges_conn.execute(
                "UPDATE edges SET status='stale', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            stats.marked_stale_version += 1
            continue

        stats.active += 1

    edges_conn.commit()
    return stats


def counts_by_relation(conn: sqlite3.Connection) -> dict[str, int]:
    """One-liner for `status` / logging: {relation: total_rows}."""
    return {
        r["relation"]: r["n"]
        for r in conn.execute(
            "SELECT relation, COUNT(*) AS n FROM edges GROUP BY relation"
        )
    }
