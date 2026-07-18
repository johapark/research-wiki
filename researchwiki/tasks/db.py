"""Manage the structured state DB — rebuild from markdown, verify drift, run queries.

The DB is a derived index over the markdown wiki + papers + caches. Markdown
is canonical; this CLI rebuilds the index, detects drift, and runs read-only
adhoc queries for development.

Usage:
  researchwiki db rebuild [--verbose]           # walk wiki/, upsert all rows
  researchwiki db verify                        # report drift between fs and db
  researchwiki db query "SELECT ..."            # adhoc read-only query
  researchwiki db query --file FILE             # run query from a SQL file
  researchwiki db papers [filters]              # structured lookups over the papers mirror
  researchwiki db path                          # print resolved DB path

`db papers` answers structural/bibliometric questions ("how many cgt papers
from 2024?", "which papers lack a DOI?") straight from the frontmatter mirror —
no re-reading markdown. Composable filters (all AND-ed):
  researchwiki db papers --year 2024-2026 --category cgt
  researchwiki db papers --no-doi
  researchwiki db papers --author hassabis --json
  researchwiki db papers --venue Nature --count
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from ..db.connection import db_path, get_connection
from ..db.rebuild import rebuild as _rebuild
from ..db.verify import verify as _verify
from ..log import log


# Deny-by-default read-only enforcement for `db query`. `mode=ro` only guards
# the *main* DB — it does NOT block `ATTACH` (an attached `?mode=rwc` file is
# writable) or `VACUUM INTO 'path'` (copies the whole DB to an arbitrary path,
# exfiltrating claim bodies). A first-verb keyword blacklist can never be
# complete. An authorizer that denies ATTACH/DETACH closes all three at the
# engine level: ATTACH, DETACH, and VACUUM INTO (which attaches its target
# internally, so the authorizer sees SQLITE_ATTACH). Paired with
# `PRAGMA query_only = ON` as a second layer. Reads return SQLITE_OK untouched.
def _deny_attach_authorizer(action, *_):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _cmd_rebuild(args) -> int:
    stats = _rebuild(verbose=args.verbose)
    log(f"pages_scanned   = {stats.pages_scanned}", tag="db")
    log(f"papers_upserted = {stats.papers_upserted}", tag="db")
    log(f"claims_upserted = {stats.claims_upserted}", tag="db")
    log(f"papers_deleted  = {stats.papers_deleted}", tag="db")
    if stats.parse_errors:
        log(f"parse_errors    = {len(stats.parse_errors)}", tag="db")
        for e in stats.parse_errors:
            print(f"     - {e}")
        return 2

    # Post-rebuild: re-sync the claim-graph cache. Slugs get re-computed
    # every rebuild (rebuild.py::_upsert_claims); reconcile() flags edges
    # whose endpoints or slug-scheme-version no longer match state.db.
    # Silent no-op when the edges cache is empty (fresh install).
    try:
        from ..claim_graph import open_edges_db, reconcile
        edges = open_edges_db()
        state = get_connection()
        try:
            rstats = reconcile(edges, state)
            if rstats.scanned:
                log(f"claim-graph reconcile: scanned={rstats.scanned} "
                    f"stale+={rstats.marked_stale_missing + rstats.marked_stale_version} "
                    f"active={rstats.active}", tag="db")
        finally:
            edges.close()
            state.close()
    except Exception as e:
        log(f"claim-graph reconcile skipped: {type(e).__name__}: {e}", tag="db")

    return 0


def _cmd_verify(args) -> int:
    report = _verify()
    log(f"pages_scanned = {report.pages_scanned}", tag="db")
    log(f"papers_in_db  = {report.papers_in_db}", tag="db")
    log(f"missing       = {len(report.missing)}", tag="db")
    log(f"extra         = {len(report.extra)}", tag="db")
    log(f"stale         = {len(report.stale)}", tag="db")
    log(f"moved         = {len(report.moved)}", tag="db")
    if report.missing:
        print()
        print("missing (in wiki/, not in db — run `db rebuild`):")
        for s in report.missing:
            print(f"  - {s}")
    if report.extra:
        print()
        print("extra (in db, not in wiki/ — will be removed on next rebuild):")
        for s in report.extra:
            print(f"  - {s}")
    if report.stale:
        print()
        print("stale (page mtime > indexed_at — run `db rebuild`):")
        for stem, p_mtime, i_at in report.stale:
            print(f"  - {stem}  (page_mtime={p_mtime}, indexed_at={i_at})")
    if report.moved:
        print()
        print("moved (category in db != fs):")
        for stem, db_cat, fs_cat in report.moved:
            print(f"  - {stem}  (db: {db_cat}, fs: {fs_cat})")
    return 0 if report.is_clean else 1


def _cmd_query(args) -> int:
    if args.file:
        sql = open(args.file).read()
    elif args.sql:
        sql = args.sql
    else:
        print("researchwiki db query: provide a SQL string or --file FILE", file=sys.stderr)
        return 2

    # Fail early with a helpful message when the DB doesn't exist yet —
    # `mode=ro` refuses to auto-create, and `get_connection()`'s side-effect
    # of running `init_schema` is bypassed on the read-only path. This is an
    # environment error (nothing to query), so exit 2.
    path = db_path()
    if not path.exists():
        print(f"researchwiki db query: no state.db at {path}. "
              "Run `researchwiki db rebuild` first.", file=sys.stderr)
        return 2

    # Open the DB read-only. `mode=ro` blocks writes to the main DB;
    # `PRAGMA query_only` and the ATTACH/DETACH authorizer close the residual
    # write/exfil paths (ATTACH to a rwc file, VACUUM INTO). `resolve()` makes
    # a relative RESEARCHWIKI_DB_PATH absolute before `as_uri()` (which raises
    # on relative paths); as_uri percent-encodes spaces / `%` / `#` / `?` and
    # (on Windows) backslashes and drive letters so the URI opens correctly.
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_deny_attach_authorizer)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
    except (sqlite3.Warning, sqlite3.Error) as e:
        # Any error executing the user's SQL is a user-input error (exit 1):
        # write refusal, ATTACH/DETACH/VACUUM refusal (authorizer raises the
        # DatabaseError parent, not OperationalError), multi-statement, syntax
        # error, or no-such-table. Render a friendly one-liner rather than a
        # Python traceback in every case. NOTE: a multi-statement paste raises
        # sqlite3.ProgrammingError on Python 3.11+ but sqlite3.Warning on 3.10,
        # and Warning is a *sibling* of Error (not a subclass) — both must be
        # caught to stay traceback-free across the supported 3.10–3.12 range.
        code = getattr(e, "sqlite_errorcode", None)
        msg = str(e).lower()
        if "not authorized" in msg or "authorization denied" in msg:
            print("researchwiki db query: refusing ATTACH/DETACH/VACUUM statement. "
                  "The CLI is read-only.", file=sys.stderr)
        elif code == getattr(sqlite3, "SQLITE_READONLY", -1) or (
            "readonly" in msg or "read-only" in msg or "attempt to write" in msg
        ):
            print("researchwiki db query: refusing write statement. "
                  "The CLI is read-only; use `db rebuild` to update the index.",
                  file=sys.stderr)
        else:
            print(f"researchwiki db query: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    _render_rows(cols, rows, args.json)
    return 0


def _render_rows(cols: list[str], rows: list, as_json: bool) -> None:
    """Shared TSV / JSON table renderer (used by `query` and `papers`)."""
    if as_json:
        print(json.dumps([{c: r[c] for c in cols} for r in rows],
                         indent=2, default=str, ensure_ascii=False))
        return
    if cols:
        print("\t".join(cols))
        print("\t".join("-" * len(c) for c in cols))
    for r in rows:
        print("\t".join(str(r[c]) if r[c] is not None else "" for c in cols))
    print(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")


def _cmd_papers(args) -> int:
    """Structured lookups over the `papers` frontmatter mirror. Builds a
    parameterized WHERE from composable filters — no hand-written SQL, no
    re-parsing markdown."""
    where: list[str] = []
    params: list = []

    if args.year:
        y = args.year.strip()
        if "-" in y and all(part.strip().isdigit() for part in y.split("-", 1)):
            lo, hi = (int(part) for part in y.split("-", 1))
            where.append("year BETWEEN ? AND ?")
            params += [lo, hi]
        elif y.isdigit():
            where.append("year = ?")
            params.append(int(y))
        else:
            print(f"researchwiki db papers: --year expects YYYY or YYYY-YYYY (got {y!r})",
                  file=sys.stderr)
            return 2
    if args.category:
        where.append("category = ?")
        params.append(args.category)
    if args.page_type:
        where.append("page_type = ?")
        params.append(args.page_type)
    if args.no_doi:
        where.append("(doi IS NULL OR TRIM(doi) = '' OR LOWER(doi) IN ('todo', 'none'))")
    if args.venue:
        where.append("venue LIKE ?")
        params.append(f"%{args.venue}%")
    if args.author:
        where.append("(authors LIKE ? OR senior_authors LIKE ?)")
        params += [f"%{args.author}%", f"%{args.author}%"]
    if args.status:
        where.append("publication_status LIKE ?")
        params.append(f"%{args.status}%")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    conn = get_connection()
    try:
        if args.count:
            n = conn.execute(f"SELECT COUNT(*) AS n FROM papers{clause}", params).fetchone()["n"]
            print(n)
            return 0
        cols = ["stem", "category", "page_type", "year", "venue", "doi"]
        sql = (f"SELECT {', '.join(cols)} FROM papers{clause} "
               f"ORDER BY year DESC, stem LIMIT ?")
        rows = conn.execute(sql, [*params, args.limit]).fetchall()
    finally:
        conn.close()
    _render_rows(cols, rows, args.json)
    return 0 if rows or args.json else 1


def _cmd_path(_args) -> int:
    print(db_path())
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki db", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_rebuild = subs.add_parser("rebuild", help="Rebuild the DB from markdown")
    p_rebuild.add_argument("--verbose", action="store_true")
    p_rebuild.set_defaults(func=_cmd_rebuild)

    p_verify = subs.add_parser("verify", help="Detect drift between db and wiki/")
    p_verify.set_defaults(func=_cmd_verify)

    p_query = subs.add_parser("query", help="Run a read-only SELECT query")
    p_query.add_argument("sql", nargs="?", help="SQL string (or use --file)")
    p_query.add_argument("--file", help="Path to a .sql file")
    p_query.add_argument("--json", action="store_true", help="Output JSON instead of TSV")
    p_query.set_defaults(func=_cmd_query)

    p_papers = subs.add_parser("papers", help="Structured lookups over the papers mirror")
    p_papers.add_argument("--year", help="Exact year (YYYY) or range (YYYY-YYYY)")
    p_papers.add_argument("--category", help="Content category (wiki/ subdir)")
    p_papers.add_argument("--page-type", dest="page_type", help="Frontmatter type: (paper/synthesis/...)")
    p_papers.add_argument("--no-doi", action="store_true", help="Only pages missing a DOI")
    p_papers.add_argument("--venue", help="Substring match on venue")
    p_papers.add_argument("--author", help="Substring match on authors or senior_authors")
    p_papers.add_argument("--status", help="Substring match on publication_status (e.g. preprint)")
    p_papers.add_argument("--count", action="store_true", help="Print the match count only")
    p_papers.add_argument("--limit", type=int, default=200, help="Max rows (default: 200)")
    p_papers.add_argument("--json", action="store_true", help="Output JSON instead of TSV")
    p_papers.set_defaults(func=_cmd_papers)

    p_path = subs.add_parser("path", help="Print the resolved DB path and exit")
    p_path.set_defaults(func=_cmd_path)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
