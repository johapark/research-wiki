"""Expose read-only wiki queries over MCP.

Serves three wiki-inspection tools to MCP clients (Claude Desktop, IDE
integrations, other agents):

  - **search** — BM25 / hybrid search over `wiki/` pages.
  - **claims** — grounded-citation search over the pre-graded claims table.
  - **check_grounding** — the structural grounding gate against a wiki page.

Bibliometric queries (year/venue/author filters over the papers table)
stay on the CLI — reach for `researchwiki db papers` when you want them.

**Read-only by design.** No ingest, no synthesize, no evolve, no lint-fix.
Write operations stay CLI-only where they can be reviewed. If an MCP client
wants those, they can shell out to `researchwiki <cmd>` themselves — but
this server won't proxy them.

**Transport: stdio.** Matches the intended use (`claude mcp add
researchwiki-local -- python -m researchwiki mcp-serve`). No auth, no HTTP;
if remote MCP is ever needed that's a separate design.

Usage:
  researchwiki mcp-serve

Requires the optional `mcp` extra: `pip install -e '.[mcp]'`. The module
imports FastMCP lazily so `researchwiki --help` and other subcommands work
even when the extra isn't installed.

Pattern lifted from hermes-agent `mcp_serve.py` (FastMCP over SessionDB
queries). Adapted for research-wiki's smaller, purely-read surface.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _do_search(query: str, limit: int, mode: str) -> list[dict]:
    """Query the search backend and return hit dicts.

    Mirrors the resolution logic in `tasks/search.py` — `auto` picks the
    strongest mode the environment supports, and requests fall back
    gracefully when an index is missing.
    """
    from . import search as search_task
    from ..search import get_default_backend, SearchBackendUnavailable

    resolved = search_task._resolve_mode(mode)
    if resolved == "hybrid":
        from ..search import hybrid as hybrid_mod
        hits = hybrid_mod.query(query, limit=limit)
        return [search_task._hybrid_as_dict(h) for h in hits]
    # Semantic-only mode returns a different hit shape and is rarely useful
    # standalone (hybrid supersedes it). Fall through to BM25 for a
    # consistent output shape — the user can still get BM25's snippets.
    try:
        backend = get_default_backend()
        return [search_task._hit_as_dict(h) for h in backend.query(query, limit=limit)]
    except SearchBackendUnavailable as e:
        return [{"error": str(e)}]


def _do_claims(query: str | None, by_stem: str | None, k: int,
               include_context: bool) -> list[dict]:
    """Grounded-citation search or per-paper claim dump.

    Exactly one of `query` / `by_stem` must be provided (matches the CLI's
    argparse constraint). Both parameters live on one tool rather than two
    because the shape of a "claim hit" is identical and MCP clients pick
    the mode by which arg they populate.
    """
    from ..search import claim_lookup, claims_by_stem

    if bool(query) == bool(by_stem):
        return [{"error": "pass exactly one of `query` or `by_stem`"}]
    if by_stem:
        return claims_by_stem(by_stem, include_context=include_context)
    return claim_lookup(query, k=k, include_context=include_context)


def _do_check_grounding(page_path: str, strict: bool) -> dict:
    """Structural grounding gate. Returns the same JSON shape as
    `researchwiki check-grounding --json` so MCP callers can consume the
    two surfaces interchangeably.
    """
    from ..grade import grounding

    p = Path(page_path).expanduser()
    if not p.is_absolute():
        # Interpret relative paths against the wiki root, not the MCP
        # server's cwd (which is likely the client's cwd — could be
        # anywhere). Client sends `wiki/synthesis/foo.md`, server resolves
        # it against `wiki_root()`.
        from ..paths import wiki_root
        p = (wiki_root() / p).resolve()
    if not p.exists():
        return {"error": f"file not found: {p}"}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return {"error": str(e)}
    permissive = not strict
    report = grounding.check(text, permissive=permissive)
    return {
        "total_claims": report.total_claims,
        "grounded_claims": report.grounded_claims,
        "model_prior_claims": report.model_prior_claims,
        "ungrounded_claims": len(report.ungrounded_units),
        "coverage": round(report.coverage, 3),
        "permissive": permissive,
        "units": [
            {
                "index": u.index,
                "line_start": u.line_start,
                "kind": u.kind,
                "is_claim": u.is_claim,
                "has_citation": u.has_citation,
                "is_model_prior": u.is_model_prior,
                "citations": u.citations,
                "flag_reason": u.flag_reason,
                "preview": (u.text[:160].replace("\n", " ")
                            + ("…" if len(u.text) > 160 else "")),
            }
            for u in report.units
        ],
    }


def build_server():
    """Construct and return a FastMCP server with all read-only tools
    registered. Split out so tests can introspect the tool set without
    launching the server's stdio loop.

    Raises ImportError if the `mcp` extra is not installed.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("researchwiki")

    @server.tool(description="Search wiki pages. `mode=auto` picks hybrid over BM25 when both indexes exist.")
    def search(query: str, limit: int = 10, mode: str = "auto") -> list[dict]:
        return _do_search(query, limit, mode)

    @server.tool(description="Grounded-citation search over the claims table. Pass `query` for topic search, or `by_stem` to dump one paper's claims.")
    def claims(query: str | None = None, by_stem: str | None = None,
               k: int = 5, include_context: bool = False) -> list[dict]:
        return _do_claims(query, by_stem, k, include_context)

    @server.tool(description="Check whether each claim-bearing unit in a wiki page carries a citation. `strict=true` treats the `*(model prior)*` marker as ungrounded.")
    def check_grounding(page_path: str, strict: bool = False) -> dict:
        return _do_check_grounding(page_path, strict)

    return server


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="researchwiki mcp-serve",
        description="Serve read-only wiki queries over MCP (stdio transport).",
    )
    # Reserved for future --transport switch. Today stdio is the only mode.
    parser.parse_args(argv)

    try:
        server = build_server()
    except ImportError:
        print(
            "researchwiki mcp-serve: the `mcp` package is not installed.\n"
            "  Install with: pip install -e '.[mcp]'  "
            "(or: pip install mcp)",
            file=sys.stderr,
        )
        return 2

    # FastMCP.run() blocks on stdio until the client disconnects. Client
    # config example (Claude Desktop mcpServers block):
    #   {
    #     "researchwiki-local": {
    #       "command": "python",
    #       "args": ["-m", "researchwiki", "mcp-serve"]
    #     }
    #   }
    server.run()
    return 0
