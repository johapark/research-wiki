"""MCP server surface: read-only wiki queries via FastMCP.

Hermetic: no live MCP client, no stdio loop, no network. Tool handlers
are exercised via the module-level `_do_*` helpers so we don't need to
marshal through the MCP protocol just to test the response shape.
"""

from __future__ import annotations

import asyncio

import pytest

from researchwiki.tasks import mcp_serve


def test_main_reports_missing_mcp_cleanly(monkeypatch, capsys):
    """Missing `mcp` extra → main() returns 2 with an install hint, no
    traceback."""
    def raise_import(*a, **kw):
        raise ImportError("no module named 'mcp'")
    monkeypatch.setattr(mcp_serve, "build_server", raise_import)
    rc = mcp_serve.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "mcp" in err
    assert "install" in err.lower()


pytest.importorskip("mcp", reason="mcp SDK not installed")


@pytest.fixture
def server():
    return mcp_serve.build_server()


def test_exactly_three_read_only_tools_registered(server):
    """The MCP surface is a fixed set. A write tool sneaking in would
    defeat the module's whole design."""
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"search", "claims", "check_grounding"}


def test_do_claims_rejects_ambiguous_input():
    """Both `query` and `by_stem`, or neither, → error dict (not raise)."""
    assert "error" in mcp_serve._do_claims("foo", "bar", 5, False)[0]
    assert "error" in mcp_serve._do_claims(None, None, 5, False)[0]


def _capture_claim_query(monkeypatch):
    import researchwiki.search as search_pkg

    calls = []

    def fake_claim_query(query, k=5, *, mode="bm25", include_context=False):
        calls.append({"query": query, "k": k, "mode": mode,
                      "include_context": include_context})
        return [{"claim_slug": "kc-1", "paper_stem": "chen-2023-fake"}]

    monkeypatch.setattr(search_pkg, "claim_query", fake_claim_query)
    return calls


def test_do_claims_defaults_to_bm25_like_the_cli(monkeypatch):
    calls = _capture_claim_query(monkeypatch)
    mcp_serve._do_claims("off-target rate", None, 5, False)
    assert calls == [{"query": "off-target rate", "k": 5, "mode": "bm25",
                      "include_context": False}]


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_do_claims_forwards_explicit_mode(monkeypatch, mode):
    calls = _capture_claim_query(monkeypatch)
    result = mcp_serve._do_claims("off-target rate", None, 8, True, mode)
    assert calls == [{"query": "off-target rate", "k": 8, "mode": mode,
                      "include_context": True}]
    assert result[0]["claim_slug"] == "kc-1"


def test_do_claims_rejects_unknown_mode(monkeypatch):
    """Bad mode → error dict naming the valid choices, not a ValueError
    escaping into the MCP transport."""
    calls = _capture_claim_query(monkeypatch)
    result = mcp_serve._do_claims("off-target rate", None, 5, False, "auto")
    assert "error" in result[0]
    assert "hybrid" in result[0]["error"]
    assert calls == []


def test_claims_tool_exposes_mode_parameter(server):
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    props = tools["claims"].inputSchema["properties"]
    assert props["mode"]["default"] == "bm25"


def test_do_search_hybrid_mode_returns_fused_hits(monkeypatch):
    """When both indexes are built, `auto` resolves to hybrid and the
    handler must reach the real fusion entry point, not a missing alias."""
    from researchwiki.search import hybrid as hybrid_mod
    from researchwiki.tasks import search as search_task

    hit = hybrid_mod.HybridHit(
        key="cgt/chen-2023-fake", stem="chen-2023-fake", category="cgt",
        page_type="paper", title="Fake", rrf_score=0.03,
        bm25_rank=1, bm25_score=4.2, semantic_rank=2, semantic_score=0.71,
    )
    calls = []

    def fake_hybrid_query(text, limit=10):
        calls.append((text, limit))
        return [hit]

    monkeypatch.setattr(search_task, "_resolve_mode", lambda mode: "hybrid")
    monkeypatch.setattr(hybrid_mod, "hybrid_query", fake_hybrid_query)

    result = mcp_serve._do_search("prime editing", 7, "auto")

    assert calls == [("prime editing", 7)]
    assert result[0]["key"] == "cgt/chen-2023-fake"
    assert result[0]["rrf_score"] == 0.03


def test_do_check_grounding_missing_file(tmp_path):
    assert "error" in mcp_serve._do_check_grounding(str(tmp_path / "nope.md"), False)


def test_do_check_grounding_grounded_page(tmp_path):
    """Page with a `[[wikilink]]` on its only claim → zero ungrounded."""
    page = tmp_path / "grounded.md"
    page.write_text(
        "# Findings\n\n"
        "Prime editors reduce off-target effects [[cgt/chen-2023-fake]].\n"
    )
    result = mcp_serve._do_check_grounding(str(page), strict=False)
    assert result["ungrounded_claims"] == 0
    assert "total_claims" in result


def test_do_check_grounding_rejects_generic_author_model(tmp_path):
    page = tmp_path / "generic.md"
    page.write_text(
        "---\ntitle: X\ntype: synthesis\nauthor_model: gpt-5.6\n---\n\n"
        "## Findings\n\nGrounded claim [[cgt/chen-2023-fake]].\n",
        encoding="utf-8",
    )

    result = mcp_serve._do_check_grounding(str(page), strict=False)

    assert result["finding"] == "missing_author_model"


def test_do_check_grounding_names_malformed_yaml(tmp_path):
    page = tmp_path / "broken.md"
    page.write_text(
        "---\ntitle: X\ntype: synthesis\nauthor_model: claude-opus-4-7\n"
        "tags: [a, b\n---\n\n## Findings\n\nClaim [[cgt/chen-2023-fake]].\n",
        encoding="utf-8",
    )

    result = mcp_serve._do_check_grounding(str(page), strict=False)

    assert result["finding"] == "invalid_frontmatter"
