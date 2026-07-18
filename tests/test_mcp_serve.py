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
