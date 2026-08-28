"""Canonical scout dispatch and the audit compatibility surface."""

from __future__ import annotations

from researchwiki import __main__ as cli
from researchwiki.scouting import citations
from researchwiki.tasks import audit, scout


def test_scout_is_auto_discovered_without_removing_audit_alias():
    tasks = cli._discover_tasks()
    assert tasks["scout"] == "scout"
    assert tasks["audit"] == "audit"


def test_bare_scout_defaults_to_citations(monkeypatch):
    seen = []
    monkeypatch.setattr(
        citations,
        "main",
        lambda argv, **kwargs: seen.append((argv, kwargs)) or 0,
    )

    assert scout.main(["--json"]) == 0
    assert seen == [(["--json"], {"prog": "researchwiki scout"})]


def test_explicit_citations_mode_strips_the_mode_token(monkeypatch):
    seen = []
    monkeypatch.setattr(
        citations,
        "main",
        lambda argv, **kwargs: seen.append((argv, kwargs)) or 0,
    )

    assert scout.main(["citations", "--refresh-cache", "7"]) == 0
    assert seen == [
        (["--refresh-cache", "7"], {"prog": "researchwiki scout citations"})
    ]


def test_web_mode_delegates_to_agent_handoff_cli(monkeypatch):
    seen = []
    monkeypatch.setattr(
        scout.web_cli,
        "main",
        lambda argv: seen.append(argv) or 0,
    )

    assert scout.main(["web", "query"]) == 0
    assert seen == [["query"]]


def test_audit_alias_preserves_legacy_identity(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(
        citations,
        "main",
        lambda argv, **kwargs: seen.append((argv, kwargs)) or 0,
    )

    assert audit.main(["--json"]) == 0
    assert seen == [(
        ["--json"],
        {
            "prog": "researchwiki audit",
            "log_tag": "audit",
            "report_title": "# Semantic Scholar Audit Report",
        },
    )]
    assert "now `researchwiki scout citations`" in capsys.readouterr().err
