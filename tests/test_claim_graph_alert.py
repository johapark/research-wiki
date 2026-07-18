"""Tests for the ingest-time contradiction alert hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.claim_graph import alert as alert_mod


def test_alert_prints_contradiction_line(monkeypatch, capsys):
    """When the judge returns disagreements involving the new paper, alert
    prints one `⚠ contradicts [[stem#slug]]` line per hit."""
    hits = [{
        "verdict": "disagree_numeric",
        "rationale": "40% vs 60% on the same benchmark",
        "similarity": 0.92,
        "pair": [
            {"claim_slug": "res-a1b2c3d4", "paper_stem": "new-2026-x",
             "section": "results", "position": 0, "text": "40% efficiency"},
            {"claim_slug": "res-e5f6g7h8", "paper_stem": "old-2024-y",
             "section": "results", "position": 3, "text": "60% efficiency"},
        ],
    }]
    monkeypatch.setattr(
        "researchwiki.tasks.lint.cross_paper.find_cross_paper_contradictions",
        lambda **kw: hits,
    )
    # Stub slug resolution too so we don't need a live state.db.
    monkeypatch.setattr(alert_mod, "_resolve_slug",
                        lambda stem, section, position: "res-oldslug1")

    result = alert_mod.alert_after_ingest("new-2026-x", Path("wiki/ai/new-2026-x.md"))
    out = capsys.readouterr().out

    assert result == {"stem": "new-2026-x", "n_alerts": 1}
    assert "⚠ contradicts [[old-2024-y#res-oldslug1]]" in out
    assert "disagree_numeric" in out
    assert "40% vs 60%" in out


def test_alert_silent_when_no_hits(monkeypatch, capsys):
    monkeypatch.setattr(
        "researchwiki.tasks.lint.cross_paper.find_cross_paper_contradictions",
        lambda **kw: [],
    )
    result = alert_mod.alert_after_ingest("new-2026-x", Path("wiki/ai/new-2026-x.md"))
    out = capsys.readouterr().out
    assert result == {"stem": "new-2026-x", "n_alerts": 0}
    assert "⚠" not in out


def test_alert_swallows_exceptions(monkeypatch, capsys):
    """The hook must never break ingest — any exception is logged and swallowed."""
    def boom(**kw):
        raise RuntimeError("judge exploded")
    monkeypatch.setattr(
        "researchwiki.tasks.lint.cross_paper.find_cross_paper_contradictions",
        boom,
    )
    result = alert_mod.alert_after_ingest("new-2026-x", Path("wiki/ai/new-2026-x.md"))
    assert result is None
    out = capsys.readouterr().out
    assert "⚠" not in out


def test_alert_picks_the_other_side_of_the_pair(monkeypatch, capsys):
    """If the new paper is the B side of the pair, the alert should still
    cite the OTHER paper — not the new one — as the contradicting counterpart."""
    hits = [{
        "verdict": "disagree_direction",
        "rationale": "opposite signs",
        "similarity": 0.9,
        "pair": [
            # A = old paper, B = new one. The alert should cite A.
            {"claim_slug": "res-oldslug", "paper_stem": "old-y",
             "section": "results", "position": 0, "text": "up 5×"},
            {"claim_slug": "res-newslug", "paper_stem": "new-x",
             "section": "results", "position": 0, "text": "down 5×"},
        ],
    }]
    monkeypatch.setattr(
        "researchwiki.tasks.lint.cross_paper.find_cross_paper_contradictions",
        lambda **kw: hits,
    )
    monkeypatch.setattr(alert_mod, "_resolve_slug",
                        lambda stem, section, position: "res-oldslug")
    alert_mod.alert_after_ingest("new-x", Path("wiki/ai/new-x.md"))
    out = capsys.readouterr().out
    assert "[[old-y#res-oldslug]]" in out
    assert "[[new-x" not in out
