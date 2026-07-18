"""Grounding must fail safe when state.db is unreachable (review Tier 3).

A DB failure while resolving `[[stem#slug]]` claim anchors must NOT be read as
"every anchor is dangling" (which would spuriously fail an otherwise-grounded
page). Instead the anchors are counted permissively and the report flags the
environment condition so the CLI can return exit 2.
"""

from __future__ import annotations

import researchwiki.grade.grounding as g


def test_db_unavailable_flags_and_counts_permissively(monkeypatch):
    def boom(pairs):
        raise g.ClaimDBUnavailable("simulated locked db")

    monkeypatch.setattr(g, "_resolve_claim_anchors", boom)
    text = "A grounded claim with an anchor [[cgt/smith-2024-x#kc-abc12345]]."
    report = g.check(text)
    assert report.anchor_db_unavailable is True
    # Permissive fallback: the anchor counts as a citation, so no spurious
    # ungrounded failure.
    assert report.ungrounded_units == []


def test_resolved_empty_is_not_flagged(monkeypatch):
    # A genuine empty resolution (DB up, nothing matches) is NOT a DB failure —
    # the flag distinguishes the two, unlike the old empty-set conflation.
    monkeypatch.setattr(g, "_resolve_claim_anchors", lambda pairs: set())
    text = "A claim with an unresolved anchor [[cgt/smith-2024-x#kc-deadbeef]]."
    report = g.check(text)
    assert report.anchor_db_unavailable is False
