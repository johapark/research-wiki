"""Author-draft temperature spread stays inside Anthropic's valid [0, 1] range.

The runner spreads draft temperatures as `0.2 + 0.4*i`, which exceeds 1.0 once
i >= 2 (n_drafts >= 4). Anthropic rejects temperature > 1.0, so the spread is
clamped. This pins both the clamp and the spread formula.
"""

import pytest


def _temps(n_drafts: int) -> list[float]:
    return [min(1.0, 0.2 + 0.4 * i) for i in range(n_drafts)]


def test_default_single_draft():
    # Default n_drafts=1 → one draft at the base temperature; the tournament
    # is opt-in (-n 2+).
    assert _temps(1) == pytest.approx([0.2])


def test_two_draft_tournament_spread():
    assert _temps(2) == pytest.approx([0.2, 0.6])


def test_three_drafts_reach_but_dont_exceed_one():
    assert _temps(3) == pytest.approx([0.2, 0.6, 1.0])


@pytest.mark.parametrize("n", [4, 5, 8])
def test_never_exceeds_anthropic_max(n):
    temps = _temps(n)
    assert all(0.0 <= t <= 1.0 for t in temps), temps
