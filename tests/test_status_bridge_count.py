"""`researchwiki status`'s concept-hub trigger must not report a crash as zero.

CLAUDE.md makes the count `status` prints the trigger for scaffolding a hub
("if it prints a nonzero `Concept-hub candidates: N …` line, that's your
trigger"). `status` prints nothing at zero — so when the scan raised and
`n_bridge_candidates` returned 0, "the scan died" was byte-identical to
"nothing to do", and the documented trigger silently stopped firing forever.

This is not hypothetical: `test_concept_neardupe.py::
test_page_body_fallback_does_not_crash` documents a real instance — an
unimported `read_page` in the empty-`state.db` fallback made a fresh corpus
report zero candidates indefinitely, because the NameError was swallowed.

The fix returns `None` on failure so the caller can say so out loud. These
tests pin all three arms, since the whole point is that they're distinguishable.

Hermetic: `collect_candidates` is stubbed; no state.db, no embeddings, no LLM.
"""

from __future__ import annotations

import pytest

from researchwiki import concepts
from researchwiki.concepts import candidates as cand


# ---------- n_bridge_candidates: the three outcomes ----------

def test_returns_count_on_success(monkeypatch):
    monkeypatch.setattr(cand, "collect_candidates",
                        lambda **kw: [{"term": "a"}, {"term": "b"}])
    assert cand.n_bridge_candidates() == 2


def test_returns_zero_when_genuinely_empty(monkeypatch):
    monkeypatch.setattr(cand, "collect_candidates", lambda **kw: [])
    assert cand.n_bridge_candidates() == 0


def test_returns_none_not_zero_when_scan_raises(monkeypatch, capsys):
    """The regression. 0 and None must be different values, because `status`
    renders them differently and only one of them is a real answer."""
    def boom(**kw):
        raise NameError("name 'read_page' is not defined")
    monkeypatch.setattr(cand, "collect_candidates", boom)

    assert cand.n_bridge_candidates() is None
    # The reason isn't lost — `log` puts it on stderr, so the failure stays
    # diagnosable without polluting `status`'s stdout report.
    err = capsys.readouterr().err
    assert "bridge-candidate scan failed" in err
    assert "NameError" in err


def test_asks_for_bridges_only(monkeypatch):
    """Bridge tier means span ≥ 2 categories. Dropping the flag would inflate
    the count with same-category terms and make the trigger cry wolf."""
    seen = {}
    monkeypatch.setattr(cand, "collect_candidates",
                        lambda **kw: seen.update(kw) or [])
    cand.n_bridge_candidates()
    assert seen.get("bridges_only") is True


def test_interrupt_still_propagates(monkeypatch):
    """The catch is `except Exception`, not bare — so Ctrl-C during a long scan
    still aborts `status` instead of being reported as an unknown count. The
    swallowing exists to keep a dashboard panel from killing the report, not to
    make the command uninterruptible."""
    monkeypatch.setattr(cand, "collect_candidates",
                        lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        cand.n_bridge_candidates()


# ---------- what status actually prints ----------
#
# `status` no longer prints a concept-hub line: proposals replaced hubs as the
# discovery path, so its nudge is the proposal queue. `n_bridge_candidates`
# stays callable for `candidates concepts`, and is pinned above.


def test_status_module_uses_the_synced_proposal_queue():
    """Concept discovery remains callable, but proposals own the status nudge."""
    from pathlib import Path
    src = Path(concepts.__file__).parent.parent / "tasks" / "status.py"
    text = src.read_text()
    assert "from ..proposals import load_proposals" in text
    assert "Proposal queue:" in text
    assert "Concept-hub candidates:" not in text
