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

def _status_concept_line(monkeypatch, capsys, result):
    """Drive just the concept-hub block of `status` and return its output.

    `status.main` touches the DB, the index and the filesystem, so rather than
    stand a whole wiki up we exercise the same three-branch render the module
    performs, against the module's own threshold constant.
    """
    monkeypatch.setattr(concepts, "n_bridge_candidates", lambda: result)
    n_bridges = concepts.n_bridge_candidates()
    if n_bridges is None:
        print("Concept-hub candidates: scan failed — count unknown "
              "(`researchwiki candidates concepts --bridges` for the error)")
    elif n_bridges >= concepts.TRIAGE_THRESHOLD:
        print(f"Concept-hub candidates: {n_bridges} bridge term(s) — likely "
              "dominated by extraction noise.")
    elif n_bridges > 0:
        print(f"Concept-hub candidates: {n_bridges} bridge term(s) with no hub yet")
    return capsys.readouterr().out


def test_status_announces_a_failed_scan(monkeypatch, capsys):
    out = _status_concept_line(monkeypatch, capsys, None)
    assert "scan failed" in out
    assert "count unknown" in out


def test_status_stays_silent_on_a_real_zero(monkeypatch, capsys):
    # Silence is correct here — nothing to scaffold. It's only wrong when it
    # means "we don't know".
    assert _status_concept_line(monkeypatch, capsys, 0) == ""


def test_status_fires_the_trigger_on_a_small_count(monkeypatch, capsys):
    out = _status_concept_line(monkeypatch, capsys, 3)
    assert "3 bridge term(s)" in out
    assert "no hub yet" in out


def test_status_suggests_triage_above_threshold(monkeypatch, capsys):
    out = _status_concept_line(monkeypatch, capsys, concepts.TRIAGE_THRESHOLD + 5)
    assert "extraction noise" in out


def test_status_module_renders_the_failed_scan_branch(monkeypatch, capsys):
    """Belt-and-braces: the branch above is a transcription of `status.py`'s.
    Assert the real module still contains all three arms, so a refactor that
    drops the None arm doesn't leave this file testing a fiction."""
    from pathlib import Path
    src = Path(concepts.__file__).parent.parent / "tasks" / "status.py"
    text = src.read_text()
    assert "if n_bridges is None:" in text
    assert "scan failed" in text
    assert "elif n_bridges > 0:" in text
