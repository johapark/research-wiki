"""Fine-grained anchor verification in `grade synthesis`.

Tests the assertion from the plan: `--fine-grained` fails a page citing
`[[stem#slug]]` when a numeric token in the sentence isn't in the specific
claim's text, EVEN when the paper as a whole contains it (paper-level
misattribution would pass).
"""

from __future__ import annotations

import sqlite3

import pytest

from researchwiki.grade.fidelity.synthesis import (
    _check_anchor_misattribution,
    _extract_anchor_pairs,
)


# ---------- anchor extraction ----------


def test_extract_anchor_strips_category_prefix():
    text = "As shown in [[compbio/foo#kc-abcd1234]] and [[bar#res-99887766]]."
    got = _extract_anchor_pairs(text)
    assert got == [("foo", "kc-abcd1234"), ("bar", "res-99887766")]


def test_extract_ignores_plain_wikilinks():
    text = "See [[foo]] and [[bar/baz]]."
    assert _extract_anchor_pairs(text) == []


def test_extract_dedups():
    text = "First [[foo#kc-abcd]] then again [[foo#kc-abcd]]."
    assert _extract_anchor_pairs(text) == [("foo", "kc-abcd")]


# ---------- fine-grained misattribution check ----------


@pytest.fixture
def state_db_with_claims(tmp_path, monkeypatch):
    """Stand up a minimal state.db with a few claims for anchor resolution."""
    p = tmp_path / "state.db"
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE claims (paper_stem TEXT, claim_slug TEXT, text TEXT,
                             UNIQUE(paper_stem, claim_slug));
    """)
    # The paper has TWO claims. Only kc-01 mentions "40%"; kc-02 doesn't.
    conn.execute("INSERT INTO claims VALUES ('foo', 'kc-01', 'Prime editing at 40% efficiency in HEK cells.')")
    conn.execute("INSERT INTO claims VALUES ('foo', 'kc-02', 'The method scales to 96-well plate assays.')")
    conn.commit()

    def fake_get_conn():
        return conn

    monkeypatch.setattr("researchwiki.db.connection.get_connection", fake_get_conn)
    yield conn
    conn.close()


def test_no_anchors_returns_empty(state_db_with_claims):
    assert _check_anchor_misattribution("A sentence with [[foo]] but no anchor.") == []


def test_matching_number_in_specific_claim_is_ok(state_db_with_claims):
    """Number IS in the cited claim's text → not anchor-misattributed."""
    sentence = "Prime editing reaches 40% efficiency. [[foo#kc-01]]"
    assert _check_anchor_misattribution(sentence) == []


def test_missing_number_in_cited_claim_flags_anchor_misattribution(state_db_with_claims):
    """Number is NOT in the cited claim (though it's in the OTHER claim of the
    same paper, which paper-level would silently accept). Should flag."""
    sentence = "The method scales to 40% efficiency. [[foo#kc-02]]"
    got = _check_anchor_misattribution(sentence)
    assert len(got) == 1
    assert got[0]["stem"] == "foo"
    assert got[0]["slug"] == "kc-02"
    assert "40" in got[0]["numeric_tokens_missing"]


def test_stale_slug_gets_no_flag(state_db_with_claims):
    """A `[[stem#slug]]` pointing at a slug not in state.db can't be checked;
    no fine-grained flag is emitted (the dangling-anchor lint owns that
    surface)."""
    sentence = "Something 99. [[foo#kc-deadbeef]]"
    # The specific claim isn't in state.db → return empty (silently skip).
    assert _check_anchor_misattribution(sentence) == []


def test_dates_are_stripped_before_checking(state_db_with_claims):
    """ISO dates in editorial headers must not be treated as claim numerics."""
    sentence = "Updated 2026-01-15: the method achieves 40% efficiency. [[foo#kc-01]]"
    assert _check_anchor_misattribution(sentence) == []
