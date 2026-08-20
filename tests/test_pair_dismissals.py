"""Tests for the discovery-pair suppression list.

`claim-overlap --discover` is stateless — it re-derives its ranked queue every
call — so a pair reviewed and rejected resurfaces at the same rank forever
unless the decision is recorded. This is the pair-level analogue of
`concepts.declines`: manual, permanent, atomic, order-independent.

`monkeypatch.chdir(tmp_path)` isolates `wiki_root()` (= `Path.cwd()`) so
`.pair-dismissals.json` lands in the temp dir rather than the real repo.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.tasks import pair_dismissals as pd
from researchwiki.tasks.pair_dismissals import (
    DISMISSALS_FILENAME,
    add_dismissal,
    add_dismissals,
    dismissed_pairs,
    is_stale,
    load_dismissals,
    pair_key,
    remove_dismissal,
)

A = "parks-2018-using-controls-to-limit-false"
B = "van-iterson-2017-controlling-bias-and-inflation-in-epigenome"
C = "rose-1998-deterministic-annealing-for-clustering-compression"


@pytest.fixture
def wiki(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------- key ordering ----------

def test_pair_key_is_order_independent():
    assert pair_key(A, B) == pair_key(B, A)


def test_pair_key_separates_distinct_pairs():
    assert pair_key(A, B) != pair_key(A, C)


# ---------- persistence ----------

def test_missing_file_reads_as_empty(wiki):
    assert load_dismissals() == {}
    assert dismissed_pairs() == set()


def test_corrupt_file_reads_as_empty_rather_than_raising(wiki):
    (wiki / DISMISSALS_FILENAME).write_text("{not json", encoding="utf-8")
    assert load_dismissals() == {}
    assert dismissed_pairs() == set()


def test_add_then_load_round_trip(wiki):
    add_dismissal(A, B, "vocabulary only; neither engages the other")
    entries = load_dismissals()
    assert len(entries) == 1
    entry = next(iter(entries.values()))
    assert entry["stems"] == sorted((A, B))
    assert entry["reason"] == "vocabulary only; neither engages the other"
    assert entry["source"] == "manual"
    assert entry["dismissed_at"]


def test_dismissed_pairs_returns_sorted_tuples(wiki):
    add_dismissal(B, A, "reason")           # reversed input order
    assert dismissed_pairs() == {tuple(sorted((A, B)))}


def test_dismissing_reversed_order_does_not_duplicate(wiki):
    add_dismissal(A, B, "first")
    add_dismissal(B, A, "second")
    assert len(load_dismissals()) == 1
    assert next(iter(load_dismissals().values()))["reason"] == "second"


def test_batch_write_records_every_pair(wiki):
    keys = add_dismissals([(A, B, "r1"), (A, C, "r2")])
    assert len(keys) == 2
    assert len(load_dismissals()) == 2
    assert dismissed_pairs() == {tuple(sorted((A, B))), tuple(sorted((A, C)))}


def test_empty_batch_is_a_noop(wiki):
    assert add_dismissals([]) == []
    assert not (wiki / DISMISSALS_FILENAME).exists()


def test_source_is_recorded(wiki):
    add_dismissal(A, B, "auto", source="llm-triage")
    assert next(iter(load_dismissals().values()))["source"] == "llm-triage"


# ---------- removal ----------

def test_remove_is_order_independent(wiki):
    add_dismissal(A, B, "reason")
    assert remove_dismissal(B, A) is True     # reversed
    assert dismissed_pairs() == set()


def test_remove_missing_pair_returns_false(wiki):
    assert remove_dismissal(A, B) is False


def test_remove_leaves_other_entries_intact(wiki):
    add_dismissals([(A, B, "r1"), (A, C, "r2")])
    remove_dismissal(A, B)
    assert dismissed_pairs() == {tuple(sorted((A, C)))}


# ---------- file shape ----------

def test_written_file_is_valid_json_object(wiki):
    add_dismissal(A, B, "reason")
    data = json.loads((wiki / DISMISSALS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert pair_key(A, B) in data


def test_a_json_list_is_rejected_as_empty(wiki):
    # Defensive: the loader must not hand a list to callers expecting a dict.
    (wiki / DISMISSALS_FILENAME).write_text("[]", encoding="utf-8")
    assert load_dismissals() == {}


# ---------- evidence fingerprint ----------
#
# A dismissal is a judgment about the claims that were on the table, so it must
# not outlive them. Claim slugs are content-addressed (`blake2s(normalize(text))`
# per claim_graph.slug), which makes the slug set a fingerprint that survives a
# `db rebuild` but changes on a real edit.

def test_entry_without_a_fingerprint_is_never_stale(wiki):
    # Written before the field existed: suppression is the safe failure mode
    # for a list of human decisions.
    assert is_stale({"stems": [A, B], "reason": "legacy"}) is False


def test_entry_with_a_mismatched_fingerprint_is_stale(monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "current-hash")
    assert is_stale({"stems": [A, B], "claims_fingerprint": "old-hash"}) is True


def test_entry_with_a_matching_fingerprint_is_not_stale(monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "same-hash")
    assert is_stale({"stems": [A, B], "claims_fingerprint": "same-hash"}) is False


def test_uncomputable_fingerprint_leaves_the_entry_valid(monkeypatch):
    # No DB / no claims: "could not tell" must not silently un-suppress.
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: None)
    assert is_stale({"stems": [A, B], "claims_fingerprint": "old-hash"}) is False


def test_malformed_stems_are_not_stale(monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "x")
    assert is_stale({"stems": [A], "claims_fingerprint": "old"}) is False
    assert is_stale({"claims_fingerprint": "old"}) is False


def test_stale_entries_drop_out_of_the_filter_set(wiki, monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "hash-at-dismiss")
    add_dismissal(A, B, "judged on the evidence of the day")
    assert dismissed_pairs() == {tuple(sorted((A, B)))}

    # The claims changed underneath it → the pair returns to the queue.
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "hash-after-regrade")
    assert dismissed_pairs() == set()
    # ...but the record itself survives, for listing and re-judgment.
    assert len(load_dismissals()) == 1


def test_honor_fingerprints_false_returns_stale_entries_too(wiki, monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "one")
    add_dismissal(A, B, "reason")
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "two")
    assert dismissed_pairs() == set()
    assert dismissed_pairs(honor_fingerprints=False) == {tuple(sorted((A, B)))}


def test_fingerprint_is_recorded_on_write(wiki, monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: "abc123")
    add_dismissal(A, B, "reason")
    assert next(iter(load_dismissals().values()))["claims_fingerprint"] == "abc123"


def test_fingerprint_key_is_omitted_when_uncomputable(wiki, monkeypatch):
    monkeypatch.setattr(pd, "claims_fingerprint", lambda a, b: None)
    add_dismissal(A, B, "reason")
    assert "claims_fingerprint" not in next(iter(load_dismissals().values()))
