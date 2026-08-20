"""`warm_claim_embeddings` — the append-only cache writer.

Two incidents motivate this function, and each gets a test here, because both
failure modes are silent: the cache is a derived artifact that rebuilds on demand,
so corrupting it produces wrong numbers rather than an error.

  1. **Eviction.** `get_claim_embeddings` ends in `_persist(rows, ...)`, which
     rewrites the cache meta to *exactly* `rows`. Calling it with a narrow row set
     therefore drops every other claim: during the semantic-member calibration
     that silently evicted 489 limitations claims (3,027 -> 2,538) and would have
     forced the next `claim-overlap` run to re-embed them.
  2. **The model load.** `get_claim_embeddings` calls `is_available()`, which
     constructs the SentenceTransformer (~3 s) even when every row is a cache hit
     — the documented reason `load_cached_claim_embeddings` exists. This function
     runs once per paper on the ingest path, so paying that on a full hit would
     undo most of what the cache buys.
"""

from __future__ import annotations

import numpy as np
import pytest

from researchwiki.index import claim_embeddings as ce


def _rows(*specs):
    return [{"paper_stem": s, "section": "key_contributions", "position": p,
             "text": t} for s, p, t in specs]


@pytest.fixture
def fake_embedder(monkeypatch):
    """Deterministic 4-dim vectors, and a call counter for the model load."""
    state = {"available_calls": 0, "embedded": []}

    def _is_available():
        state["available_calls"] += 1
        return True

    def _embed(texts):
        state["embedded"].extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts],
                        dtype=np.float32)

    monkeypatch.setattr("researchwiki.index.embeddings.is_available",
                        _is_available, raising=True)
    monkeypatch.setattr("researchwiki.index.embeddings.embed_texts",
                        _embed, raising=True)
    return state


def test_narrow_second_call_does_not_evict_the_first(fake_embedder):
    """The E1 incident, pinned: a caller with a subset must not shrink the cache."""
    wide = _rows(("paperA", 0, "alpha"), ("paperA", 1, "beta"),
                 ("paperB", 0, "gamma"))
    assert ce.warm_claim_embeddings(wide) is not None

    # A caller asking about one section only — the calibration's exact shape.
    narrow = _rows(("paperA", 0, "alpha"))
    assert ce.warm_claim_embeddings(narrow) is not None

    # Every original row must still be readable from the cache.
    got = ce.load_cached_claim_embeddings(wide)
    assert got is not None
    _, covered = got
    assert sorted(covered) == [0, 1, 2], "narrow call evicted rows it never asked about"


def test_full_cache_hit_never_constructs_the_model(fake_embedder):
    """Second incident: ~3 s of model construction on a path that needs none."""
    rows = _rows(("paperA", 0, "alpha"), ("paperB", 0, "gamma"))
    ce.warm_claim_embeddings(rows)
    assert fake_embedder["available_calls"] == 1
    before = fake_embedder["available_calls"]

    ce.warm_claim_embeddings(rows)                       # identical rows -> all hits
    assert fake_embedder["available_calls"] == before, \
        "model was constructed despite a complete cache hit"


def test_only_the_misses_are_embedded(fake_embedder):
    """Cost scales with new claims, not corpus size — the ingest-path property."""
    ce.warm_claim_embeddings(_rows(("paperA", 0, "alpha")))
    fake_embedder["embedded"].clear()

    ce.warm_claim_embeddings(_rows(("paperA", 0, "alpha"), ("paperB", 0, "gamma")))
    assert fake_embedder["embedded"] == ["gamma"]


def test_edited_text_is_re_embedded_and_replaced_in_place(fake_embedder):
    """A text edit invalidates just that row (hash mismatch), not the cache."""
    ce.warm_claim_embeddings(_rows(("paperA", 0, "alpha"), ("paperB", 0, "gamma")))
    fake_embedder["embedded"].clear()

    edited = _rows(("paperA", 0, "alpha rewritten"), ("paperB", 0, "gamma"))
    out = ce.warm_claim_embeddings(edited)
    assert fake_embedder["embedded"] == ["alpha rewritten"]
    # The new vector, not the stale one: fake embedder encodes text length.
    assert out[0][0] == pytest.approx(float(len("alpha rewritten")))
    # And the row count did not grow — an edit replaces, it doesn't append.
    got = ce.load_cached_claim_embeddings(edited)
    assert got is not None and len(got[1]) == 2


def test_returns_vectors_aligned_to_the_requested_rows(fake_embedder):
    rows = _rows(("paperA", 0, "a"), ("paperB", 0, "bbbb"), ("paperC", 0, "cc"))
    out = ce.warm_claim_embeddings(rows)
    assert out.shape == (3, 4)
    # Row order follows `rows`, regardless of cache insertion order.
    assert [v[0] for v in out] == pytest.approx([1.0, 4.0, 2.0])


def test_empty_rows_and_missing_numpy_return_none(fake_embedder, monkeypatch):
    assert ce.warm_claim_embeddings([]) is None
    monkeypatch.setattr(ce, "_NUMPY", False)
    assert ce.warm_claim_embeddings(_rows(("paperA", 0, "alpha"))) is None


def test_unavailable_embedder_returns_none_rather_than_a_partial_array(monkeypatch):
    """Callers have one skip path; a cold model must not yield zero vectors."""
    monkeypatch.setattr("researchwiki.index.embeddings.is_available",
                        lambda: False, raising=True)
    assert ce.warm_claim_embeddings(_rows(("paperA", 0, "alpha"))) is None
