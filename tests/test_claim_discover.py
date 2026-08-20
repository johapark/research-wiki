"""Tests for the zero-token claim-pair discovery tier.

Behaviour contract: `--discover` ranks cross-paper claim pairs inside a cosine
*band* by IDF-weighted shared-term mass, and judges/writes nothing. It exists
because lowering the auto-link threshold does not work — at cosine 0.70, 80% of
all possible paper pairs qualify, and the relation that motivated the module
(Parks 2018 vs van Iterson 2017 on empirical nulls) peaks at 0.743.

See researchwiki/tasks/claim_discover.py and PLAN-bottom-up-synthesis.md
(evidence items E5/E6).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from researchwiki.tasks.claim_discover import (
    DEFAULT_COS_HI,
    DEFAULT_COS_LO,
    DISCOVERY_DECAY_DAYS,
    DISCOVERY_THRESHOLD,
    DiscoveredPair,
    _BLOCK,
    _tokens,
    discovery_stamp_age_days,
    discovery_warning,
    write_discovery_stamp,
)


def _idf_mass(a: str, b: str, corpus: list[str]) -> float:
    """Mirror of the ranker's scoring, for asserting relative order."""
    docs = [_tokens(t) for t in corpus]
    df: Counter = Counter()
    for d in docs:
        df.update(d)
    idf = {w: math.log(len(docs) / (1 + c)) for w, c in df.items()}
    return sum(idf.get(w, 0.0) for w in _tokens(a) & _tokens(b))


# ---------- tokenization ----------

def test_short_and_stop_words_carry_no_mass():
    toks = _tokens("The method shows that results improve with the approach")
    for w in ("the", "that", "with", "method", "shows", "results", "approach"):
        assert w not in toks


def test_hyphenated_compound_also_yields_its_parts():
    # Regression: "empirical-null" and "empirical"+"null" were disjoint, which
    # dropped the motivating pair from rank 74 to rank 369.
    toks = _tokens("a Bayesian empirical-null method")
    assert "empirical-null" in toks
    assert "empirical" in toks
    assert "null" in toks


def test_hyphen_parts_below_length_floor_are_dropped():
    # "d-dimer" keeps the compound and "dimer"; the one-letter part is noise.
    toks = _tokens("transient d-dimer elevations")
    assert "d-dimer" in toks and "dimer" in toks
    assert "d" not in toks


def test_hyphen_split_unifies_the_motivating_pair():
    corpus = [
        "The empirical control distribution had long tails inconsistent with a normal distribution",
        "Introduced a Bayesian empirical-null method that fits a three-component normal mixture",
        "Superpixel segmentation achieves boundary recall on the benchmark",
        "The transformer encoder attends over gene tokens for expression prediction",
    ]
    assert _idf_mass(corpus[0], corpus[1], corpus) > 0
    # ... and beats an unrelated pairing drawn from the same corpus.
    assert _idf_mass(corpus[0], corpus[1], corpus) > _idf_mass(corpus[0], corpus[3], corpus)


def test_shared_rare_terms_outrank_shared_common_ones():
    corpus = [
        "chromatin accessibility peaks at distal enhancer regions",
        "distal enhancer regions show chromatin accessibility",
        "prediction prediction prediction network network tasks",
        "network tasks prediction",
    ]
    rare = _idf_mass(corpus[0], corpus[1], corpus)
    common = _idf_mass(corpus[2], corpus[3], corpus)
    assert rare > common


# ---------- band contract ----------

def test_band_sits_strictly_below_the_auto_link_threshold():
    # The tier must not re-surface what `claim-overlap` already judges at 0.83,
    # or the queue is half things already handled.
    assert DEFAULT_COS_LO < DEFAULT_COS_HI
    assert DEFAULT_COS_HI <= 0.83


# ---------- the result type ----------

def _pair(cat_a="ai", cat_b="ai") -> DiscoveredPair:
    return DiscoveredPair(
        stem_a="a-2020-x", stem_b="b-2021-y",
        category_a=cat_a, category_b=cat_b,
        slug_a="kc-1111aaaa", slug_b="res-2222bbbb",
        text_a="…", text_b="…", cosine=0.75, idf_mass=20.0,
        shared_terms=["alpha", "beta"],
    )


def test_cross_category_flag():
    assert _pair("ai", "compbio").cross_category
    assert not _pair("ai", "ai").cross_category


def test_citations_use_the_durable_slug_anchor_form():
    p = _pair()
    assert p.citation_a() == "[[a-2020-x#kc-1111aaaa]]"
    assert p.citation_b() == "[[b-2021-y#res-2222bbbb]]"


# ---------- blocked upper-triangle scan ----------

def _blocked_pairs(vecs, stems, lo, hi, block):
    """The scan in discover_pairs, isolated from DB/cache so the geometry can
    be asserted directly against a brute-force reference."""
    n = len(stems)
    out = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        b = vecs[start:stop] @ vecs.T
        keep = (b >= lo) & (b < hi)
        keep &= stems[start:stop, None] != stems[None, :]
        keep &= np.arange(n)[None, :] > np.arange(start, stop)[:, None]
        bi, bj = np.where(keep)
        out.extend(zip((bi + start).tolist(), bj.tolist()))
    return sorted(out)


def _reference_pairs(vecs, stems, lo, hi):
    n = len(stems)
    sims = vecs @ vecs.T
    m = (sims >= lo) & (sims < hi) & (stems[:, None] != stems[None, :])
    ii, jj = np.where(np.triu(m, 1))
    return sorted(zip(ii.tolist(), jj.tolist()))


@pytest.mark.parametrize("block", [1, 3, 7, _BLOCK])
def test_blocked_scan_matches_full_matrix_for_any_block_size(block):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    stems = np.array([f"p{i // 4}" for i in range(40)])
    assert (_blocked_pairs(v, stems, 0.2, 0.9, block)
            == _reference_pairs(v, stems, 0.2, 0.9))


def test_blocked_scan_never_emits_self_pairs_or_same_paper_pairs():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(24, 6)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    stems = np.array([f"p{i // 6}" for i in range(24)])
    for a, b in _blocked_pairs(v, stems, -1.0, 1.0, 5):
        assert a < b                      # upper triangle only
        assert stems[a] != stems[b]       # cross-paper only


# ---------- nudge ----------

def test_nudge_bar_is_higher_and_slower_than_the_coverage_backlog():
    # Discovery is an opportunity signal, not a coverage gap: nothing is wrong
    # when the queue has entries, so it must nag less than the backlog nudge.
    from researchwiki.tasks.claim_overlap import (
        BACKLOG_DECAY_DAYS, BACKLOG_THRESHOLD,
    )
    assert DISCOVERY_THRESHOLD > BACKLOG_THRESHOLD
    assert DISCOVERY_DECAY_DAYS > BACKLOG_DECAY_DAYS


def test_stamp_round_trip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert discovery_stamp_age_days() is None
    write_discovery_stamp()
    age = discovery_stamp_age_days()
    assert age is not None and age < 1.0


def test_unreadable_stamp_reads_as_absent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claim-discovery-stamp").write_text("not-a-timestamp", encoding="utf-8")
    assert discovery_stamp_age_days() is None


def test_warning_is_silent_on_an_empty_corpus(monkeypatch, tmp_path):
    # Advisory surface: no DB, no claims, no crash, no output.
    monkeypatch.chdir(tmp_path)
    assert discovery_warning(touch=False) is None
