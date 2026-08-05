"""Near-duplicate claim-set detection (`lint`'s `duplicate_claim_sets`).

The failure this guards is a *commentary* on a paper — a Nature Genetics
"Research highlight", a News & Views, an editorial — ingested as `type: paper`.
Claim extraction then credits the original authors' contributions to the
commentary, and both fidelity gates pass, because the claims really are in the
commentary's PDF. The observable residue is structural: two pages assert the
same body of work.

Pinned here:
  - the metric (reciprocal top-1 concentration) fires on a synthetic
    derivative/source pair and stays quiet on an unrelated page
  - the score is the MIN of the two directional shares, and both directions
    are reported so a reviewer can tell which page is derivative
  - the threshold and the short-page guard both hold
  - a cold / thin embedding cache makes the check report "skipped" (None)
    rather than silently reporting nothing — and never loads the bi-encoder
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from researchwiki.index import claim_embeddings as ce
from researchwiki.tasks.lint import db_checks
from researchwiki.tasks.lint.db_checks import (
    _DUP_CLAIM_MIN_CLAIMS,
    _DUP_CLAIM_THRESHOLD,
    _duplicate_claim_pairs,
    find_duplicate_claim_sets,
)


def _ray(angle: float) -> list[float]:
    """A unit vector on the unit circle, embedded in 3-D."""
    return [math.cos(angle), math.sin(angle), 0.0]


def _corpus(n_source=10, n_deriv=10, n_other=10, deriv_offset=0.02):
    """source/derivative pages that mirror each other, plus an unrelated page.

    `source` claim i and `derivative` claim i sit `deriv_offset` radians apart —
    closer to each other than to any other claim on either page — so every
    claim's nearest OFF-page neighbour is its counterpart. The `other` page lives
    on the far side of the circle.
    """
    stems, vecs = [], []
    for i in range(n_source):
        stems.append("source-2026-original-work")
        vecs.append(_ray(i * 0.20))
    for i in range(n_deriv):
        stems.append("deriv-2026-research-highlight")
        vecs.append(_ray(i * 0.20 + deriv_offset))
    for i in range(n_other):
        stems.append("other-2024-unrelated-paper")
        vecs.append(_ray(math.pi + i * 0.20))
    return stems, np.array(vecs, dtype=np.float32)


# ---------- the metric ----------

def test_mirrored_pages_are_reported():
    stems, vecs = _corpus()
    hits = _duplicate_claim_pairs(stems, vecs)
    assert len(hits) == 1
    pair = {p["stem"] for p in hits[0]["pages"]}
    assert pair == {"source-2026-original-work", "deriv-2026-research-highlight"}
    assert hits[0]["score"] == 1.0


def test_unrelated_page_is_not_dragged_in():
    """The third page's claims all nearest-match each other's page pair only if
    it genuinely mirrors one — it doesn't, so it must not appear."""
    stems, vecs = _corpus()
    hits = _duplicate_claim_pairs(stems, vecs)
    reported = {p["stem"] for h in hits for p in h["pages"]}
    assert "other-2024-unrelated-paper" not in reported


def test_both_directions_are_reported_with_counts():
    """A reviewer needs the per-direction share to judge which page is
    derivative — the shorter page carrying the higher share is the suspect."""
    # 20-claim source, 10-claim derivative: every derivative claim points at the
    # source (share 1.0), but only half the source's claims point back — its
    # other half sits next to a third page instead.
    stems, vecs = [], []
    for i in range(10):
        stems.append("source")
        vecs.append(_ray(i * 0.20))
    for i in range(10):
        stems.append("source")
        vecs.append(_ray(math.pi + i * 0.20))
    for i in range(10):
        stems.append("deriv")
        vecs.append(_ray(i * 0.20 + 0.02))
    for i in range(10):
        stems.append("third")
        vecs.append(_ray(math.pi + i * 0.20 - 0.01))
    hits = _duplicate_claim_pairs(stems, np.array(vecs, dtype=np.float32),
                                  threshold=0.4, min_claims=5)
    by_pair = {tuple(sorted(p["stem"] for p in h["pages"])): h for h in hits}
    hit = by_pair[("deriv", "source")]
    by_stem = {p["stem"]: p for p in hit["pages"]}
    assert by_stem["deriv"]["top1_share"] == 1.0
    assert by_stem["deriv"]["n_claims"] == 10
    assert by_stem["source"]["n_claims"] == 20
    assert by_stem["source"]["top1_share"] == 0.5
    # Score is the MIN of the two shares: a big page can't dilute the signal
    # away, but neither can one page's total dependence inflate it.
    assert hit["score"] == 0.5


def test_below_threshold_pair_is_dropped():
    """Only 2 of 10 derivative claims mirror the source; the rest sit on the
    unrelated page's side. 0.2 < 0.25, so nothing is reported."""
    stems, vecs = [], []
    for i in range(10):
        stems.append("source")
        vecs.append(_ray(i * 0.20))
    for i in range(10):
        stems.append("deriv")
        # First two mirror source claims; the rest mirror the far-side page.
        vecs.append(_ray(i * 0.20 + 0.02) if i < 2 else _ray(math.pi + i * 0.20 + 0.02))
    for i in range(10):
        stems.append("other")
        vecs.append(_ray(math.pi + i * 0.20))
    hits = _duplicate_claim_pairs(stems, np.array(vecs, dtype=np.float32))
    scores = {tuple(sorted(p["stem"] for p in h["pages"])): h["score"] for h in hits}
    assert ("deriv", "source") not in scores


def test_short_pages_are_excluded():
    """With a handful of claims one shared neighbour is already a high share,
    so pages under the claim floor never pair up."""
    n = _DUP_CLAIM_MIN_CLAIMS - 1
    stems, vecs = [], []
    for i in range(n):
        stems.append("tiny-a")
        vecs.append(_ray(i * 0.20))
    for i in range(n):
        stems.append("tiny-b")
        vecs.append(_ray(i * 0.20 + 0.02))
    assert _duplicate_claim_pairs(stems, np.array(vecs, dtype=np.float32)) == []


def test_threshold_constant_is_the_tuned_value():
    """Tuned against the live corpus: 0.25 → 6 pairs (reviewable), 0.20 → 16,
    0.30 → 3 and loses the known-bad research-highlight pair. Bumping this is a
    deliberate act, not a drive-by."""
    assert _DUP_CLAIM_THRESHOLD == 0.25


def test_pairs_are_sorted_by_descending_score():
    stems, vecs = [], []
    for i in range(10):                       # perfect mirror -> score 1.0
        stems.append("aaa-source")
        vecs.append(_ray(i * 0.20))
    for i in range(10):
        stems.append("aaa-deriv")
        vecs.append(_ray(i * 0.20 + 0.02))
    for i in range(10):                       # partial mirror -> lower score
        stems.append("bbb-source")
        vecs.append(_ray(math.pi + i * 0.20))
    for i in range(10):
        stems.append("bbb-deriv")
        vecs.append(_ray(math.pi + i * 0.20 + (0.02 if i < 6 else 1.5)))
    hits = _duplicate_claim_pairs(stems, np.array(vecs, dtype=np.float32))
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


# ---------- cache-only embedding loader ----------

def _rows(*specs):
    return [{"paper_stem": s, "section": sec, "position": p, "text": t}
            for s, sec, p, t in specs]


def test_cache_loader_returns_none_without_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "semantic_cache_dir", lambda: tmp_path / "nope")
    assert ce.load_cached_claim_embeddings(_rows(("a", "results", 0, "x"))) is None


def test_cache_loader_covers_only_cached_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "semantic_cache_dir", lambda: tmp_path)
    rows = _rows(("a", "results", 0, "one"), ("a", "results", 1, "two"))
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(tmp_path / "claims.npy", vecs)
    (tmp_path / "claims_meta.json").write_text(json.dumps([
        {"id": "a·results·0", "hash": ce._text_hash("one")},
        {"id": "a·results·1", "hash": ce._text_hash("EDITED")},   # stale hash
    ]), encoding="utf-8")

    loaded = ce.load_cached_claim_embeddings(rows + _rows(("b", "results", 0, "new")))
    assert loaded is not None
    out, idxs = loaded
    assert idxs == [0]                    # stale hash and uncached row both skipped
    assert out.shape == (1, 2)
    assert pytest.approx(float(np.linalg.norm(out[0])), abs=1e-6) == 1.0


def test_cache_loader_never_rewrites_the_cache(tmp_path, monkeypatch):
    """`lint` is read-only: the loader must not mutate derived state the way
    `get_claim_embeddings` does (it rewrites the cache to its row set)."""
    monkeypatch.setattr(ce, "semantic_cache_dir", lambda: tmp_path)
    np.save(tmp_path / "claims.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    meta = [{"id": "a·results·0", "hash": ce._text_hash("one")}]
    (tmp_path / "claims_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    before = (tmp_path / "claims.npy").read_bytes(), (tmp_path / "claims_meta.json").read_text()

    ce.load_cached_claim_embeddings(_rows(("a", "results", 0, "one"),
                                         ("zz", "results", 0, "uncached")))

    assert ((tmp_path / "claims.npy").read_bytes(),
            (tmp_path / "claims_meta.json").read_text()) == before


def test_cache_loader_does_not_load_the_bi_encoder(tmp_path, monkeypatch):
    """A model construction inside `lint` would cost ~3s on every run."""
    monkeypatch.setattr(ce, "semantic_cache_dir", lambda: tmp_path)
    np.save(tmp_path / "claims.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    (tmp_path / "claims_meta.json").write_text(
        json.dumps([{"id": "a·results·0", "hash": ce._text_hash("one")}]), encoding="utf-8")

    def boom(*a, **kw):
        raise AssertionError("bi-encoder must not be constructed")
    monkeypatch.setattr("researchwiki.index.embeddings._get_model", boom)

    assert ce.load_cached_claim_embeddings(_rows(("a", "results", 0, "one"))) is not None


# ---------- the check's IO shell ----------

class _FakeRow(dict):
    def __getitem__(self, k):          # sqlite3.Row-style access
        return dict.__getitem__(self, k)


def _fake_claims(n=10, n_pages=2):
    return [_FakeRow(paper_stem=f"p{i % n_pages}", section="results",
                     position=i, text=f"claim {i}") for i in range(n)]


def _scattered(n_pages=20, per_page=25, dim=32, seed=0):
    """Claims whose nearest off-page neighbours scatter across many pages, so
    no pair clears the threshold — the normal state of a healthy corpus.

    Page and claim counts matter: the metric has a chance floor set by how many
    other pages a claim could point at. At 10 pages × 12 claims, 3-of-12 lands
    on 0.25 by luck; the live corpus is 389 pages × ~30 claims, where it
    doesn't. That's why this check is advisory and threshold-tuned per corpus.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_pages * per_page, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    stems = [f"p{i % n_pages}" for i in range(n_pages * per_page)]
    return stems, v


def test_check_reports_skipped_when_cache_is_cold(monkeypatch):
    """None, not [] — `lint` renders "skipped" so a cold cache can't be mistaken
    for a clean corpus (same convention as `invalid_frontmatter`)."""
    monkeypatch.setattr(db_checks, "safe_read", None, raising=False)
    monkeypatch.setattr("researchwiki.db.safe.safe_read",
                        lambda fn, *, default, label: _fake_claims())
    monkeypatch.setattr("researchwiki.index.claim_embeddings.load_cached_claim_embeddings",
                        lambda rows: None)
    assert find_duplicate_claim_sets() is None


def test_check_reports_skipped_when_cache_is_thin(monkeypatch):
    """Under 50% coverage the top-1 ranking is computed against a corpus with
    holes in it, so the shares stop meaning anything."""
    rows = _fake_claims(10)
    monkeypatch.setattr("researchwiki.db.safe.safe_read",
                        lambda fn, *, default, label: rows)
    monkeypatch.setattr(
        "researchwiki.index.claim_embeddings.load_cached_claim_embeddings",
        lambda r: (np.eye(3, dtype=np.float32), [0, 1, 2]),      # 3 of 10
    )
    assert find_duplicate_claim_sets() is None


def test_check_returns_empty_list_when_nothing_duplicates(monkeypatch):
    stems, vecs = _scattered()
    rows = [_FakeRow(paper_stem=s, section="results", position=i, text=f"claim {i}")
            for i, s in enumerate(stems)]
    monkeypatch.setattr("researchwiki.db.safe.safe_read",
                        lambda fn, *, default, label: rows)
    monkeypatch.setattr(
        "researchwiki.index.claim_embeddings.load_cached_claim_embeddings",
        lambda r: (vecs, list(range(len(rows)))),
    )
    assert find_duplicate_claim_sets() == []


def test_check_is_wired_into_the_lint_orchestrator():
    """The JSON key and the prose section both have to exist, or the finding
    never reaches a reviewer."""
    from researchwiki.tasks import lint as lint_pkg
    assert lint_pkg.find_duplicate_claim_sets is find_duplicate_claim_sets
    src = (lint_pkg.__file__ and open(lint_pkg.__file__, encoding="utf-8").read()) or ""
    assert '"duplicate_claim_sets": kw["duplicate_claim_sets"]' in src
    assert "Near-duplicate claim sets" in src
