"""Near-duplicate aggregation of concept-hub candidates.

Covers the deterministic layer (singularization, co-occurrence-gated canonical
merge in the keywords detector, alias-aware hub exclusion, canonical decline
filter) and the LLM `alias` verdict layer (via the `judge_fn` seam). No
provider is touched; `monkeypatch.chdir(tmp_path)` isolates `wiki_root()` for
decline writes.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.concepts import candidates as C
from researchwiki.concepts.candidates import (
    _canonical_key, _singularize, _term_slug,
    find_candidates_from_keywords, _load_hub_aliases,
)
from researchwiki.concepts import triage as T
from researchwiki.concepts.declines import (
    add_decline, declined_canon, declined_slugs,
)


# --- _singularize ----------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("models", "model"), ("studies", "study"), ("batches", "batch"),
    ("boxes", "box"), ("variants", "variant"), ("datasets", "dataset"),
    # invariant / singular-ending-in-s (must NOT be stripped)
    ("bias", "bias"), ("analysis", "analysis"), ("virus", "virus"),
    ("species", "species"), ("series", "series"), ("gas", "gas"),
    ("bus", "bus"), ("atlas", "atlas"), ("alias", "alias"), ("lens", "lens"),
])
def test_singularize(word, expected):
    assert _singularize(word) == expected


def test_irregular_plural_stays_distinct_not_merged():
    # We don't map analyses→analysis (irregular), but crucially the two must
    # NOT collapse to the same canonical key — the pair simply stays split.
    assert _canonical_key("analyses") != _canonical_key("analysis")


# --- _canonical_key --------------------------------------------------------

def test_canonical_key_folds_plural_and_hyphen():
    assert _canonical_key("foundation models") == "foundation-model"
    assert _canonical_key("foundation-model") == "foundation-model"
    assert _canonical_key("Foundation Models") == "foundation-model"


def test_canonical_key_singularizes_last_token_only():
    # "systems" is not the head noun; only the last segment is singularized.
    assert _canonical_key("systems biology") == "systems-biology"


def test_term_slug_unchanged_by_feature():
    # The frozen filename/decline/edge key must not singularize.
    assert _term_slug("Foundation Models") == "foundation-models"


# --- keywords detector: co-occurrence-gated merge --------------------------

def _paper(stem, keywords, category="compbio"):
    return {"stem": stem, "category": category, "keywords": keywords, "tags": []}


def test_cooccurrence_merge_collapses_and_rescues():
    # 2 papers say "foundation model", 2 say "foundation models". Each variant
    # alone is below the 3-page floor; merged they surface as ONE row (4 pages).
    meta = [
        _paper("a", ["foundation model"]),
        _paper("b", ["foundation model"]),
        _paper("c", ["foundation models"]),
        _paper("d", ["foundation models"]),
    ]
    rows = find_candidates_from_keywords(meta, set())
    fm = [r for r in rows if _canonical_key(r["term"]) == "foundation-model"]
    assert len(fm) == 1
    assert fm[0]["pages"] == 4                      # unioned stems, not summed twice
    # representative is a real surface form whose slug round-trips
    assert fm[0]["slug"] == _term_slug(fm[0]["term"])


def test_lone_variant_not_rewritten():
    # "biases" appears alone (no "bias") in 3 papers → single canonical group →
    # emitted verbatim, never singularized/rewritten.
    meta = [_paper(s, ["biases"]) for s in ("a", "b", "c")]
    rows = find_candidates_from_keywords(meta, set())
    assert any(r["slug"] == "biases" for r in rows)


# --- alias-aware hub exclusion ---------------------------------------------

def test_existing_canon_excludes_singular_of_hub():
    meta = [_paper(s, ["protein language model"]) for s in ("a", "b", "c")]
    existing_canon = {_canonical_key("protein language models")}  # the hub's topic_seed
    rows = find_candidates_from_keywords(meta, set(), existing_canon=existing_canon)
    assert not any(_canonical_key(r["term"]) == "protein-language-model" for r in rows)


def test_existing_canon_excludes_alias():
    meta = [_paper(s, ["FH"]) for s in ("a", "b", "c")]
    existing_canon = {_canonical_key("FH")}   # from a hub's topic_seed_aliases
    rows = find_candidates_from_keywords(meta, set(), existing_canon=existing_canon)
    assert not any(r["term"] == "FH" for r in rows)


def test_existing_canon_none_is_backcompat_noop():
    # Passing no existing_canon (the way current unit tests call it) keeps the
    # pre-feature behavior: only the exact-slug exclusion applies.
    meta = [_paper(s, ["protein language model"]) for s in ("a", "b", "c")]
    rows = find_candidates_from_keywords(meta, set())   # no existing_canon
    assert any(_canonical_key(r["term"]) == "protein-language-model" for r in rows)


# --- _load_hub_aliases (DB-sourced, no filesystem) -------------------------

class _FakeConn:
    def __init__(self, rows): self._rows = rows
    def execute(self, sql):
        assert "page_type = 'concept'" in sql
        return self
    def fetchall(self): return self._rows
    def close(self): pass


def test_load_hub_aliases_reads_seed_and_aliases(monkeypatch):
    fm = {"topic_seed": "protein language models",
          "topic_seed_aliases": ["pLM", "protein LM"]}
    rows = [{"raw_frontmatter": json.dumps(fm)}]
    monkeypatch.setattr("researchwiki.db.connection.get_connection",
                        lambda: _FakeConn(rows), raising=True)
    got = _load_hub_aliases()
    assert set(got) == {"protein language models", "pLM", "protein LM"}


def test_load_hub_aliases_db_failure_returns_empty(monkeypatch):
    def boom(): raise RuntimeError("no db")
    monkeypatch.setattr("researchwiki.db.connection.get_connection", boom, raising=True)
    assert _load_hub_aliases() == []


# --- decline filter: canonical ∪ exact -------------------------------------

def test_declined_canon_canonicalizes_term(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    add_decline("foundation models", "noise")
    assert declined_canon() == {"foundation-model"}       # canonical of the term
    assert "foundation-models" in declined_slugs()         # exact slug still recorded


def test_decline_backcompat_sourceless_entry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".concept-declines.json").write_text(json.dumps({
        "old-slug": {"term": "old term", "reason": "r", "declined_at": "2026-01-01T00:00:00"}
    }) + "\n")
    add_decline("new models", "r2", source="llm-triage")
    assert "old-slug" in declined_slugs()                  # legacy exact still present
    assert "new-model" in declined_canon()                 # new one canonicalized


# --- Layer 2: LLM alias verdict --------------------------------------------

def _cand(term):
    return {"term": term, "slug": _term_slug(term), "pages": 4, "categories": 2,
            "weighted": 4.0, "sections": {}, "label": "concept-ready (bridge)",
            "source": "keywords"}


def test_alias_verdict_declines_with_canonical_reason(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cands = [_cand("language models")]
    judge = lambda chunk: [{"term": "language models", "verdict": "alias",
                            "reason": "subsumed", "canonical": "genomic language model"}]
    results = T.triage_candidates(cands, judge_fn=judge)
    assert results[0]["verdict"] == "alias"
    summary = T.apply_triage(results, dry_run=False)
    d = summary["declined"][0]
    assert d["term"] == "language models"                  # slug-safe: own term
    assert "genomic language model" in d["reason"]         # canonical decorates reason
    from researchwiki.concepts.declines import load_declines
    assert load_declines()[_term_slug("language models")]["source"] == "llm-triage"


def test_alias_is_noise_verdict():
    assert "alias" in T.NOISE_VERDICTS
    assert "alias" in T._VALID_VERDICTS


# --- cross-source canonical dedup ------------------------------------------

def test_dedup_by_canonical_collapses_cross_source():
    rows = [
        {"term": "foundation models", "slug": "foundation-models", "pages": 4, "source": "keywords"},
        {"term": "foundation model", "slug": "foundation-model", "pages": 9, "source": "claims"},
        {"term": "single-cell foundation models", "slug": "single-cell-foundation-models",
         "pages": 3, "source": "keywords"},
    ]
    out = C._dedup_by_canonical(rows)
    fm = [r for r in out if _canonical_key(r["term"]) == "foundation-model"]
    assert len(fm) == 1
    assert fm[0]["source"] == "keywords"          # keywords wins over claims despite fewer pages
    # a distinct canonical key (different modifier) is NOT collapsed
    assert any(_canonical_key(r["term"]) == "single-cell-foundation-model" for r in out)


def test_dedup_lone_term_verbatim():
    rows = [{"term": "biases", "slug": "biases", "pages": 3, "source": "keywords"}]
    assert C._dedup_by_canonical(rows) == rows      # single-member group untouched
