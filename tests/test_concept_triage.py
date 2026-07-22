"""Batch-LLM concept-candidate triage (`researchwiki.concepts.triage`).

The LLM is replaced by an injected `judge_fn` stub, so tests are
deterministic and never touch a provider. `monkeypatch.chdir(tmp_path)`
isolates `wiki_root()` (= `Path.cwd()`) so `.concept-declines.json` lands in
a temp dir.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.concepts import triage as t
from researchwiki.concepts.declines import declined_slugs, load_declines, add_decline
from researchwiki.concepts.candidates import _term_slug


def _cand(term, *, pages=4, categories=2, label="concept-ready (bridge)"):
    return {"term": term, "slug": _term_slug(term), "pages": pages,
            "categories": categories, "weighted": float(pages),
            "sections": {}, "label": label, "source": "keywords"}


@pytest.fixture
def wiki(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- triage_candidates: DI seam + slug-safe merge ---------------------------

def test_verdicts_classified_and_split(wiki):
    cands = [_cand("protein language models"), _cand("three datasets"),
             _cand("Smith-Waterman"), _cand("population structure")]

    def judge(chunk):
        return [
            {"term": "protein language models", "verdict": "concept", "reason": "one substrate, three jobs"},
            {"term": "three datasets", "verdict": "fragment", "reason": "extraction fragment"},
            {"term": "Smith-Waterman", "verdict": "glossary", "reason": "algorithm"},
            {"term": "population structure", "verdict": "uncertain", "reason": "cannot tell"},
        ]

    results = t.triage_candidates(cands, judge_fn=judge)
    by_term = {r["term"]: r["verdict"] for r in results}
    assert by_term == {
        "protein language models": "concept",
        "three datasets": "fragment",
        "Smith-Waterman": "glossary",
        "population structure": "uncertain",
    }
    summary = t.apply_triage(results, dry_run=False)
    # noise → declined; concept/uncertain → kept
    assert {d["term"] for d in summary["declined"]} == {"three datasets", "Smith-Waterman"}
    assert {k["term"] for k in summary["kept"]} == {"protein language models", "population structure"}
    # declines written with provenance + slugs that match the candidates
    dec = load_declines()
    assert _term_slug("three datasets") in dec
    assert dec[_term_slug("Smith-Waterman")]["source"] == "llm-triage"
    assert declined_slugs() >= {_term_slug("three datasets"), _term_slug("Smith-Waterman")}


def test_dry_run_writes_nothing(wiki):
    cands = [_cand("three datasets")]
    judge = lambda chunk: [{"term": "three datasets", "verdict": "fragment", "reason": "x"}]
    summary = t.apply_triage(t.triage_candidates(cands, judge_fn=judge), dry_run=True)
    assert summary["dry_run"] is True
    assert summary["declined"][0]["term"] == "three datasets"
    assert not (wiki / ".concept-declines.json").exists()   # nothing written
    assert declined_slugs() == set()


def test_slug_safety_uses_candidate_term(wiki):
    """The model echoes a mangled term; the decline must still key off the
    candidate's canonical slug/term, never the echoed string."""
    cands = [_cand("Smith-Waterman")]
    judge = lambda chunk: [{"term": "  smith-waterman  ", "verdict": "glossary", "reason": "algo"}]
    results = t.triage_candidates(cands, judge_fn=judge)
    assert len(results) == 1 and results[0]["term"] == "Smith-Waterman"   # canonical, not echoed
    t.apply_triage(results, dry_run=False)
    assert _term_slug("Smith-Waterman") in load_declines()


def test_invented_term_discarded_and_missing_kept(wiki):
    cands = [_cand("real term one"), _cand("real term two")]

    def judge(chunk):
        return [
            {"term": "real term one", "verdict": "fragment", "reason": "noise"},
            {"term": "hallucinated term", "verdict": "concept", "reason": "invented"},  # not in chunk
            # "real term two" omitted entirely
        ]

    results = t.triage_candidates(cands, judge_fn=judge)
    by_term = {r["term"]: r["verdict"] for r in results}
    assert "hallucinated term" not in by_term          # invented term discarded
    assert by_term["real term one"] == "fragment"
    assert by_term["real term two"] == "uncertain"     # omitted → fail-safe keep


def test_chunking_merges_all(wiki):
    cands = [_cand(f"term {i}") for i in range(90)]
    seen_chunks = []

    def judge(chunk):
        seen_chunks.append(len(chunk))
        return [{"term": c["term"], "verdict": "uncertain", "reason": "r"} for c in chunk]

    results = t.triage_candidates(cands, judge_fn=judge, chunk_size=40)
    assert seen_chunks == [40, 40, 10]        # 3 chunks
    assert len(results) == 90


# --- degradation: never raises ---------------------------------------------

def test_judge_raising_keeps_all(wiki):
    cands = [_cand("a"), _cand("b")]
    def judge(chunk):
        raise RuntimeError("provider down")
    results = t.triage_candidates(cands, judge_fn=judge)
    assert all(r["verdict"] == "uncertain" for r in results)
    assert t.apply_triage(results, dry_run=False)["declined"] == []
    assert not (wiki / ".concept-declines.json").exists()


def test_judge_returning_none_keeps_all(wiki):
    cands = [_cand("a")]
    results = t.triage_candidates(cands, judge_fn=lambda chunk: None)
    assert results[0]["verdict"] == "uncertain"


def test_empty_candidates(wiki):
    assert t.triage_candidates([], judge_fn=lambda c: []) == []


def test_invalid_verdict_value_ignored(wiki):
    cands = [_cand("a")]
    judge = lambda chunk: [{"term": "a", "verdict": "banana", "reason": "r"}]
    results = t.triage_candidates(cands, judge_fn=judge)
    assert results[0]["verdict"] == "uncertain"   # invalid enum dropped → fail-safe keep


# --- _parse_verdicts --------------------------------------------------------

def test_parse_fenced_json():
    text = '```json\n{"verdicts": [{"term": "x", "verdict": "concept", "reason": "r"}]}\n```'
    out = t._parse_verdicts(text)
    assert out == [{"term": "x", "verdict": "concept", "reason": "r"}]


def test_parse_embedded_in_prose():
    text = 'Here you go: {"verdicts": [{"term": "x", "verdict": "fragment", "reason": "r"}]} — done.'
    out = t._parse_verdicts(text)
    assert out and out[0]["verdict"] == "fragment"


def test_parse_garbage_returns_none():
    assert t._parse_verdicts("not json at all") is None
    assert t._parse_verdicts("") is None
    assert t._parse_verdicts('{"no_verdicts_key": 1}') is None


# --- add_decline back-compat ------------------------------------------------

def test_add_decline_source_backcompat(wiki):
    # a hand-built source-less entry (pre-feature shape) coexists
    p = wiki / ".concept-declines.json"
    p.write_text(json.dumps({"old-slug": {"term": "old term", "reason": "r",
                                          "declined_at": "2026-01-01T00:00:00"}}) + "\n")
    add_decline("new term", "r2", source="llm-triage")
    dec = load_declines()
    assert dec["old-slug"].get("source", "manual") == "manual"   # default on read
    assert dec[_term_slug("new term")]["source"] == "llm-triage"
    # default source when omitted (the manual --decline path)
    add_decline("manual term", "r3")
    assert load_declines()[_term_slug("manual term")]["source"] == "manual"
