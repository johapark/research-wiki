"""Pure-logic tests for the salience grader (PDF-anchor recall).

End-to-end `score_salience` against a synthetic page body exercises the full
synthesize → grade.scorer.score_text path; no PDFs or indices required.
"""

from researchwiki.grade.salience import (
    SalienceReport,
    _is_limitation,
    _first_caption_sentence,
    _split_sentences,
    score_salience,
    synthesize_fixture,
)


# ---------- helpers ----------

def test_split_sentences_basic():
    assert _split_sentences("First. Second sentence! Third one?") == [
        "First.", "Second sentence!", "Third one?",
    ]


def test_split_sentences_empty():
    assert _split_sentences("   ") == []


def test_split_sentences_strips_line_number_prefix():
    """Nature accelerated-preview line numbers ('26 The cycle...') prefix
    every line of the extracted text. Strip them before splitting so the
    prefix doesn't contaminate anchor token-overlap scoring."""
    # Real PDF extraction produces one line-number per line; the test input
    # mirrors that structure.
    text = (
        "26 The cycle of scientific discovery is bottlenecked.\n"
        "27 We present ERA."
    )
    sents = _split_sentences(text)
    assert sents == [
        "The cycle of scientific discovery is bottlenecked.",
        "We present ERA.",
    ]


def test_split_sentences_preserves_inline_numbers():
    """Strip leading line-number prefixes only; keep mid-sentence numbers
    (e.g. "5,377,346") intact."""
    text = "We surveyed 5,377,346 researchers across 41.3M papers."
    assert _split_sentences(text) == [
        "We surveyed 5,377,346 researchers across 41.3M papers.",
    ]


def test_first_caption_sentence_strips_pipe_header():
    block = "Fig. 3 | Folddisco workflow. The system identifies motifs."
    assert _first_caption_sentence(block) == "Folddisco workflow."


def test_first_caption_sentence_extended_data():
    block = "Extended Data Fig. 4 | Variant analysis across cohorts."
    assert _first_caption_sentence(block) == "Variant analysis across cohorts."


def test_limitation_cue_detection():
    assert _is_limitation("This has limitations on out-of-distribution data.")
    assert _is_limitation("However, the method fails when N is small.")
    assert _is_limitation("Future work will explore generalization.")
    assert not _is_limitation("The method achieves state-of-the-art accuracy.")


# ---------- synthesize_fixture ----------

def test_synthesize_fixture_abstract_seeds_headline_claims():
    pdf_text = "irrelevant"
    sections = {
        "abstract": "We introduce Folddisco. It searches structural motifs. "
                    "Performance is 20× faster than prior tools. "
                    "We benchmark on PDB.",
    }
    f = synthesize_fixture("foo-2026-bar", pdf_text, sections)
    # Every abstract sentence is a critical anchor — 4 sentences here.
    assert len(f.headline_claims) == 4
    assert all(it.importance == "critical" for it in f.headline_claims
               if it.id.startswith("abstract-"))
    assert "Folddisco" in f.headline_claims[0].verbalization


def test_synthesize_fixture_introduction_no_longer_seeds_critical():
    """Introduction is intentionally NOT a critical anchor source — it
    overlaps with the abstract. A sections dict with `introduction` but no
    `abstract` should produce zero critical headline_claims."""
    sections = {
        "introduction": "We introduce Folddisco. It searches structural motifs.",
    }
    f = synthesize_fixture("stem", "", sections)
    assert all(it.importance != "critical" for it in f.headline_claims)
    # No abstract → no critical anchors at all.
    assert not any(it.id.startswith("abstract-") for it in f.headline_claims)


def test_synthesize_fixture_results_added_with_high_importance():
    sections = {
        "results": "Folddisco recovers 95% of true motifs. It scales to PDB-wide search.",
    }
    f = synthesize_fixture("stem", "", sections)
    ids = [it.id for it in f.headline_claims]
    assert "results-lead-0" in ids
    assert all(it.importance == "high" for it in f.headline_claims)


def test_synthesize_fixture_routes_limitation_cue():
    sections = {
        "discussion": "The method generalizes to multi-domain proteins. "
                      "However, the approach fails on disordered regions. "
                      "We close with broader implications. "
                      "Future work will address transmembrane proteins.",
    }
    f = synthesize_fixture("stem", "", sections)
    # Two limitation cues ("However ... fails", "Future work") — both routed
    # to the limitations axis. The non-cue sentences land in headline_claims.
    lim_texts = [it.verbalization for it in f.limitations]
    assert any("fails on disordered" in t for t in lim_texts)
    assert any("Future work" in t for t in lim_texts)
    assert any("generalizes to multi-domain" in it.verbalization
               for it in f.headline_claims)


def test_synthesize_fixture_captions_to_capabilities():
    sections = {
        "figure_captions": "Fig. 1 | Workflow overview. Steps include indexing.\n\n"
                           "Fig. 2 | Benchmark results. We test on PDB.",
        "extended_data": "Extended Data Fig. 1 | Ablation study.",
    }
    f = synthesize_fixture("stem", "", sections)
    cap_ids = [it.id for it in f.capabilities]
    assert "fig-0" in cap_ids and "fig-1" in cap_ids
    assert "ed-0" in cap_ids
    importances = {it.id: it.importance for it in f.capabilities}
    assert importances["fig-0"] == "high"
    assert importances["ed-0"] == "normal"


def test_synthesize_fixture_empty_sections_yields_empty_axes():
    f = synthesize_fixture("stem", "", {})
    assert f.headline_claims == []
    assert f.capabilities == []
    assert f.limitations == []


# ---------- score_salience end-to-end ----------

def test_score_salience_zero_anchors_returns_none_score():
    r = score_salience("stem", "", {}, "page body content")
    assert isinstance(r, SalienceReport)
    assert r.n_anchors == 0
    assert r.salience_score is None
    assert r.missed_anchors == []


def test_score_salience_full_match():
    sections = {
        "abstract": "Folddisco is a structural motif search tool. "
                    "It is 20-fold faster than pyScoMotif.",
    }
    # Page body repeats every significant token from each anchor — should
    # match cleanly (overlap >= 0.75 and numbers preserved).
    page_body = (
        "## Summary\n\n"
        "Folddisco is a structural motif search tool that achieves "
        "20-fold speedup over pyScoMotif on PDB-scale benchmarks.\n"
    )
    r = score_salience("stem", "", sections, page_body)
    assert r.n_anchors == 2
    assert r.salience_score is not None and r.salience_score > 0.5
    assert r.n_match >= 1


def test_score_salience_miss_surfaces_missed_anchors():
    sections = {
        "results": "The benchmark shows kcat/KM values from 180 to 53000 across enzymes.",
    }
    # Page body deliberately omits the numbers and most significant tokens.
    page_body = "## Summary\n\nThe paper discusses enzyme catalysis broadly.\n"
    r = score_salience("stem", "", sections, page_body)
    assert r.n_anchors == 1
    assert r.n_miss >= 1
    assert r.missed_anchors  # non-empty when there's at least one miss
    assert "id" in r.missed_anchors[0] and "axis" in r.missed_anchors[0]


def test_score_salience_per_axis_only_populated_axes():
    sections = {
        "abstract": "Method foo solves the bar problem efficiently.",
        # No results, discussion, captions
    }
    r = score_salience("stem", "", sections,
                      "Method foo solves the bar problem efficiently.")
    # Only headline_claims should appear in per_axis (capabilities and
    # limitations had zero items so they're omitted).
    assert "headline_claims" in r.per_axis
    assert "capabilities" not in r.per_axis
    assert "limitations" not in r.per_axis


# ── Semantic-cosine path ──────────────────────────────────────────────


def test_heuristic_verdict_semantic_promotes_low_overlap_match():
    """A high cosine should promote a low-overlap claim to match — the
    paraphrase-tolerance the semantic path was added to provide."""
    from researchwiki.benchmark.fixture import FixtureItem
    from researchwiki.grade.scorer import _heuristic_verdict

    item = FixtureItem(
        id="t1", importance="critical",
        verbalization="The paper introduces a structural motif search tool.",
    )
    # Page body shares almost no significant tokens with the anchor.
    page_body = "Different completely unrelated wording about disparate topics."
    # Without semantic: heavy miss.
    v_token_only = _heuristic_verdict(item, page_body)
    assert v_token_only.verdict == "miss"
    # With high semantic cosine (paraphrase): promoted to match.
    v_with_sem = _heuristic_verdict(item, page_body, semantic_score=0.85)
    assert v_with_sem.verdict == "match"
    assert "semantic 0.85" in v_with_sem.rationale


def test_heuristic_verdict_semantic_partial_threshold():
    """A mid-range cosine (0.50–0.75) should at least promote to partial,
    even when token overlap is below the partial threshold."""
    from researchwiki.benchmark.fixture import FixtureItem
    from researchwiki.grade.scorer import _heuristic_verdict

    item = FixtureItem(
        id="t1", importance="high",
        verbalization="The paper introduces a structural motif search tool.",
    )
    page_body = "Wholly different topic with no token overlap whatsoever."
    v = _heuristic_verdict(item, page_body, semantic_score=0.60)
    assert v.verdict == "partial"


def test_heuristic_verdict_semantic_does_not_override_numeric_drift():
    """A high cosine cannot rescue a claim with multiple missing numbers —
    numeric integrity is the hard signal."""
    from researchwiki.benchmark.fixture import FixtureItem
    from researchwiki.grade.scorer import _heuristic_verdict

    item = FixtureItem(
        id="t1", importance="critical",
        verbalization="Achieves 85.7% accuracy with 1234 parameters across 42 trials.",
    )
    # Page has high cosine (paraphrase) but ZERO of the numbers.
    page_body = "The method demonstrates strong accuracy with many parameters in extensive trials."
    v = _heuristic_verdict(item, page_body, semantic_score=0.90)
    # 3 missing numbers (>1) → can't even land partial; stays miss despite
    # high cosine. The numeric-integrity guard is preserved.
    assert v.verdict == "miss"


def test_heuristic_verdict_token_path_unchanged_without_semantic():
    """Backward-compat: no semantic_score → identical behavior to before."""
    from researchwiki.benchmark.fixture import FixtureItem
    from researchwiki.grade.scorer import _heuristic_verdict

    item = FixtureItem(
        id="t1", importance="high",
        verbalization="The Folddisco method searches structural motifs efficiently.",
    )
    page_body = "Folddisco method searches structural motifs efficiently in PDB."
    v = _heuristic_verdict(item, page_body)  # no semantic_score
    assert v.verdict == "match"
    assert "semantic" not in v.rationale


def test_score_salience_use_semantic_false_falls_back_to_token():
    """`use_semantic=False` should produce the same scores as before the
    semantic path was added — preserves the token-only behavior path."""
    sections = {
        "abstract": "Folddisco is a structural motif search tool. "
                    "It is 20-fold faster than pyScoMotif.",
    }
    page_body = (
        "## Summary\n\n"
        "Folddisco is a structural motif search tool that achieves "
        "20-fold speedup over pyScoMotif on PDB-scale benchmarks.\n"
    )
    r = score_salience("stem", "", sections, page_body, use_semantic=False)
    # Token overlap is high here; both anchors should match without
    # semantic.
    assert r.n_match >= 1
