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
    # Sentences are full-length on purpose: real abstract prose runs a median
    # 164 chars per sentence, and anchors below `_ANCHOR_MIN_CHARS` are dropped
    # as extraction artifacts (see `anchor_is_substantive`).
    sections = {
        "abstract": "We introduce Folddisco, a structural motif search tool "
                    "for proteome-scale queries. It searches structural motifs "
                    "by indexing residue-pair geometry rather than aligning "
                    "whole structures. Performance is 20-fold faster than "
                    "prior tools on the same hardware. We benchmark on the "
                    "full PDB and on AlphaFold-predicted structures.",
    }
    f = synthesize_fixture("foo-2026-bar", pdf_text, sections)
    # Each substantive abstract sentence is a critical anchor — 4 here.
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
        "abstract": "Folddisco is a structural motif search tool that indexes "
                    "residue-pair geometry for proteome-scale queries. "
                    "It is 20-fold faster than pyScoMotif on the same "
                    "PDB-scale benchmark set.",
    }
    # Page body repeats every significant token from each anchor — should
    # match cleanly (overlap >= 0.75 and numbers preserved).
    page_body = (
        "## Summary\n\n"
        "Folddisco is a structural motif search tool that indexes residue-pair "
        "geometry for proteome-scale queries, and it is 20-fold faster than "
        "pyScoMotif on the same PDB-scale benchmark set.\n"
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
    abstract = ("Method foo solves the bar problem efficiently by reusing the "
                "baz decomposition at every recursion level.")
    sections = {
        "abstract": abstract,
        # No results, discussion, captions
    }
    r = score_salience("stem", "", sections, abstract)
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
        "abstract": "Folddisco is a structural motif search tool that indexes "
                    "residue-pair geometry for proteome-scale queries. "
                    "It is 20-fold faster than pyScoMotif on the same "
                    "PDB-scale benchmark set.",
    }
    page_body = (
        "## Summary\n\n"
        "Folddisco is a structural motif search tool that indexes residue-pair "
        "geometry for proteome-scale queries, and it is 20-fold faster than "
        "pyScoMotif on the same PDB-scale benchmark set.\n"
    )
    r = score_salience("stem", "", sections, page_body, use_semantic=False)
    # Token overlap is high here; both anchors should match without
    # semantic.
    assert r.n_match >= 1


# ────────────────────────────────────────────────────────────────────
# Abstract-anchor guards.
#
# `critical` carries weight 3, and in this fixture the critical tier IS the
# abstract — a median 75% of the weighted denominator (p90 94%) on a 353-paper
# measurement. So whatever `extract_abstract` over-reaches into becomes the
# score. Each case below is a shape observed in the live corpus.
# ────────────────────────────────────────────────────────────────────

from researchwiki.grade.salience import (
    _MAX_ABSTRACT_ANCHORS,
    _MIN_WORDS_PER_SENTENCE,
    anchor_is_substantive,
)


def _abs_ids(fixture):
    return [it.id for it in fixture.headline_claims if it.id.startswith("abstract-")]


# ---------- guard 1: prose check ----------

def test_reference_slab_abstract_is_rejected_wholesale():
    """cheng-2023: extraction returns bibliography text where the abstract
    should be — "sentences" averaging ~4 words. No page can ever cover these
    anchors, so the score is unreachable rather than merely low (measured
    ceiling ~0.21).

    The two title-only entries here are the reason the prose check exists as a
    separate guard: they're long enough in characters and shaped like claims, so
    `anchor_is_substantive` passes them. Only the slab-level words-per-sentence
    ratio identifies them as bibliography.
    """
    slab = " ".join([
        "Genome-wide association analysis of coronary artery disease.",
        "Molecular architecture of the human chromatin remodelling complex.",
        "A global reference for human genetic variation. Nature 526, 68 (2015).",
        "Deep learning of genomic contexts from sequence alone. Cell 187, 1 (2024).",
        "Mach. Learn. Res. 12, 2825 (2011).", "Genome Res. 27, 722 (2017).",
        "Bioinformatics 34, 3094 (2018).", "J. Comput. Biol. 7, 203 (2000).",
        "Nat. Methods 15, 475 (2018).", "Science 381, eadg7492 (2023).",
    ])
    # Premise: the per-sentence filter alone would let entries through, so a
    # pass below can only come from the slab-level prose check.
    survivors = [s for s in _split_sentences(slab) if anchor_is_substantive(s)]
    assert len(survivors) >= 2, survivors
    f = synthesize_fixture("cheng-2023-x", "", {"abstract": slab})
    assert _abs_ids(f) == []


def test_one_word_abstract_fragment_is_rejected():
    """yi-2026: a single title fragment came through as the whole abstract.
    Caught by either guard — the per-sentence length floor gets there first."""
    f = synthesize_fixture("yi-2026-x", "", {"abstract": "Introduction"})
    assert _abs_ids(f) == []


def test_ordinary_prose_abstract_clears_the_prose_check():
    prose = (
        "We present a method for detecting somatic variants in low-coverage "
        "sequencing data using a learned error model. Across 12 tumour types "
        "the model improved precision at fixed recall relative to two widely "
        "used callers."
    )
    f = synthesize_fixture("stem", "", {"abstract": prose})
    assert len(_abs_ids(f)) == 2
    assert (len(prose.split()) / 2) >= _MIN_WORDS_PER_SENTENCE


# ---------- guard 2: substance filter + cap ----------

def test_masthead_and_author_lines_are_not_anchors():
    """bjornsson-2020's abstract region opens with the journal masthead, a DOI
    line, a copyright notice and a credential-laden author list. All are
    guaranteed misses that inflate the weighted denominator."""
    for junk in (
        "Circulation: Genomic and Precision Medicine is available at "
        "www.ahajournals.org/journal/circgen Circ Genom Precis Med. 2021;14:e00",
        "DOI: 10.1161/CIRCGEN.120.003029 February 2021 Correspondence to: "
        "Unnur Thorsteinsdottir, PhD, deCODE Genetics/Amgen, Inc",
        "For Sources of Funding and Disclosures, see page 47. "
        "© 2020 American Heart Association, Inc.",
        "Olafsdottir, MD; Sebastian Niehus, MSc; Birte Kehr, PhD; "
        "Gardar Sveinbjornsson, MSc; Steinunn Gudmundsdottir, MSc",
        "Key Words: cardiovascular disease genetics lipids microRNA "
        "polyadenylation Downloaded from http://ahajournals.org by on June 1",
    ):
        assert anchor_is_substantive(junk) is False, junk[:50]


def test_real_finding_survives_the_substance_filter():
    assert anchor_is_substantive(
        "Mean level of LDL cholesterol was 74% lower in del2.5 carriers than "
        "in 101851 noncarriers, a difference of 2.48 mmol/L."
    ) is True


def test_cap_counts_eligible_sentences_not_raw_index():
    """The ordering regression, from bjornsson-2020: its abstract region runs 28
    sentences whose leading 12 are masthead and author list, with "BACKGROUND:"
    landing at index 12 — exactly at the cap. A cap on raw sentence index keeps
    only the junk and discards every finding (measured 0.24 -> 0.06); filtering
    first inverts that to 0.24 -> 0.34.

    The junk block here is one longer than `_MAX_ABSTRACT_ANCHORS` so
    cap-before-filter yields an empty abstract tier, which is the failure mode
    this pins. bjornsson-2020 is the only paper in the corpus with a junk lead
    that long, so the margin is thin in practice and worth a guard.
    """
    junk = [
        "Circulation: Genomic and Precision Medicine is available at "
        f"www.ahajournals.org/journal/circgen Circ Genom Precis Med, page {i}."
        for i in range(_MAX_ABSTRACT_ANCHORS - 2)
    ] + [
        "DOI: 10.1161/CIRCGEN.120.003029 February 2021 Correspondence to: "
        "Unnur Thorsteinsdottir, PhD, deCODE Genetics.",
        "Halldorsson, MSc; Asgeir Sigurdsson, BSc; Hakon Jonsson, PhD; "
        "Eva F. Olafsdottir, MD; Birte Kehr, PhD.",
        "© 2020 American Heart Association, Inc. For Sources of Funding and "
        "Disclosures, see page 47 of this issue.",
    ]
    real = [
        "To date, a gain-of-function mutation in LDLR with a large effect on "
        "LDL cholesterol levels has not been described.",
        "We analyzed whole-genome sequencing data from 43202 Icelanders and "
        "genotyped structural variants against that reference.",
        "We discovered a 2.5-kb deletion overlapping the 3' untranslated "
        "region of LDLR in seven heterozygous carriers.",
    ]
    assert len(junk) > _MAX_ABSTRACT_ANCHORS  # premise of the regression
    f = synthesize_fixture("bjornsson-2020-x", "", {"abstract": " ".join(junk + real)})
    kept = [it.verbalization for it in f.headline_claims
            if it.id.startswith("abstract-")]
    assert len(kept) == len(real)
    for r in real:
        assert any(r.split(",")[0][:40] in k for k in kept), r[:40]


def test_abstract_anchors_are_capped_at_the_configured_maximum():
    """Over-capture into the introduction is the common case: 42% of corpus
    abstracts hit extract_abstract's 4000-char cap, median 17 sentences / 467
    words where a real abstract is 5-12 / 150-350."""
    sents = [
        f"Finding number {i} shows that the described approach improves "
        f"downstream accuracy on the held-out evaluation split."
        for i in range(25)
    ]
    f = synthesize_fixture("stem", "", {"abstract": " ".join(sents)})
    assert len(_abs_ids(f)) == _MAX_ABSTRACT_ANCHORS


def test_short_abstract_is_not_capped_and_ids_track_source_position():
    """Ids stay pinned to the original sentence index, so an anchor id remains
    traceable back into the extracted abstract even where the filter left gaps."""
    sents = [
        "Copyright: 2019 the Authors, distributed under a Creative Commons "
        "Attribution License permitting unrestricted reuse.",
        "We introduce a graph-based caller that resolves segmental duplications "
        "missed by linear-reference pipelines.",
        "Precision improved from 0.71 to 0.93 on the benchmark truth set "
        "across all three evaluated sample preparations.",
    ]
    f = synthesize_fixture("stem", "", {"abstract": " ".join(sents)})
    # Sentence 0 is boilerplate → dropped; 1 and 2 keep their source indices.
    assert _abs_ids(f) == ["abstract-1", "abstract-2"]
