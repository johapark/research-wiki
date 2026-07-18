"""Pure-logic tests for the structural-conformance ('coherence') scorer.

Operates on synthetic draft strings — no PDF, no LLM, no DB. Verifies the
paper-page contract checks fire on the right shapes.
"""

from researchwiki.grade.coherence import (
    PAPER_WEIGHTS,
    CoherenceReport,
    score_coherence,
)


# A synthetic well-formed paper-page body that hits every contract check.
# Section bodies are prose to satisfy "no empty section bodies"; bullets
# in Key Contributions and Limitations are substantive (≥6 / ≥10 words).
_WELL_FORMED = """\
## Summary

The paper introduces a new method that achieves substantial improvements
over prior work, demonstrated across three benchmarks with clear ablation
analysis. Numbers and named comparators are reported throughout.

## Key Contributions

- Introduces a novel architecture combining attention with structural
  inductive bias for the protein design task.
- Demonstrates state-of-the-art performance on the three established
  benchmarks used by the field.
- Releases code, model weights, and evaluation harness publicly to
  facilitate independent reproduction.
- Provides extensive ablations isolating the contribution of each
  architectural choice for downstream practitioners.

## Methodology and Architecture

The model is a transformer with structural conditioning. Inputs are
encoded as discrete tokens with positional embeddings derived from the
backbone geometry. Training uses the standard cross-entropy objective
plus an auxiliary reconstruction term.

## Results

Performance is reported on three benchmarks. The main result table shows
consistent improvements over prior baselines, with the largest gains on
the hardest split.

## Limitations

- The method requires a pretrained backbone encoder, which limits
  deployment in low-compute settings without access to GPU clusters.
- Generalization across non-canonical amino acids has not been
  experimentally validated and may require additional retraining.

## Related Papers

- [[compbio/jumper-2021-highly-accurate-protein-structure-prediction]]
- [[compbio/abramson-2024-accurate-structure-prediction-of-biomolecular]]
""" + "\nFiller paragraph here. " * 200   # pad word count above the 600-word floor


def test_well_formed_paper_scores_one():
    r = score_coherence(_WELL_FORMED, page_type="paper")
    assert isinstance(r, CoherenceReport)
    assert r.score == 1.0
    assert r.violations == []
    assert "Summary" in r.sections_present
    assert "Limitations" in r.sections_present


def test_missing_limitations_drops_two_weights():
    body = _WELL_FORMED.replace("## Limitations\n\n- The method requires"
                                " a pretrained backbone encoder, which limits\n"
                                "  deployment in low-compute settings without"
                                " access to GPU clusters.\n"
                                "- Generalization across non-canonical amino"
                                " acids has not been\n"
                                "  experimentally validated and may require"
                                " additional retraining.", "")
    r = score_coherence(body, page_type="paper")
    # Lost: sections_present (0.30), limitations_substantive (0.10).
    expected = 1.0 - PAPER_WEIGHTS["sections_present"] - PAPER_WEIGHTS["limitations_substantive"]
    assert abs(r.score - expected) < 1e-6
    assert any("limitations" in v.lower() for v in r.violations) or \
           any("missing sections" in v.lower() for v in r.violations)


def test_one_paragraph_collapse_fails_word_count_and_sections():
    body = "## Summary\n\nThe paper introduces a method.\n"
    r = score_coherence(body, page_type="paper")
    # Loses sections_present, word_count_in_band, key_contribs, limitations,
    # no_empty_sections (Summary body itself is non-empty so this still
    # passes — there are no other sections to be empty), wikilink_density.
    # Result: only `no_empty_sections` (0.15) passes.
    assert r.score < 0.5
    assert any("missing sections" in v.lower() for v in r.violations)
    assert any("word count" in v.lower() for v in r.violations)


def test_wikilink_stuffing_fails_density_check():
    """A draft that pads outside-of-Related-Papers prose with wikilinks
    should fail the density check."""
    # 12 wikilinks across ~150 words outside Related Papers → 1 per ~12
    # words, well above the 1-per-40 cap.
    stuffed = (
        "## Summary\n\n"
        + " ".join([f"see [[a/p-{i}-x]]" for i in range(12)])
        + "\n\n## Key Contributions\n\n"
        + "- Introduces a novel architecture combining attention with "
          "structural inductive bias for protein design.\n"
        + "- Demonstrates state-of-the-art performance on three benchmarks.\n"
        + "- Releases code, weights, and evaluation harness publicly.\n"
        + "\n## Methodology and Architecture\n\n"
        + "The model is a transformer with structural conditioning.\n"
        + "\n## Results\n\nPerformance is reported on benchmarks.\n"
        + "\n## Limitations\n\n- The method requires a pretrained backbone "
          "encoder which limits deployment.\n"
        + "\n## Related Papers\n\n- [[compbio/abramson-2024-x]]\n"
        + "Filler. " * 200
    )
    r = score_coherence(stuffed, page_type="paper")
    assert any("density" in v.lower() for v in r.violations)
    # Score should be 1.0 - 0.10 = 0.90 if only density fails; allow
    # slightly less if word-count band is also tight.
    assert r.score < 1.0


def test_unknown_page_type_returns_neutral_pass():
    """v1 only contracts paper pages; other types return 1.0 so the
    rest of the pipeline doesn't have to branch."""
    r = score_coherence("any body text", page_type="synthesis")
    assert r.score == 1.0
    assert r.violations == []


def test_key_contributions_under_three_bullets_fails():
    """Three bullets are required; two should fail the check."""
    body = """\
## Summary

The paper introduces a method that achieves improvements over prior work
across multiple benchmarks with clear ablation analysis.

## Key Contributions

- Introduces a novel architecture combining attention with structural
  inductive bias for protein design.
- Demonstrates state-of-the-art performance on three benchmarks.

## Methodology and Architecture

The model is a transformer with structural conditioning.

## Results

Performance is reported on three benchmarks.

## Limitations

- The method requires a pretrained backbone encoder which limits
  deployment in low-compute settings.

## Related Papers

- [[compbio/x-2024-y]]
""" + "\nFiller paragraph " * 60
    r = score_coherence(body, page_type="paper")
    assert any("Key Contributions" in v for v in r.violations)
