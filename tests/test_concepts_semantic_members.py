"""Tests for the semantic recall tier on concept-hub member discovery.

Behaviour contract: the semantic pass *proposes*, it never decides. Cosine
alone cannot carry membership — on the calibration case every true member
outranked the first false positive by 0.003, and a floor low enough to admit
them admits 31 distinct papers for a term like "ATAC-seq". So the pass
contributes a reviewed candidate list plus alias suggestions mined from the
matching claim text, and `find_members` stays untouched.

See researchwiki/concepts/semantic_members.py and PLAN-bottom-up-synthesis.md
(evidence item E1/E2) for the failure this exists to fix.
"""

from __future__ import annotations

from researchwiki.concepts.semantic_members import (
    SemanticCandidate,
    _head_token,
    _suggest_alias,
    suggested_alias_set,
)


def _cand(stem: str, score: float, alias: str | None) -> SemanticCandidate:
    return SemanticCandidate(
        stem=stem, category="ai", claim_slug="kc-0000dead",
        section="key_contributions", text="…", score=score,
        suggested_alias=alias,
    )


# ---------- head-token selection ----------

def test_head_token_picks_the_distinguishing_word_not_the_generic_head():
    # "model"/"mechanism" are the grammatical heads and the useless ones.
    assert _head_token("mixture model") == "mixture"
    assert _head_token("attention mechanism") == "attention"
    assert _head_token("chromatin accessibility") == "accessibility"


def test_head_token_empty_on_punctuation_only_term():
    assert _head_token("---") == ""
    assert _head_token("") == ""


# ---------- alias mining: the real corpus cases ----------

def test_mines_parenthesized_acronym_when_expansion_mentions_the_concept():
    # The sarthi-2024 claim that lexical matching missed.
    text = ("Implemented soft clustering with Gaussian Mixture Models (GMMs) "
            "and UMAP-based dimensionality reduction.")
    assert _suggest_alias("mixture model", text, {"mixture model"}) == "GMM"


def test_ignores_acronym_whose_expansion_is_unrelated_to_the_concept():
    # Regression: this returned "HSIC" — a parenthetical that merely shared a
    # sentence with a semantically-near claim is not an alias for the term.
    text = ("Employs Hilbert-Schmidt Independence Criterion (HSIC) to enforce "
            "statistical independence among learned latent variables.")
    assert _suggest_alias("mixture model", text, {"mixture model"}) is None


def test_mines_qualifier_bigram():
    # The van-iterson-2017 claim — the page's central disagreement, and
    # invisible to a literal search for "mixture model".
    text = ("BACON models observed association z-statistics as a "
            "three-component normal mixture estimated by Gibbs sampling.")
    assert _suggest_alias("mixture model", text, {"mixture model"}) == "normal mixture"


def test_never_suggests_an_alias_the_caller_already_has():
    text = "A two-Gaussian mixture model on log-transformed counts."
    known = {"mixture model", "gaussian mixture"}
    got = _suggest_alias("mixture model", text, known)
    assert got not in ("gaussian mixture", "mixture model")


def test_stopword_qualifiers_are_not_aliases():
    text = "The mixture is estimated from the data using the mixture weights."
    assert _suggest_alias("mixture model", text, {"mixture model"}) is None


def test_no_alias_when_head_token_absent():
    text = "Compares specificity-improvement strategies as complementary trade-offs."
    assert _suggest_alias("mixture model", text, {"mixture model"}) is None


# ---------- paste-ready alias list ----------

def test_alias_set_dedupes_and_preserves_rank_order():
    cands = [
        _cand("a-2020", 0.78, "normal mixture"),
        _cand("b-2021", 0.76, "GMM"),
        _cand("c-2022", 0.74, "normal mixture"),   # duplicate
        _cand("d-2023", 0.72, None),
    ]
    assert suggested_alias_set(cands) == ["normal mixture", "GMM"]


def test_alias_set_is_capped_so_tail_noise_stays_out_of_the_paste_line():
    # An alias mined from a false-positive paper is a false alias; the
    # per-candidate suggestion stays visible, the paste-ready list does not
    # inherit the whole tail.
    cands = [_cand(f"p-{i}", 0.80 - i * 0.01, f"alias{i}") for i in range(9)]
    assert suggested_alias_set(cands, top=3) == ["alias0", "alias1", "alias2"]


def test_alias_set_empty_when_nothing_mined():
    assert suggested_alias_set([_cand("a-2020", 0.75, None)]) == []
    assert suggested_alias_set([]) == []


# ---------- an empty substrate must announce itself ----------

def test_zero_claims_logs_the_cause_instead_of_returning_quietly(monkeypatch, capsys):
    # A migrated corpus whose H2 headings don't match the extractor has no
    # claims at all; silence would read as "no candidates found".
    import researchwiki.concepts.semantic_members as sm
    monkeypatch.setattr(sm, "_contribution_claims", lambda: [])
    assert sm.semantic_member_candidates("mixture model") == []
    err = capsys.readouterr().err
    assert "no contribution claims" in err
    assert "zero_claim_papers" in err
