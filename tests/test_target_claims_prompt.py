"""Target-claims prompt assembly — the extractor is fed the whole substantive
paper (references excluded) up to a char budget, priority-trimmed so
Results/Discussion/captions survive when a long paper overflows.

Regression guard for the bug where a blind `full_text[:18000]` prefix + tiny
per-section caps meant the extractor never saw the deep Results table where
benchmark ratios live, and emitted only qualitative claims.
"""
from __future__ import annotations

from researchwiki.agents.phases.target_claims import (
    _allocate,
    _build_prompt,
    _rich_sections,
    render_for_author_prompt,
    TargetClaim,
    TargetClaimsOutput,
)
from researchwiki.agents.phases.draft import _build_author_prompt

BIG = 120_000


def _long_paper() -> str:
    """Synthetic PDF text: a huge Methods section pushes Results deep, with a
    benchmark number ~10K chars into Results and a limitation in Discussion.
    A References section carries a decoy number that must NOT be extracted."""
    methods_filler = "methods detail sentence. " * 2000          # ~50K chars
    results_pad = "result narrative sentence. " * 400            # ~10K chars
    return (
        "Abstract\nWe present a caller.\n\n"
        "Introduction\nBackground prose.\n\n"
        "Methods\n" + methods_filler + "\n\n"
        "Results\n" + results_pad +
        "On the benchmark VariantMedium reached AUPRC 0.918 here.\n\n"
        "Discussion\nWe note LIMITMARK indel calling is limited.\n\n"
        "References\n1. Author et al. Journal DECOY 0.404 (2020).\n"
    )


def test_deep_results_number_reaches_the_prompt():
    full = _long_paper()
    assert full.find("0.918") > 50000           # deep past the old windows
    prompt = _build_prompt({"title": "X"}, {}, full, BIG)
    assert "0.918" in prompt and "AUPRC" in prompt


def test_discussion_limitation_reaches_the_prompt():
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), BIG)
    assert "LIMITMARK" in prompt


def test_references_are_excluded():
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), BIG)
    assert "DECOY" not in prompt and "0.404" not in prompt
    assert "References" not in prompt


def test_budget_trims_low_priority_first():
    """Under a tight budget, Results/Discussion survive in full and low-density
    Methods is trimmed to whatever budget is left over — never the reverse."""
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), 12000)
    assert "0.918" in prompt          # Results number kept
    assert "LIMITMARK" in prompt      # Discussion kept
    # Methods gets only leftover budget, not its full ~2000 sentences.
    assert prompt.count("methods detail sentence") < 100


def test_allocate_orders_by_priority_and_respects_budget():
    rich = {"results": "R" * 5000, "discussion": "D" * 5000,
            "methods": "M" * 5000, "references": "X" * 5000}
    got = _allocate(rich, 8000)
    assert len(got.get("results", "")) == 5000       # highest priority, full
    assert len(got.get("discussion", "")) == 3000    # partial — budget ran out
    assert "methods" not in got                       # nothing left
    assert "references" not in got                    # never included


def test_rich_sections_falls_back_without_full_text():
    fallback = {"results": "precapped results"}
    assert _rich_sections(fallback, None, cap=BIG) is fallback


def test_critical_render_header_demands_verbatim_numbers():
    out = TargetClaimsOutput(claims=[
        TargetClaim(type="headline", content="AUPRC 0.918",
                    importance="critical", location="Results"),
    ])
    block = render_for_author_prompt(out)
    assert "verbatim" in block.lower() and "numeric" in block.lower()
    assert "0.918" in block


def _unhealthy_paper() -> str:
    """No findings heading; a giant Introduction pushes the result to EOF."""
    return (
        "Abstract\nWe present a ranker.\n\n"
        "Introduction\n" + "background-only prose. " * 2200
        + "\nLATE_RESULT GenRec reduces the token budget to one third.\n"
        "References\nREFERENCE_DECOY should never reach a prompt.\n"
    )


def test_unhealthy_target_context_is_stratified_not_head_only():
    prompt = _build_prompt({"title": "X"}, {}, _unhealthy_paper(), 6000)
    assert "document-stratified fallback" in prompt
    assert "LATE_RESULT" in prompt
    assert "REFERENCE_DECOY" not in prompt


def test_unhealthy_author_context_is_stratified_not_head_only():
    prompt = _build_author_prompt(
        {"title": "X"}, {}, [], pdf_full_text=_unhealthy_paper(),
    )
    assert "document-stratified fallback" in prompt
    assert "LATE_RESULT" in prompt
    assert "REFERENCE_DECOY" not in prompt
