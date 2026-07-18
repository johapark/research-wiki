"""Source-derived keyword extraction (Slice 2 Option B).

`propose_keywords` was previously draft-derived: keywords described what the
*wiki page* covered, which inherited any omissions the draft made. The
RUFUS gap on Jang 2025 was the motivating case — a primary somatic variant
caller named 25× in the source PDF that the draft omitted, and therefore the
keyword list also omitted, with no structural signal exposing the gap.

The new contract feeds source sections (abstract + Introduction / Results /
Discussion — Methods is excluded, as its reagent catalogs and reporting-summary
checklists pollute the keyword signal) as the primary input. Keywords reflect the *paper's* central
content, and the body-coverage log line in the runner surfaces any source→body
asymmetry. These tests exercise the input-selection logic and the runner's
gap detector — the actual LLM behavior is covered by the smoke test against
Jang 2025.
"""

from __future__ import annotations

from researchwiki.agents.phases.commit import (
    KeywordsOutput,
    _build_keyword_input,
    propose_keywords,
)
from researchwiki.agents.runner import _keyword_body_gaps


# ---------- _build_keyword_input ----------


def test_full_pdf_text_is_tier1_preferred_path():
    """When full_pdf_text is available, it's used over the (truncated) sections.
    Critical because pre-truncated sections may have already cut the named
    entities we want to surface (e.g., RUFUS past the 4000-char Results cap).
    Uses a Results section — Methods is deliberately excluded from keyword
    extraction (see `_build_keyword_input` / `_SECTION_BUDGETS`)."""
    full_text = (
        "Results\n\n"
        "We benchmarked Mutect2 in three modes. " * 200  # ~7000 chars of preamble
        + "Calling somatic mutations using RUFUS\n"
        + "RUFUS was run in paired mode with 25-bp k-mers. "
    )
    sections = {"results": full_text[:4000]}  # pre-truncated, doesn't include RUFUS
    assert "RUFUS" not in sections["results"]
    excerpt, label = _build_keyword_input(
        sections=sections, draft_text="(unused)", full_pdf_text=full_text,
    )
    assert label == "Source PDF excerpts (abstract + main text)"
    assert "RUFUS" in excerpt   # the wide-window re-anchor surfaces it


def test_sections_used_when_full_text_absent():
    sections = {
        "methods": "We compared Mutect2 and RUFUS at 1,360× coverage.",
        "results": "RUFUS detected 0.3% VAF mutations; Mutect2 missed them.",
        "introduction": "Somatic mutation calling at sub-1% VAF is hard.",
    }
    excerpt, label = _build_keyword_input(
        sections=sections, draft_text="(unused)", full_pdf_text=None,
    )
    assert label == "Source PDF excerpts (abstract + main text)"
    assert "RUFUS" in excerpt
    assert "Mutect2" in excerpt
    # Methods is excluded from keyword extraction; Introduction / Results are pulled.
    assert "## Methods" not in excerpt
    assert "## Introduction" in excerpt
    assert "## Results" in excerpt


def test_section_budget_caps_per_section():
    long_methods = "Mutect2 " * 3000  # ~24000 chars; methods cap is 12000
    excerpt, _ = _build_keyword_input(
        sections={"methods": long_methods}, draft_text="(x)", full_pdf_text=None,
    )
    # Methods cap 12000 + header overhead.
    assert len(excerpt) <= 12200


def test_falls_back_to_draft_when_sections_empty():
    excerpt, label = _build_keyword_input(
        sections={},
        draft_text="## Summary\nA paper about RUFUS and somatic calling.",
        full_pdf_text=None,
    )
    assert "fallback" in label.lower()
    assert "RUFUS" in excerpt


def test_falls_back_to_draft_when_all_inputs_empty_or_none():
    excerpt, label = _build_keyword_input(
        sections=None,
        draft_text="## Summary\nA paper about RUFUS and somatic calling.",
        full_pdf_text=None,
    )
    assert "fallback" in label.lower()
    assert "RUFUS" in excerpt


def test_fallback_when_all_section_bodies_empty():
    """A sections dict with empty string values should still trigger fallback —
    not produce a vacuous source-mode prompt with no content."""
    excerpt, label = _build_keyword_input(
        sections={"methods": "", "results": "  "},
        draft_text="## Summary\nFallback content.",
        full_pdf_text=None,
    )
    assert "fallback" in label.lower()
    assert "Fallback" in excerpt


# ---------- propose_keywords stub mode (preserves existing behavior) ----------


def test_stub_mode_returns_empty():
    out = propose_keywords(
        metadata={"title": "T", "year": 2025},
        draft_text="(any)",
        sections={"methods": "stuff"},
        full_pdf_text=None,
        use_stub=True,
    )
    assert out.keywords == []
    assert isinstance(out, KeywordsOutput)


# ---------- runner._keyword_body_gaps ----------


def test_body_gaps_finds_keyword_absent_from_body():
    keywords = ["RUFUS", "Mutect2", "GATK PoN"]
    body = "## Summary\nWe used Mutect2 in three modes against the GATK panel of normals."
    missing = _keyword_body_gaps(keywords, body)
    # "RUFUS" should be flagged (body never names it); "Mutect2" present;
    # "GATK PoN" matches via first-token "GATK" so not flagged.
    assert "RUFUS" in missing
    assert "Mutect2" not in missing
    assert "GATK PoN" not in missing


def test_body_gaps_case_insensitive():
    body = "We applied bqsr after the calling step."
    missing = _keyword_body_gaps(["BQSR"], body)
    assert missing == []


def test_body_gaps_handles_multi_word_keywords():
    """First-token matching is the rule — handles paraphrase / abbreviation."""
    body = "The PairHMM realignment is computed per haplotype."
    # 'PairHMM realignment' first token is 'PairHMM' which is in the body.
    missing = _keyword_body_gaps(["PairHMM realignment"], body)
    assert missing == []


def test_body_gaps_strips_punctuation_from_first_token():
    body = "We use the FilterMutectCalls tool."
    missing = _keyword_body_gaps(["FilterMutectCalls,"], body)  # trailing comma
    assert missing == []


def test_body_gaps_empty_keyword_list():
    assert _keyword_body_gaps([], "any body text") == []


def test_body_gaps_empty_keyword_string_dropped():
    """A keyword that's just whitespace shouldn't crash or appear in the gap list."""
    missing = _keyword_body_gaps(["   ", "RUFUS"], "no relevant content")
    assert missing == ["RUFUS"]
