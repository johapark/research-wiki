"""Critic phase — recall signal (missed salience anchors), not just precision.

Every `ClaimDetail.is_weak()` predicate is a precision test: it can only flag a
claim that is already *on* the page. So a draft that simply omitted the paper's
load-bearing content produced no revision signal, the critic returned "no weak
claims", and the runner broke out of the evolve loop — most often on exactly the
pages with the worst coverage (on a live-corpus sample, drafts under 0.5
salience got a silent critic 82% of the time vs 60% above it). Writing less was
rewarded with less critique.

These tests pin the fix: eligible critical misses reach the critic, and they
gate the evolve loop alongside weak claims. The eligibility filter carries most
of the risk — a coverage gap becomes an *additive* instruction, so a bad anchor
makes the page worse rather than merely unchanged. Each filter case below is a
shape observed in the real corpus.
"""

from types import SimpleNamespace

import pytest

from researchwiki.agents.phases.revise import (
    _COVERAGE_GAP_TRIGGER,
    _build_critic_prompt,
    coverage_gaps,
    critic,
)


def _anchor(text, *, importance="critical", id="abstract-1", axis="headline_claims"):
    return {"axis": axis, "id": id, "importance": importance, "text": text}


# A substantive abstract sentence — the shape that SHOULD reach the critic.
# Taken from the beisel-2026 page, whose four critical misses were all real.
_REAL = (
    "While both studies focused on AsCas12a, the ability to use the DNA guide "
    "extended to other orthologs, suggesting that this is a general property of "
    "the Cas12a family rather than a quirk of one enzyme."
)
_REAL_2 = (
    "The authors report that guide DNA supported editing at efficiencies "
    "comparable to guide RNA across the tested loci, with no detectable loss of "
    "specificity in the accompanying off-target assay."
)


def _draft(anchors, claim_details=()):
    return SimpleNamespace(
        scores={"missed_anchors": list(anchors)},
        claim_details=list(claim_details),
        text="## Key Contributions\n- something\n",
    )


# ---------- eligibility filter ----------

def test_substantive_critical_anchor_is_eligible():
    assert coverage_gaps({"missed_anchors": [_anchor(_REAL)]}) == [_anchor(_REAL)]


def test_non_critical_anchors_dropped():
    """Only `critical` (= abstract-sentence) anchors qualify. Figure captions and
    extended-data anchors are legitimately skippable, so their absence isn't a
    defect worth an evolve round."""
    anchors = [
        _anchor(_REAL, importance="high", axis="capabilities"),
        _anchor(_REAL, importance="normal", axis="capabilities"),
    ]
    assert coverage_gaps({"missed_anchors": anchors}) == []


@pytest.mark.parametrize("text", [
    # arnold-2026 (an opinion piece): the "abstract" region is body prose, so
    # rhetorical asides surface as critical misses.
    "We can't measure everything.",
    "This is where we actually spend most of our time.",
])
def test_short_rhetorical_fragments_dropped(text):
    assert coverage_gaps({"missed_anchors": [_anchor(text)]}) == []


@pytest.mark.parametrize("text", [
    # eddy-2011: all three top critical misses were journal front-matter.
    "PLoS Comput Biol 7(10): e1002195. doi:10.1371/journal.pcbi.1002195 "
    "Editor: William R. Pearson, University of Virginia, United States",
    "Received April 27, 2011; Accepted July 29, 2011; Published October 20, "
    "2011 Copyright: 2011 Eddy. This is an open-access article distributed "
    "under the terms of the Creative Commons Attribution License",
    "HHMI had no role in study design, data collection and analysis, decision "
    "to publish, or preparation of the manuscript.",
    "Competing interests: The author has declared that no competing interests "
    "exist in relation to the work described in this manuscript.",
])
def test_journal_front_matter_dropped(text):
    """Front-matter sits in the abstract region so it scores as a critical miss,
    but a page is *right* to omit it. Instructing the author to cover a funding
    disclaimer would actively damage the page."""
    assert coverage_gaps({"missed_anchors": [_anchor(text)]}) == []


def test_lowercase_initial_fragment_dropped():
    """benegas-2025: a sentence-splitter artifact starting mid-clause. The
    "missing" text is a fragment of a sentence the page may already cover."""
    text = (
        "near zero as recall increases, which indicates that the classifier "
        "retains little discriminative power in the high-recall regime."
    )
    assert coverage_gaps({"missed_anchors": [_anchor(text)]}) == []


def test_digit_initial_anchor_kept():
    text = (
        "17 of the 24 tested variants showed a measurable shift in editing "
        "efficiency relative to the wild-type control."
    )
    assert coverage_gaps({"missed_anchors": [_anchor(text)]}) == [_anchor(text)]


def test_missing_or_malformed_scores_are_tolerated():
    for scores in (None, {}, {"missed_anchors": None}, {"missed_anchors": []}):
        assert coverage_gaps(scores) == []
    # A row with no text at all must not raise on the `text[0]` lookahead.
    assert coverage_gaps({"missed_anchors": [{"importance": "critical"}]}) == []


# ---------- trigger threshold ----------

def test_trigger_is_two():
    """Pinned deliberately: the value is calibrated (44-page sample, seed 7) and
    changing it changes how often ingest spends an extra evolve+grade round."""
    assert _COVERAGE_GAP_TRIGGER == 2


def test_single_gap_alone_does_not_fire(monkeypatch):
    """One uncovered anchor is as likely to be an extraction artifact as a real
    omission — not worth a round on its own."""
    calls = []
    monkeypatch.setattr(
        "researchwiki.agents.phases.revise.llm.call",
        lambda **kw: calls.append(kw) or SimpleNamespace(
            text="x", model="m", input_tokens=1, output_tokens=1),
    )
    out = critic(draft=_draft([_anchor(_REAL)]), metadata={})
    assert calls == []
    assert out.coverage_gaps == []
    assert out.model == "(skipped)"


def test_two_gaps_fire_with_zero_weak_claims(monkeypatch):
    """The core regression: before the fix this returned "no weak claims" and the
    runner broke out of the evolve loop without ever revising."""
    calls = []
    monkeypatch.setattr(
        "researchwiki.agents.phases.revise.llm.call",
        lambda **kw: calls.append(kw) or SimpleNamespace(
            text="- add-to Key Contributions: ...", model="m",
            input_tokens=10, output_tokens=5),
    )
    gaps = [_anchor(_REAL, id="abstract-4"), _anchor(_REAL_2, id="abstract-9")]
    out = critic(draft=_draft(gaps), metadata={})

    assert len(calls) == 1, "critic must call the LLM on a coverage gap"
    assert out.weak_claims == []
    assert len(out.coverage_gaps) == 2
    assert out.model == "m"


def test_sub_trigger_gap_rides_along_when_a_claim_is_weak(monkeypatch):
    """The critic is already firing, so a lone gap costs nothing extra to
    include and the author may as well see it."""
    monkeypatch.setattr(
        "researchwiki.agents.phases.revise.llm.call",
        lambda **kw: SimpleNamespace(text="x", model="m", input_tokens=1, output_tokens=1),
    )
    weak = SimpleNamespace(
        section="Key Contributions", position=1, text="a claim",
        bm25=2.0, semantic=0.20, negation_mismatch=False, numeric_unmatched=[],
        is_weak=lambda: True,
    )
    out = critic(draft=_draft([_anchor(_REAL)], claim_details=[weak]), metadata={})
    assert len(out.weak_claims) == 1
    assert len(out.coverage_gaps) == 1


def test_ineligible_anchors_never_fire(monkeypatch):
    """Two front-matter anchors clear the count but not the filter."""
    calls = []
    monkeypatch.setattr(
        "researchwiki.agents.phases.revise.llm.call",
        lambda **kw: calls.append(kw) or SimpleNamespace(
            text="x", model="m", input_tokens=1, output_tokens=1),
    )
    anchors = [
        _anchor("Received April 27, 2011; Accepted July 29, 2011; Published "
                "October 20, 2011 Copyright: 2011 Eddy.", id="abstract-13"),
        _anchor("HHMI had no role in study design, data collection and "
                "analysis, decision to publish.", id="abstract-17"),
    ]
    out = critic(draft=_draft(anchors), metadata={})
    assert calls == []
    assert out.coverage_gaps == []


# ---------- prompt ----------

def test_prompt_carries_gaps_and_triage_instruction():
    prompt = _build_critic_prompt([], "draft body", {}, [_anchor(_REAL)])
    assert "Coverage gaps" in prompt
    assert _REAL[:60] in prompt
    # The critic must be told to triage rather than obey — the anchors are
    # extracted structurally and some are not findings at all.
    assert "Triage" in prompt
    # With no weak claims the flags block must say so explicitly rather than
    # sitting empty under its header.
    assert "(none" in prompt


def test_prompt_omits_gap_block_when_there_are_none():
    prompt = _build_critic_prompt([], "draft body", {}, [])
    assert "Coverage gaps" not in prompt
