"""Revision phases — critic, evolve, debug.

These three phases all consume a `Draft` plus grader output and produce a
revised draft. They share enough scaffolding (LLM call, prompt assembly,
output dataclass shape) that grouping them in one module keeps the runner's
import surface tight.

Naming note: the runner's `evolve` phase here is the *agent's revision step*
in response to critic notes — distinct from the "memory evolution" step
(cross-page edits at ingest), which lives in its own module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...grade.salience import anchor_is_substantive
from .. import llm, prompt_lib
from .draft import Draft
from .grade import ClaimDetail


# ---------- critic ----------

@dataclass
class CritiqueOutput:
    """Result of a critic call. notes is the user-facing message; weak_claims
    is the deterministic list of claims the grader flagged as weak, and
    coverage_gaps the eligible load-bearing PDF anchors the draft omitted.
    Together they are the source of truth for whether to skip evolve — the
    first covers precision defects, the second recall defects."""
    notes: str
    weak_claims: list[ClaimDetail]
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    coverage_gaps: list[dict] = field(default_factory=list)


# Minimum eligible critical misses before a coverage gap alone justifies an
# evolve round. Calibrated on a 44-page random sample of the live corpus
# (seed 7), counting post-filter: at 2, the trigger fires on 39% of pages,
# catches all 7 scoring below 0.55 salience (median salience 0.58 among firing
# pages vs 0.72 among quiet ones), and trips only 1 of the 12 healthiest pages
# (salience >= 0.75). At 1 it fires on ~60%, mostly on pages whose single miss
# is an extraction artifact; at 3 it misses genuine cases like hickey-2023 at
# 0.54. A false fire costs one evolve+grade round but cannot damage the page:
# `fitness.is_evolve_improvement` still gates whether the revision is kept.
_COVERAGE_GAP_TRIGGER = 2


def coverage_gaps(scores: dict | None) -> list[dict]:
    """Eligible missed salience anchors from a graded draft's score dict.

    The grader measures recall (`salience_score`) but every `ClaimDetail.
    is_weak()` predicate is a *precision* test — it can only flag claims that
    are already on the page. So an omission used to produce no revision signal
    at all, and the evolve loop broke out on "no weak claims" precisely on the
    pages with the worst coverage: on a live-corpus sample, drafts scoring
    below 0.5 salience got a silent critic 82% of the time versus 60% above
    it. Writing less was rewarded with less critique.

    Only `critical` anchors qualify. In the synthetic fixture those are exactly
    the abstract sentences (`grade/salience.py`), which is the one tier where
    "the page doesn't mention this" is a defect rather than a choice — figure
    captions and extended-data anchors are legitimately skippable. The list
    `_missed_anchors` returns is already sorted critical-first and truncated to
    `_TOP_K_MISSED`, so counting criticals in it decides the trigger without
    needing a wider report.

    Filtering is deliberately conservative: these anchors become *additive*
    instructions, and a bad one makes the page worse rather than merely
    unchanged. The LLM triages what survives (see `_CRITIC_SYSTEM`); this pass
    only removes shapes no author should ever be told to cover.

    The substance test itself lives in `grade.salience` — the same rule now
    filters the fixture at synthesis time, so a shape that can't earn credit in
    the denominator also can't reach the author as an instruction. Anchors from
    pages graded before that landed still carry the junk, which is why this
    filter runs on read as well.
    """
    anchors = (scores or {}).get("missed_anchors") or []
    return [m for m in anchors if _anchor_is_eligible(m)]


def _anchor_is_eligible(m: dict) -> bool:
    """One anchor's eligibility as an additive revision instruction."""
    if m.get("importance") != "critical":
        return False
    return anchor_is_substantive(m.get("text") or "")


def critic(
    *,
    draft: Draft,
    metadata: dict,
    use_stub: bool = False,
) -> CritiqueOutput:
    """Critic phase — identify weak claims and coverage gaps in a winning draft.

    The deterministic grader has already flagged weak claims (low semantic
    similarity, negation mismatch, or numeric drift) and the load-bearing PDF
    anchors the draft never covered. The critic's job is to translate that
    mechanical signal into actionable revision notes the author phase can
    consume on the next pass. We deliberately give the critic the FACTS (which
    claim, what the grader said) rather than asking it to discover weaknesses
    on its own — that keeps the LLM accountable to the grader's evidence rather
    than inventing concerns.

    Coverage gaps are the one place the critic exercises judgment rather than
    translation: the anchors are extracted structurally, so some are
    front-matter or rhetorical asides that a page is right to omit. The critic
    triages them and stays silent on the ones that aren't real findings.
    """
    weak = [c for c in draft.claim_details if c.is_weak()]
    gaps = coverage_gaps(draft.scores)
    if len(gaps) < _COVERAGE_GAP_TRIGGER:
        # Below the trigger, gaps ride along only when the critic is already
        # firing for weak claims — one uncovered anchor is as likely to be an
        # extraction artifact as a real omission, and isn't worth a round on
        # its own.
        gaps = gaps if weak else []
    if not weak and not gaps:
        return CritiqueOutput(
            notes="no weak claims or coverage gaps detected; draft passes grader thresholds",
            weak_claims=[],
            model="(skipped)",
            input_tokens=0,
            output_tokens=0,
            coverage_gaps=[],
        )

    prompt = _build_critic_prompt(weak, draft.text, metadata, gaps)
    resp = llm.call(
        phase="critic",
        prompt=prompt,
        system=_CRITIC_SYSTEM,
        use_stub=use_stub,
    )
    return CritiqueOutput(
        notes=resp.text,
        weak_claims=weak,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
        cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
        coverage_gaps=gaps,
    )


_CRITIC_SYSTEM = """\
You are a critic for a research-paper wiki. The grader gives you two kinds of
evidence about a draft page, and they pull in opposite directions.

## Grader flags — claims that ARE on the page but are weakly supported
Flags include: low semantic similarity to the retrieved evidence chunks,
numeric values not present in the PDF, and asserted negations the source
doesn't echo. For each flagged claim, output one bullet:
- Quote the problematic phrase from the claim verbatim.
- Say whether to (a) remove the claim, (b) soften / rephrase to match what the
  PDF actually supports, or (c) replace the unmatched number with a value
  that does appear in the PDF.

Do NOT introduce new concerns about these claims that the grader didn't flag.
On this axis the grader is the source of truth; you are its translator.

## Coverage gaps — sentences from the PDF the page never covered
These are pulled structurally from the paper's own abstract, so the extraction
is imperfect and you MUST triage before instructing anything. Stay silent on a
gap when it is:
- journal front-matter (editors, submission dates, copyright, funding
  disclaimers, competing-interest statements);
- a rhetorical aside, quotation, or narrative sentence rather than a finding
  (common in commentary and opinion pieces);
- a fragment of a sentence, or content the draft already states in other words.

For a gap that IS a real, checkable finding the page omits, output one bullet:
- Name where in the draft it belongs (which section).
- State the finding to add in one clause, grounded in the anchor's wording.
- Never invent numbers or entities the anchor doesn't contain.

Emitting nothing for the coverage-gap block is a correct and expected outcome.
Adding filler to satisfy a bad anchor is worse than leaving the gap.

Keep notes terse — the author re-reads the draft, not your prose.
"""


def _build_critic_prompt(
    weak: list[ClaimDetail],
    draft_text: str,
    metadata: dict,
    gaps: list[dict] | None = None,
) -> str:
    parts = [
        "# Draft (unchanged after tournament)",
        draft_text[:4000],
        "",
        "# Grader flags (ordered by section + position)",
    ]
    if not weak:
        parts.append("(none — no claim on the page is weakly supported)")
    for c in weak:
        sem_s = f"{c.semantic:.2f}" if c.semantic is not None else "n/a"
        line = f"- [{c.section}#{c.position}] (sem={sem_s}, BM25={c.bm25:.1f}"
        if c.negation_mismatch:
            line += ", negation mismatch"
        if c.numeric_unmatched:
            line += f", numeric drift: {c.numeric_unmatched}"
        line += ")"
        parts.append(line)
        parts.append(f"  text: {c.text[:300]}")
    if gaps:
        parts.extend([
            "",
            "# Coverage gaps — PDF sentences the draft never covered",
            "Triage each: skip front-matter, asides, fragments, and anything "
            "the draft already says. Instruct only on real omitted findings.",
        ])
        for g in gaps:
            parts.append(f"- [{g.get('id', '?')}] ({g.get('axis', '?')})")
            parts.append(f"  text: {(g.get('text') or '')[:400]}")
    parts.extend([
        "",
        "# Output format",
        "One bullet per flagged claim: quote the problem phrase verbatim, say "
        "remove / soften-to / replace-with. Then one bullet per coverage gap "
        "you judged real: say add-to <section> and what to add. No prose intro "
        "or outro.",
    ])
    return "\n".join(parts)


# ---------- evolve ----------

@dataclass
class EvolveOutput:
    """Result of an evolve call — a revised draft consuming critic notes."""
    text: str
    model: str
    temperature: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def evolve(
    *,
    draft: Draft,
    critique: CritiqueOutput,
    metadata: dict,
    sections: dict,
    use_stub: bool = False,
) -> EvolveOutput:
    """Evolve phase — produce a revised draft addressing the critic's notes."""
    prompt = _build_evolve_prompt(draft.text, critique.notes, metadata, sections)
    paper_type = (metadata or {}).get("paper_type")
    resp = llm.call(
        phase="evolve",
        prompt=prompt,
        system=prompt_lib.load_author_system(paper_type),
        use_stub=use_stub,
    )
    return EvolveOutput(
        text=resp.text,
        model=resp.model,
        temperature=resp.temperature,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
        cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
    )


def _build_evolve_prompt(prior_draft: str, critic_notes: str, metadata: dict, sections: dict) -> str:
    return "\n".join([
        "# Prior draft (passed tournament but the critic flagged issues)",
        prior_draft[:4000],
        "",
        "# Critic's revision notes",
        critic_notes,
        "",
        "# PDF excerpts (use these to ground revisions)",
        "## Methods",
        sections.get("methods", "")[:2000],
        "## Results",
        sections.get("results", "")[:2000],
        "## Discussion",
        sections.get("discussion", "")[:1200],
        "",
        "---",
        "",
        "Output the FULL revised draft — Summary, Key Contributions, Results — "
        "addressing each critic note. Keep claims the critic did not flag "
        "essentially unchanged. Format identical to the prior draft.",
    ])


# ---------- debug ----------

@dataclass
class DebugOutput:
    """Result of a debug call — repaired draft addressing structural gate
    failures (numeric drift, too few KC bullets, too few graded claims).
    Same shape as EvolveOutput so the runner can treat them uniformly."""
    text: str
    model: str
    temperature: float
    input_tokens: int
    output_tokens: int
    issues_addressed: list[str]   # issue codes the prompt tried to fix
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# Mapping from gate-rejection text → structured issue code.
# Only structural problems DEBUG can plausibly fix; semantic issues (low
# semantic similarity, broken wikilinks) don't go through DEBUG —
# verify_crosslinks handles the latter, and the semantic floor is fundamental.
_GATE_ISSUE_PATTERNS: list[tuple[str, str]] = [
    ("numeric drift", "drift"),
    ("Key Contribution bullets", "n_kc"),
    ("graded claims", "n_graded"),
]


def detect_structural_gate_issues(gate_reasons: list[str]) -> list[str]:
    """Return the subset of gate-rejection codes DEBUG should try to fix."""
    issues: list[str] = []
    for r in gate_reasons or []:
        for needle, code in _GATE_ISSUE_PATTERNS:
            if needle in r and code not in issues:
                issues.append(code)
    return issues


def debug(
    *,
    draft: Draft,
    issues: list[str],
    gate_reasons: list[str],
    metadata: dict,
    sections: dict,
    use_stub: bool = False,
) -> DebugOutput:
    """DEBUG phase — repair a draft that failed structural gate checks.

    Targeted at three issue classes (per `detect_structural_gate_issues`):
      'drift'     — agent invented a number not in the PDF; remove or replace
      'n_kc'      — too few Key Contribution bullets; add more from the PDF
      'n_graded'  — too few gradable claims overall; expand KC or Results

    Crucially, DEBUG must not invent new facts to pad sections. If the PDF
    doesn't support an additional KC bullet, the agent says so and accepts
    the gate failure rather than fabricating one.
    """
    if not issues:
        return DebugOutput(
            text=draft.text, model="(noop)", temperature=0.0,
            input_tokens=0, output_tokens=0, issues_addressed=[],
        )

    drift_details: list[str] = []
    if "drift" in issues:
        for cd in (draft.claim_details or []):
            if cd.numeric_unmatched:
                drift_details.append(
                    f"  • [{cd.section} #{cd.position}] claim text: "
                    f"{cd.text[:160]!r}\n     unmatched numbers: "
                    f"{', '.join(cd.numeric_unmatched)}"
                )

    prompt = _build_debug_prompt(
        prior_draft=draft.text,
        issues=issues,
        gate_reasons=gate_reasons,
        drift_details=drift_details,
        sections=sections,
    )
    paper_type = (metadata or {}).get("paper_type")
    resp = llm.call(
        phase="debug",
        prompt=prompt,
        system=prompt_lib.load_author_system(paper_type),
        use_stub=use_stub,
    )
    return DebugOutput(
        text=resp.text,
        model=resp.model,
        temperature=resp.temperature,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        issues_addressed=issues,
        cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
        cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
    )


def _build_debug_prompt(
    *,
    prior_draft: str,
    issues: list[str],
    gate_reasons: list[str],
    drift_details: list[str],
    sections: dict,
) -> str:
    instructions: list[str] = []
    if "drift" in issues:
        instructions.append(
            "DRIFT — one or more numeric claims are not supported by the "
            "PDF. For each unmatched number: either REMOVE the claim "
            "(preferred), or REPLACE it with a verified number you can find "
            "verbatim in the PDF excerpts. Do NOT fabricate a replacement "
            "number. Specific drifting claims:"
        )
        instructions.extend(drift_details)
    if "n_kc" in issues:
        instructions.append(
            "N_KC — the draft has too few Key Contribution bullets (the gate "
            "needs ≥ 4). Add bullets ONLY for contributions explicitly "
            "stated in the PDF excerpts below. If the paper genuinely has "
            "fewer than 4 distinct contributions, leave the section as-is "
            "and prepend a one-line note '(paper has limited contribution "
            "set; gate threshold not met)'."
        )
    if "n_graded" in issues:
        instructions.append(
            "N_GRADED — too few gradable claims overall. The grader scores "
            "Key Contribution + Results bullets; if either section is light, "
            "expand it from the PDF excerpts. Numbers in Results must be "
            "verbatim from the PDF."
        )

    return "\n".join([
        "# Prior draft (passed tournament but failed the structural gate)",
        prior_draft[:4000],
        "",
        "# Gate rejection reasons",
        *(f"- {r}" for r in gate_reasons),
        "",
        "# Repair instructions",
        *instructions,
        "",
        "# PDF excerpts (use these to ground repairs — verbatim numbers only)",
        "## Methods",
        (sections.get("methods") or "")[:2000],
        "## Results",
        (sections.get("results") or "")[:2000],
        "## Discussion",
        (sections.get("discussion") or "")[:1200],
        "",
        "---",
        "",
        "Output the FULL repaired draft — Summary, Key Contributions, "
        "Methodology and Architecture, Results, Limitations, Related Papers — "
        "fixing only the listed issues. Leave sections that the gate did NOT "
        "flag essentially unchanged. Do not invent facts. Format identical to "
        "the prior draft.",
    ])
