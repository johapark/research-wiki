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

from dataclasses import dataclass

from .. import llm, prompt_lib
from .draft import Draft
from .grade import ClaimDetail


# ---------- critic ----------

@dataclass
class CritiqueOutput:
    """Result of a critic call. notes is the user-facing message; weak_claims
    is the deterministic list of claims the grader flagged as weak (used as
    the source of truth for whether to skip evolve)."""
    notes: str
    weak_claims: list[ClaimDetail]
    model: str
    input_tokens: int
    output_tokens: int


def critic(
    *,
    draft: Draft,
    metadata: dict,
    use_stub: bool = False,
) -> CritiqueOutput:
    """Critic phase — identify weak claims in a winning draft.

    The deterministic grader has already flagged weak claims (low semantic
    similarity, negation mismatch, or numeric drift). The critic's job is
    to translate that mechanical signal
    into actionable revision notes the author phase can consume on the next
    pass. We deliberately give the critic the FACTS (which claim, what the
    grader said) rather than asking it to discover weaknesses on its own —
    that keeps the LLM accountable to the grader's evidence rather than
    inventing concerns.
    """
    weak = [c for c in draft.claim_details if c.is_weak()]
    if not weak:
        return CritiqueOutput(
            notes="no weak claims detected; draft passes grader thresholds",
            weak_claims=[],
            model="(skipped)",
            input_tokens=0,
            output_tokens=0,
        )

    prompt = _build_critic_prompt(weak, draft.text, metadata)
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
    )


_CRITIC_SYSTEM = """\
You are a critic for a research-paper wiki. The grader has flagged specific
claims as weakly supported by the source PDF — flags include: low semantic
similarity to the retrieved evidence chunks, numeric values not present in
the PDF, and asserted negations the source doesn't echo. Your job is to
translate those mechanical flags into concrete revision instructions the
author can act on.

For each flagged claim, output one bullet:
- Quote the problematic phrase from the claim verbatim.
- Say whether to (a) remove the claim, (b) soften / rephrase to match what the
  PDF actually supports, or (c) replace the unmatched number with a value
  that does appear in the PDF.
- Keep notes terse — the author re-reads the draft, not your prose.

Do NOT introduce new concerns the grader didn't flag. The grader is the source
of truth; you are its translator.
"""


def _build_critic_prompt(weak: list[ClaimDetail], draft_text: str, metadata: dict) -> str:
    parts = [
        "# Draft (unchanged after tournament)",
        draft_text[:4000],
        "",
        "# Grader flags (ordered by section + position)",
    ]
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
    parts.extend([
        "",
        "# Output format",
        "One bullet per flagged claim. Quote the problem phrase verbatim. Say "
        "remove / soften-to / replace-with. No prose intro or outro.",
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
