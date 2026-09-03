"""Target-claims extraction phase (L3).

Runs after extract, before crosslinks. Calls an LLM over the PDF text and
returns a structured list of claims a thorough wiki page should cover.
The author phase consumes this list as a coverage target — the directive
is "consider preserving these items," not "include every one of them
verbatim," so the author retains synthesis discretion while having a
concrete checklist of source-grounded content.

Why a separate phase rather than baking this into the author prompt
directly: extracting a claim list and authoring a wiki page are
different tasks with different failure modes. Extraction wants
specificity (capture every named instance, every kcat/KM value, every
n-count); authoring wants synthesis (pick the load-bearing claims,
weave them into prose under section caps). Separating them lets each
phase optimize for its own constraint and lets the operator inspect
the target-claims list independently of the rendered page (useful
when diagnosing "the author had access to claim X but didn't include
it" — that's a synthesis-side issue, distinguishable from "the agent
never extracted X" which is upstream).

Cost: one LLM call per ingest. Empirically ~5K input + ~1K output =
~$0.030 added per ingest at Sonnet/Haiku rates. Phase config in
config/models.yaml; defaults to the `extractor` role (same as reconcile —
this is the second structured-extraction call in the pipeline).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import llm
from ...errors import EnvironmentFailure


CLAIM_TYPES = ("headline", "capability", "limitation")
IMPORTANCE_TIERS = ("critical", "high", "normal")


@dataclass
class TargetClaim:
    """One target claim extracted from the PDF."""
    type: str           # 'headline' | 'capability' | 'limitation'
    content: str        # one-sentence claim, source-grounded
    importance: str     # 'critical' | 'high' | 'normal'
    location: str       # 'Abstract' / 'Fig. 2' / 'Methods §off-target' / etc.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetClaimsOutput:
    """Result of one target-claims call."""
    claims: list[TargetClaim] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    error: str | None = None

    def is_empty(self) -> bool:
        return not self.claims

    def by_importance(self, tier: str) -> list[TargetClaim]:
        return [c for c in self.claims if c.importance == tier]


_SYSTEM_PROMPT = """\
You are reading a research paper and producing a structured list of the
claims a thorough wiki summary should preserve. Each claim falls into
one of three categories:

  headline   — a quantitative result or specific outcome the paper
               makes. Examples: "20× faster querying than pyScoMotif",
               "n=5 patients enrolled", "median follow-up 23.0 months",
               "AUC 0.837 on SCOPe40 sensitivity benchmark".

  capability — a demonstrated tool/method ability, named instance, or
               specific use case. Examples: "identifies zinc-finger
               motifs in metagenomic proteins", "predicts splicing for
               11 modalities at 1bp resolution", "distinguishes active
               from inactive β-adrenergic receptor structures".

  limitation — a constraint, weakness, or scope boundary the paper
               acknowledges, OR an obvious gap a careful reader would
               flag. Examples: "20 Å connectivity cutoff misses long-
               range allosteric pockets", "single-arm phase 1 trial",
               "validated only on human cell lines".

Importance tiers (use them as a TRIAGE — most claims are NOT critical):
  critical   — central to the paper's contribution; the page would be
               materially wrong without this. Headline benchmark numbers,
               primary endpoint outcomes, the main mechanistic claim.
               Reserve for the load-bearing few — typically 3-6 per paper,
               never the majority. If you are marking most claims critical,
               you are mis-triaging; demote the supporting ones to high.
  high       — significant supporting result; the page should cover it.
               Notable demonstrated capability, important caveat,
               specific instance the paper highlights.
  normal     — context / secondary detail. Worth flagging but the page
               can summarize at higher level.

Output strict JSON: {"claims": [{"type": "...", "content": "...", "importance": "...", "location": "..."}]}

Constraints:
  - Cap at 35 claims total. Prefer SPECIFIC (numbers + names + venues)
    over generic. A claim like "the paper proposes a method" is too
    generic; "Folddisco indexes 53M structures of AFDB50 in <25h on
    64 cores" is right.
  - Each claim must be SOURCE-GROUNDED — cite the location (Abstract,
    Fig. 2, Methods §X, ED Fig. 3, Results) so a reader can verify.
  - ONE entry per distinct fact. Do not emit the same result twice — not
    across types (if it has a number, prefer "headline"; if it's a
    demonstrated use case, prefer "capability"), and not with different
    `location` values. If a fact appears in both the Abstract and the main
    text, emit it ONCE citing the most specific location (the figure/table
    or Methods §, not the Abstract). Two entries whose `content` restates
    the same number or finding is a duplicate — collapse them.
  - Limitations should reflect what the paper itself says OR what a
    peer reviewer would call a methodological caveat. Don't invent
    limitations the paper doesn't support.
  - For review papers, "headline" claims are the review's framing
    statements (e.g., "field has shifted from physics-based to ML-
    based design") and "capability" claims are the categories of work
    surveyed.

Output JSON only, no prose.
"""


_JSON_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "content", "importance"],
                "properties": {
                    "type": {"type": "string", "enum": list(CLAIM_TYPES)},
                    "content": {"type": "string"},
                    "importance": {"type": "string", "enum": list(IMPORTANCE_TIERS)},
                    "location": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def extract_target_claims(
    *,
    metadata: dict,
    sections: dict,
    pdf_full_text: str | None = None,
    use_stub: bool = False,
) -> TargetClaimsOutput:
    """Run one target-claims LLM call. Returns the parsed list.

    On an untyped call error, JSON parse failure, or schema mismatch, returns
    an empty TargetClaimsOutput with an error string set. Typed provider and
    environment failures propagate: authoring without this checklist after an
    outage would make a degraded ingest indistinguishable from a good one.
    """
    if use_stub:
        return TargetClaimsOutput(
            claims=[
                TargetClaim(
                    type="headline", content="(stub) example headline claim",
                    importance="high", location="Abstract",
                ),
            ],
            model="(stub)",
        )

    from .. import model_config
    prompt = _build_prompt(
        metadata, sections, pdf_full_text,
        max_chars=model_config.target_claims_max_chars(),
    )
    try:
        resp = llm.call(
            phase="target_claims",
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            schema=_JSON_SCHEMA,
        )
    except EnvironmentFailure:
        raise  # house rule 1 (errors.py): never author from a hidden outage
    except Exception as e:
        return TargetClaimsOutput(
            error=f"LLM call failed: {type(e).__name__}: {e}",
        )

    raw = resp.text.strip()
    # Strip code fences if the model wrapped its JSON.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return TargetClaimsOutput(
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
            cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
            error=f"no JSON object in LLM output: {raw[:120]!r}",
        )
    raw_json = json_match.group(0)
    try:
        data = json.loads(raw_json)
        claims_raw = data.get("claims") or []
        partial = False
    except json.JSONDecodeError:
        # Truncated output — extract complete claim objects via regex rather
        # than failing entirely. Each object ends with `}` before a `,` or `]`.
        claims_raw = [
            m for m in (
                _try_parse(s) for s in re.findall(r"\{[^{}]+\}", raw_json)
            )
            if m is not None
        ]
        partial = True

    _TIER_ORDER = {"critical": 0, "high": 1, "normal": 2}
    _MAX_CLAIMS = 35

    parsed: list[TargetClaim] = []
    seen: set[str] = set()
    for item in claims_raw:
        if not isinstance(item, dict):
            continue
        ctype = (item.get("type") or "").strip().lower()
        importance = (item.get("importance") or "normal").strip().lower()
        content = (item.get("content") or "").strip()
        location = (item.get("location") or "").strip() if item.get("location") else ""
        if not content or ctype not in CLAIM_TYPES or importance not in IMPORTANCE_TIERS:
            continue
        # Backstop dedup: the extractor sometimes emits the same fact twice
        # (e.g. once cited to the Abstract, once to the main text). Collapse
        # on a normalized content key so the author isn't fed near-identical
        # claims that turn into repeated Key Contributions bullets.
        key = _content_key(content)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(TargetClaim(
            type=ctype, content=content,
            importance=importance, location=location,
        ))

    # Sort by importance tier before capping so critical claims survive
    # truncation even when the model emits them late in the output.
    parsed.sort(key=lambda c: _TIER_ORDER.get(c.importance, 99))
    parsed = parsed[:_MAX_CLAIMS]

    return TargetClaimsOutput(
        claims=parsed,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
        cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
        error="partial JSON — truncated output recovered" if partial else None,
    )


def _try_parse(s: str) -> dict | None:
    """Return parsed dict if `s` is valid JSON, else None."""
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def _content_key(content: str) -> str:
    """Normalize a claim's content for dedup. Lowercases, drops the
    trailing location parenthetical (`... (Main §2.2)`), strips
    punctuation, and collapses whitespace so the same fact cited to two
    different locations maps to one key. Conservative — only exact
    normalized restatements collapse, not merely similar claims."""
    s = re.sub(r"\s*\([^()]*\)[\s.]*$", "", content.strip())
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# Sections rendered to the extractor, in natural reading order. `references`
# is deliberately absent — bibliography is pure noise for claim extraction.
_RENDER_ORDER = ("abstract", "introduction", "methods", "results",
                 "discussion", "figure_captions", "extended_data")

# Budget is allocated in this order (highest claim density first), so when a
# paper exceeds the char budget the low-density prose (intro, methods) is
# trimmed before Results/Discussion/captions — never the reverse.
_PRIORITY = ("results", "discussion", "figure_captions", "extended_data",
             "abstract", "methods", "introduction")

# Hard per-section sub-caps for the noise-prone sections, so one giant
# figure-legend dump or intro can't eat the whole budget. Sections not
# listed (results/discussion/abstract/methods) are bounded only by the
# global budget, so they're included in full whenever the budget allows.
_SUB_CAP = {"figure_captions": 8000, "extended_data": 6000, "introduction": 4000}

_HEADERS = {
    "abstract": "# Abstract",
    "introduction": "# Introduction",
    "methods": "# Methods",
    "results": "# Results (extract every benchmark number here verbatim)",
    "discussion": "# Discussion (acknowledged limitations live here)",
    "figure_captions": "# Figure / Table captions",
    "extended_data": "# Extended Data captions",
}


def _rich_sections(sections: dict, pdf_full_text: str | None, cap: int) -> dict:
    """Re-derive sections from the full PDF text so each is complete up to
    `cap` chars (extract_sections() output is pre-capped at 4000 — sized for the
    author phase, far too tight for claim extraction). Falls back to the
    pre-capped sections when full text isn't available."""
    if not pdf_full_text:
        return sections or {}
    try:
        from ...pdf.sections import anchor_sections
        rich = anchor_sections(pdf_full_text, max_chars=cap)
        return rich or (sections or {})
    except Exception:
        return sections or {}


def _allocate(rich: dict, budget: int) -> dict[str, str]:
    """Choose how much of each section to include within `budget` chars, by
    priority. References are excluded entirely. Returns {section_key: text}."""
    included: dict[str, str] = {}
    remaining = budget
    for key in _PRIORITY:
        if remaining <= 0:
            break
        body = (rich or {}).get(key) or ""
        if not body:
            continue
        sub = _SUB_CAP.get(key)
        if sub:
            body = body[:sub]
        if len(body) > remaining:
            body = body[:remaining]
        included[key] = body
        remaining -= len(body)
    return included


def _build_prompt(
    metadata: dict, sections: dict, pdf_full_text: str | None, max_chars: int,
) -> str:
    """Assemble the extraction prompt: the whole substantive paper (references
    excluded) up to `max_chars`, priority-trimmed so Results/Discussion/captions
    survive when a long paper overflows the budget."""
    parts = [
        "# Paper metadata",
        f"- Title: {metadata.get('title') or 'unknown'}",
        f"- Authors: {metadata.get('authors') or 'unknown'}",
        f"- Year: {metadata.get('year') or 'unknown'}",
        f"- Venue: {metadata.get('venue') or 'unknown'}",
        f"- Type: {metadata.get('paper_type') or 'research'}",
        "",
    ]
    if pdf_full_text:
        from ...pdf.sections import assess_section_health, stratified_text_sample
        health = assess_section_health(pdf_full_text, sections)
        if not health.healthy:
            parts.extend([
                "# Full PDF text (document-stratified fallback — section extraction unhealthy)",
                f"# Extraction-health reasons: {', '.join(health.reasons)}",
                "# References are excluded when their boundary is detectable.",
                stratified_text_sample(pdf_full_text, max_chars),
                "",
                "---",
                "Output the structured target-claims list now.",
            ])
            return "\n".join(parts)

    rich = _rich_sections(sections, pdf_full_text, cap=max_chars)
    included = _allocate(rich, max_chars)
    # First-page preamble as insurance when the abstract header wasn't detected
    # (Nature-family papers often omit it); small, kept outside the budget.
    if not included.get("abstract") and metadata.get("pdf_text_preview"):
        parts.extend(["# PDF first-page preamble",
                      metadata["pdf_text_preview"][:2500], ""])
    for key in _RENDER_ORDER:
        body = included.get(key)
        if body:
            parts.extend([_HEADERS[key], body, ""])
    if not any(included.get(k) for k in ("results", "discussion", "methods",
                                         "abstract")) and pdf_full_text:
        # Section parser found nothing substantive — fall back to a
        # budget-capped raw slice so extraction still has something to chew on.
        parts.extend(["# Full PDF text (fallback — sections not detected)",
                      pdf_full_text[:max_chars], ""])
    parts.append("---")
    parts.append("Output the structured target-claims list now.")
    return "\n".join(parts)


def render_for_author_prompt(out: TargetClaimsOutput, *, max_normal: int = 5) -> str:
    """Render target claims for inclusion in the author prompt.

    Critical and high importance items are listed in full; normal items
    are capped at `max_normal` to avoid drowning the author in low-
    priority context. Returns a markdown-formatted block; empty string
    when the output is empty (so the author prompt assembly can skip it).
    """
    if out.is_empty():
        return ""
    lines: list[str] = []
    for tier in ("critical", "high"):
        items = out.by_importance(tier)
        if not items:
            continue
        if tier == "critical":
            lines.append(
                "## Critical — reproduce verbatim, INCLUDING every numeric "
                "value (benchmark ratios, n-counts, p-values). Do not round or "
                "restate a number as a qualitative claim:"
            )
        else:
            lines.append("## High — preserve these:")
        for c in items:
            loc = f" ({c.location})" if c.location else ""
            lines.append(f"  - [{c.type}] {c.content}{loc}")
        lines.append("")
    normals = out.by_importance("normal")[:max_normal]
    if normals:
        lines.append(f"## Normal-importance (cover at higher level if space):")
        for c in normals:
            loc = f" ({c.location})" if c.location else ""
            lines.append(f"  - [{c.type}] {c.content}{loc}")
        lines.append("")
    return "\n".join(lines)
