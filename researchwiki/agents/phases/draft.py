"""Draft container + author phase + tournament phase.

`Draft` is the in-memory carrier for an author-phase output (used downstream
by grade, critic, evolve, debug, tournament, commit). `author` is the LLM call
that produces it; `tournament` is the deterministic argmax that picks one
draft from N.

`_wrap_with_frontmatter` lives here because it's the canonical "make a Draft
look like a wiki page" helper — used by grade (for scoring) and by commit
(for final output) and by the runner's sandbox writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import llm, prompt_lib
from .crosslinks import CrosslinkCandidate


@dataclass
class Draft:
    """One author-phase output. iteration_id, scores, claim_details filled by runner."""
    text: str
    model: str
    temperature: float
    input_tokens: int
    output_tokens: int
    iteration_id: int = 0
    scores: dict = field(default_factory=dict)
    claim_details: list = field(default_factory=list)   # list[ClaimDetail]
    stance: str = "balanced"   # drafting stance that produced it (see DRAFT_STANCES)
    handle: str = ""   # 1-4 word short name, from the author trailer
    hook: str = ""     # result-first catalog gloss, from the author trailer


# Instruction-level draft diversity. Each parallel author draft is given a
# different *stance* — a short directive appended after the shared prompt — so
# the tournament chooses among genuinely different drafts rather than relying on
# the model's sampling noise alone. The (now near-vestigial) temperature spread
# layers on top. Slot 0 is the neutral baseline (empty suffix), so the
# most-likely winner is the standard draft; the cache breakpoint sits before the
# suffix, so varying it per draft keeps the shared-prefix cache discount.
DRAFT_STANCES: list[tuple[str, str]] = [
    ("balanced", ""),
    (
        "skeptical",
        "DRAFTING STANCE — skeptical/precision-first: use the paper's exact "
        "reported numbers, never round or approximate; make no claim the PDF "
        "doesn't directly support; write a thorough Limitations section that "
        "names the paper's acknowledged weaknesses and obvious gaps.",
    ),
    (
        "comprehensive",
        "DRAFTING STANCE — breadth-first coverage: ensure every distinct "
        "contribution and every headline result in the paper is represented in "
        "Key Contributions or Results; prefer covering one more real, "
        "PDF-supported claim over polishing prose. Do not invent claims to pad "
        "coverage.",
    ),
]


def stance_for_slot(slot: int) -> tuple[str, str]:
    """Return the (name, instruction) stance for draft slot `slot`, cycling if
    there are more drafts than defined stances."""
    return DRAFT_STANCES[slot % len(DRAFT_STANCES)]


def author(
    *,
    metadata: dict,
    sections: dict,
    temperature: float,
    candidates: list[CrosslinkCandidate] | None = None,
    use_stub: bool = False,
    system_prompt_override: str | None = None,
    stance: tuple[str, str] | None = None,
    pdf_full_text: str | None = None,
    target_claims=None,
) -> Draft:
    """Generate a wiki-page draft (Summary + Key Contributions + Methodology +
    Results + Limitations + Related Papers).

    The draft is the *body* of the page — YAML frontmatter is added by the
    commit phase using `metadata`. The grader scores KC + Results; Related
    Papers is graded separately by the cross-link verifier at commit.

    The author also emits a HANDLE/HOOK trailer after the sections, which
    `split_gloss_trailer` parses off here: `Draft.text` is always the clean body,
    and the two catalog fields ride on `Draft.handle` / `Draft.hook`. Producing
    them here rather than in a follow-up proposer call costs no extra request and
    gives the gloss the full source sections as context instead of only the
    already-drafted page.

    System prompt selection (via `prompt_lib.load_author_system`):
      - paper_type == 'review'  → prompts/author-system-review.md
      - else                    → prompts/author-system-research.md
      - system_prompt_override  → use that file path directly (eval A/B path)

    `candidates` (verified cross-link surface) is presented to the author as a
    whitelist for the Related Papers section. Anything not on the list will be
    stripped at commit time, so the author has no incentive to invent links.

    `stance` is an optional (name, instruction) pair appended to the prompt so
    parallel drafts differ by drafting stance, not just sampling noise (see
    DRAFT_STANCES). The neutral "balanced" stance has an empty instruction and
    leaves the prompt unchanged.
    """
    candidates = candidates or []
    stance_name, stance_text = stance or ("balanced", "")
    prompt = _build_author_prompt(
        metadata, sections, candidates,
        pdf_full_text=pdf_full_text,
        target_claims=target_claims,
    )
    if stance_text:
        prompt = f"{prompt}\n\n{stance_text}"
    paper_type = (metadata or {}).get("paper_type")
    system = prompt_lib.load_author_system(
        paper_type, override_path=system_prompt_override
    )
    # cache_prompt lets drafts sharing the same prompt reuse the prefix at
    # ~0.1× input cost. The neutral baseline draft writes it; same-stance
    # drafts read it. Stance-varied drafts have a different prompt and just
    # write their own entry — acceptable, and far simpler than a cache-split.
    resp = llm.call(
        phase="author",
        prompt=prompt,
        temperature=temperature,   # explicit override — varies per parallel draft
        system=system,
        use_stub=use_stub,
        cache_prompt=True,
    )
    body, handle, hook = split_gloss_trailer(resp.text)
    return Draft(
        text=body,
        model=resp.model,
        temperature=resp.temperature,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        stance=stance_name,
        handle=handle,
        hook=hook,
    )


def _build_author_prompt(
    metadata: dict,
    sections: dict,
    candidates: list[CrosslinkCandidate],
    *,
    pdf_full_text: str | None = None,
    target_claims=None,
) -> str:
    # L1 (2026-06-13): also include `pdf_full_text` (budget-capped) when
    # available. The curated section excerpts above are kept because they
    # carry "where to look first" signal — the parser found these as the
    # paper's structural anchors. The full text below catches what the
    # 4000-char anchor cap drops, especially for short-format papers
    # (Brief Communications, clinical-trial Articles) where load-bearing
    # results, ED-figure captions, and quantitative anchors live past the
    # cap. Validated against benchmark-fixtures/{kim,yang,lai}-2026-* — see
    # benchmark-fixtures/PLAN.md.
    parts = [
        f"# Paper metadata",
        f"- Title: {metadata.get('title') or 'unknown'}",
        f"- Year: {metadata.get('year') or 'unknown'}",
        f"- DOI: {metadata.get('doi') or 'unknown'}",
        f"- Venue: {metadata.get('venue') or 'unknown'}",
        f"- Authors: {metadata.get('authors') or 'unknown'}",
        "",
        "# PDF first-page preamble (for Summary)",
        metadata.get("pdf_text_preview", "")[:3000],
        "",
        "# Methods section excerpt",
        sections.get("methods", "")[:2500],
        "",
        "# Results section excerpt",
        sections.get("results", "")[:2500],
        "",
        "# Discussion section excerpt",
        sections.get("discussion", "")[:1500],
        "",
    ]
    # L2 (2026-06-13): figure / table / Extended Data captions extracted as
    # labeled blocks. Brief Communications and short-format Articles often
    # push load-bearing quantitative results into figure captions (review-
    # paper enzyme kcat/KM tables in Fig. 3d, trial benchmark tables in
    # Fig. 1, ED Fig results sections at the back of the paper). L1's full
    # PDF text already contains these but unstructured; L2 surfaces them
    # as named, labeled blocks so the author can target them when the
    # fixture-curated content (e.g. "TIM-barrel, β-barrel, repeat proteins")
    # lives there. Only included when present — silently skipped for papers
    # using period-style captions or papers without figures.
    if sections.get("figure_captions"):
        parts.extend([
            "# Figure / Table captions (main text)",
            "The following are figure and table captions from the main text,",
            "extracted as labeled blocks. Captions often carry the paper's",
            "specific quantitative anchors (kcat/KM tables, benchmark numbers,",
            "named instances of categories, residue annotations) — search here",
            "first when looking for specific numbers or enumerated names.",
            "",
            sections["figure_captions"],
            "",
        ])
    if sections.get("extended_data"):
        parts.extend([
            "# Extended Data figure / table captions",
            "Extended Data figures sit beyond the main text and frequently",
            "carry the paper's most detailed empirical results — ablation",
            "tables, partial-match analyses, secondary cohort breakdowns,",
            "off-target evaluations. Treat these as first-class results,",
            "not appendix material.",
            "",
            sections["extended_data"],
            "",
        ])
    if pdf_full_text:
        # Budget: 30K chars. The Anthropic input window is large enough to
        # hold the full PDF for most papers, but capping bounds cost and
        # latency for ~50–100-page outliers (textbooks, very long reviews).
        # Cap is applied as a head-of-document slice — papers' headline
        # claims live in the front half; appendices/extended references in
        # the back. ED figures sit between, and Brief Communications fit
        # entirely under 30K. Revisit if a fixture surfaces a back-of-paper
        # claim that gets truncated.
        FULL_PDF_BUDGET = 30_000
        truncated_note = ""
        body = pdf_full_text
        if len(body) > FULL_PDF_BUDGET:
            body = body[:FULL_PDF_BUDGET]
            truncated_note = (
                f"\n[truncated at {FULL_PDF_BUDGET} chars; "
                f"full PDF is {len(pdf_full_text)} chars]"
            )
        parts.extend([
            "# Full PDF text (supplementary context — search this for ANY",
            "# numeric claim, figure-caption content, Extended Data figure",
            "# results, or detail not present in the curated excerpts above.",
            "# The excerpts above are anchored at section headings; this is",
            "# everything else.)",
            body + truncated_note,
            "",
        ])
    # L3 (2026-06-13): target-claims block. A pre-extraction phase
    # produces a list of source-grounded items the page should preserve,
    # graded by importance. The block sits between content and output
    # instructions: the author has already seen the source content
    # (sections / captions / full text); this is the coverage target
    # against which to write.
    #
    # Treat as constraint-shaped (per the L4 retro): "preserve these
    # critical/high items" rather than additive bullet templates that
    # crowd out discretionary content. Empty target_claims → block
    # silently skipped → behavior matches pre-L3 prompt shape.
    if target_claims is not None:
        try:
            from .target_claims import render_for_author_prompt
            block = render_for_author_prompt(target_claims)
        except Exception:
            block = ""
        if block:
            parts.extend([
                "# Target claims — the paper's load-bearing content, extracted",
                "# from the PDF. This is a WHOLE-PAGE coverage menu, NOT a Key",
                "# Contributions checklist and NOT content to copy verbatim.",
                "# Distribute these across every section where each fits —",
                "# Summary, Key Contributions, Methodology, Results,",
                "# Limitations. Most numeric items belong in the Results table",
                "# or Methodology prose, not as Key Contributions bullets. Key",
                "# Contributions still obeys its hard ≤10-bullet cap: when",
                "# several target claims describe one finding, MERGE them into a",
                "# single bullet rather than emitting one bullet per claim.",
                "",
                block,
                "",
            ])
    parts.append("# Related-Papers candidates (verified — safe to use as [[wikilinks]])")
    if candidates:
        parts.append("Each candidate is a wiki page already in this wiki — "
                     "cross-checked against the source paper's reference "
                     "graph (citation-graph candidates) or its semantic "
                     "neighborhood with LLM judgment (topical candidates). "
                     "Use the `[[wikilink]]` form shown.")
        for c in candidates:
            year_s = f", {c.year}" if c.year else ""
            if c.kind == "cited_by_source":
                relation = "cited by THIS paper"
            elif c.kind == "cites_source":
                relation = "cites THIS paper"
            else:
                relation = f"topical match — {c.relationship}" if c.relationship else "topical match"
            title_s = f": {c.title[:120]}" if c.title else ""
            parts.append(f"  - [[{c.wikilink}]] ({relation}{year_s}){title_s}")
        parts.append("")
        parts.append("RULES for the Related Papers section:")
        parts.append("  • You may only use [[wikilinks]] FROM THIS LIST. No other wikilinks.")
        parts.append("  • Don't include every candidate — pick the most relevant ≤6.")
        parts.append("  • For citation-graph candidates: one line stating relationship + direction.")
        parts.append("  • For topical candidates: one line stating the methodological link "
                     "(do not assert citation — neither paper cites the other).")
    else:
        parts.append("(no verified candidates — write '(none)' for the Related Papers section)")
    parts.extend([
        "",
        "---",
        "",
        "Output ONLY the six sections below, in this exact order. Omit "
        "Methodology and Architecture only if the paper has no novel method "
        "(pure review, evaluation-only, or commentary).",
        "",
        "## Summary",
        "<≤150 words>",
        "",
        "## Key Contributions",
        "- <bullet>",
        "- <bullet>",
        "...",
        "",
        "## Methodology and Architecture",
        "<≤300 words. The differentiating parts only — skip generic background.>",
        "",
        "## Results",
        "<table or prose, ≤300 words>",
        "",
        "## Limitations",
        "- <bullet>",
        "- <bullet>",
        "...",
        "",
        "## Related Papers",
        "<≤6 [[wikilink]] entries from the candidate list above, OR '(none)'>",
        "",
        "Then, after the sections, a final line `---` followed by exactly these "
        "two catalog fields. They are not page sections — they are metadata for "
        "the wiki's index, and they are stripped from the page body before it is "
        "graded.",
        "",
        "HANDLE: <1-4 word handle a researcher would use to refer to this paper "
        "in conversation — prefer the system / tool / method name (e.g. 'Bridge "
        "RNAs', 'MMseqs2', 'AlphaFold 3', 'PE8'). Write TODO if there is no "
        "obvious handle.>",
        "HOOK: <one to two sentences, ≤400 characters, on ONE line. This is the "
        "paper's line in a catalog of ~400 papers, so its job is to separate it "
        "from the ~40 others in its category. Write it RESULT-FIRST: method + "
        "scale + the distinguishing finding. Lead with what the paper found, not "
        "what it asked. Keep the concrete numbers that distinguish it (cohort "
        "size, error rate, speedup). Do NOT restate the research question and do "
        "NOT open with 'This study/paper/review'.",
        "",
        "  Good HOOK: Updated NanoSeq compatible with whole-exome capture (<5 "
        "errors per 10^9 bp); 1,042 oral epithelium samples reveal 46 genes under "
        "positive selection and >62,000 driver mutations.",
        "  Bad HOOK:  This study investigates somatic mutation and selection in "
        "normal tissues using duplex sequencing.>",
    ])
    return "\n".join(parts)


# The author emits HANDLE/HOOK as a trailer after a final `---` rule. Parsed and
# stripped in `author()` so every downstream consumer — tournament, critic,
# verify_crosslinks, the graders, promote — sees only the six sections. Stripping
# at the source is what keeps a format slip from leaking metadata into the page.
_TRAILER_RE = re.compile(
    r"\n-{3,}\s*\n+\s*HANDLE\s*:(?P<handle>.*?)\n+\s*HOOK\s*:(?P<hook>.*?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_HOOK_MAX_CHARS = 400


def split_gloss_trailer(text: str) -> tuple[str, str, str]:
    """-> (body without trailer, handle, hook).

    Degrades per-field rather than raising: a missing or malformed trailer
    yields ('', '') and leaves the body untouched, so the handle falls back to
    'TODO' and `hook:` is simply omitted from YAML — which puts the page on
    `lint`'s `missing_hook` queue instead of committing a bad gloss.
    """
    m = _TRAILER_RE.search(text)
    if not m:
        return text, "", ""
    handle = " ".join(m.group("handle").split()).strip().strip("\"'*").rstrip(".,;:")
    hook = " ".join(m.group("hook").split()).strip().strip("\"'*")
    if len(handle) > 40:
        handle = ""
    if len(hook) > _HOOK_MAX_CHARS * 2:
        # Wildly over budget means the model ignored the format, not that it was
        # merely verbose; a 1000-char "hook" is a paragraph and not usable here.
        hook = ""
    return text[: m.start()].rstrip() + "\n", handle, hook


def _wrap_with_frontmatter(
    body: str,
    metadata: dict,
    stem: str,
    *,
    category: str | None = None,
    category_strength: str | None = None,
) -> str:
    """Wrap a draft body in minimal YAML frontmatter for the grader's parser
    and the sandbox writer.

    `category` defaults to None → the placeholder `[TODO]` is written, signalling
    the user must fill it in. The sandbox writer passes a search-index suggestion
    here so a sandboxed page carries a useful category hint instead of forcing
    the user to recategorize from scratch on a false-positive sandbox.
    `category_strength`: 'weak' marks first-of-kind / low-confidence suggestions
    so a downstream reviewer knows to re-check before promoting.
    """
    cat_value = f"[{category}]" if category else "[TODO]"
    # Quote the title: paper titles routinely contain a colon-space subtitle
    # ("MemGPT: Towards ..."), which is invalid as an unquoted YAML scalar and
    # breaks frontmatter parsing. Mirrors promote.py's quoting of the title.
    title = (metadata.get("title") or "unknown").replace('"', '\\"')
    lines = [
        "---",
        f'title: "{title}"',
        f"authors: {metadata.get('authors') or 'unknown'}",
        f"year: {metadata.get('year') or 'unknown'}",
        f"doi: {metadata.get('doi') or 'unknown'}",
        "type: paper",
        f"category: {cat_value}",
    ]
    if category_strength == "weak":
        lines.append("category_suggestion_strength: weak  # first-of-kind — review")
    lines += [
        f'pdf_path: "[[{stem}.pdf]]"',
        "source_collection: external",
        "tags: []",
        "---",
        "",
        body,
    ]
    return "\n".join(lines)


def tournament(drafts: list[Draft]) -> tuple[Draft, str]:
    """Pick the winning draft via deterministic argmax on the author fitness.

    Ranking is the AUTHOR lens from `agents.fitness.tournament_key`: paraphrase
    fidelity first, then *coverage breadth* (graded-claim count) as the near-tie
    tie-break, then numeric integrity, BM25, and the weakest claim. semantic_score
    is None when the embedding model isn't installed; we degrade gracefully to
    BM25-primary in that case.
    """
    from ..fitness import tournament_key

    if not drafts:
        raise ValueError("tournament called with no drafts")
    if len(drafts) == 1:
        return drafts[0], "single draft, no contest"

    winner = max(drafts, key=tournament_key)
    others = [d for d in drafts if d is not winner]

    def _fmt(d: Draft) -> str:
        sem = d.scores.get("semantic_score")
        sem_s = f"{sem:.2f}" if sem is not None else "n/a"
        return (f"#{d.iteration_id} sem={sem_s} graded={d.scores.get('n_graded') or 0} "
                f"drift={d.scores.get('n_drift') or 0} "
                f"bm25={(d.scores.get('mean_bm25') or 0.0):.1f}")

    rationale = "kept " + _fmt(winner) + "; beat " + " | ".join(_fmt(d) for d in others)
    return winner, rationale
