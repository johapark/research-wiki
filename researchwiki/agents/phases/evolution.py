"""Memory Evolution proposals.

A-Mem's Memory Evolution: when a new memory is added, neighboring memories
get their context/tags refined in light of it. The wiki analog: when a new
paper P lands, top-k semantically-related synthesis pages should be edited
to reference P (or to update an existing claim P contradicts or refines).

This module produces *proposals*. By default each is a structured descriptor
of an edit and nothing is written to `wiki/`. The reviewer (user / Claude
Step 6) decides whether to materialize it. The narrowest auto-apply gate
ships behind `auto_apply_proposal()`: only `refine` verdicts in the
bullet-append flavor, at high confidence with passing structural checks,
bypass review. Line-replace refines and every enhance / contrast verdict
always go through human review.

Verdict shapes:
  refine   — mechanical patch. Two flavors: append a cited bullet under a
             specific section, OR rewrite one specific line to weave the
             new paper's data in. Bullet-append is the only auto-applyable
             flavor; line-rewrite always goes through human review.
  enhance  — the paragraph's *framing* needs updating in light of the new
             paper (integrated context, qualifier, numeric update). LLM
             rewrites one specific existing paragraph. Never auto-applies;
             proposal-only.
  contrast — flag a contradiction; don't auto-edit, just note for the author
  none     — P doesn't change N

Call shape:
  proposals = propose_evolution(source_stem="cgt/du-2025-...", k=8)
  for prop in proposals:
      # prop has: target_key, verdict, confidence, patch, rationale, claim_ids
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ...fsatomic import write_text_atomic
from ...paths import wiki_dir
from ...index import pages_semantic as semantic_pages
from ...wiki import Page, extract_section, read_page
from .. import llm
from ...log import log
from . import evolve_ledger


# Page types eligible for evolution. Paper pages are immutable per their PDF
# (Rule 3 says re-extract from PDF, don't synthesize); only synthesis pages
# get refined when neighbors arrive.
EVOLVABLE_TYPES = ("synthesis",)

VALID_VERDICTS = {"refine", "enhance", "contrast", "none"}

# Cosine threshold below which the per-neighbor LLM call is skipped. A
# lightweight prefilter that cuts ~half the candidates the KNN returns and
# saves ~$0.005 per skipped neighbor. Calibrated against early dogfooding
# data: genuine "refine" / "enhance" matches sit at 0.75+; under 0.65 the
# LLM almost always returns "none" — paying ~$0.005 to confirm a no-op
# isn't worth it. Override per-call via `min_cosine`.
DEFAULT_MIN_COSINE = 0.65

# Adaptive neighbor selection (replaces the old flat top-8). The A/B (2026-07-14)
# found two things that pin these defaults:
#   1. Actionable proposals occur across cos 0.71–0.83 with no clean cliff — so
#      `gap` must be >= the observed 0.10 spread between a top hit and an
#      accepted-at-0.73 neighbor, and `min_keep` guards against under-judging.
#      Net: gap-trimming only removes a clear marginal *tail* (e.g. two 0.69s
#      below a 0.83 top); on the common clustered distribution it trims nothing.
#   2. Extending the cap past 8 was measured to *increase* first-pass cost, not
#      recall: clustered neighbors all sit within `gap`, so a bigger cap just
#      judges more marginal "none" pairs. So `max_keep` stays at the old 8 — no
#      regression — and the real re-run savings come from the judged-pair ledger.
DEFAULT_GAP = 0.12       # keep neighbors within this cosine gap of the top hit
DEFAULT_MIN_KEEP = 4     # always judge at least this many above the floor
DEFAULT_MAX_KEEP = 8     # hard cap on judged neighbors (also the candidate pool size)


@dataclass
class EvolutionProposal:
    """One proposed edit to one neighbor page in light of a source paper.

    `confidence` is the LLM's own self-rated certainty in [0, 1]. Used by a
    future auto-apply gate; for proposal-only mode it's recorded but not
    actioned. `claim_ids` lists graded claim IDs from the source paper's
    PDF that ground the proposed change — empty when the verdict is
    `contrast` (no specific source claim to point at).

    `patch` is the structured edit body. Shape depends on `verdict`:
      refine:    {add_bullet_under, bullet_text}            (bullet-append)
                 or {target_line_match, new_line}           (line-replace)
      enhance:   {target_section, target_paragraph_match, new_paragraph}
      contrast:  {target_line_match, note_for_author}
      none:      {} — caller filters these before materializing

    `input_tokens` / `output_tokens` carry the LLM cost for the call that
    produced this proposal. Summed by the runner for the cost-telemetry log
    so the C2 dashboard reflects evolution overhead alongside ingest cost.
    `model` is the resolved model string the call actually ran on (from the
    LLMResponse), so telemetry records the real model rather than a literal.
    """
    source_key: str                 # category/stem of the new paper
    target_key: str                 # category/stem of the page to edit
    verdict: str                    # refine | enhance | contrast | none
    confidence: float
    rationale: str                  # one-sentence summary
    patch: dict = field(default_factory=dict)
    claim_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""                 # resolved model for the judge call

    def is_actionable(self) -> bool:
        """True when this proposal would actually change the target page."""
        return self.verdict in {"refine", "enhance", "contrast"}


def propose_evolution(
    source_key: str,
    *,
    k: int = DEFAULT_MAX_KEEP,
    min_cosine: float = DEFAULT_MIN_COSINE,
    gap: float = DEFAULT_GAP,
    min_keep: int = DEFAULT_MIN_KEEP,
    use_stub: bool = False,
    ledger_conn=None,
) -> tuple[list[EvolutionProposal], dict]:
    """Generate edit proposals for neighbors of `source_key`.

    Steps:
      1. Load the source paper page; build a probe text from its summary.
      2. Semantic-neighbor pool among synthesis pages, excluding any that
         already reference the source (`k` = pool size / hard cap).
      3. **Adaptive selection** (`_adaptive_select`) — cosine floor + keep
         within `gap` of the top hit, bounded by `[min_keep, k]`. Trims the
         marginal tail on sparse papers; gives hub papers headroom past the old
         flat 8. Dropped candidates are logged, never silently truncated.
      4. **Judged-pair ledger** — skip the LLM call for any (source, target)
         pair already judged "none" at the current page content hashes (see
         `evolve_ledger`); re-judge everything else.
      5. Per-neighbor LLM call producing one structured verdict + patch.

    Returns `(proposals, stats)`. `stats` carries the telemetry counts:
    `n_knn`, `n_above_threshold`, `n_judged`, `n_cached_skipped`,
    `n_dropped_adaptive`, `n_actionable`, `input_tokens`, `output_tokens`,
    `model`.
    """
    source = read_page(wiki_dir() / f"{source_key}.md")
    if source is None:
        raise FileNotFoundError(f"source page not found: wiki/{source_key}.md")

    candidates = _select_neighbors(source_key, source, k=k)
    stats = {
        "n_knn": len(candidates),
        "n_above_threshold": 0,
        "n_judged": 0,
        "n_cached_skipped": 0,
        "n_dropped_adaptive": 0,
        "n_actionable": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "",  # resolved judge model — all calls share phase "memory_evolution"
    }
    if not candidates:
        return [], stats

    kept, dropped = _adaptive_select(
        candidates, min_cosine=min_cosine, gap=gap, min_keep=min_keep, max_keep=k)
    stats["n_above_threshold"] = len(kept)
    stats["n_dropped_adaptive"] = len(dropped)
    if dropped:
        by_reason: dict[str, int] = {}
        for _, reason in dropped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        log(f"adaptive-select kept {len(kept)}, dropped {len(dropped)} {by_reason}",
            tag="propose_evolution")

    # Judged-pair ledger: skip cached "none" pairs. Opened lazily (real mode
    # only); a caller may inject `ledger_conn` to share/observe it.
    conn = ledger_conn
    own_conn = False
    if conn is None and not use_stub:
        conn = evolve_ledger.open_ledger()
        own_conn = True
    src_hash = evolve_ledger.page_hash(source.body) if conn is not None else ""

    proposals: list[EvolutionProposal] = []
    try:
        for neighbor_hit in kept:
            neighbor = read_page(wiki_dir() / f"{neighbor_hit.key}.md")
            if neighbor is None:
                continue

            if use_stub:
                proposals.append(_stub_proposal(source_key, neighbor_hit.key,
                                                neighbor_hit.score))
                stats["n_judged"] += 1
                continue

            tgt_hash = evolve_ledger.page_hash(neighbor.body) if conn is not None else ""
            if conn is not None and evolve_ledger.is_cached_none(
                    conn, source_key, neighbor_hit.key, src_hash, tgt_hash):
                stats["n_cached_skipped"] += 1
                continue

            prop = _judge_one(source_key, source, neighbor)
            if prop is None:
                continue
            proposals.append(prop)
            stats["n_judged"] += 1
            stats["input_tokens"] += prop.input_tokens
            stats["output_tokens"] += prop.output_tokens
            if prop.model:
                stats["model"] = prop.model
            if conn is not None:
                if prop.verdict == "none":
                    evolve_ledger.record_none(
                        conn, source_key, neighbor_hit.key, src_hash, tgt_hash)
                else:
                    # Actionable now — drop any stale "none" so it can't shadow
                    # this proposal on a future run.
                    evolve_ledger.clear(conn, source_key, neighbor_hit.key)
    finally:
        if own_conn and conn is not None:
            conn.close()

    stats["n_actionable"] = sum(1 for p in proposals if p.is_actionable())
    return proposals, stats


def _select_neighbors(source_key: str, source: Page, k: int):
    """Top-k semantic neighbors among evolvable page types, excluding ones
    that already cite the source (no need to propose redundant edits).

    Uses `query_text` rather than `query_stem` so this works for newly-promoted
    papers that haven't been added to the page-level semantic index yet —
    the runner integration calls evolve immediately after promote, before any
    reindex. We embed the source's canonical index text on the fly; for
    already-indexed sources the result is functionally equivalent to a
    `query_stem` lookup (same canonical concat → same vector).
    """
    if not semantic_pages.index_exists():
        return []
    probe = semantic_pages.page_index_text(source)
    if not probe.strip():
        return []
    hits = semantic_pages.query_text(
        probe,
        k=k * 2,
        page_types=EVOLVABLE_TYPES,
        exclude_keys=frozenset({source_key}),
    )
    out = []
    src_link = f"[[{source_key}]]"
    for h in hits:
        target = read_page(wiki_dir() / f"{h.key}.md")
        if target is None:
            continue
        if src_link in target.body:
            continue
        out.append(h)
        if len(out) >= k:
            break
    return out


def _adaptive_select(hits, *, min_cosine: float, gap: float,
                     min_keep: int, max_keep: int):
    """Choose which neighbors to judge. Returns `(kept, dropped)` where
    `dropped` is a list of `(hit, reason)` for logging — nothing is silently
    truncated.

    Rules, applied to hits sorted by cosine descending:
      - below `min_cosine`            → dropped ("below_floor")
      - rank < `min_keep`             → always kept (safety floor)
      - rank >= `max_keep`            → dropped ("cap")
      - within `gap` of the top score → kept
      - otherwise                     → dropped ("gap")

    Trims the marginal tail on sparse papers while `max_keep` (> the old flat 8)
    lets a hub paper get fuller coverage.
    """
    above = sorted((h for h in hits if h.score >= min_cosine),
                   key=lambda h: h.score, reverse=True)
    dropped = [(h, "below_floor") for h in hits if h.score < min_cosine]
    if not above:
        return [], dropped
    top = above[0].score
    kept = []
    for i, h in enumerate(above):
        if i < min_keep:
            kept.append(h)
        elif i >= max_keep:
            dropped.append((h, "cap"))
        elif h.score >= top - gap:
            kept.append(h)
        else:
            dropped.append((h, "gap"))
    return kept, dropped


def _stub_proposal(source_key: str, target_key: str, score: float) -> EvolutionProposal:
    """Deterministic placeholder for offline tests."""
    return EvolutionProposal(
        source_key=source_key,
        target_key=target_key,
        verdict="refine" if score > 0.75 else "none",
        confidence=score,
        rationale=f"semantic neighbor (cos={score:.2f}); review",
        patch={"add_bullet_under": "(stub)", "bullet_text": f"[[{source_key}]] — stub"} if score > 0.75 else {},
    )


def _judge_one(
    source_key: str,
    source: Page,
    neighbor: Page,
) -> EvolutionProposal | None:
    """One LLM call; returns the parsed proposal or None on failure.

    The prompt deliberately leads with the NEIGHBOR body (not the source) —
    we want the LLM thinking "what would change about this page," not "is
    this paper related to this page." Different framing, different output.
    """
    try:
        resp = llm.call(
            phase="memory_evolution",
            # No cache_prefix: measured inert on Haiku 4.5 (deployed) — the
            # ~3.4k-token prompt is below the ~4096 min cacheable prefix, so the
            # marker is silently ignored (cache_create=0). And caching the
            # neighbor would be wrong anyway: it varies per call, so the constant
            # part is source+system. On Haiku no prefix can cache this phase.
            prompt=_neighbor_block(neighbor) + "\n\n" + _source_task_block(source_key, source),
            system=_EVOLUTION_SYSTEM,
            schema=_EVOLUTION_SCHEMA,
            # Structured verdict+patch — no extended thinking needed. Off
            # explicitly so an adaptive-thinking model (Sonnet 5, Opus 4.6+)
            # can't spend the whole max_tokens budget on thinking and return an
            # empty answer (measured: stop_reason=max_tokens, content=[thinking]).
            disable_thinking=True,
        )
    except Exception as e:
        log(f"judge call failed for {neighbor.key}: {e}", tag="propose_evolution")
        return None

    parsed = _parse_evolution_response(resp.text)
    if parsed is None:
        return None
    verdict = (parsed.get("verdict") or "").strip().lower()
    if verdict not in VALID_VERDICTS:
        return None
    return EvolutionProposal(
        source_key=source_key,
        target_key=neighbor.key,
        verdict=verdict,
        confidence=float(parsed.get("confidence") or 0.0),
        rationale=(parsed.get("rationale") or "").strip()[:300],
        patch=parsed.get("patch") or {},
        claim_ids=parsed.get("claim_ids") or [],
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        model=resp.model,
    )


# JSON Schema for the evolution proposer envelope. Honored by chat-relay;
# ignored by other providers. `verdict` is enum-constrained — downstream code
# rejects anything not in VALID_VERDICTS — and `patch` is loosely typed
# because its shape varies per verdict (refine/enhance/contrast each pull
# different keys; "none" gives `patch: {}`). The downstream `_judge_one`
# logic still validates patch shape per-verdict; the schema only ensures
# the envelope is well-formed.
_EVOLUTION_SCHEMA = {
    "type": "object",
    "required": ["verdict", "rationale", "patch"],
    "properties": {
        "verdict":    {"type": "string",
                       "enum": ["refine", "enhance", "contrast", "none"]},
        "confidence": {"type": ["number", "null"]},
        "rationale":  {"type": "string"},
        "patch":      {"type": "object"},
        "claim_ids":  {"type": "array", "items": {"type": "string"}},
    },
}


_EVOLUTION_SYSTEM = """\
You are a wiki maintainer. A new paper has been added; you decide whether
an existing synthesis page should be EDITED in light of it.

Be surgical. Most pairings will be "none". The bar for any non-"none"
verdict is that you can name a *specific line, bullet, or paragraph* in
the target page that is now incomplete, refinable, mis-framed, or
contradicted by the new paper. Vague "this is related" is a "none".

Rule 1 (load-bearing): every edit you propose must be supported by content
that is plausibly in the new paper's PDF — not just its title. If you
can't quote what the new paper said, return "none".

Verdict-selection heuristic (choose the LEAST invasive that fits):
  - Missing citation + one factoid   → refine (append-bullet)
  - Single stale/incomplete line     → refine (line-replace)
  - Paragraph's framing needs update → enhance
  - Contradiction / tension          → contrast
  - No change                        → none

Output JSON only. No prose around it. The `patch` field shape DEPENDS on the verdict:

For verdict="refine" (bullet-append flavor) — target page has a list/section
the new paper belongs in as a new bullet:
{
  "verdict": "refine",
  "confidence": 0.85,
  "rationale": "one sentence",
  "patch": {
    "add_bullet_under": "## Approaches",       // exact section heading from target
    "bullet_text": "[[source/key]] — uses RNP delivery to reduce off-targets 3.2× (Results, Table 2)."
  }
}

For verdict="refine" (line-replace flavor) — a specific line should be rewritten:
{
  "verdict": "refine",
  "confidence": 0.75,
  "rationale": "one sentence",
  "patch": {
    "target_line_match": "Off-target rates remain >10⁻⁴ across most assays.",   // quote from target
    "new_line": "Off-target rates remain >10⁻⁴ across most assays, though [[source/key]] reports 3.2× reduction with RNP delivery (Results, Table 2)."
  }
}

For verdict="enhance" — integrate the new paper into an existing paragraph
by rewriting that paragraph. Use ONLY when a bullet or line-replace would
lose the integrated framing. Scope is ONE paragraph — do not touch
headings, list bullets, or tables:
{
  "verdict": "enhance",
  "confidence": 0.75,
  "rationale": "one sentence explaining why refine (bullet/line) wouldn't carry the same information",
  "patch": {
    "target_section": "## Evidence from the wiki",           // section the paragraph lives under
    "target_paragraph_match": "verbatim first ~120 chars of the existing paragraph — the anchor by which the reviewer locates it",
    "new_paragraph": "Full rewrite of that paragraph. MUST cite [[source/key]]. MUST preserve every [^footnote-id] reference present in the original paragraph — do not drop citations. Do NOT introduce numbers not stated in the new paper or the target's cited papers."
  }
}

For verdict="contrast" — the new paper contradicts an existing claim. NEVER auto-edit;
flag for the human author:
{
  "verdict": "contrast",
  "confidence": 0.7,
  "rationale": "one sentence",
  "patch": {
    "target_line_match": "X improves with parameter Y.",
    "note_for_author": "Source paper's Fig 4 shows opposite trend; reconciliation needed."
  }
}

For verdict="none" — no edit warranted:
{"verdict": "none", "confidence": 0.9, "rationale": "one sentence", "patch": {}}

Hard rules for `patch`:
  - For refine and enhance, `bullet_text` / `new_line` / `new_paragraph` MUST
    contain `[[source/key]]` using the source key shown in the user prompt.
  - `add_bullet_under`, `target_line_match`, `target_section`, and
    `target_paragraph_match` MUST be quoted verbatim from the target page
    body (verbatim prefix suffices for `target_paragraph_match`).
  - `new_paragraph` MUST preserve every `[^footnote-id]` reference present
    in `target_paragraph_match` — dropping a footnote breaks the target
    page's grounding.
  - If you cannot satisfy these constraints from the material in the prompt,
    return "none" instead.
"""


def _neighbor_block(neighbor: Page) -> str:
    """The target-page half of the judge prompt (kept first, for the
    deliberate "what would change about THIS page" framing).

    Split out from the source half so a future batched judge can reuse it, and
    once (if ever) the deployed model has a large enough prompt to cache. NB:
    prompt caching is *inert* for this phase on Haiku 4.5 (the whole prompt is
    below the ~4096-token min cacheable prefix — measured), so the split is not
    currently used for caching. Sends only `body[:3500]` — a big synthesis is
    truncated, a known limitation the batching redesign should revisit.
    """
    return "\n".join([
        "# Target page (the one that might be edited)",
        f"  [[{neighbor.key}]]  type={neighbor.fm.get('type','')}",
        f"  Title: {neighbor.fm.get('title','')}",
        "",
        "Full target page body:",
        neighbor.body[:3500],
    ])


def _source_task_block(source_key: str, source: Page) -> str:
    """The variable half: the new paper + the decision instruction."""
    src_summary = extract_section(source.body, "Summary")[:1200]
    src_kc = extract_section(source.body, "Key Contributions")[:1200]
    src_results = extract_section(source.body, "Results")[:800]

    return "\n".join([
        "# New paper (the trigger for evolution)",
        f"  [[{source_key}]]",
        f"  Title: {source.fm.get('title','')}",
        f"  Year:  {source.fm.get('year','')}",
        "",
        "Summary:",
        src_summary,
        "",
        "Key Contributions:",
        src_kc,
        "",
        "Results excerpt:",
        src_results,
        "",
        "---",
        "",
        f"Decide: should the target page above be edited in light of this new "
        f"paper? If yes, what specific line/bullet, and what does it become? "
        f"Output JSON per the system prompt.",
    ])


def _parse_evolution_response(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------- proposal serialization ----------

def render_proposal_md(prop: EvolutionProposal) -> str:
    """One markdown file per proposal. Goes to .ingest/{source-stem}-evolution-proposals/."""
    lines = [
        "---",
        f"source: {prop.source_key}",
        f"target: {prop.target_key}",
        f"verdict: {prop.verdict}",
        f"confidence: {prop.confidence:.2f}",
        "---",
        "",
        f"# {prop.verdict.upper()} — [[{prop.target_key}]]",
        "",
        f"**Rationale:** {prop.rationale}",
        "",
    ]
    if prop.verdict == "refine" and prop.patch.get("add_bullet_under"):
        lines.extend([
            "## Patch (refine — append bullet)",
            "",
            f"- Cite `[[{prop.source_key}]]` in the body (inline, and add a "
            f"`[^id]: [[{prop.source_key}]]` footnote under `## References` if "
            f"the page uses footnotes). Synthesis/idea pages have no "
            f"`referenced_papers:` field — the body is the citation source.",
            f"- Section: **{prop.patch.get('add_bullet_under','(unspecified)')}**",
            "- New bullet:",
            f"  > {prop.patch.get('bullet_text','(missing)')}",
        ])
    elif prop.verdict == "refine":
        lines.extend([
            "## Patch (refine — replace line)",
            "",
            "Replace the line:",
            f"> {prop.patch.get('target_line_match','(missing)')}",
            "",
            "With:",
            f"> {prop.patch.get('new_line','(missing)')}",
        ])
    elif prop.verdict == "enhance":
        lines.extend([
            "## Patch (enhance — rewrite paragraph)",
            "",
            f"- Cite `[[{prop.source_key}]]` in the body (inline, and add a "
            f"`[^id]: [[{prop.source_key}]]` footnote under `## References` if "
            f"the page uses footnotes). Synthesis/idea pages have no "
            f"`referenced_papers:` field — the body is the citation source.",
            f"- Section: **{prop.patch.get('target_section','(unspecified)')}**",
            "- Locate the paragraph starting with:",
            f"  > {prop.patch.get('target_paragraph_match','(missing)')}",
            "",
            "Replace that paragraph with:",
            "",
            prop.patch.get("new_paragraph","(missing)"),
            "",
            "**Reviewer checklist before applying:**",
            "  - New paragraph cites the source paper.",
            "  - Every `[^footnote-id]` from the original paragraph appears in the new one.",
            "  - No numbers introduced that aren't in the new paper or the target's cited papers.",
            "  - After applying, run `researchwiki check-grounding` and `researchwiki grade synthesis`.",
        ])
    elif prop.verdict == "contrast":
        lines.extend([
            "## Patch (FLAG — DO NOT AUTO-APPLY)",
            "",
            "Target line:",
            f"> {prop.patch.get('target_line_match','(missing)')}",
            "",
            f"**Note for author:** {prop.patch.get('note_for_author','(missing)')}",
        ])
    if prop.claim_ids:
        lines.extend(["", "**Grounding claims:** " + ", ".join(prop.claim_ids)])
    return "\n".join(lines) + "\n"


# ---------- auto-apply gate (refine bullet-append only, high confidence) ----------
#
# The narrowest auto-apply path: refine proposals in the bullet-append
# flavor at confidence >= 0.9, with all structural checks passing, get
# materialized in place. Refine's line-replace flavor and every enhance /
# contrast proposal keep going through human review — they modify or
# integrate existing prose, which has higher blast radius.

_AUTO_APPLY_MIN_CONFIDENCE = 0.9


def _split_frontmatter(body: str) -> tuple[str, str]:
    """Split a wiki page into (frontmatter_block, rest). The frontmatter
    block includes the trailing `---\\n` delimiter; rest starts immediately
    after. Returns ("", body) if no frontmatter — a malformed page that
    auto-apply must refuse."""
    if not body.startswith("---\n"):
        return ("", body)
    end_idx = body.find("\n---\n", 4)
    if end_idx < 0:
        return ("", body)
    fm_end = end_idx + 5  # include the closing `---\n`
    return (body[:fm_end], body[fm_end:])


def _insert_in_referenced_papers(fm: str, wikilink: str) -> tuple[str, bool]:
    """Append `wikilink` to the YAML `referenced_papers:` list. Returns
    (new_fm, did_insert). Idempotent: if the link is already present in
    the frontmatter, returns (fm, False). Refuses with (fm, False) when
    no `referenced_papers:` list exists."""
    if wikilink in fm:
        return (fm, False)

    lines = fm.split("\n")
    in_list = False
    last_list_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("referenced_papers:"):
            in_list = True
            continue
        if in_list:
            if line.startswith("  - "):
                last_list_idx = i
            elif line.strip() == "":
                # blank line — list might continue, keep walking
                continue
            elif not line.startswith("  "):
                # de-dented line = next top-level YAML key, list ended
                break
    if not in_list or last_list_idx < 0:
        return (fm, False)

    lines.insert(last_list_idx + 1, f"  - {wikilink}")
    return ("\n".join(lines), True)


def _insert_bullet_under(body: str, heading: str, bullet_text: str) -> tuple[str, bool]:
    """Append `- {bullet_text}` at the end of the section identified by
    `heading` (exact line, e.g. '## Evidence from the wiki'). Section
    ends at the next heading of the same or higher level (lower count of
    `#` chars), or end of body. Returns (new_body, did_insert)."""
    target = heading.strip()
    if not target.startswith("#"):
        return (body, False)
    heading_level = len(target) - len(target.lstrip("#"))

    lines = body.split("\n")
    heading_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == target:
            heading_idx = i
            break
    if heading_idx < 0:
        return (body, False)

    # Find end of section
    end_idx = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        s = lines[j].rstrip()
        if s.startswith("#"):
            this_level = len(s) - len(s.lstrip("#"))
            if this_level <= heading_level:
                end_idx = j
                break

    # Walk back over trailing blank lines so the new bullet sits right
    # after the section's existing content.
    insert_idx = end_idx
    while insert_idx > heading_idx + 1 and not lines[insert_idx - 1].strip():
        insert_idx -= 1

    lines.insert(insert_idx, f"- {bullet_text}")
    return ("\n".join(lines), True)


def _update_generated_at(fm: str, date_str: str) -> str:
    """Replace `generated_at: <whatever>` in the frontmatter with the new
    date. If no existing line, insert before the closing `---`."""
    lines = fm.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("generated_at:"):
            lines[i] = f"generated_at: {date_str}"
            return "\n".join(lines)
    # No existing key — insert before the closing `---`
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "---":
            lines.insert(i, f"generated_at: {date_str}")
            return "\n".join(lines)
    return fm


def auto_apply_proposal(
    prop: EvolutionProposal,
    *,
    today: str | None = None,
    wiki_root_dir: Path | None = None,
) -> tuple[bool, str]:
    """Apply an evolution proposal in place if it passes the auto-apply gate.

    Gate (all must hold):
      - verdict == "refine" AND patch carries the append-bullet shape
        (`add_bullet_under` + `bullet_text`). Line-replace refines,
        enhance, and contrast never auto-apply — they modify existing
        prose and require human review.
      - confidence >= 0.9.
      - `bullet_text` contains the source wikilink.
      - Target page exists.
      - Source wikilink not already in target body (no duplicate).
      - Target has YAML frontmatter with a `referenced_papers:` list.
      - Section heading from `add_bullet_under` exists in target body.

    On pass: target page is modified in place — bullet appended under the
    named section, source wikilink added to `referenced_papers:`,
    `generated_at:` updated. Returns `(True, "applied")`.

    On any check failing: target page is untouched. Returns
    `(False, "<reason>")`. Reason is a one-line human-readable string
    suitable for log lines or the CLI summary.
    """
    if prop.verdict != "refine":
        return (False,
                f"verdict={prop.verdict} (only 'refine' with bullet-append supports auto-apply)")
    if not prop.patch.get("add_bullet_under"):
        return (False,
                "auto-apply requires the bullet-append refine shape "
                "(add_bullet_under + bullet_text); line-replace refines "
                "and enhance proposals always require human review")
    if prop.confidence < _AUTO_APPLY_MIN_CONFIDENCE:
        return (False,
                f"confidence {prop.confidence:.2f} < {_AUTO_APPLY_MIN_CONFIDENCE:.2f} threshold")

    section_heading = (prop.patch.get("add_bullet_under") or "").strip()
    bullet_text = (prop.patch.get("bullet_text") or "").strip()
    if not section_heading or not bullet_text:
        return (False, "patch missing add_bullet_under or bullet_text")

    source_link = f"[[{prop.source_key}]]"
    if source_link not in bullet_text:
        return (False, f"bullet_text doesn't contain {source_link}")

    target_root = wiki_root_dir if wiki_root_dir is not None else wiki_dir()
    target_path = target_root / f"{prop.target_key}.md"
    if not target_path.exists():
        return (False, f"target page missing: {prop.target_key}")

    body = target_path.read_text(encoding="utf-8")

    if source_link in body:
        return (False, "source already linked in target body")

    fm, rest = _split_frontmatter(body)
    if not fm:
        return (False, "target has no YAML frontmatter")

    # referenced_papers was dropped from synthesis/idea pages — the citation now
    # lives in the body bullet. Extend the YAML list only where it still exists
    # (concept pages); don't abort when it doesn't. The body bullet is the
    # required insert.
    fm_new, _ = _insert_in_referenced_papers(fm, source_link)

    rest_new, body_changed = _insert_bullet_under(rest, section_heading, bullet_text)
    if not body_changed:
        return (False, f"section {section_heading!r} not found in target body")

    import datetime
    today_str = today or datetime.date.today().isoformat()
    fm_new = _update_generated_at(fm_new, today_str)

    write_text_atomic(target_path, fm_new + rest_new)
    return (True, "applied")
