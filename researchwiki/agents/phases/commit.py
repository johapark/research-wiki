"""Commit-time LLM utilities for the winning draft.

  - propose_short_name → 1-4 word handle for the index.md auto-entry
  - propose_keywords   → 5-10 retrieval tokens for YAML `keywords:`

The sandbox/promote write itself lives in `runner._phase_commit`, which
composes `verify_crosslinks` + `promote` + `_wrap_with_frontmatter` directly.
A `commit()` function here duplicated that path but was never called by
anything, so it was removed rather than left as an untested second
implementation of the promotion write.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .. import llm
from ...log import log


# ---------- short-name proposal ----------

@dataclass
class ShortNameOutput:
    name: str
    model: str
    input_tokens: int
    output_tokens: int
    hook: str = ""


def propose_short_name(
    *,
    metadata: dict,
    draft_text: str,
    use_stub: bool = False,
) -> ShortNameOutput:
    """Derive a paper's short handle and catalog hook from an existing page.

    **Not on the ingest path.** A fresh ingest gets both fields for free from the
    author's HANDLE/HOOK trailer (`phases.draft.split_gloss_trailer`), which has
    the source sections in context and costs no extra request. This is the
    standalone route for pages that already exist without them: backfilling the
    wiki's pre-`hook:` papers, or importing pages from another framework where
    re-running the full author phase would be wasteful and would rewrite prose
    that is already reviewed.

    Takes `draft_text` — the page body — as its only prose input. That keeps it
    Rule 1-clean by construction: the body is already PDF-grounded, and no
    Semantic Scholar `tldr`/`abstract` reaches the hook, so a persisted `hook:`
    needs no provenance field. Never pass upstream prose here.

    Failures degrade per-field: the handle falls back to 'TODO', the hook to ''
    (then omitted from YAML, leaving the page on `lint`'s `missing_hook` queue).
    The hook is never salvaged from a Summary slice — that yields the paper's
    question instead of its finding, which is the failure this field replaced.
    """
    if use_stub:
        return ShortNameOutput(name="TODO", model="stub", input_tokens=0, output_tokens=0)

    prompt = "\n".join([
        f"Title: {metadata.get('title') or '(unknown)'}",
        f"Year: {metadata.get('year') or '(unknown)'}",
        f"Authors: {metadata.get('authors') or '(unknown)'}",
        "",
        "Page excerpt:",
        _extract_gloss_context(draft_text)[:3000],
        "",
        "Task: produce two things for this paper's entry in a catalog of ~400 papers.",
        "",
        "1. HANDLE — a recognisable 1-4 word handle a researcher would use to refer "
        "to it in conversation. Examples from other papers in this wiki: "
        "'Co-Scientist', 'PaperOrchestra', 'AlphaFold 3', 'Bridge RNAs', 'SHAPEIT5', "
        "'CRISPResso2', 'PE8', 'MMseqs2'. Prefer the system / tool / method name "
        "('PaperOrchestra' > 'Multi-Agent Paper Writing System'). If there is no "
        "obvious handle, output 'TODO'.",
        "",
        "2. HOOK — a one-to-two sentence gloss, at most 400 characters. Its job is to "
        "separate this paper from the ~40 others in its category, so write it "
        "RESULT-FIRST: method + scale + the distinguishing finding. Lead with what "
        "the paper found, not what it asked. Keep concrete numbers where they "
        "distinguish (cohort size, error rate, speedup). Do NOT restate the "
        "research question, do NOT open with 'This study/paper/review', and do not "
        "write a generic topic description.",
        "",
        "   Good: 'Updated NanoSeq compatible with whole-exome capture (<5 errors "
        "per 10^9 bp); 1,042 oral epithelium samples reveal 46 genes under positive "
        "selection and >62,000 driver mutations.'",
        "   Bad:  'This study investigates somatic mutation and selection in normal "
        "tissues using duplex sequencing.'",
        "",
        "Output format — exactly two lines, no markdown, no quotes, no bullets:",
        "HANDLE: <handle>",
        "HOOK: <hook>",
    ])

    try:
        resp = llm.call(
            phase="short_name",
            prompt=prompt,
            use_stub=False,
        )
    except Exception as e:
        log(f"short_name LLM call failed: {type(e).__name__}: {e}", tag="agent")
        return ShortNameOutput(name="TODO", model="(failed)", input_tokens=0, output_tokens=0)

    # Both fields are parsed defensively and independently. A thinking model can
    # spend its whole budget on reasoning and return no content at all — indexing
    # into an empty splitlines() there used to crash the ingest after a good page
    # had already been drafted.
    name, hook = _parse_gloss_response(resp.text)
    if not name:
        log(f"short_name unusable (got {resp.text[:60]!r}); falling back to TODO",
            tag="agent")
        name = "TODO"
    if not hook:
        log("hook unusable; leaving it unset for lint to flag", tag="agent")
    return ShortNameOutput(
        name=name,
        hook=hook,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


_HANDLE_LINE_RE = re.compile(r"^\s*HANDLE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_HOOK_LINE_RE = re.compile(r"^\s*HOOK\s*:\s*(.+?)\s*\Z", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_HOOK_MAX_CHARS = 400


def _parse_gloss_response(text: str) -> tuple[str, str]:
    """-> (handle, hook), either possibly ''. Mirrors `draft.split_gloss_trailer`
    so both routes accept the same output shape and enforce the same bounds."""
    m = _HANDLE_LINE_RE.search(text)
    handle = ""
    if m:
        handle = m.group(1).strip().strip("\"'*").rstrip(".,;:")
        if len(handle) > 40:
            handle = ""

    m = _HOOK_LINE_RE.search(text)
    hook = ""
    if m:
        hook = " ".join(m.group(1).split()).strip().strip("\"'*")
        # Wildly over budget means the format was ignored, not merely verbose;
        # a 1000-char "hook" is a paragraph and unusable in a catalog line.
        if len(hook) > _HOOK_MAX_CHARS * 2:
            hook = ""
    return handle, hook


def _extract_gloss_context(draft_text: str) -> str:
    """Page context for the handle + hook prompt: Summary plus Key Contributions.

    Summary alone is what the retired index-line generator used, and it reliably
    yields the paper's *question*. The contributions carry the findings and the
    numbers a result-first hook needs, so both sections go in.
    """
    wanted = ("Summary", "Key Contributions")
    out: list[str] = []
    for heading in wanted:
        m = re.search(
            rf"^##\s+{re.escape(heading)}\s*$\s*(.+?)(?=^##\s+|\Z)",
            draft_text, re.MULTILINE | re.DOTALL,
        )
        if m:
            out.append(f"## {heading}\n{m.group(1).strip()}")
    return "\n\n".join(out) if out else draft_text[:3000]


# ---------- keyword proposal ----------

@dataclass
class KeywordsOutput:
    keywords: list[str] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# Keyword tokens are *retrieval* tokens — terms a researcher would type in a
# search bar to find this paper. They sit alongside `tags:` in YAML but play a
# different role: tags are categorical labels (`crispr`, `tool`); keywords are
# specific phrases (`homology-directed repair`, `off-target rate`).
# Public: `tasks/lint/yaml_checks.py` and `tasks/backfill.py` both need the
# floor. The writer owns it because the writer is what enforces it — a list
# below MIN_KEYWORDS is not written at all (see `render_keywords_yaml`).
MAX_KEYWORDS = 10
MIN_KEYWORDS = 5
_MAX_KEYWORD_LEN = 50           # arbitrary; long phrases hurt BM25 IDF
_KEYWORD_DENY = frozenset({
    # Banal terms that add no retrieval signal — every paper has these.
    "paper", "study", "method", "approach", "results", "analysis",
})

# JSON Schema for the batched keywords response. Honored by the chat-relay
# provider (validates `structured` and retries on mismatch); other providers
# silently ignore it. Per-item filtering still runs on accepted responses,
# so schema validation here is about the envelope shape (presence of `items`
# array, each element has `key` and `keywords: [string]`), not keyword
# quality.
_KEYWORDS_BATCH_SCHEMA = {
    "type": "object",
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key", "keywords"],
                "properties": {
                    "key": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}


def propose_keywords(
    *,
    metadata: dict,
    draft_text: str,
    sections: dict | None = None,
    full_pdf_text: str | None = None,
    use_stub: bool = False,
) -> KeywordsOutput:
    """Ask the LLM for 5-10 retrieval-token keywords for the paper.

    **Source-derived (preferred path).** When `full_pdf_text` is provided
    (the extract phase's full extracted text, no cap), keywords are
    extracted from the *source PDF* via a larger-window section re-anchor
    (12000-char per-section cap, vs. the 4000-char cap the extract phase
    uses for author-grounding). Falling back to the pre-truncated
    `sections` would miss named entities that live past the 4000-char
    Methods cut — Jang 2025 is the canonical example: RUFUS first appears
    at PDF char position ~16,200 and is invisible to the 4000-cap view.

    **Sections-derived (intermediate fallback).** When `full_pdf_text` is
    not available but `sections` is, use the pre-truncated sections.
    Acceptable when the named entities of interest live near the section
    starts; loses recall when they don't.

    **Draft-derived (last-resort fallback).** When neither source view is
    available (e.g., backfill path operating on already-committed pages
    without a PDF in scope), falls back to the draft's Summary + Key
    Contributions. Loses the omission-detection property.

    Why source-derived is the right default: the prior draft-derived
    behavior produced keyword lists that faithfully reflected what the
    draft happened to cover — including its gaps. Jang 2025 was the
    motivating case: the PDF mentions RUFUS 25× as a primary somatic
    variant caller alongside Mutect2, but the draft omitted it; under the
    old contract the keyword list also omitted it, and there was no
    structural signal anywhere in the YAML or body that something central
    was missing. With source-derived keywords, RUFUS lands in
    `keywords:` and the body's silence becomes visible at a glance.

    Failures degrade to an empty list — the page is still committable, lint
    will surface it as a missing-keywords warning on the next pass.
    """
    if use_stub:
        return KeywordsOutput()

    # Pick the input excerpt: full-text re-anchor preferred, sections next, draft last.
    excerpt, source_label = _build_keyword_input(
        sections=sections, draft_text=draft_text, full_pdf_text=full_pdf_text,
    )

    prompt = "\n".join([
        f"Title: {metadata.get('title') or '(unknown)'}",
        f"Year: {metadata.get('year') or '(unknown)'}",
        "",
        f"{source_label}:",
        excerpt,
        "",
        "Task: produce 5-10 retrieval-token keywords for this paper. "
        "These go in YAML `keywords:` and are indexed for full-text search. "
        "A reader looking for this paper would type one of these terms.",
        "",
        "Rules:",
        "  - 5-10 items, ordered by importance.",
        "  - Each item: 1-4 words. No commas inside an item.",
        "  - Specific phrases ('homology-directed repair', 'off-target rate') "
        "    not categorical labels ('crispr', 'tool') — those go in `tags:`.",
        "  - Names of methods, datasets, benchmarks, key proteins, or distinctive "
        "    metrics from the paper are good fits — even if the wiki page may "
        "    not yet describe them in detail. The keyword list reflects the "
        "    paper's central content; gaps between keywords and body text are "
        "    a deliberate coverage signal for downstream review.",
        "  - Skip stop-words and banal terms ('paper', 'study', 'method', "
        "    'analysis', 'approach').",
        "  - GROUND every keyword in the paper's OWN work: it must name "
        "    something this paper introduces, builds, uses, or studies — "
        "    evidenced by the Title, the Abstract, or a main-text sentence "
        "    about what the authors did. Sanity-check each candidate against "
        "    the Title above: if it doesn't fit what this paper is about, "
        "    drop it.",
        "  - Do NOT lift terms from enumerated lists or standardized form "
        "    fields that aren't prose about this study — e.g. a journal "
        "    reporting-summary or methods checklist ('ChIP-seq / Flow "
        "    cytometry / MRI-based neuroimaging ...'), author-contribution "
        "    or competing-interest statements, a table of contents, or the "
        "    titles of cited references. These enumerate generic categories "
        "    (often boilerplate identical across unrelated papers) that may "
        "    have nothing to do with this work. A term appearing only in "
        "    such a list — never in the Title or Abstract — is not a keyword.",
        "  - Output JSON only: {\"keywords\": [\"...\", \"...\"]}",
    ])

    try:
        resp = llm.call(
            phase="keywords",
            prompt=prompt,
            use_stub=False,
        )
    except Exception as e:
        log(f"keywords LLM call failed: {type(e).__name__}: {e}", tag="agent")
        return KeywordsOutput(model="(failed)")

    keywords = _parse_keywords_response(resp.text)
    if not keywords:
        # An empty list is nearly always a transient formatting miss (prose
        # instead of JSON, truncated object), not a paper with no keywords —
        # so spend one more call before giving up. Cheap: `keywords` is a
        # short-output role.
        log("keywords empty after parse; retrying once", tag="agent")
        try:
            retry = llm.call(phase="keywords", prompt=prompt, use_stub=False)
        except Exception as e:
            log(f"keywords retry failed: {type(e).__name__}: {e}", tag="agent")
        else:
            retry_keywords = _parse_keywords_response(retry.text)
            if retry_keywords:
                return KeywordsOutput(
                    keywords=retry_keywords,
                    model=retry.model,
                    input_tokens=resp.input_tokens + retry.input_tokens,
                    output_tokens=resp.output_tokens + retry.output_tokens,
                )
        log("WARNING: keywords still empty after retry — page will need "
            "`researchwiki backfill keywords`", tag="agent")
    return KeywordsOutput(
        keywords=keywords,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# Section-budget caps (chars). Abstract leads — it states the paper's own
# contribution in the authors' words and is the lowest-noise anchor for "what
# is this paper about." Results / Discussion / Introduction carry the findings
# and recurring named entities (datasets, tools, metrics) worth indexing.
# Methods is deliberately EXCLUDED from keyword input: it is the dominant noise
# source (antibody / reagent catalogs, and journal "Reporting Summary" method
# checklists — 'ChIP-seq / Flow cytometry / MRI-based neuroimaging ...' — that
# anchor_sections sweeps into a single Methods block when body structure is
# sparse). The ghareeb 2026 paper is the motivating case: its only anchored
# section was a Methods block containing the antibody table + reporting-summary
# checklist and NONE of the actual findings, so the keyword phase was fed pure
# noise. Trade-off: a tool named ONLY in Methods (cf. RUFUS in Jang 2025) can be
# missed — but a tool central enough to index almost always recurs in
# Results / Discussion, and the abstract+body signal is worth far more than the
# reagent-catalog noise Methods drags in. Budgets stay generous (cheap input
# tokens; output still capped at ≤10 keywords).
_SECTION_BUDGETS = {
    "abstract": 4000,
    "results": 12000,
    "introduction": 4000,
    "discussion": 4000,
}


def _build_keyword_input(
    *,
    sections: dict | None,
    draft_text: str,
    full_pdf_text: str | None = None,
) -> tuple[str, str]:
    """Pick the LLM input for keyword extraction. Abstract + main text only.

    The input always LEADS with the abstract (the cleanest statement of the
    paper's own contribution) and adds Introduction / Results / Discussion.
    Methods is excluded — see `_SECTION_BUDGETS` for why (reagent catalogs and
    journal reporting-summary checklists pollute it). Three-tier waterfall:

      1. `full_pdf_text` (preferred) — re-anchor MAIN-TEXT sections from the
         unbounded PDF text with the larger keyword budgets, prepending the
         abstract carried by the extract phase. Surfaces named entities past
         the 4000-char per-section author-grounding cut.
      2. `sections` (intermediate) — pre-truncated sections from the extract
         phase. Loses recall on entities past 4000 chars.
      3. Draft Summary+KC (last-resort) — when no source view is available;
         loses the omission-detection property.

    Returns (excerpt, label) — `label` is the prompt header text the LLM sees.
    """
    # Abstract is carried by the extract phase as a dedicated section even when
    # body anchoring fails (ghareeb: full-text re-anchor yields only a noisy
    # Methods block, but `sections['abstract']` is clean). Always lead with it.
    abstract = ((sections or {}).get("abstract") or "").strip()
    abstract_block = (
        f"## Abstract\n{abstract[:_SECTION_BUDGETS['abstract']]}" if abstract else ""
    )
    main_keys = ("introduction", "results", "discussion")

    # Tier 1: full-text re-anchor with the larger keyword budget.
    if full_pdf_text:
        from ...pdf.sections import anchor_sections
        # max_chars guards against pathologically long single sections; the
        # per-section sub-budgets below tighten it further per tier importance.
        wider_sections = anchor_sections(full_pdf_text, max_chars=20000)
        parts: list[str] = [p for p in (abstract_block,) if p]
        for key in main_keys:
            body = (wider_sections.get(key) or "").strip()
            if body:
                parts.append(f"## {key.title()}\n{body[:_SECTION_BUDGETS[key]]}")
        if parts:
            return "\n\n".join(parts), "Source PDF excerpts (abstract + main text)"

    # Tier 2: pre-truncated sections from the extract phase.
    if sections:
        parts = [p for p in (abstract_block,) if p]
        for key in main_keys:
            body = (sections.get(key) or "").strip()
            if body:
                parts.append(f"## {key.title()}\n{body[:_SECTION_BUDGETS[key]]}")
        if parts:
            return "\n\n".join(parts), "Source PDF excerpts (abstract + main text)"

    # Tier 3: draft Summary+KC fallback.
    return _extract_summary_and_contributions(draft_text)[:2000], "Draft (fallback — source sections unavailable)"


def propose_keywords_batch(
    *,
    items: list[dict],
    use_stub: bool = False,
) -> dict[str, KeywordsOutput]:
    """Batched keyword proposer — one LLM call covering N papers.

    Each `items` entry must carry: `key` (page key for the result dict),
    `title`, `year`, `body` (the wiki page markdown body).

    Returns a dict keyed by `item['key']`. Items missing from the LLM's
    response or with parse failures map to `KeywordsOutput(model="(missing)")`
    — the caller decides whether to retry, skip, or log.

    The single-paper `propose_keywords` is preserved for the ingest hot path
    (one paper at a time during `agent ingest`); this batched variant exists
    for backfill flows where N pages need keywords in one go and routing
    each through the chat-relay would mean N sequential prompts.
    """
    if use_stub:
        return {item["key"]: KeywordsOutput() for item in items}
    if not items:
        return {}

    sections = []
    for item in items:
        excerpt = _extract_summary_and_contributions(item.get("body") or "")[:1500]
        sections.append("\n".join([
            f"key: {item['key']}",
            f"title: {item.get('title') or '(unknown)'}",
            f"year: {item.get('year') or '(unknown)'}",
            "page content (Summary + Key Contributions):",
            excerpt,
        ]))

    prompt = "\n".join([
        f"You will produce keywords for {len(items)} wiki papers. "
        f"Each paper is delimited by a `---` line.",
        "",
        "\n\n---\n\n".join(sections),
        "",
        "---",
        "",
        "Task: for each paper above, produce 5-10 retrieval-token keywords "
        "for YAML `keywords:`, indexed for full-text search. A reader looking "
        "for one of these papers would type one of these terms.",
        "",
        "Rules per paper:",
        "  - 5-10 items, ordered by importance.",
        "  - Each item: 1-4 words. No commas inside an item.",
        "  - Specific phrases ('homology-directed repair', 'off-target rate') "
        "    not categorical labels ('crispr', 'tool').",
        "  - Names of methods, datasets, benchmarks, key proteins, or distinctive "
        "    metrics from the paper are good fits.",
        "  - Skip stop-words and banal terms ('paper', 'study', 'method').",
        "",
        "Output JSON only — one entry per input paper, echoing the input `key`:",
        '{"items": [{"key": "<echoed>", "keywords": ["...", "..."]}, ...]}',
    ])

    # Output budget: ~150 tokens per paper for the JSON (10 keywords × ~6 tokens
    # average + key + braces) plus headroom. Floor at 2000 to absorb formatting
    # overhead on small batches.
    max_tokens = max(2000, 200 * len(items))

    try:
        resp = llm.call(
            phase="keywords",
            prompt=prompt,
            max_tokens=max_tokens,
            use_stub=False,
            schema=_KEYWORDS_BATCH_SCHEMA,
        )
    except Exception as e:
        log(f"keywords batch LLM call failed: {type(e).__name__}: {e}", tag="agent")
        return {item["key"]: KeywordsOutput(model="(failed)") for item in items}

    parsed = _parse_keywords_batch_response(
        resp.text, expected_keys={item["key"] for item in items},
    )

    # Amortize token counts across items so per-page reporting still tells
    # a roughly correct cost story.
    n = max(1, len(items))
    out: dict[str, KeywordsOutput] = {}
    for item in items:
        kws = parsed.get(item["key"])
        if kws is None:
            out[item["key"]] = KeywordsOutput(model="(missing)")
            continue
        out[item["key"]] = KeywordsOutput(
            keywords=kws,
            model=resp.model,
            input_tokens=resp.input_tokens // n,
            output_tokens=resp.output_tokens // n,
        )
    return out


def _parse_keywords_batch_response(
    text: str, expected_keys: set[str]
) -> dict[str, list[str]]:
    """Parse a batched response into a key→filtered-keywords dict.

    Tolerates fenced code blocks and prose preamble. Drops entries whose
    `key` isn't in `expected_keys` (defends against the model hallucinating
    extra papers). Per-item filtering reuses `_filter_keyword_list` so the
    quality bar matches the single-paper path.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        log(f"keywords batch response carried no JSON object "
            f"(len={len(raw)}, head={raw[:80]!r})", tag="agent")
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log(f"keywords batch response was not valid JSON: {e}", tag="agent")
        return {}
    items = obj.get("items") or []
    out: dict[str, list[str]] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not isinstance(key, str) or key not in expected_keys:
            continue
        out[key] = _filter_keyword_list(entry.get("keywords") or [])
    return out


def _extract_summary_and_contributions(draft_text: str) -> str:
    """Concatenate the Summary and Key Contributions sections for the prompt."""
    parts = []
    for heading in ("Summary", "Key Contributions"):
        m = re.search(rf"^##\s+{heading}\s*$\s*(.+?)(?=^##\s+|\Z)",
                      draft_text, re.MULTILINE | re.DOTALL)
        if m:
            parts.append(m.group(1).strip())
    return "\n\n".join(parts) if parts else draft_text[:2000]


def _parse_keywords_response(text: str) -> list[str]:
    """Parse the JSON-shaped LLM response into a clean keyword list.

    Tolerates fenced code blocks, leading/trailing prose, and stray quotes.
    Drops keywords that fail the deny-list, length cap, or contain commas
    (which would break our YAML list rendering).
    """
    raw = text.strip()
    # Strip code fences if the model wrapped its JSON.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    # Find the first JSON object — tolerates pre/post prose.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    # A parse miss here is not "this paper has no keywords" — it's a malformed
    # response. Returning [] silently is how 38% of the corpus ended up with no
    # `keywords:` field despite `lint` requiring one, so both misses are logged.
    if not m:
        log(f"keywords response carried no JSON object "
            f"(len={len(raw)}, head={raw[:80]!r})", tag="agent")
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log(f"keywords response was not valid JSON: {e}", tag="agent")
        return []
    return _filter_keyword_list(obj.get("keywords") or [])


def _filter_keyword_list(items: list) -> list[str]:
    """Apply the deny-list, length cap, dedup, and max-count rules to a raw
    keyword list. Shared by the single-paper and batched parsers."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        kw = item.strip().strip('"').strip("'").strip(".,;:")
        if not kw or len(kw) > _MAX_KEYWORD_LEN:
            continue
        if "," in kw:                       # YAML rendering shortcut
            continue
        if kw.lower() in _KEYWORD_DENY:
            continue
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
        if len(out) >= MAX_KEYWORDS:
            break
    return out


def render_keywords_yaml(keywords: list[str]) -> str | None:
    """Render keywords as a YAML list line, or None if too few.

    Returns `keywords: [a, b, c]` style — same shape as `tags:` so Obsidian's
    Properties panel renders both consistently. None when fewer than
    `MIN_KEYWORDS` items pass quality filtering — better to write no field
    than to commit a half-list that lint would immediately flag.
    """
    if len(keywords) < MIN_KEYWORDS:
        return None
    inner = ", ".join(keywords)
    return f"keywords: [{inner}]"
