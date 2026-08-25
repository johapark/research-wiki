"""Batch-LLM triage of concept-hub candidates.

`candidates.collect_candidates()` is a cheap heuristic (span ≥ 2, pages ≥ 3,
minus a bare-acronym/ubiquity filter). At corpus scale its output is
dominated by generic-bigram extraction noise ("three datasets", "current
models", "without fine-tuning") that clears the heuristic but is never a
concept. Manually `--decline`-ing them is whack-a-mole.

This module classifies the *whole* candidate set in one (chunked) LLM pass
against the concept-vs-glossary thesis test (prompts/concept-page-author.md)
and auto-writes the noise verdicts to the decline list (tagged
`source="llm-triage"`, reversible via `--undecline`). It never scaffolds —
`concept`/`uncertain` verdicts stay surfaced for the human `--thesis` gate.

Design mirrors `tasks/suggest_splits.py` (LLM judgment as an explicit,
reviewable command) but lives in `concepts/` and never imports from
`tasks/` — the ~8-line JSON parse recipe is duplicated on purpose to keep
that package boundary clean. Untyped LLM failures degrade to a no-op (verdict
`uncertain` = keep); typed environment/configuration failures propagate so the
CLI preserves its exit-2 contract.
"""

from __future__ import annotations

import json
import re
import sys

from ..errors import EnvironmentFailure
from .candidates import _term_slug, collect_candidates
from .declines import add_declines

TRIAGE_THRESHOLD = 12        # status recommends --triage at/above this bridge count
CHUNK_SIZE = 40              # candidates per LLM call — bounds output tokens
_MAX_TOKENS = 3500           # ~50-60 output tokens/verdict × CHUNK_SIZE + headroom
TRIAGE_SYSTEM_FILENAME = "concept-triage-system.md"

NOISE_VERDICTS = frozenset({"glossary", "fragment", "redundant", "alias"})
KEEP_VERDICTS = frozenset({"concept", "uncertain"})
_VALID_VERDICTS = NOISE_VERDICTS | KEEP_VERDICTS

# Honored only by the chat-relay provider (validated + retried); anthropic/
# openai ignore it, so the prompt restates the output contract in prose.
_TRIAGE_SCHEMA = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["term", "verdict", "reason"],
                "properties": {
                    "term": {"type": "string"},
                    "verdict": {"type": "string", "enum": sorted(_VALID_VERDICTS)},
                    "reason": {"type": "string"},
                    # Advisory only (for the `alias` verdict): the more-canonical
                    # term this one duplicates. Decorates the decline reason; the
                    # decline still acts on the candidate's own term, never this.
                    "canonical": {"type": "string"},
                },
            },
        }
    },
}


def _build_prompt(cands: list[dict]) -> str:
    """Lean enumerated table — term + the heuristic context (pages, span,
    tier label) the thesis test needs. The term string alone already
    settles the dominant fragment noise."""
    lines = [
        "Classify each candidate term below. Reply with one verdict per term.",
        "",
    ]
    for i, c in enumerate(cands, 1):
        cats = c.get("categories") or 1
        label = c.get("label") or "candidate"
        lines.append(
            f'{i}. "{c["term"]}" — {c.get("pages", "?")} papers, '
            f'{cats} categor{"y" if cats == 1 else "ies"}, label={label}'
        )
    return "\n".join(lines)


def _parse_verdicts(text: str) -> list[dict] | None:
    """Parse `{"verdicts": [...]}` out of an LLM response. Mirrors the
    fence-strip + brace-regex fallback in tasks/suggest_splits._call_judge.
    Returns the verdicts list, or None if nothing parses (→ safe no-op)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return None
    verdicts = obj.get("verdicts")
    return verdicts if isinstance(verdicts, list) else None


def _judge_chunk(chunk: list[dict], *, use_stub: bool) -> list[dict] | None:
    """One LLM call over a chunk.

    Returns parsed verdicts or None on an untyped call failure, a missing prompt,
    or unparsable output. Typed environment/configuration failures propagate to
    the CLI funnel.
    """
    from ..agents import llm
    from ..paths import wiki_root

    try:
        system = (wiki_root() / "prompts" / TRIAGE_SYSTEM_FILENAME).read_text(encoding="utf-8")
    except OSError as e:
        print(f"  concept-triage: prompt file unreadable ({e}) — keeping chunk",
              file=sys.stderr)
        return None
    try:
        resp = llm.call(
            phase="concept_triage",
            prompt=_build_prompt(chunk),
            system=system,
            max_tokens=_MAX_TOKENS,
            schema=_TRIAGE_SCHEMA,
            use_stub=use_stub,
        )
    except EnvironmentFailure:
        raise
    except Exception as e:
        print(f"  concept-triage: LLM call failed ({e}) — keeping chunk",
              file=sys.stderr)
        return None
    parsed = _parse_verdicts(resp.text)
    if parsed is None:
        print("  concept-triage: unparsable LLM response — keeping chunk",
              file=sys.stderr)
        print(f"  {resp.text[:300]}", file=sys.stderr)
    return parsed


def triage_candidates(
    cands: list[dict] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    judge_fn=None,
    use_stub: bool = False,
) -> list[dict]:
    """Classify every candidate. Returns `[{**cand, verdict, reason}]`.

    `judge_fn(chunk) -> list[dict] | None` replaces the LLM entirely
    (dependency-injection seam for tests); it receives the chunk's
    candidate dicts and returns raw `{term, verdict, reason}` dicts.

    Safety: verdicts are matched back to candidates by SLUG (never the
    model's echoed string), so an invented term is discarded and the
    canonical `cand["term"]`/`cand["slug"]` travel with the verdict.
    Any candidate with no valid verdict defaults to `uncertain` (keep).
    """
    if cands is None:
        cands = collect_candidates()
    if not cands:
        return []

    results: list[dict] = []
    for start in range(0, len(cands), max(1, chunk_size)):
        chunk = cands[start:start + max(1, chunk_size)]
        by_slug = {c["slug"]: c for c in chunk}
        try:
            raw = judge_fn(chunk) if judge_fn is not None else _judge_chunk(chunk, use_stub=use_stub)
        except EnvironmentFailure:
            raise
        except Exception as e:   # an untyped failed chunk keeps its terms uncertain
            print(f"  concept-triage: judge failed ({e}) — keeping chunk", file=sys.stderr)
            raw = None

        seen: set[str] = set()
        for v in raw or []:
            if not isinstance(v, dict):
                continue
            verdict = str(v.get("verdict", "")).strip().lower()
            term = v.get("term")
            if verdict not in _VALID_VERDICTS or not isinstance(term, str):
                continue
            slug = _term_slug(term)
            cand = by_slug.get(slug)
            if cand is None:          # model invented / mangled a term — never act on it
                continue
            if slug in seen:          # ignore duplicate verdicts for one term
                continue
            seen.add(slug)
            reason = str(v.get("reason") or "").strip() or "(no reason given)"
            results.append({**cand, "verdict": verdict, "reason": reason,
                            "canonical": str(v.get("canonical") or "").strip()})

        for slug, cand in by_slug.items():   # fail-safe: unjudged terms are kept
            if slug not in seen:
                results.append({**cand, "verdict": "uncertain",
                                "reason": "no verdict returned"})
    return results


def apply_triage(results: list[dict], *, dry_run: bool = False) -> dict:
    """Write the noise verdicts to the decline list (unless `dry_run`).

    Declines using the candidate's own `term` (so the resulting slug
    provably equals the `slug` collect_candidates filters on). Returns a
    summary: per-verdict counts, the (would-be) declined terms, and the
    kept concept/uncertain terms.
    """
    counts: dict[str, int] = {}
    declined: list[dict] = []
    kept: list[dict] = []
    to_write: list[tuple[str, str]] = []
    for r in results:
        v = r["verdict"]
        counts[v] = counts.get(v, 0) + 1
        if v in NOISE_VERDICTS:
            # For an alias verdict, record the canonical it duplicates. The
            # decline still keys off the candidate's OWN term (slug-safe) — the
            # echoed `canonical` only decorates the human-readable reason.
            reason = (f"near-duplicate of '{r.get('canonical') or '?'}'"
                      if v == "alias" else r["reason"])
            to_write.append((r["term"], reason))
            declined.append({"term": r["term"], "verdict": v, "reason": reason})
        else:
            kept.append({"term": r["term"], "verdict": v, "reason": r["reason"]})
    # One atomic write for the whole batch — a triage run can decline hundreds
    # of terms, and per-term writes would rewrite the growing file each time.
    if not dry_run:
        add_declines(to_write, source="llm-triage")
    return {
        "dry_run": dry_run,
        "counts": counts,
        "declined": declined,
        "kept": kept,
        "total": len(results),
    }
