"""LLM editorial-scoping — decides per-member scope-fit for each candidate.

One LLM call per candidate, judging every member as `in_scope`,
`tangential`, or `out_of_scope`. Batched per candidate (not per member)
because 5–8 members fit easily in one prompt, and a single call lets the
judge reason about the cluster as a whole (e.g., realize that 4 members
share a different scope than the target synthesis and reject them together).
Per-member calls would lose that comparative context.

The public entry point is `judge_candidate` — it mutates the passed-in
`Candidate` in place (member_verdicts, judge_topic, token counts, judged
flag). Called from `detect.find_candidates` when `judge=True`. Silently
no-ops if the LLM module can't be imported (stub test harnesses) or the
call fails.
"""

from __future__ import annotations

import json
import re

from ..log import log
from ..wiki import Page, extract_section
from .detect import Candidate, MemberVerdict


VALID_VERDICTS = {"in_scope", "tangential", "out_of_scope"}


# JSON Schema for the synthesis-judge envelope. Covers both the extend judge
# (no `topic` field expected) and the new judge (adds `topic`). `topic` is
# optional in the schema so one shape accepts both prompts; downstream code
# only reads `topic` when it makes sense for new-cluster judging. Honored by
# chat-relay; ignored by other providers.
_JUDGMENT_SCHEMA = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "topic": {"type": ["string", "null"]},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key", "verdict"],
                "properties": {
                    "key":       {"type": "string"},
                    "verdict":   {"type": "string",
                                  "enum": ["in_scope", "tangential",
                                           "out_of_scope"]},
                    "rationale": {"type": ["string", "null"]},
                },
            },
        },
    },
}


_EXTEND_JUDGE_SYSTEM = """\
You are a wiki editor reviewing whether proposed papers fit an existing
synthesis page's scope. The synthesis has a specific stated topic; a cluster
detector found these papers via citation/semantic similarity, but similarity
isn't the same as scope-fit.

For each proposed member, decide one of:
  - in_scope:     belongs in the synthesis's main structure (table cell,
                  primary section). Should be added with a one-line entry.
  - tangential:   related but on a different axis from the synthesis's
                  thesis. Belongs in a "tensions" or "open questions"
                  section, not the main table.
  - out_of_scope: different topic or axis. Should not be added at all.

Be strict. False positives dilute the synthesis's editorial focus.
Common out_of_scope patterns to watch for:
  - synthesis is about *prediction*, candidate is about *mitigation* — different axis
  - synthesis is about *Cas9*, candidate is about *prime editor* / *base editor*
  - candidate is the on-target predecessor of an already-cited paper

For each verdict provide a 1-sentence rationale (≤25 words).

Output JSON only:
{"verdicts": [
  {"key": "category/stem", "verdict": "in_scope|tangential|out_of_scope",
   "rationale": "..."},
  ...
]}
"""


_NEW_JUDGE_SYSTEM = """\
You are a wiki editor evaluating whether a cluster of papers warrants a
new synthesis page. The cluster was assembled by a graph-clustering
algorithm; not every member necessarily fits the emergent topic.

For each member, decide one of:
  - in_scope:     clear member of the cluster's emergent topic; should
                  anchor a synthesis page if one is created.
  - tangential:   weakly related; could be referenced peripherally but
                  is not a primary member.
  - out_of_scope: cluster noise; doesn't belong in this synthesis.

Also propose a 4–8 word topic title that captures the in_scope members'
shared theme. Title should be specific (e.g., "DNA foundation models in
the Hyena/Evo lineage"), not generic ("CRISPR research").

Be strict. A new synthesis with one or two noisy members is worse than
no synthesis.

Output JSON only:
{
  "topic": "<4-8 word topic title>",
  "verdicts": [
    {"key": "category/stem", "verdict": "in_scope|tangential|out_of_scope",
     "rationale": "..."},
    ...
  ]
}
"""


def _build_member_blurb(p: Page) -> str:
    """One member's metadata block fed to the judge prompt."""
    title = (p.fm.get("title", "") or "").strip()[:200]
    keywords = p.str_field("keywords").strip()
    summary = extract_section(p.body, "Summary").strip()[:600]
    return (
        f"### [[{p.key}]]\n"
        f"Title: {title}\n"
        + (f"Keywords: {keywords}\n" if keywords else "")
        + f"Summary: {summary}\n"
    )


def _build_extend_judge_prompt(
    target_title: str,
    target_body: str,
    member_blurbs: list[str],
) -> str:
    return "\n".join([
        f"# Target synthesis page: {target_title}",
        "",
        target_body[:3500],
        "",
        "---",
        "",
        f"# Proposed members to add ({len(member_blurbs)})",
        "",
        *member_blurbs,
        "",
        "---",
        "",
        "Output JSON per the system prompt. One verdict per member.",
    ])


def _build_new_judge_prompt(c: Candidate, member_blurbs: list[str]) -> str:
    return "\n".join([
        "# Cluster context",
        f"  members: {len(c.members)}",
        f"  density: {c.density:.2f}",
        f"  common keywords across cluster: "
        f"{', '.join(c.common_keywords[:10]) or '(none)'}",
        "",
        f"# Cluster members ({len(member_blurbs)})",
        "",
        *member_blurbs,
        "",
        "---",
        "",
        "Output JSON per the system prompt. Topic title + one verdict per member.",
    ])


def _parse_judgment_response(text: str) -> dict | None:
    """Tolerate fenced code, leading prose, or stray quotes around the JSON."""
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


def judge_candidate(
    c: Candidate,
    syntheses: list[Page],
    paper_pages: list[Page],
) -> None:
    """One LLM call to judge a candidate's members. Mutates `c` in place.

    Skipped silently when the LLM module isn't importable (e.g., in stub
    test harnesses) — c.judged stays False and downstream rendering
    treats the candidate as un-judged.
    """
    try:
        from ..agents import llm
    except ImportError:
        return

    by_key = {p.key: p for p in paper_pages}
    members_to_judge = (
        c.members_missing_from_nearest if c.verdict == "extend" else c.members
    )
    member_blurbs: list[str] = []
    for k in members_to_judge:
        p = by_key.get(k)
        if p is not None:
            member_blurbs.append(_build_member_blurb(p))
    if not member_blurbs:
        return

    if c.verdict == "extend":
        target_body = ""
        target_title = c.nearest_synthesis or "(unknown)"
        for s in syntheses:
            if s.key == c.nearest_synthesis:
                target_body = s.body
                target_title = s.fm.get("title", target_title)
                break
        prompt = _build_extend_judge_prompt(target_title, target_body, member_blurbs)
        system = _EXTEND_JUDGE_SYSTEM
    else:
        prompt = _build_new_judge_prompt(c, member_blurbs)
        system = _NEW_JUDGE_SYSTEM

    try:
        resp = llm.call(
            phase="synthesis_judge",
            prompt=prompt,
            system=system,
            schema=_JUDGMENT_SCHEMA,
        )
    except Exception as e:
        log(f"LLM call failed for {c.slug}: {e}", tag="judge")
        return

    parsed = _parse_judgment_response(resp.text)
    if parsed is None:
        log(f"could not parse JSON for {c.slug}", tag="judge")
        return

    raw_verdicts = parsed.get("verdicts") or []
    member_keys = set(members_to_judge)
    out: list[MemberVerdict] = []
    for v in raw_verdicts:
        if not isinstance(v, dict):
            continue
        key = (v.get("key") or "").strip()
        verdict = (v.get("verdict") or "").strip().lower()
        rationale = (v.get("rationale") or "").strip()[:300]
        if key not in member_keys or verdict not in VALID_VERDICTS:
            continue
        out.append(MemberVerdict(key=key, verdict=verdict, rationale=rationale))

    c.member_verdicts = out
    if c.verdict == "new":
        c.judge_topic = (parsed.get("topic") or "").strip()[:120]
    c.judge_input_tokens = resp.input_tokens
    c.judge_output_tokens = resp.output_tokens
    c.judged = True
