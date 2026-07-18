"""Concept-hub authored-prose contract lint.

A concept hub is a mini-synthesis around one term. Its *value* is the human
(or LLM)-authored prose — the Definition section (how the corpus frames the
term) and, on bridge hubs, the Cross-domain connections section (the payoff
paragraph). Without that authored content the hub is just an annotated
bibliography with a topic keyword.

These checks enforce the minimum contract:

  concept_definition_thin
    `## Definition` < DEFINITION_MIN_WORDS (40) content words after stripping
    markdown, HTML comments, wikilinks, and footnote refs. Signals that the
    author dropped a stub and moved on without actually writing the section.

  concept_missing_bridge_section
    Hub's `concept_span` ≥ 2 (bridge tier) but no `## Cross-domain
    connections` H2 exists in the body (or exists but is empty). The whole
    point of a bridge hub is the bridge story — if it's missing, the hub
    isn't earning its keep as a bridge.

  concept_definition_paraphrases_claim
    The Definition text has > DEFINITION_OVERLAP_MAX (0.7) token overlap
    (Jaccard-like) with any single spoke's source-claim hint text. Signals
    that the author copied one member paper's claim into the Definition
    rather than synthesizing across the corpus. Uses the hint comment
    on the spoke bullet (`<!-- how this paper uses <term>. hint: "..." -->`)
    as the source-claim proxy.

Checks are **warn-only** — reported by `lint --json` under
`concept_contract_violations` but never flip the exit code. Promote to
defect after two calibration rounds on real hubs.
"""

from __future__ import annotations

import re
from pathlib import Path

from .walk import all_pages


# Contract thresholds. Deliberately conservative — a hub with 39 authored
# words is probably fine, so a stricter cut would false-positive. Tighten
# after seeing real usage patterns.
DEFINITION_MIN_WORDS = 40
DEFINITION_OVERLAP_MAX = 0.7

# Section headings we recognize as the required Cross-domain block. Variants
# stay accepted — the point is that some form of the section exists with
# real content.
_BRIDGE_SECTION_HEADINGS = (
    "## Cross-domain connections",
    "## Cross-domain",
    "## Bridge",
    "## Bridges",
)

# Structural markup we strip before counting words.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_FOOTNOTE_REF_RE = re.compile(r"\[\^[^\]\s]+\]")
_MD_MARKERS_RE = re.compile(r"[*_`~]+")
_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)

# Content-word tokenization. Same shape as grounding's word count — keeps
# the two checks consistent about what "a word" is.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")

# Common English stopwords stripped from the overlap comparison so a hub's
# Definition doesn't get flagged for reusing "the", "of", "and", etc. from
# a spoke's hint. Small and stable — big lists would hide real overlap.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "our", "their",
    "which", "who", "what", "when", "where", "than", "then",
    "we", "they", "he", "she", "here", "there",
    "how", "why", "into", "onto", "over", "under",
    "not", "no", "any", "all", "each", "every", "both", "some",
})


def _section_text(body: str, headings: tuple[str, ...]) -> tuple[str | None, str]:
    """Extract the body of the first H2 section whose heading matches any of
    `headings` (case-insensitive, prefix match on the H2 name). Returns
    (matched_heading, section_body). Returns (None, "") when no match.
    """
    headers = list(_H2_RE.finditer(body))
    if not headers:
        return None, ""
    accepted = {h.lstrip("#").strip().lower() for h in headings}
    for i, m in enumerate(headers):
        title = m.group(1).strip().lower()
        # Prefix match against any accepted form.
        if any(title == a or title.startswith(a) for a in accepted):
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
            return m.group(0), body[start:end]
    return None, ""


def _content_words(text: str) -> list[str]:
    """Strip markup and return the lowercase content words (no stopwords)."""
    t = _HTML_COMMENT_RE.sub(" ", text)
    t = _WIKILINK_RE.sub(" ", t)
    t = _FOOTNOTE_REF_RE.sub(" ", t)
    t = _MD_MARKERS_RE.sub(" ", t)
    tokens = _WORD_RE.findall(t)
    return [t.lower() for t in tokens if t.lower() not in _STOPWORDS]


def _extract_spoke_hints(body: str) -> list[str]:
    """Pull each spoke bullet's hint text — the `hint: "..."` payload inside
    the trailing HTML comment. Returns [hint_text_1, hint_text_2, ...].
    """
    section = _section_text(
        body, ("## How it appears across the corpus",),
    )[1]
    if not section:
        return []
    # Hint comment shape: `<!-- how this paper uses <term>. hint: "..." -->`
    hint_re = re.compile(r'hint:\s*"([^"]+)"')
    return [m.group(1).strip() for m in hint_re.finditer(section)]


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """Ratio of |a ∩ b| / min(|a|, |b|). Directional overlap-coefficient
    style — 1.0 when one set is a subset of the other. Chosen over Jaccard
    because we want to detect "Definition contains most of a claim's words"
    even when the Definition is longer.
    """
    if not a or not b:
        return 0.0
    inter = a & b
    return len(inter) / min(len(a), len(b))


def _concept_span(fm) -> int | None:
    """Extract `concept_span:` from the frontmatter. Handles int, str, and
    the line-parser's stringified forms. Returns None when unparseable."""
    v = fm.get("concept_span")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return None


def check_hub(path: Path, body: str, fm: dict) -> list[dict]:
    """Run every contract check on one hub. Returns a list of violation dicts:
    {page, kind, detail}. Empty when the hub passes."""
    violations: list[dict] = []

    # (a) definition thin
    def_head, def_body = _section_text(body, ("## Definition",))
    if def_body is not None:
        words = _content_words(def_body)
        if def_head is None:
            violations.append({
                "page": path,
                "kind": "concept_missing_definition",
                "detail": "no `## Definition` H2 found",
            })
        elif len(words) < DEFINITION_MIN_WORDS:
            violations.append({
                "page": path,
                "kind": "concept_definition_thin",
                "detail": f"{len(words)} content words "
                          f"(< {DEFINITION_MIN_WORDS})",
            })

    # (b) missing bridge section on span ≥ 2 hubs
    span = _concept_span(fm)
    if span is not None and span >= 2:
        bridge_head, bridge_body = _section_text(body, _BRIDGE_SECTION_HEADINGS)
        if bridge_head is None or not _content_words(bridge_body):
            violations.append({
                "page": path,
                "kind": "concept_missing_bridge_section",
                "detail": f"concept_span={span} but no populated "
                          f"`## Cross-domain connections` section",
            })

    # (c) definition paraphrases a single spoke's claim
    if def_body:
        def_tokens = set(_content_words(def_body))
        for i, hint in enumerate(_extract_spoke_hints(body)):
            hint_tokens = set(_content_words(hint))
            if len(hint_tokens) < 5:
                # Ignore very short hints — the overlap ratio is unstable.
                continue
            ratio = _overlap_ratio(def_tokens, hint_tokens)
            if ratio > DEFINITION_OVERLAP_MAX:
                violations.append({
                    "page": path,
                    "kind": "concept_definition_paraphrases_claim",
                    "detail": f"Definition token-overlap with spoke "
                              f"#{i + 1} = {ratio:.2f} "
                              f"(> {DEFINITION_OVERLAP_MAX})",
                })
                break  # one flag per hub — the author can inspect the rest

    return violations


def find_concept_contract_violations(
    pages: list[Path], pages_body: dict[Path, str], pages_fm: dict[Path, dict],
) -> list[dict]:
    """Run contract checks on every wiki/concepts/*.md. Non-concept pages
    are skipped silently. Idempotent, no DB reads — every input is already
    on disk by the time lint gets here."""
    out: list[dict] = []
    for md in pages:
        # Concept hubs live under wiki/concepts/ (per CLAUDE.md invariant).
        if md.parent.name != "concepts":
            continue
        fm = pages_fm.get(md, {}) or {}
        if str(fm.get("type", "")).strip("\"'") != "concept":
            continue
        body = pages_body.get(md, "")
        out.extend(check_hub(md, body, fm))
    return out
