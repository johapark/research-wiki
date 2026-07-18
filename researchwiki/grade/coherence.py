"""Structural-conformance ('coherence') scoring — does the page look like a
properly-shaped wiki page?

The third axis alongside fidelity and salience. The per-PDF graders
(`grade.fidelity.paper`, `grade.fidelity.synthesis`, `grade.salience`)
measure *content* relative to the source PDF — what's grounded, what's
misattributed, what's missing. None of them measure whether the page is
**shaped** like a wiki page should be.

Consumed by the agent runner (`agents.runner`) as a tournament-ranking
signal alongside fidelity and salience, but not agent-specific — could
equally be called from a CLI grader against any markdown file.

A draft can score well on fidelity and salience while still being a single
1500-word paragraph with no `## Limitations`, no bullets in
`## Key Contributions`, and a block of unused wikilinks pasted at the
bottom. Coherence catches that.

This module is regex/heuristic only — no LLM, no PDF read. It operates on
the draft body string and the page type. v1 covers paper pages
(produced by `agent ingest`); synthesis and idea pages have their own
contracts (declared in CLAUDE.md) but don't go through this path.

Thresholds and weights are first-cut and provisional, same posture as
`fidelity.BM25_FLOOR` / `salience` overlap thresholds. Calibrate once we
have observations from real ingests.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Paper-page contract (v1) ─────────────────────────────────────────
#
# H2 sections every paper page must declare. Order matters in the contract
# (CLAUDE.md ingest template) but isn't enforced here — we only check
# presence. Keep names in sync with `prompts/ingest-digest.md` and the
# agent author prompt.
PAPER_REQUIRED_SECTIONS: tuple[str, ...] = (
    "Summary",
    "Key Contributions",
    "Methodology and Architecture",
    "Results",
    "Limitations",
    "Related Papers",
)

# Word-count band: tight enough that a single-paragraph collapse (~200
# words) fails; loose enough that a thorough page on a review paper
# (~3500 words) still passes. Outside this band signals trouble.
PAPER_WORD_MIN = 600
PAPER_WORD_MAX = 4000

# Bullet quality in `## Key Contributions`. The contract is "≥3 atomic
# claims," each substantive enough to be graded. <3 bullets means the
# author collapsed contributions into prose; <6 words per bullet usually
# means a label rather than a claim.
KEY_CONTRIB_MIN_BULLETS = 3
KEY_CONTRIB_MIN_WORDS_PER_BULLET = 6

# `## Limitations` must have at least one substantive bullet.
LIMITATIONS_MIN_WORDS_PER_BULLET = 10

# Wikilink density: stuffing wikilinks is a known failure mode of LLM
# authors trying to look well-cited. Cap density outside the
# `## Related Papers` section. 1 link per 40 words is loose enough that
# heavily-cited prose passes; tight enough to catch a draft that's 30%
# wikilinks by token volume.
WIKILINK_DENSITY_MIN_WORDS_PER_LINK = 40

# Weights sum to 1.0; score is the sum of weights of passing checks.
PAPER_WEIGHTS: dict[str, float] = {
    "sections_present": 0.30,
    "word_count_in_band": 0.20,
    "key_contributions_bullets": 0.15,
    "no_empty_sections": 0.15,
    "limitations_substantive": 0.10,
    "wikilink_density_sane": 0.10,
}


_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Multi-line-aware: a bullet runs from `[-*+] ` to the next bullet at any
# indent or to the next blank line. Indented continuation lines belong to
# the same bullet — common LLM-author output wraps long claims across
# lines.
_BULLET_RE = re.compile(
    r"^[ \t]*[-*+][ \t]+(.+?)(?=\n[ \t]*[-*+][ \t]|\n[ \t]*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class CoherenceReport:
    """Per-draft structural-conformance score.

    `score` is 0..1: sum of `weight` over passing checks. `violations`
    names the failing checks for debugging / log output. `sections_present`
    and `word_count` are surfaced because they're useful for downstream
    diagnostics regardless of pass/fail.
    """
    score: float
    violations: list[str] = field(default_factory=list)
    sections_present: list[str] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_code(body: str) -> str:
    return _FENCED_CODE_RE.sub("", body)


def _word_count(body: str) -> int:
    cleaned = _strip_code(body)
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", cleaned))


def _split_sections(body: str) -> dict[str, str]:
    """Map H2 title → section body (text up to next H2 or EOF)."""
    sections: dict[str, str] = {}
    headers = list(_H2_RE.finditer(body))
    for i, m in enumerate(headers):
        title = m.group(1).strip()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        sections[title] = body[m.end():end].strip()
    return sections


def _bullets_in(section_body: str) -> list[str]:
    return [m.group(1).strip() for m in _BULLET_RE.finditer(section_body)]


def _word_count_of(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", text))


def _score_paper(body: str) -> CoherenceReport:
    sections = _split_sections(body)
    section_titles = list(sections.keys())
    violations: list[str] = []
    score = 0.0

    # Check 1: required sections present. Match by case-insensitive prefix
    # so a descriptive suffix ("## Methodology and Architecture — overview")
    # still counts.
    title_lower = [t.lower() for t in section_titles]
    missing = [
        s for s in PAPER_REQUIRED_SECTIONS
        if not any(t.startswith(s.lower()) for t in title_lower)
    ]
    if missing:
        violations.append(f"missing sections: {missing}")
    else:
        score += PAPER_WEIGHTS["sections_present"]

    # Check 2: word count in band.
    n_words = _word_count(body)
    if PAPER_WORD_MIN <= n_words <= PAPER_WORD_MAX:
        score += PAPER_WEIGHTS["word_count_in_band"]
    else:
        violations.append(
            f"word count {n_words} outside [{PAPER_WORD_MIN}, {PAPER_WORD_MAX}]"
        )

    # Resolve actual section titles by required-name prefix for downstream
    # checks (the page might use "Key Contributions" or "Key Contributions
    # — TLDR"; both should map to the same slot).
    def _resolve(required: str) -> str | None:
        for t in section_titles:
            if t.lower().startswith(required.lower()):
                return t
        return None

    # Check 3: Key Contributions bullets.
    kc = _resolve("Key Contributions")
    if kc is not None:
        bullets = _bullets_in(sections[kc])
        substantive = [b for b in bullets if _word_count_of(b) >= KEY_CONTRIB_MIN_WORDS_PER_BULLET]
        if len(substantive) >= KEY_CONTRIB_MIN_BULLETS:
            score += PAPER_WEIGHTS["key_contributions_bullets"]
        else:
            violations.append(
                f"Key Contributions has {len(substantive)} substantive bullet(s) "
                f"(need ≥{KEY_CONTRIB_MIN_BULLETS} of ≥{KEY_CONTRIB_MIN_WORDS_PER_BULLET} words)"
            )
    elif "Key Contributions" not in missing:
        # Section name resolved earlier but not here — defensive; shouldn't fire.
        violations.append("Key Contributions present but unparseable")

    # Check 4: no empty section bodies.
    empties = [t for t, b in sections.items() if not b.strip()]
    if not empties:
        score += PAPER_WEIGHTS["no_empty_sections"]
    else:
        violations.append(f"empty sections: {empties}")

    # Check 5: Limitations substantive.
    lim = _resolve("Limitations")
    if lim is not None:
        bullets = _bullets_in(sections[lim])
        substantive = [b for b in bullets if _word_count_of(b) >= LIMITATIONS_MIN_WORDS_PER_BULLET]
        if substantive:
            score += PAPER_WEIGHTS["limitations_substantive"]
        else:
            violations.append(
                f"Limitations has no substantive bullet "
                f"(need ≥1 of ≥{LIMITATIONS_MIN_WORDS_PER_BULLET} words)"
            )

    # Check 6: wikilink density sane.
    rel = _resolve("Related Papers")
    rel_body = sections.get(rel, "") if rel else ""
    rel_links = len(_WIKILINK_RE.findall(rel_body))
    # Count links and words OUTSIDE the Related Papers section.
    outside_text = body
    if rel is not None:
        # Re-find the Related Papers slice and excise it.
        m = re.search(rf"^##\s+{re.escape(rel)}\s*$", body, re.MULTILINE)
        if m:
            outside_text = body[:m.start()]
    outside_links = len(_WIKILINK_RE.findall(outside_text))
    outside_words = _word_count(outside_text)
    density_ok = (
        rel_links >= 1 and
        (outside_words == 0
         or outside_words / max(outside_links, 1) >= WIKILINK_DENSITY_MIN_WORDS_PER_LINK)
    )
    if density_ok:
        score += PAPER_WEIGHTS["wikilink_density_sane"]
    else:
        if rel_links < 1:
            violations.append("Related Papers section has no wikilinks")
        else:
            ratio = outside_words / max(outside_links, 1)
            violations.append(
                f"wikilink density too high outside Related Papers "
                f"({outside_links} links / {outside_words} words = "
                f"1 per {ratio:.0f} words; need ≥1 per "
                f"{WIKILINK_DENSITY_MIN_WORDS_PER_LINK})"
            )

    return CoherenceReport(
        score=round(score, 4),
        violations=violations,
        sections_present=section_titles,
        word_count=n_words,
    )


def score_coherence(draft_body: str, page_type: str = "paper") -> CoherenceReport:
    """Score a draft body against its page-type contract.

    v1 supports `page_type='paper'` only; other types return a no-op
    1.0-scored report so the rest of the agent pipeline doesn't have to
    branch. When more page-types are wired through agent ingest, add
    their contracts here.
    """
    if page_type != "paper":
        # Unknown / unsupported page type — return a neutral pass.
        return CoherenceReport(
            score=1.0,
            violations=[],
            sections_present=[],
            word_count=_word_count(draft_body),
        )
    return _score_paper(draft_body)
