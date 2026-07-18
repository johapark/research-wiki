"""Salience scoring — does the wiki page capture the most important content
of its source PDF?

The fidelity graders (`coverage.py`, `fidelity.py`) ask: *is every claim on the
page findable in the PDF?* — precision over content. Salience is the orthogonal
recall axis: *did the page include what a reader would most want to know?*

The mechanism reuses `benchmark-fixture`'s machinery. A `ContentFixture` (the
hand-curated yaml format under `benchmark-fixtures/`) declares headline_claims,
capabilities, limitations, related_papers, and the `grade.scorer` runs
token-overlap (or LLM-judge) verdicts against a page. We **synthesize** a
fixture from PDF structure — abstract proxy, Results / Discussion lead-ins,
figure captions, extended-data captions — and feed it to the same scorer. No
LLM, no S2 lookup, fully deterministic.

What this measures vs. the curated path: the synthetic fixture is a *lower
bound* on salience. It catches anchors any reasonable page should hit (the
paper's lead claims and labelled-figure findings) but doesn't surface the
domain-judgment items a curator adds to `benchmark-fixtures/{stem}.yaml`. A page
with strong synthetic-salience may still miss curator-graded items; a page
with weak synthetic-salience is missing things the *PDF itself* flagged as
load-bearing.

Anchor sources and axis assignments (each silently skipped when absent):

  abstract (all)      → headline_claims (importance: critical) — the
                        author-condensed summary; every abstract sentence
                        is treated as a critical anchor. Extraction handles
                        both explicit `Abstract` headers and the headerless
                        Nature-family case via `sections.extract_abstract`.
  results §1 (1-2 s)  → headline_claims (high)
  discussion §1+§last → headline_claims (high), routed to limitations when
                        the sentence carries a limitations cue
  figure_captions[*]  → capabilities (high) — first sentence per caption block
  extended_data[*]    → capabilities (normal)

Introduction is intentionally NOT used as a critical anchor source: in
practice it overlaps heavily with the abstract (more verbose restatement),
so anchoring on it would double-count the same content with different
phrasing. The abstract is the authoritative critical signal.

Importance tiers and limitations cue regex are first-cut and provisional —
same posture as `fidelity.BM25_FLOOR`. Calibrate against
`benchmark-fixtures/{stem}.yaml` before tightening.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Direct (non-lazy) imports work now that `benchmark/__init__.py` no longer
# auto-imports `replicate.py` — the previous cycle was
#   benchmark/__init__.py
#     → benchmark.replicate
#       → agents.phases.grade_draft
#         → grade.fidelity.paper
#           → grade.salience
#             → benchmark.fixture (loops back into benchmark/__init__.py)
# Replicate is still in benchmark/ but consumers (tasks/benchmark_fixture.py)
# import it directly via `from researchwiki.benchmark.replicate import ...`.
from ..benchmark.fixture import ContentFixture, FixtureItem
from .scorer import score_text


# Sentence-end heuristic. Avoids splitting on common abbreviations
# ("Fig. 3", "et al.") by requiring whitespace+capital after the period.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\[])")

# Line-number prefix common in Nature accelerated previews and some arXiv
# preprints: every line begins with `\d+ ` (1-3 digits + whitespace) where
# the digits are running line numbers (1..~500). Strip from each line at
# anchor-extraction time so token-overlap scoring against the page body
# isn't dragged down by "26", "27" prefixes the page never repeats.
# 1-3 digit cap keeps real numerics like "5,377,346" from being eaten;
# applies at line starts only (multiline mode).
_LINE_NUMBER_PREFIX = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]+")

# Limitations cue: phrases that typically introduce a limitation, caveat, or
# unresolved future-work item. When a discussion sentence matches, we route
# it to the `limitations` axis instead of `headline_claims`.
_LIMITATION_CUE_RE = re.compile(
    r"\b(limitations?|caveats?|however|fails?|failed|failure|"
    r"does not generalize|out of scope|future work|"
    r"remains? (an )?open|unresolved)\b",
    re.IGNORECASE,
)

# Caption header — strip "Fig. 1 | Workflow" prefix to get the actual sentence.
# Mirrors `sections.CAPTION_START_RE` shape but anchored at start of block.
_CAPTION_HEADER_RE = re.compile(
    r"^\s*(?:Extended\s+Data\s+)?(?:Fig\.|Figure|Table)\s+\d+\w?\s*\|\s*",
    re.IGNORECASE,
)

# Default top-K missed anchors to surface in the report.
_TOP_K_MISSED = 5


@dataclass
class SalienceReport:
    n_anchors: int
    salience_score: float | None     # None when n_anchors == 0; else 0..1
    n_match: int
    n_partial: int
    n_miss: int
    per_axis: dict[str, dict[str, int]]   # {axis: {match, partial, miss}}
    missed_anchors: list[dict[str, Any]]  # top-K miss items: {axis, id, label, text}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter; good enough for first-/last-N selection.

    Strips line-number prefixes (e.g. "26 The cycle..." → "The cycle...")
    before splitting so anchors don't carry text-extraction noise into the
    overlap check downstream.
    """
    text = _LINE_NUMBER_PREFIX.sub("", text).strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


def _take_first(text: str, n: int) -> list[str]:
    return _split_sentences(text)[:n]


def _take_last(text: str, n: int) -> list[str]:
    sents = _split_sentences(text)
    return sents[-n:] if len(sents) >= n else sents


def _first_caption_sentence(block: str) -> str:
    """Strip the `Fig. N |` / `Table N |` header, return the first sentence."""
    body = _CAPTION_HEADER_RE.sub("", block, count=1).strip()
    sents = _split_sentences(body)
    return sents[0] if sents else ""


def _is_limitation(sentence: str) -> bool:
    return bool(_LIMITATION_CUE_RE.search(sentence))


def _make_item(item_id: str, importance: str, text: str) -> FixtureItem:
    """Wrap a sentence as a FixtureItem. `id` is informational here (no
    diff-keying as in curated fixtures)."""
    return FixtureItem(
        id=item_id,
        importance=importance,  # type: ignore[arg-type]
        verbalization=text,
        location=None,
    )


def synthesize_fixture(
    stem: str,
    pdf_text: str,
    sections: dict[str, str],
) -> ContentFixture:
    """Build a ContentFixture from PDF structure — no LLM, no S2.

    `pdf_text` is the full extracted text (only used as a fallback signal
    today; reserved for an S2-style abstract heuristic later). `sections`
    is the dict returned by `researchwiki.pdf.sections.anchor_sections`.

    Returns a ContentFixture whose `related_papers` is empty by design — the
    scorer's `_score_related_papers` then has no items, contributes nothing
    to the weighted denominator, and the recall is computed across the
    populated axes only.
    """
    headline: list = []
    capabilities: list = []
    limitations: list = []

    abstract = sections.get("abstract", "").strip()
    if abstract:
        # Every abstract sentence is a critical anchor — the abstract is the
        # author-condensed summary, so each sentence carries equal weight as
        # a load-bearing claim the page should cover.
        for i, s in enumerate(_split_sentences(abstract)):
            headline.append(_make_item(f"abstract-{i}", "critical", s))

    results = sections.get("results", "").strip()
    if results:
        for i, s in enumerate(_take_first(results, 2)):
            headline.append(_make_item(f"results-lead-{i}", "high", s))

    discussion = sections.get("discussion", "").strip()
    if discussion:
        # First two sentences set the headline framing; last two surface the
        # paper's own caveats. Route by limitations cue.
        for i, s in enumerate(_take_first(discussion, 2)):
            item = _make_item(f"discussion-lead-{i}", "high", s)
            (limitations if _is_limitation(s) else headline).append(item)
        for i, s in enumerate(_take_last(discussion, 2)):
            item = _make_item(f"discussion-tail-{i}", "high", s)
            (limitations if _is_limitation(s) else headline).append(item)

    fig_caps = sections.get("figure_captions", "").strip()
    if fig_caps:
        # Each caption block is separated by a blank line in `anchor_sections`.
        for i, block in enumerate(b for b in fig_caps.split("\n\n") if b.strip()):
            sent = _first_caption_sentence(block)
            if sent:
                capabilities.append(_make_item(f"fig-{i}", "high", sent))

    ed_caps = sections.get("extended_data", "").strip()
    if ed_caps:
        for i, block in enumerate(b for b in ed_caps.split("\n\n") if b.strip()):
            sent = _first_caption_sentence(block)
            if sent:
                capabilities.append(_make_item(f"ed-{i}", "normal", sent))

    return ContentFixture(
        paper_stem=stem,
        paper_type="paper",
        title="(synthetic salience anchors from PDF structure)",
        notes="Auto-generated by researchwiki.grade.salience; "
              "not a substitute for a hand-curated benchmark-fixtures/ entry.",
        headline_claims=headline,
        capabilities=capabilities,
        limitations=limitations,
        related_papers=[],
    )


def _per_axis_counts(score_report) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for axis, ar in score_report.axes.items():
        # Skip axes the synthetic fixture never populates so the report stays
        # focused on what salience actually measured.
        if ar.n_items == 0:
            continue
        out[axis] = {"match": ar.n_match, "partial": ar.n_partial, "miss": ar.n_miss}
    return out


def _missed_anchors(score_report, fixture, top_k: int) -> list[dict[str, Any]]:
    """Top-K `miss` items, paired with the fixture's verbalization for context.
    `critical` misses surface first; ties broken by axis order then id."""
    by_id = {it.id: it for _, it in fixture.all_items()}
    importance_rank = {"critical": 0, "high": 1, "normal": 2}
    misses: list[tuple[int, str, dict[str, Any]]] = []
    for axis, ar in score_report.axes.items():
        for v in ar.items:
            if v.verdict != "miss":
                continue
            item = by_id.get(v.item_id)
            if item is None:
                continue
            misses.append((
                importance_rank.get(v.importance, 3),
                axis,
                {
                    "axis": axis,
                    "id": v.item_id,
                    "importance": v.importance,
                    "text": item.verbalization,
                },
            ))
    misses.sort(key=lambda t: (t[0], t[1], t[2]["id"]))
    return [m[2] for m in misses[:top_k]]


def score_salience(
    stem: str,
    pdf_text: str,
    sections: dict[str, str],
    page_body: str,
    *,
    top_k_missed: int = _TOP_K_MISSED,
    use_semantic: bool = True,
) -> SalienceReport:
    """Synthesize a fixture from PDF anchors and score the page against it.

    `use_semantic` (default True): adds bi-encoder cosine alongside token
    overlap so paraphrased anchors still score. Falls back gracefully to
    token-only when the embedding model isn't available.

    Falls back to a zero-anchor report when no PDF structure is recoverable
    (e.g., scanned-image PDF, parsing failure) — `salience_score` is None and
    callers can distinguish "no signal" from "nothing matched".
    """
    fixture = synthesize_fixture(stem, pdf_text, sections)
    n_anchors = (
        len(fixture.headline_claims)
        + len(fixture.capabilities)
        + len(fixture.limitations)
    )

    if n_anchors == 0:
        return SalienceReport(
            n_anchors=0,
            salience_score=None,
            n_match=0, n_partial=0, n_miss=0,
            per_axis={},
            missed_anchors=[],
        )

    report = score_text(
        fixture, page_body,
        page_path=f"(salience:{stem})",
        use_llm=False,
        use_semantic=use_semantic,
    )

    n_match = sum(ar.n_match for ar in report.axes.values())
    n_partial = sum(ar.n_partial for ar in report.axes.values())
    n_miss = sum(ar.n_miss for ar in report.axes.values())

    return SalienceReport(
        n_anchors=n_anchors,
        salience_score=report.overall_weighted_recall,
        n_match=n_match,
        n_partial=n_partial,
        n_miss=n_miss,
        per_axis=_per_axis_counts(report),
        missed_anchors=_missed_anchors(report, fixture, top_k_missed),
    )
