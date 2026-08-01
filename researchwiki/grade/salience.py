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

  abstract (first N)  → headline_claims (importance: critical) — the
                        author-condensed summary. Extraction handles both
                        explicit `Abstract` headers and the headerless
                        Nature-family case via `sections.extract_abstract`,
                        then two filters run (see `anchor_is_substantive` and
                        `_MAX_ABSTRACT_ANCHORS`): non-findings are dropped and
                        the survivors capped, because the extracted region
                        routinely over-reaches into the masthead or the
                        introduction.
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
from dataclasses import asdict, dataclass
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

# ---------- abstract-anchor guards ----------
#
# The abstract tier dominates this score: `critical` carries weight 3 in
# `scorer.IMPORTANCE_WEIGHTS`, so on a 353-paper corpus measurement the
# abstract accounted for a median 75% of the weighted denominator (p90 94%).
# That makes the abstract region's extraction quality the single biggest
# driver of `salience_score`, and the region is noisy in two distinct ways —
# hence two guards, applied in this order.

# Guard 1: prose check. A handful of PDFs yield a non-prose slab where the
# abstract should be — `cheng-2023` returned bibliography text ("Nature 562,
# 203-209 (2018). doi:... pmid:...", "Mach.", "Learn." — 100 "sentences" at 4.2
# words each), `yi-2026` a one-word title fragment. Their implied salience
# ceiling was ~0.21: no page could ever cover those anchors, so the score is
# unreachable rather than merely low. Corpus-wide, words-per-sentence has a
# median of 23.5 and only 3/353 papers fall below 10 — a clean separator with no
# cluster near the line.
#
# Scope, measured honestly: on the current corpus this guard is not what saves
# those papers. Of 268 papers with a recoverable abstract, one (`yi-2026`) falls
# below the floor, and the per-sentence filter below already empties it;
# `cheng-2023` now extracts no abstract at all. What guard 1 uniquely catches is
# the shape the per-sentence filter is blind to by construction: bibliography
# entries that are long in *characters* and claim-shaped ("Molecular
# architecture of the human chromatin remodelling complex.") but short in words.
# Only the slab-level ratio identifies those as a reference list. Kept as a
# categorical backstop — cheap, and when it trips every draft of that paper is
# otherwise scored against noise.
_MIN_WORDS_PER_SENTENCE = 10.0

# Guard 2: anchor count cap, applied to the *substantive* sentences (see
# `anchor_is_substantive`) in document order — NOT to the raw sentence index.
# The ordering is load-bearing. Extraction over-reaches at both ends: usually
# into the introduction (42% of abstracts hit `extract_abstract`'s 4000-char
# cap; median 17 sentences / 467 words where a real abstract is 5-12 / 150-350),
# but sometimes at the *front* — `bjornsson-2020` yields 27 sentences whose
# first 12 are the journal masthead and author list, with "BACKGROUND:" landing
# at index 12. A raw first-12 cap there keeps only the junk and discards the
# whole abstract (measured: 0.24 → 0.06). Filtering first inverts that to
# 0.24 → 0.34.
#
# 12 is calibrated against the hand-curated fixtures in `benchmark-fixtures/`:
# of the curated headline items that localize to an abstract sentence at all,
# 100% land in the first 10 (max index 8), so the cap discards nothing a human
# curator judged load-bearing. Re-measured with the shipped rules against an
# all-sentences baseline on a 58-paper sample (token-overlap verdicts): median
# 0.322 → 0.344, mean 0.319 → 0.340, 29 papers up / 11 down / 18 unchanged. The
# gains are the structurally-penalized tail — chen-2025 +0.16, christian-2026
# +0.16, jaganathan-2019 +0.16, bjornsson-2020 +0.10. The 11 that scored lower
# moved little (worst -0.07, chin-2019) and are pages that had been earning
# credit against introduction bleed.
_MAX_ABSTRACT_ANCHORS = 12

# Minimum anchor length. On commentary and opinion pieces the "abstract" region
# is body prose and short rhetorical asides come through as load-bearing ("We
# can't measure everything." — 28 chars). Cheap because real abstract prose is
# long: across a 50-paper sample the median abstract sentence runs 164 chars and
# only 7.1% fall under 60. Inspecting that slice, the clear majority are
# artifacts — running citations ("Circulation. 2026;153:1928-1939."), author
# fragments, emails, "All rights reserved", mid-clause splitter debris — against
# a minority of real-but-minor findings ("We also report results on the BEIR
# benchmark."). Net-positive on the denominator, at the cost of occasionally
# forgiving a terse claim.
_ANCHOR_MIN_CHARS = 60

# Front-matter the PDF's first block routinely bleeds into the abstract region.
# These are guaranteed misses — a page *should* omit a funding disclaimer — so
# they inflate the weighted denominator with content no page can earn credit
# for. Measured contribution: 62% of papers carry at least one, a median 4.0%
# of the weighted denominator, p90 14.5%, max 79%.
_BOILERPLATE_RE = re.compile(
    r"""
    doi:\s*10\.                     # inline DOI
  | \bEditor:\s                     # handling-editor line
  | \bReceived\b.{0,40}\bAccepted\b # submission-history run
  | \bCopyright:?\s                 # copyright notice
  | ©\s*\d{4}                       # copyright glyph + year
  | \bAll\ rights\ reserved\b
  | \bhad\ no\ role\ in\ study\ design
  | \bCompeting\ interests?\b
  | \bopen[-\ ]access\ article\b
  | \bcreativecommons\.org
  | \bcorrespondence\ (to|should\ be\ addressed)\b
  | \be-?mail:\s
  | \bis\ available\ at\ (www|http)  # journal masthead line
  | \bDownloaded\ from\ http         # PDF footer
  | \bKey\ ?[Ww]ords:\s              # trailing keyword list
  | \bSources\ of\ Funding\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Academic credentials — two or more in one "sentence" means the splitter is
# chewing through an author list ("Halldorsson, MSc; Asgeir Sigurdsson, BSc;
# Gudny A."), not a claim.
_CREDENTIAL_RE = re.compile(
    r"\b(PhD|MSc|BSc|MD|DPhil|MPH|DVM|PharmD|MBBS)\b"
)


def anchor_is_substantive(text: str) -> bool:
    """Is this sentence a checkable finding, rather than an extraction artifact?

    Applied to abstract-derived anchors before they enter the fixture, so junk
    stays out of the weighted denominator. Also consumed by the critic
    (`agents.phases.revise.coverage_gaps`), where a missed anchor becomes an
    *additive* instruction to the author — there a bad anchor makes the page
    worse rather than merely mis-scored, so both callers want the same rule and
    it lives here, next to the extraction it compensates for.

    Deliberately conservative: it removes shapes that are never findings, not
    shapes that merely look unimportant.
    """
    text = (text or "").strip()
    if len(text) < _ANCHOR_MIN_CHARS:
        return False
    if _BOILERPLATE_RE.search(text):
        return False
    # Lowercase-initial anchors start mid-clause ("near zero as recall
    # increases, which indicates...") — a splitter artifact, so the "missing"
    # content is a fragment of a sentence the page may already cover in full.
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    if len(_CREDENTIAL_RE.findall(text)) >= 2:
        return False
    if text.count(";") >= 3:
        return False
    return True


def _abstract_is_prose(text: str, sentences: list[str]) -> bool:
    """Reject a non-prose slab masquerading as an abstract (guard 1)."""
    if not sentences:
        return False
    return (len(text.split()) / len(sentences)) >= _MIN_WORDS_PER_SENTENCE


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
        # Abstract sentences are the `critical` tier — the author-condensed
        # summary, where "the page doesn't mention this" is a defect rather
        # than a choice. Two guards keep the tier honest; see their constants
        # for the calibration. Anchor ids stay tied to the ORIGINAL sentence
        # index so an id remains traceable back to the extracted abstract even
        # though the sequence now has gaps.
        sentences = _split_sentences(abstract)
        if _abstract_is_prose(abstract, sentences):
            eligible = [
                (i, s) for i, s in enumerate(sentences)
                if anchor_is_substantive(s)
            ]
            for i, s in eligible[:_MAX_ABSTRACT_ANCHORS]:
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
