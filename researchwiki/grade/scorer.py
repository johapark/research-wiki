"""Fixture-based page scoring.

Originally written for the curated `benchmark-fixtures/{stem}.yaml` benchmark
suite. Now also the engine behind `grade/salience.py` — a synthetic
fixture built from PDF anchors flows through the same verdict logic as
a hand-curated fixture. Living in `grade/` reflects this reality: the
scorer is production grader infrastructure, not benchmark-only.

Five axes per fixture: headline_claims, capabilities, limitations,
related_papers, comparator_fidelity. Each item is independently judged
`match` / `partial` / `miss`. Aggregate is recall, importance-weighted
(critical=3, high=2, normal=1).

Three verdict paths combine per item:

  - **token-overlap** (always runs): significant-token Jaccard between
    fixture verbalization and page body. ≥75% match, ≥45% partial,
    else miss. Modulated by numeric integrity — every numeric token in
    the verbalization should appear in the page.

  - **bi-encoder cosine** (when `use_semantic=True`): page body is
    paragraph-chunked and embedded once via BAAI/bge-small-en-v1.5;
    each anchor's max cosine over chunks supplies a paraphrase-tolerant
    signal alongside token overlap. Either ≥0.75 cosine OR ≥0.75
    overlap satisfies match. Numeric drift remains a hard check.

  - **llm-judge** (when `use_llm=True`): a per-item JSON-schema LLM
    call asks "does this page assert this fixture item?" Tolerant of
    paraphrase and rewording. Cost: O(items) per page. Off by default.

Mechanical paths handle `related_papers` (wikilink presence check) and
`comparator_fidelity` (ratio + comparator pairing); they don't go
through the verdict combinators above.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ..benchmark.fixture import ContentFixture, FixtureItem


Verdict = Literal["match", "partial", "miss"]


IMPORTANCE_WEIGHTS = {"critical": 3, "high": 2, "normal": 1}
VERDICT_SCORES = {"match": 1.0, "partial": 0.5, "miss": 0.0}


WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
# Matches numeric tokens with optional unit/multiplier suffix and Unicode
# superscript exponent. Handles forms commonly found in research-paper prose:
#   "20×", "20-fold", "23.0", "12.4 ± 1.0", "n=5", "67.4%", "5,202", "10⁶".
#
# Two changes vs. the original regex (R2 fix on 2026-06-13):
#   1. Trailing `\b` removed. Word-boundary required a word→non-word
#      transition, which fails when the suffix itself is non-word
#      (×, %, °, ⁰-⁹). Result: "20×" was matching "20" (suffix dropped),
#      breaking numerical-fidelity check on ratios. Now the suffix is
#      captured.
#   2. Unicode superscript digits added inline so "10⁶" matches as one
#      token instead of "10". Common in scientific notation.
#
# Limitations (acknowledged):
#   - Mantissa-and-exponent like "1×10⁶" still fragments to ["1×", "10⁶"]
#     — semantic equivalence to "1,000,000" or "1 million" needs a
#     normalization layer the heuristic doesn't have.
#   - Dates "2026-01-01" still fragment to ["2026", "01", "01"]. In
#     practice this is fine: pages tend to repeat the year and the parts
#     that match are usually enough to satisfy the heuristic.
NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*[⁰¹²³⁴⁵⁶⁷⁸⁹]*"     # main + optional superscript exponent
    r"(?:[×x]|%|‰|°|-fold|fold)?",        # optional unit/multiplier
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "a", "an", "of", "for", "with", "and", "or", "in", "on", "at", "to",
    "from", "by", "as", "across", "over", "all", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "can", "could", "may", "might", "will", "would",
    "should", "must", "shall", "than", "when", "where", "which", "who", "whom",
    "whose", "what", "why", "how", "it", "its", "their", "they", "them",
}


@dataclass
class ItemVerdict:
    axis: str
    item_id: str
    importance: str
    verdict: Verdict
    rationale: str = ""
    matched_excerpt: str = ""

    def weighted(self) -> tuple[float, float]:
        """Return (achieved, possible) score weighted by importance."""
        w = IMPORTANCE_WEIGHTS[self.importance]
        return VERDICT_SCORES[self.verdict] * w, 1.0 * w


@dataclass
class AxisReport:
    axis: str
    n_items: int
    n_match: int
    n_partial: int
    n_miss: int
    weighted_recall: float
    items: list[ItemVerdict] = field(default_factory=list)


@dataclass
class ScoreReport:
    paper_stem: str
    page_path: str
    use_llm: bool
    overall_weighted_recall: float
    axes: dict[str, AxisReport]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes"] = {a: asdict(r) for a, r in self.axes.items()}
        return d


# ── token-overlap heuristic (fallback when no LLM) ──────────────────


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text) if t.lower() not in STOPWORDS}


def _significant_tokens(verbalization: str) -> set[str]:
    """Tokens carrying meaning — drops stopwords and very short tokens."""
    return {t for t in _tokenize(verbalization) if len(t) >= 4 or t.isupper()}


# Semantic-cosine thresholds for paraphrase tolerance. When a precomputed
# bi-encoder cosine is available alongside token overlap, either signal can
# satisfy the verdict — token-only authoring still passes, and prose that
# rewords the claim still scores.
#
# Calibrated 2026-06 against the 6 curated content fixtures (145 verdicts).
# 25% of all matches depend on the cosine path (overlap < 0.65 + cosine ≥
# 0.75); the threshold sits exactly at the median of the match-vs-partial
# boundary. Lowering would introduce false matches on weak-paraphrase
# items; raising would lose paraphrase tolerance. See
# `benchmark-fixtures/CALIBRATION-2026-06.md` for the full per-verdict
# inspection and the score-distribution evidence.
_SEM_MATCH_THRESHOLD = 0.75
_SEM_PARTIAL_THRESHOLD = 0.50


def _heuristic_verdict(
    item: FixtureItem,
    page_body: str,
    *,
    semantic_score: float | None = None,
) -> ItemVerdict:
    """Token-overlap + optional bi-encoder cosine.

    Token-overlap path (always runs): count fixture-significant tokens that
    appear in the page. ≥75% → match, ≥45% → partial, else miss. Numbers in
    the verbalization weigh extra — each missing number drops the verdict.

    Semantic path (when `semantic_score` is provided): a high cosine
    promotes a low-overlap claim. Verdict is the *better* of the two paths,
    capped by numeric integrity (a missing-number drift still drops match
    to partial; ≥2 missing numbers still drop to miss).
    """
    sig = _significant_tokens(item.verbalization)
    page_tokens = _tokenize(page_body)
    if not sig:
        return ItemVerdict(
            axis="(unset)", item_id=item.id, importance=item.importance,
            verdict="miss", rationale="empty verbalization",
        )
    overlap = sig & page_tokens
    overlap_ratio = len(overlap) / len(sig)

    nums_in_fixture = set(NUMBER_RE.findall(item.verbalization))
    nums_in_page = set(NUMBER_RE.findall(page_body))
    num_miss = nums_in_fixture - nums_in_page

    sem_match = semantic_score is not None and semantic_score >= _SEM_MATCH_THRESHOLD
    sem_partial = semantic_score is not None and semantic_score >= _SEM_PARTIAL_THRESHOLD

    if (overlap_ratio >= 0.75 or sem_match) and not num_miss:
        verdict: Verdict = "match"
    elif (overlap_ratio >= 0.45 or sem_partial) and len(num_miss) <= 1:
        verdict = "partial"
    else:
        verdict = "miss"

    rationale = f"token overlap {len(overlap)}/{len(sig)} ({overlap_ratio:.0%})"
    if semantic_score is not None:
        rationale += f"; semantic {semantic_score:.2f}"
    if num_miss:
        rationale += f"; numbers missing: {sorted(num_miss)}"
    return ItemVerdict(
        axis="(unset)", item_id=item.id, importance=item.importance,
        verdict=verdict, rationale=rationale,
    )


# ── related_papers (mechanical wikilink check) ──────────────────────


def _score_related_papers(items: list[FixtureItem], page_body: str) -> AxisReport:
    found = {m.strip() for m in WIKILINK_RE.findall(page_body)}
    verdicts: list[ItemVerdict] = []
    for it in items:
        link = it.link or ""
        if link in found:
            v = ItemVerdict(
                axis="related_papers", item_id=it.id, importance=it.importance,
                verdict="match", rationale=f"wikilink present: [[{link}]]",
            )
        else:
            v = ItemVerdict(
                axis="related_papers", item_id=it.id, importance=it.importance,
                verdict="miss", rationale=f"wikilink not on page: [[{link}]]",
            )
        verdicts.append(v)
    return _axis_report("related_papers", verdicts)


# ── comparator_fidelity (mechanical: ratio paired with comparator) ──


def _comparator_check(items: list[FixtureItem], page_body: str) -> AxisReport:
    """For each headline claim with a `relation`, verify the page actually
    pairs the expected ratio with the expected comparator.

    The previous implementation used a 300-char proximity window which
    produced false positives when a page listed multiple comparators in
    nearby prose (e.g., a 2-row scalability table where each row has its
    own comparator). The 300-char window crossed table-row boundaries and
    matched ratios with whichever comparator was closest in scan order.

    v2 algorithm (fixes the bug):
      For each occurrence of the ratio in the page, take a ±100-char
      window. Within that window, identify ALL comparator-like words from
      the fixture's `relation.comparator` field. Match only if the
      EXPECTED comparator is the *closest* comparator-like word to the
      ratio. This rejects cases where the ratio is closer to a different
      comparator (e.g., 20× sitting next to RCSB while pyScoMotif appears
      farther away in the same window).

    Returns "match" when ratio + correct-comparator pairing is found,
    "partial" when ratio exists on the page but isn't paired with the
    expected comparator, "miss" when the ratio doesn't appear at all.
    """
    # A relation with a blank/whitespace comparator can't be comparator-checked
    # (and `comparator.split()[0]` below would IndexError on it), so exclude it.
    # Carry the relation alongside the item so the loops below don't have to
    # re-narrow `it.relation` away from None on every access — the filter here
    # already guarantees it. (This used to need an `assert rel is not None` in
    # each loop, which `python -O` strips.)
    relational = [(it, it.relation) for it in items
                  if it.relation and it.relation.comparator.split()]
    # Collect ALL distinct comparator first-words across the fixture so we can
    # check whether an OTHER fixture's comparator is closer to a given ratio
    # — a useful proxy for "page paired ratio with the wrong baseline."
    all_comparators: set[str] = set()
    for _it, rel in relational:
        first = re.sub(r"[^\w-].*$", "", rel.comparator.split()[0])
        if first:
            all_comparators.add(first.lower())

    verdicts: list[ItemVerdict] = []
    for it, rel in relational:
        ratio_pat = re.compile(re.escape(rel.ratio), re.IGNORECASE)
        comparator_first = re.sub(
            r"[^\w-].*$", "", rel.comparator.split()[0]
        ) or rel.comparator.split()[0]
        comp_pat = re.compile(r"\b" + re.escape(comparator_first) + r"\b", re.IGNORECASE)

        verdict: Verdict = "miss"
        rationale = ""
        excerpt = ""
        ratio_hits = list(ratio_pat.finditer(page_body))

        for m in ratio_hits:
            # ±100 chars: tight enough to keep within a single sentence or
            # table row; loose enough to catch normal prose pairings like
            # "X is 20× faster than Y".
            window_start = max(0, m.start() - 100)
            window_end = min(len(page_body), m.end() + 100)
            window = page_body[window_start:window_end]

            # Where does the EXPECTED comparator appear (if at all)?
            expected_offsets = [mm.start() for mm in comp_pat.finditer(window)]
            if not expected_offsets:
                continue

            # Find the closest expected-comparator occurrence to the ratio.
            ratio_offset_in_window = m.start() - window_start
            closest_expected = min(
                expected_offsets,
                key=lambda off: abs(off - ratio_offset_in_window),
            )
            expected_dist = abs(closest_expected - ratio_offset_in_window)

            # Find the closest OTHER fixture-comparator. If a different
            # comparator is closer, this is a comparator-drift signal.
            other_distances = []
            for other in all_comparators - {comparator_first.lower()}:
                other_pat = re.compile(r"\b" + re.escape(other) + r"\b", re.IGNORECASE)
                for omm in other_pat.finditer(window):
                    other_distances.append(abs(omm.start() - ratio_offset_in_window))

            if other_distances and min(other_distances) < expected_dist:
                # Drift case: a different fixture-comparator is closer. Mark
                # partial and keep scanning (some other ratio occurrence
                # might be cleanly paired).
                if verdict == "miss":
                    verdict = "partial"
                    rationale = (
                        f"ratio {rel.ratio!r} present but a different fixture "
                        f"comparator is closer to it than {rel.comparator!r}"
                    )
                continue

            verdict = "match"
            rationale = f"ratio {rel.ratio!r} paired with {rel.comparator!r}"
            excerpt = window.strip()[:240]
            break

        if verdict == "miss" and ratio_hits:
            verdict = "partial"
            rationale = (
                f"ratio {rel.ratio!r} present on page but not within 100 chars of "
                f"comparator {rel.comparator!r}"
            )
        elif verdict == "miss":
            rationale = f"ratio {rel.ratio!r} not found on page"
        verdicts.append(ItemVerdict(
            axis="comparator_fidelity", item_id=it.id, importance=it.importance,
            verdict=verdict, rationale=rationale, matched_excerpt=excerpt,
        ))
    return _axis_report("comparator_fidelity", verdicts)


# ── LLM judge (optional, for prose-shaped items) ────────────────────


_JUDGE_SYSTEM = """\
You judge whether a wiki-page summary asserts a specific reference claim from
a source paper. The fixture verbalizes one claim from the paper. The wiki page
is meant to summarize the paper. Decide whether the page asserts the same
content, in the same direction and with the same magnitude.

Output strict JSON: {"verdict": "match|partial|miss", "rationale": "<≤25 words>", "excerpt": "<verbatim page passage that supports your verdict, or empty>"}

Verdict criteria:
  - match: the page asserts the claim with the same direction, magnitude, and
    named entities. Paraphrase is fine; substantive equivalence required.
  - partial: the page asserts a RELATED but INCOMPLETE claim — softer
    framing, OMITTED number (silent on a number the claim mentions),
    omitted comparator name, or only covers part of a multi-part claim.
    Reserved for MISSING-detail cases.
  - miss: the page does not assert this claim at all, OR asserts a direct
    CONTRADICTION (different number, opposite direction, different named
    entity). A wrong number is "miss", not "partial" — partial is for
    silence about detail, not for contradicting it. Example: claim says
    "<25 hours", page says "<5 hours" → miss (wrong magnitude). Claim
    says "23.0 months follow-up", page says "follow-up exceeds 12
    months" → partial (no contradiction, just missing the precise value).
"""


def _llm_judge(item: FixtureItem, page_body: str) -> ItemVerdict | None:
    """One LLM call per item. Returns None if LLM is unavailable; caller
    falls back to the heuristic. The judge is intentionally a thin
    yes/partial/no decision — the LLM does paraphrase tolerance, the
    aggregate-scoring layer does importance weighting."""
    try:
        from ..agents import llm
    except Exception:
        return None
    try:
        cfg_available = llm.is_real_mode_available()
    except Exception:
        cfg_available = False
    if not cfg_available:
        return None

    prompt = (
        f"FIXTURE CLAIM (id={item.id}):\n"
        f"{item.verbalization.strip()}\n\n"
        f"WIKI PAGE BODY:\n"
        f"{page_body[:18000]}\n\n"
        f"Output strict JSON now."
    )
    try:
        resp = llm.call(
            phase="eval_judge",
            prompt=prompt,
            system=_JUDGE_SYSTEM,
        )
    except Exception as e:
        return ItemVerdict(
            axis="(unset)", item_id=item.id, importance=item.importance,
            verdict="miss", rationale=f"LLM error: {type(e).__name__}: {e}",
        )

    text = resp.text.strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return ItemVerdict(
            axis="(unset)", item_id=item.id, importance=item.importance,
            verdict="miss", rationale=f"LLM returned non-JSON: {text[:80]!r}",
        )
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return ItemVerdict(
            axis="(unset)", item_id=item.id, importance=item.importance,
            verdict="miss", rationale=f"LLM returned malformed JSON: {text[:80]!r}",
        )
    v = parsed.get("verdict", "miss")
    if v not in ("match", "partial", "miss"):
        v = "miss"
    return ItemVerdict(
        axis="(unset)", item_id=item.id, importance=item.importance,
        verdict=v,
        rationale=str(parsed.get("rationale", ""))[:200],
        matched_excerpt=str(parsed.get("excerpt", ""))[:240],
    )


# ── per-axis scoring ────────────────────────────────────────────────


def _score_verbalization_axis(
    axis: str,
    items: list[FixtureItem],
    page_body: str,
    use_llm: bool,
    semantic_lookup: dict[str, float] | None = None,
) -> AxisReport:
    verdicts: list[ItemVerdict] = []
    for it in items:
        v: ItemVerdict | None = None
        if use_llm:
            v = _llm_judge(it, page_body)
        if v is None:
            sem = semantic_lookup.get(it.id) if semantic_lookup else None
            v = _heuristic_verdict(it, page_body, semantic_score=sem)
        v.axis = axis
        verdicts.append(v)
    return _axis_report(axis, verdicts)


def _axis_report(axis: str, verdicts: list[ItemVerdict]) -> AxisReport:
    n_match = sum(1 for v in verdicts if v.verdict == "match")
    n_partial = sum(1 for v in verdicts if v.verdict == "partial")
    n_miss = sum(1 for v in verdicts if v.verdict == "miss")
    achieved = sum(v.weighted()[0] for v in verdicts)
    possible = sum(v.weighted()[1] for v in verdicts)
    recall = (achieved / possible) if possible else 0.0
    return AxisReport(
        axis=axis, n_items=len(verdicts),
        n_match=n_match, n_partial=n_partial, n_miss=n_miss,
        weighted_recall=recall, items=verdicts,
    )


# ── public entry ────────────────────────────────────────────────────


def _build_semantic_lookup(
    fixture: ContentFixture, body: str
) -> dict[str, float] | None:
    """Embed the page body once and compute per-item bi-encoder cosine
    against each verbalization-bearing fixture item. Returns
    {item_id: max_cosine} or None when the embedding model isn't available
    or the body has no usable chunks.

    Page chunks: paragraph-split (blank-line boundaries), drop very short
    chunks (<20 chars after stripping). For typical paper pages (≤1500
    words / 5–15 paragraphs) embedding cost is single-digit seconds on CPU
    on first call; cached chunk embeddings aren't worth it at this scale.
    """
    from ..index import embeddings as semantic_mod
    if not semantic_mod.is_available():
        return None
    chunks = [
        c.strip()
        for c in re.split(r"\n[ \t]*\n+", body)
        if len(c.strip()) >= 20
    ]
    if not chunks:
        return None
    chunk_embs = semantic_mod.embed_texts(chunks)
    if chunk_embs is None or chunk_embs.shape[0] == 0:
        return None
    out: dict[str, float] = {}
    for it in (
        fixture.headline_claims + fixture.capabilities + fixture.limitations
    ):
        if not it.verbalization:
            continue
        result = semantic_mod.score_claim(
            it.verbalization, chunks, chunk_embeddings=chunk_embs,
        )
        if result is not None:
            out[it.id] = float(result.score)
    return out or None


def score_text(
    fixture: ContentFixture,
    body: str,
    *,
    page_path: str = "(in-memory)",
    use_llm: bool = False,
    use_semantic: bool = False,
) -> ScoreReport:
    """Score an in-memory page body against `fixture`. Used by the
    replication driver, which scores fresh author drafts before they're
    written to disk.

    `use_semantic`: when True, embed the page body via the bi-encoder
    once and supply per-item cosine scores to the verdict logic. Either
    a high overlap or a high cosine satisfies the match threshold —
    paraphrase-tolerant. Falls back gracefully (no semantic, token-only)
    when the embedding model can't be loaded.
    """
    semantic_lookup = (
        _build_semantic_lookup(fixture, body) if use_semantic else None
    )

    axes: dict[str, AxisReport] = {}
    axes["headline_claims"] = _score_verbalization_axis(
        "headline_claims", fixture.headline_claims, body, use_llm, semantic_lookup,
    )
    axes["capabilities"] = _score_verbalization_axis(
        "capabilities", fixture.capabilities, body, use_llm, semantic_lookup,
    )
    axes["limitations"] = _score_verbalization_axis(
        "limitations", fixture.limitations, body, use_llm, semantic_lookup,
    )
    axes["related_papers"] = _score_related_papers(fixture.related_papers, body)
    axes["comparator_fidelity"] = _comparator_check(fixture.headline_claims, body)

    achieved = sum(
        sum(v.weighted()[0] for v in a.items) for a in axes.values()
    )
    possible = sum(
        sum(v.weighted()[1] for v in a.items) for a in axes.values()
    )
    overall = (achieved / possible) if possible else 0.0

    return ScoreReport(
        paper_stem=fixture.paper_stem,
        page_path=page_path,
        use_llm=use_llm,
        overall_weighted_recall=overall,
        axes=axes,
    )


def score_page(
    fixture: ContentFixture,
    page_path: Path | str,
    *,
    use_llm: bool = False,
) -> ScoreReport:
    """Score `page_path` (a wiki paper page) against `fixture`.

    Returns five axes — headline_claims, capabilities, limitations,
    related_papers (mechanical), comparator_fidelity (mechanical) —
    and an overall weighted-recall score.

    `use_llm=False` (default) uses the token-overlap heuristic for
    verbalization-based axes; safe and offline. `use_llm=True` uses
    a per-item LLM judge (one call per item; ~30 calls per fixture).
    """
    body = Path(page_path).read_text(encoding="utf-8")
    return score_text(fixture, body, page_path=str(page_path), use_llm=use_llm)
