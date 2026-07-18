"""Retrieval-fixture scoring dispatch + prose reports.

Two entry points:

- `score_retrieval(fixture, embedding, backend, ...)` — runs retrieval +
  scoring for one fixture under one embedding config. Routes by
  `fixture.fixture_type` to the right scorer in `benchmark.retrieval`.
- `print_retrieval_score` / `print_retrieval_diff` — human-readable output
  for single-run and A/B modes.

Called from `researchwiki.tasks.benchmark_fixture` when the fixture's
`fixture_type` is `claims` or `pages`.
"""

from __future__ import annotations

from .fixture import RetrievalFixture
from .retrieval import (
    retrieve_claims,
    retrieve_pages,
    score_claims_fixture,
    score_pages_fixture,
)


def _short_stem(stem: str, width: int = 50) -> str:
    """Trim a long stem for inline display, keeping the leading author-year."""
    if len(stem) <= width:
        return stem
    return stem[: width - 1] + "…"


def score_retrieval(
    fixture: RetrievalFixture, embedding: str, backend: str,
    *, trust_remote_code: bool = False,
    doc_prefix: str = "", query_prefix: str = "",
):
    """Run retrieval + scoring for one fixture under one embedding.
    Routes by fixture_type."""
    kw = dict(
        trust_remote_code=trust_remote_code,
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
    )
    if fixture.fixture_type == "claims":
        retrieved = retrieve_claims(fixture, embedding, backend, **kw)
        return score_claims_fixture(fixture, retrieved, embedding, backend)
    elif fixture.fixture_type == "pages":
        retrieved = retrieve_pages(fixture, embedding, backend, **kw)
        return score_pages_fixture(fixture, retrieved, embedding, backend)
    raise ValueError(f"unknown fixture_type: {fixture.fixture_type!r}")


def print_retrieval_score(fixture: RetrievalFixture, score) -> None:
    """Per-fixture report — header, aggregates, per-item table, must-not violations."""
    lines: list[str] = []
    lines.append(f"[retrieval] {score.fixture_id}")
    lines.append(f"  query: {fixture.query!r}")
    lines.append(
        f"  embedding={score.embedding}  backend={score.backend}  k={score.k}"
    )
    lines.append("")
    lines.append(
        f"  MRR              {score.mrr:.3f}\n"
        f"  nDCG@{score.k:<14d}{score.ndcg_at_k:.3f}\n"
        f"  recall (all)     "
        f"{int(round(score.expected_recall * len(score.per_item))):d}"
        f" / {len(score.per_item):d}   ({score.expected_recall:.0%})"
    )
    n_critical = sum(1 for it in score.per_item if it.importance == "critical")
    n_critical_hit = sum(
        1 for it in score.per_item if it.importance == "critical" and it.rank is not None
    )
    lines.append(
        f"  recall (critical) {n_critical_hit:d} / {n_critical:d}   "
        f"({score.expected_recall_critical:.0%})"
    )
    must_not_str = (
        f"{score.must_not_hits:d}   ⚠" if score.must_not_hits > 0 else "0"
    )
    lines.append(f"  must_not_hits    {must_not_str}")
    if score.rank_violations > 0:
        lines.append(f"  rank_violations  {score.rank_violations:d}   ⚠")

    # Per-item detail table
    lines.append("")
    lines.append("  per-item:")
    for it in score.per_item:
        mark = "✓" if it.rank is not None else "✗"
        rank_str = f"#{it.rank}" if it.rank is not None else "—"
        if it.rank_violation:
            mark = "≠"
            rank_str += " (≠ expected)"
        rationale = f"  ({it.rationale})" if it.rationale else ""
        lines.append(
            f"    {mark} {it.importance:<8s} [{rank_str:>5s}] "
            f"{_short_stem(it.expected_key, 60)}{rationale}"
        )

    # Must-not violations: identify which retrieved items are negative anchors
    if score.must_not_hits > 0:
        neg_stems = {n.paper_stem for n in fixture.must_not_appear}
        violations = [
            (i + 1, k) for i, k in enumerate(score.retrieved_keys)
            if k.split("§", 1)[0] in neg_stems
        ]
        rationale_for = {n.paper_stem: n.rationale for n in fixture.must_not_appear}
        lines.append("")
        lines.append("  must_not violations:")
        for rank, key in violations:
            stem = key.split("§", 1)[0]
            r = rationale_for.get(stem, "")
            r_str = f"  ({r})" if r else ""
            lines.append(f"    • #{rank}  {_short_stem(key, 60)}{r_str}")

    print("\n".join(lines))


def print_retrieval_diff(fixture: RetrievalFixture, baseline, candidate, diff) -> None:
    """A/B diff report — aggregates, per-item rank changes, must-not membership."""
    lines: list[str] = []
    lines.append(f"[retrieval A/B] {diff.fixture_id}")
    lines.append(f"  query: {fixture.query!r}")
    lines.append(f"  baseline:  {diff.baseline_label}")
    lines.append(f"  candidate: {diff.candidate_label}")
    lines.append("")

    def _arrow_pp(d: float) -> str:
        return f"{d * 100:+.1f}pp"

    lines.append(
        f"  Δ MRR              {diff.delta_mrr:+.3f}  "
        f"({baseline.mrr:.3f} → {candidate.mrr:.3f})"
    )
    lines.append(
        f"  Δ nDCG@{baseline.k:<11d}{diff.delta_ndcg:+.3f}  "
        f"({baseline.ndcg_at_k:.3f} → {candidate.ndcg_at_k:.3f})"
    )
    lines.append(
        f"  Δ recall (all)     {_arrow_pp(diff.delta_recall):>7s}  "
        f"({baseline.expected_recall:.0%} → {candidate.expected_recall:.0%})"
    )
    lines.append(
        f"  Δ recall (crit.)   {_arrow_pp(diff.delta_recall_critical):>7s}  "
        f"({baseline.expected_recall_critical:.0%} → {candidate.expected_recall_critical:.0%})"
    )
    lines.append(
        f"  Δ must_not_hits    {diff.delta_must_not_hits:+d}     "
        f"({baseline.must_not_hits} → {candidate.must_not_hits})"
    )

    def _rank_change(b: int | None, c: int | None) -> str:
        b_s = f"#{b}" if b else "—"
        c_s = f"#{c}" if c else "out"
        return f"[{b_s} → {c_s}]"

    if diff.improved:
        lines.append("")
        lines.append(f"  improved ({len(diff.improved)}):")
        for key, b, c in diff.improved:
            lines.append(f"    {_rank_change(b, c):<14s}  {_short_stem(key, 60)}")
    if diff.regressed:
        lines.append("")
        lines.append(f"  regressed ({len(diff.regressed)}):")
        for key, b, c in diff.regressed:
            lines.append(f"    {_rank_change(b, c):<14s}  {_short_stem(key, 60)}")
    if diff.must_not_left:
        lines.append("")
        lines.append(f"  must_not LEFT top-K ({len(diff.must_not_left)} — improvement):")
        for stem in diff.must_not_left:
            lines.append(f"    • {_short_stem(stem, 60)}")
    if diff.must_not_entered:
        lines.append("")
        lines.append(f"  must_not ENTERED top-K ({len(diff.must_not_entered)} — regression):")
        for stem in diff.must_not_entered:
            lines.append(f"    • {_short_stem(stem, 60)}")
    if not diff.improved and not diff.regressed:
        lines.append("")
        lines.append("  (no rank changes)")

    print("\n".join(lines))
