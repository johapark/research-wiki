"""Benchmark infrastructure — fixtures + retrieval scoring + style + replication.

Renamed from `eval/` to be honest about what lives here: hand-curated
fixtures, the replication driver, retrieval-quality benchmarks, and
style reporting. The page-content scorer moved out (now in
`grade/scorer.py`) because it's used by both the benchmark suite and
the production grader; the rest is genuinely benchmark methodology.

Fixtures live on disk under `benchmark-fixtures/`:
  - root: `{stem}.yaml` — content-coverage fixtures (the existing kind)
  - retrieval/: `{kind}/{slug}.yaml` — retrieval-quality fixtures with a
    `fixture_type:` discriminator

The on-disk dir name and the `benchmark-fixture` CLI command kept their
names — they're stable user-facing surfaces. Only the Python module
renamed.

Public surface:
  - `load_fixture(stem)` → ContentFixture | RetrievalFixture
  - `score_page(fixture, page_path, *, use_llm)` → ScoreReport (content)
    *Re-exported from `researchwiki.grade.scorer`* — the content scorer
    moved into `grade/` because it's the production-grader engine
    consumed by salience, not benchmark-only infrastructure.
  - `score_claims_fixture(...)`, `score_pages_fixture(...)` → RetrievalScore
  - `find_fixtures()` → list[stem] (recurses into retrieval/)
  - `replicate_score(...)` — NOT re-exported here. It depends on
    `agents.phases` (it runs the author phase), which would create
    a circular import at the package level. Callers import it directly:
    `from researchwiki.benchmark.replicate import replicate_score`.

The `benchmark-fixture` task wraps these in a CLI; dispatch is by the
returned fixture type.
"""

from __future__ import annotations

from .fixture import (
    ContentFixture,
    ExpectedClaim,
    ExpectedPage,
    NegativeAnchor,
    RetrievalFixture,
    find_fixtures,
    load_fixture,
)
from .retrieval import (
    RetrievalDiff,
    RetrievalScore,
    diff_retrieval_scores,
    retrieve_claims,
    retrieve_pages,
    score_claims_fixture,
    score_pages_fixture,
)
from ..grade.scorer import ScoreReport, score_page, score_text
from .style import StyleReport, compute_style

__all__ = [
    # Fixtures
    "ContentFixture", "RetrievalFixture",
    "ExpectedClaim", "ExpectedPage", "NegativeAnchor",
    "find_fixtures", "load_fixture",
    # Content scoring (re-exported from grade.scorer for backward compat)
    "ScoreReport", "score_page", "score_text",
    # Retrieval scoring
    "RetrievalScore", "RetrievalDiff",
    "score_claims_fixture", "score_pages_fixture",
    "diff_retrieval_scores",
    "retrieve_claims", "retrieve_pages",
    # Style
    "StyleReport", "compute_style",
]
