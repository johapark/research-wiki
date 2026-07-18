"""Grader discrimination eval — trusted vs hallucinated.

Runs the BM25 coverage grader on:
  - 5 trusted wiki pages (the originals from the spike + the 3 Nature 2026 AAPs)
  - 3 hallucinated fixtures (one per category: cgt, genomics, compbio)

Reports per-page distributions and an aggregated trusted-vs-hallucinated
comparison so we can decide whether BM25 alone discriminates well enough,
or where exactly the semantic-similarity signal needs to add lift.

Run from the wiki root:
    python tests/grade-fixtures/run-eval.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

from researchwiki.grade.fidelity.paper import grade_page


TRUSTED = [
    "tao-2026-ai-guided-redesign-of-laboratory-evolved-reverse",
    "hofmeister-2023-accurate-rare-variant-phasing-of-whole-genome",
    "gottweis-2026-accelerating-scientific-discovery-with-co-scientist",
    "ghareeb-2026-a-multi-agent-system-for-automating",
    "aygun-2026-an-ai-system-to-help",
]

# stem -> path to hallucinated fixture
HALLUCINATED = {
    "tao-2026-ai-guided-redesign-of-laboratory-evolved-reverse":
        "tests/grade-fixtures/tao-2026-HALLUCINATED.md",
    "hofmeister-2023-accurate-rare-variant-phasing-of-whole-genome":
        "tests/grade-fixtures/hofmeister-2023-HALLUCINATED.md",
    "aygun-2026-an-ai-system-to-help":
        "tests/grade-fixtures/aygun-2026-HALLUCINATED.md",
}


def _summary(scores: list[float], label: str) -> str:
    if not scores:
        return f"{label}: (empty)"
    s = sorted(scores)
    n = len(s)
    return (
        f"{label}: n={n} "
        f"mean={statistics.fmean(s):.2f} "
        f"median={statistics.median(s):.2f} "
        f"p10={s[max(0, n // 10)]:.2f} "
        f"p25={s[max(0, n // 4)]:.2f} "
        f"p75={s[min(n - 1, 3 * n // 4)]:.2f} "
        f"p90={s[min(n - 1, 9 * n // 10)]:.2f} "
        f"min={s[0]:.2f} max={s[-1]:.2f}"
    )


def _grade(stem: str, page_path: str | None):
    """Return per-claim BM25 scores, semantic scores (or None), drift count."""
    report = grade_page(stem, page_path=page_path)
    graded = [c for c in report.claims if c.graded]
    bm25 = [c.top1_score for c in graded]
    sem = [c.semantic_score for c in graded] if report.semantic_available else []
    drift = sum(1 for c in graded if c.numeric_unmatched)
    return bm25, sem, drift


def main() -> None:
    papers_dir = Path("papers")
    missing = [s for s in TRUSTED if not (papers_dir / f"{s}.pdf").exists()]
    if missing:
        print(
            "SKIPPED: this eval requires the following PDFs under papers/ "
            "(they're not bundled with the repo):"
        )
        for stem in missing:
            print(f"  - papers/{stem}.pdf")
        print(
            "\nDrop the PDFs into papers/ (or run agent ingest on them) and re-run."
        )
        return

    trusted_bm25: list[float] = []
    trusted_sem: list[float] = []
    hallucinated_bm25: list[float] = []
    hallucinated_sem: list[float] = []
    t_drift_total = 0
    h_drift_total = 0

    print("=== Per-page results ===")
    print(f"{'kind':<13} {'stem':<55} {'n':>3} {'drift':>5} "
          f"{'BM25μ':>7} {'BM25med':>7} {'semμ':>5} {'semmed':>6}")

    for stem in TRUSTED:
        bm25, sem, drift = _grade(stem, page_path=None)
        trusted_bm25.extend(bm25)
        trusted_sem.extend([n for n in sem if n is not None])
        t_drift_total += drift
        if not bm25:
            continue
        sem_mean = statistics.fmean([n for n in sem if n is not None]) if sem else float("nan")
        sem_med = statistics.median([n for n in sem if n is not None]) if sem else float("nan")
        print(f"{'trusted':<13} {stem[:55]:<55} {len(bm25):>3} {drift:>5} "
              f"{statistics.fmean(bm25):>7.2f} {statistics.median(bm25):>7.2f} "
              f"{sem_mean:>5.2f} {sem_med:>6.2f}")

    for stem, fixture_path in HALLUCINATED.items():
        bm25, sem, drift = _grade(stem, page_path=fixture_path)
        hallucinated_bm25.extend(bm25)
        hallucinated_sem.extend([n for n in sem if n is not None])
        h_drift_total += drift
        if not bm25:
            continue
        label = f"halluc:{Path(fixture_path).stem.split('-HALL')[0][:30]}"
        sem_mean = statistics.fmean([n for n in sem if n is not None]) if sem else float("nan")
        sem_med = statistics.median([n for n in sem if n is not None]) if sem else float("nan")
        print(f"{'hallucinated':<13} {label:<55} {len(bm25):>3} {drift:>5} "
              f"{statistics.fmean(bm25):>7.2f} {statistics.median(bm25):>7.2f} "
              f"{sem_mean:>5.2f} {sem_med:>6.2f}")

    print()
    print("=== Aggregated BM25 distributions ===")
    print(_summary(trusted_bm25, "trusted     "))
    print(_summary(hallucinated_bm25, "hallucinated"))

    print()
    print("=== Aggregated semantic-similarity distributions ===")
    print(_summary(trusted_sem, "trusted     "))
    print(_summary(hallucinated_sem, "hallucinated"))

    print()
    print("=== Threshold sweep (semantic similarity) ===")
    print(f"{'thresh':>7} {'true_neg%':>10} {'false_neg%':>11} {'gap':>7}")
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        tn = sum(1 for s in hallucinated_sem if s < t) / max(1, len(hallucinated_sem))
        fn = sum(1 for s in trusted_sem if s < t) / max(1, len(trusted_sem))
        gap = tn - fn
        print(f"{t:>7.2f} {tn * 100:>9.1f}% {fn * 100:>10.1f}% {gap * 100:>6.1f}")

    print()
    print("=== Numeric drift ===")
    print(f"trusted drift count     : {t_drift_total}")
    print(f"hallucinated drift count: {h_drift_total}")


if __name__ == "__main__":
    main()
