"""Leave-one-out evaluation of the Tantivy-backed category classifier.

For each paper in the wiki:
  - Rebuild the index excluding that paper
  - Ask suggest_category() to classify it from title + summary
  - Record (actual, predicted, confidence)

Report: overall accuracy, per-category precision/recall, confusion matrix,
abstention rate. No side effects — the held-out indexes are temporary.

Run: `researchwiki eval-classifier`. Read-only over the wiki state.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from ..index.pages_bm25 import TantivySearchBackend
from ..search import build_documents_from_wiki, suggest_category


def main(argv: list[str]) -> int:
    argparse.ArgumentParser(
        prog="researchwiki eval-classifier",
        description="Leave-one-out evaluation of the category classifier.",
    ).parse_args(argv)
    return evaluate()


def evaluate() -> int:
    docs = build_documents_from_wiki()
    papers = [d for d in docs if d.page_type == "paper"]
    if not papers:
        print("No paper documents in the wiki — nothing to evaluate "
              "(ingest some papers first).")
        return 1
    print(f"Loaded {len(papers)} paper documents "
          f"(categories: {Counter(d.category for d in papers)})")
    print()

    rows: list[tuple[str, str, str | None, float, str]] = []
    # (stem, actual, predicted, confidence, mode) where mode ∈ {"correct", "wrong", "abstain"}

    for held_out in papers:
        remaining = [d for d in docs if d.stem != held_out.stem]
        with tempfile.TemporaryDirectory() as td:
            backend = TantivySearchBackend(path=Path(td))
            backend.build(remaining)
            # Match the real ingest signal: title from S2, abstract from S2.
            # The summary section is our proxy for abstract.
            seed_text_abstract = held_out.summary or held_out.body[:500]
            suggestion = suggest_category(backend, held_out.title, seed_text_abstract)
        if suggestion is None:
            rows.append((held_out.stem, held_out.category, None, 0.0, "abstain"))
        else:
            mode = "correct" if suggestion.category == held_out.category else "wrong"
            rows.append((
                held_out.stem, held_out.category, suggestion.category,
                suggestion.confidence, mode,
            ))

    total = len(rows)
    correct = sum(1 for r in rows if r[4] == "correct")
    wrong = sum(1 for r in rows if r[4] == "wrong")
    abstain = sum(1 for r in rows if r[4] == "abstain")

    modal = Counter(d.category for d in papers).most_common(1)[0][0]
    baseline_correct = sum(1 for d in papers if d.category == modal)

    print("## Per-paper results")
    for stem, actual, pred, conf, mode in rows:
        marker = {"correct": "✓", "wrong": "✗", "abstain": "?"}[mode]
        pred_s = f"{pred:>10s} ({conf:.0%})" if pred else f"{'abstain':>10s}         "
        print(f"  {marker}  actual={actual:>10s}  pred={pred_s}  {stem[:55]}")
    print()

    print("## Summary")
    print(f"  Total papers:         {total}")
    print(f"  Correct predictions:  {correct} ({correct/total:.1%})")
    print(f"  Wrong predictions:    {wrong}")
    print(f"  Abstentions (TODO):   {abstain} ({abstain/total:.1%})")
    print(f"  Baseline (always {modal!r}): {baseline_correct}/{total} = {baseline_correct/total:.1%}")
    decided = total - abstain
    if decided:
        print(f"  Accuracy when committed: {correct}/{decided} = {correct/decided:.1%}")
    print()

    print("## Confusion matrix (row=actual, col=predicted; `·` = abstention)")
    cats = sorted({r[1] for r in rows} | {r[2] for r in rows if r[2]})
    matrix: dict[str, Counter] = defaultdict(Counter)
    for _, actual, pred, _, _ in rows:
        key = pred or "·"
        matrix[actual][key] += 1
    header = "  actual\\pred | " + " | ".join(f"{c:>7s}" for c in cats + ["·"])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in cats:
        cells = " | ".join(f"{matrix[a].get(c, 0):>7d}" for c in cats + ["·"])
        print(f"  {a:>10s} | {cells}")
    print()

    print("## Per-category precision / recall")
    for cat in cats:
        tp = sum(1 for _, a, p, _, _ in rows if a == cat and p == cat)
        fp = sum(1 for _, a, p, _, _ in rows if a != cat and p == cat)
        fn = sum(1 for _, a, p, _, _ in rows if a == cat and p != cat)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        support = sum(1 for _, a, _, _, _ in rows if a == cat)
        print(f"  {cat:>10s}: precision={prec:.2f}  recall={rec:.2f}  support={support}")
    return 0
