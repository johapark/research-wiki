"""Leave-one-out evaluation of the Tantivy-backed category classifier.

For each paper in the wiki:
  - Rebuild the index excluding that paper
  - Ask suggest_category() to classify it from title + summary
  - Record (actual, predicted, confidence)

Report: overall accuracy, per-category precision/recall, confusion matrix,
abstention rate. No side effects — the held-out indexes are temporary.

Run: `researchwiki eval classifier`. Read-only over the wiki state.

The `eval-classifier` spelling still works and delegates here — kept because
`CONTRIBUTING.md` counts the CLI as a published surface, so removing a command
name is a breaking change even for one nobody has scripted.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from ..index.pages_bm25 import TantivySearchBackend
from ..search import build_documents_from_wiki, suggest_category


def main(argv: list[str]) -> int:
    argparse.ArgumentParser(
        prog="researchwiki eval-classifier",
        description="Deprecated alias for `researchwiki eval classifier`.",
    ).parse_args(argv)
    print("note: `eval-classifier` is now `researchwiki eval classifier`.",
          file=sys.stderr)
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
    # (stem, actual, predicted, confidence, mode), mode ∈
    #   correct       — committed to a category, and it was right
    #   wrong         — committed to a category, and it was wrong
    #   abstain-right — declined, and the paper does live in `other`
    #   abstain-miss  — declined, but the paper has a real category
    #
    # The two abstain modes are separate because `other` is *both* a real
    # category and the abstention bucket. Declining on a paper that belongs in
    # `other` is the classifier working; declining on a `single-cell` paper is
    # it giving up. Collapsing them is what the old accounting did, one step
    # further on: it counted an abstention as an ordinary prediction of `other`
    # and reported 0% abstention on a run that abstained ten times.

    for held_out in papers:
        remaining = [d for d in docs if d.stem != held_out.stem]
        with tempfile.TemporaryDirectory() as td:
            backend = TantivySearchBackend(path=Path(td))
            backend.build(remaining)
            # Match the real ingest signal: title from S2, abstract from S2.
            # The summary section is our proxy for abstract.
            seed_text_abstract = held_out.summary or held_out.body[:500]
            suggestion = suggest_category(backend, held_out.title, seed_text_abstract)
        # An abstention still *places* the paper: `suggest_category_llm` returns
        # category="other", and `promote` files it there. So `predicted` stays
        # "other" — the confusion matrix reports where papers actually land, and
        # hiding abstentions from it would under-report how `other` gets filled.
        # `None` is reserved for the classifier returning nothing at all.
        if suggestion is None:
            rows.append((held_out.stem, held_out.category, None, 0.0, "abstain-miss"
                         if held_out.category != "other" else "abstain-right"))
        elif suggestion.abstained:
            mode = "abstain-right" if held_out.category == "other" else "abstain-miss"
            rows.append((held_out.stem, held_out.category, "other",
                         suggestion.confidence, mode))
        else:
            mode = "correct" if suggestion.category == held_out.category else "wrong"
            rows.append((
                held_out.stem, held_out.category, suggestion.category,
                suggestion.confidence, mode,
            ))

    total = len(rows)
    correct = sum(1 for r in rows if r[4] == "correct")
    wrong = sum(1 for r in rows if r[4] == "wrong")
    abstain_right = sum(1 for r in rows if r[4] == "abstain-right")
    abstain_miss = sum(1 for r in rows if r[4] == "abstain-miss")
    abstain = abstain_right + abstain_miss

    modal = Counter(d.category for d in papers).most_common(1)[0][0]
    baseline_correct = sum(1 for d in papers if d.category == modal)

    print("## Per-paper results")
    for stem, actual, pred, conf, mode in rows:
        marker = {"correct": "✓", "wrong": "✗",
                  "abstain-right": "?", "abstain-miss": "?"}[mode]
        pred_s = (f"{pred:>10s} ({conf:.0%})" if pred
                  else f"{'abstain':>10s} ({conf:.0%})")
        print(f"  {marker}  actual={actual:>10s}  pred={pred_s}  {stem[:55]}")
    print()

    print("## Summary")
    placed_right = correct + abstain_right
    print(f"  Total papers:         {total}")
    print()
    print("  Placement — where the paper ends up on disk:")
    print(f"    correct:            {placed_right} ({placed_right/total:.1%})")
    print(f"    wrong:              {wrong + abstain_miss}")
    print()
    print("  Commitment — did the classifier name a category, or decline?")
    print(f"    committed:          {correct + wrong} ({(correct + wrong)/total:.1%})")
    print(f"    abstained:          {abstain} ({abstain/total:.1%})")
    print(f"      ├─ paper is 'other': {abstain_right}  (declining placed it right)")
    print(f"      └─ real category:    {abstain_miss}  (gave up on a placeable paper)")
    print(f"  Baseline (always {modal!r}): {baseline_correct}/{total} = {baseline_correct/total:.1%}")
    decided = correct + wrong
    if decided:
        print(f"  Accuracy when committed: {correct}/{decided} = {correct/decided:.1%}")
    print()
    print("  Confidence is the classifier's own self-report, not a vote share —")
    print("  `suggest_category` is LLM-first and only falls back to kNN counting.")
    print()

    print("## Confusion matrix (row=actual, col=predicted; `·` = no answer at all)")
    print("   Abstentions appear under `other`, which is where they actually land.")
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
        # How much of this category's *predicted* volume came from declining
        # rather than choosing. Only ever non-zero for `other`, and it is the
        # number that says whether `other` is a judgement or a shrug.
        by_abstention = sum(1 for _, _, pr, _, m in rows
                            if pr == cat and m.startswith("abstain"))
        note = f"  ({by_abstention} of {tp + fp} via abstention)" if by_abstention else ""
        print(f"  {cat:>10s}: precision={prec:.2f}  recall={rec:.2f}  "
              f"support={support}{note}")
    return 0
