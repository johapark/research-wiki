"""Score a frozen, independently nominated impact-review label set.

Run from the repository root:

    python evaluations/run_impact_review_holdout.py \
        evaluations/impact-review-2026-10-01-labels.json \
        evaluations/impact-review-2026-10-01-results.json

Receipts are redirected to a temporary directory. This does not mutate the
wiki, indexes, state database, or the user's impact-review queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from researchwiki import impact_review as impact
from researchwiki.wiki import read_pages


_CATEGORY_FIX_TARGET = "ideas/graphot-variant-aware-off-target-search"


def _labeled_page_bytes(page) -> bytes:
    """Replay the single metadata correction made after baseline labeling."""
    raw = page.path.read_bytes()
    if page.key == _CATEGORY_FIX_TARGET:
        raw = raw.replace(b"category: [cgt]", b"category: [ideas]", 1)
    return raw


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(centre - radius, 4), round(centre + radius, 4)]


def main(labels_file: Path, results_file: Path) -> None:
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    rows = labels["pairs"]
    all_pages = [
        replace(page, fm={**page.fm, "category": ["ideas"]})
        if page.key == _CATEGORY_FIX_TARGET else page
        for page in read_pages()
    ]
    page_by_key = {page.key: page for page in all_pages}
    tuned = set(labels["excluded_tuning_sources"])
    for row in rows:
        source = page_by_key.get(row["source"])
        target = page_by_key.get(row["target"])
        if source is None or target is None:
            raise RuntimeError(f"missing page in frozen set: {row['source']} → {row['target']}")
        if source.stem in tuned:
            raise RuntimeError(f"tuning source in holdout: {source.stem}")
        if hashlib.sha256(source.path.read_bytes()).hexdigest() != row["source_file_sha256"]:
            raise RuntimeError(f"source file changed since labeling: {source.key}")
        if hashlib.sha256(_labeled_page_bytes(target)).hexdigest() != row["target_file_sha256"]:
            raise RuntimeError(f"target file changed since labeling: {target.key}")
        if impact._fingerprint(impact._source_text(source)) != row["source_fingerprint"]:
            raise RuntimeError(f"source changed since labeling: {source.key}")
        if impact._fingerprint(impact._target_text(target)) != row["target_fingerprint"]:
            raise RuntimeError(f"target changed since labeling: {target.key}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    original_ingest_dir = impact.ingest_dir
    original_read_pages = impact.read_pages
    measurements: list[dict] = []
    candidate_volumes: list[int] = []
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="impact-holdout-") as scratch:
        impact.ingest_dir = lambda: Path(scratch)
        impact.read_pages = lambda: all_pages
        try:
            for index, (source_key, group) in enumerate(sorted(grouped.items()), 1):
                receipt = impact.scan(source_key)
                if receipt["state"] != "complete":
                    raise RuntimeError(f"scan incomplete for {source_key}: {receipt['error']}")
                audit = {row["target"]: row for row in receipt["audit"]}
                chosen = {row["target"] for row in receipt["candidates"]}
                candidate_volumes.append(len(chosen))
                for labeled in group:
                    target_key = labeled["target"]
                    score = audit[target_key]
                    measurements.append({
                        "source": source_key,
                        "target": target_key,
                        "label": labeled["label"],
                        "predicted": target_key in chosen,
                        "score": score["semantic_score"],
                        "page_score": score["page_score"],
                        "passage_score": score["passage_score"],
                        "shared_terms": score["shared_terms"],
                        "shared_phrases": score["shared_phrases"],
                        "already_cited": score["already_cited"],
                    })
                if index % 10 == 0 or index == len(grouped):
                    print(f"scored {index}/{len(grouped)} source papers", flush=True)
        finally:
            impact.ingest_dir = original_ingest_dir
            impact.read_pages = original_read_pages

    counts = Counter((row["label"], row["predicted"]) for row in measurements)
    tp = counts["relevant", True]
    fn = counts["relevant", False]
    fp = counts["not_relevant", True]
    tn = counts["not_relevant", False]
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    by_type = {}
    for page_type in ("synthesis", "idea", "proposal"):
        subset = [row for row in measurements
                  if page_by_key[row["target"]].page_type == page_type]
        n_positive = sum(row["label"] == "relevant" for row in subset)
        n_caught = sum(row["label"] == "relevant" and row["predicted"] for row in subset)
        by_type[page_type] = {
            "pairs": len(subset), "relevant": n_positive,
            "caught": n_caught,
            "recall": round(n_caught / n_positive, 4) if n_positive else None,
        }
    report = {
        "labels_file": str(labels_file),
        "snapshot_replay": "GraphOT category [cgt] is read as [ideas], matching the frozen pre-fix labels",
        "impact_review_sha256": hashlib.sha256(Path(impact.__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "source_papers": len(grouped),
        "targets": len({row["target"] for row in rows}),
        "pairs": len(rows),
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                             "uncertain": sum(row["label"] == "uncertain" for row in rows)},
        "recall": round(recall, 4) if recall is not None else None,
        "recall_wilson_95": _wilson(tp, tp + fn),
        "precision_on_independently_nominated_pairs": (
            round(precision, 4) if precision is not None else None),
        "candidate_volume_per_source": {
            "median": statistics.median(candidate_volumes),
            "max": max(candidate_volumes),
            "mean": round(statistics.mean(candidate_volumes), 2),
        },
        "by_target_type": by_type,
        "measurements": measurements,
    }
    results_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"recall={tp}/{tp + fn}={recall:.3f}; precision={tp}/{tp + fp}={precision:.3f}")
    print(f"misses={fn}; results={results_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python evaluations/run_impact_review_holdout.py LABELS RESULT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
