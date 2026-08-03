"""Score wiki-page fidelity against source PDFs. Three targets:

  researchwiki grade paper <stem>            # paper-page fidelity + salience
  researchwiki grade synthesis <page>        # synthesis/idea-page fidelity
  researchwiki grade regression [--missing-only]   # re-grade every paper + drift diff

paper — extracts atomic claims (Key Contributions bullets + Results lines)
  from the wiki page and BM25/semantic-queries them against the paper's PDF.
  Two orthogonal axes:
    Fidelity (precision) — "is what the page says actually in the PDF?"
    Salience (recall)    — "does the page capture what the PDF flagged as
                            load-bearing?" (disable with --no-salience)

synthesis — the fidelity gate for synthesis/idea pages, sibling to
  `check-grounding` (structural). Each claim is graded against the PDF(s)
  of the paper(s) it cites; catches cross-paper misattribution — a number
  ascribed to a paper that doesn't contain it. See `researchwiki grade
  synthesis --help` for the verdict taxonomy.

regression — re-grade every paper page and diff each against its persisted
  per-claim scores. Catches silent quality drift after model/embedding/prompt
  swaps. Default persists new baseline; --no-persist for diff-only.

Usage:
  researchwiki grade paper <stem>                    # human-readable report
  researchwiki grade paper <stem> --json             # machine-readable
  researchwiki grade paper <stem> --rebuild-index    # force-rebuild the per-PDF index
  researchwiki grade paper <stem> --weakest 3        # show top-N weakest claims
  researchwiki grade paper <stem> --no-salience      # fidelity only

  researchwiki grade synthesis <path>                # exit 0 if no misattribution
  researchwiki grade synthesis <path> --weak         # also list weak/composite
  researchwiki grade synthesis <path> --fine-grained # verify [[stem#slug]] anchors

  researchwiki grade regression                      # re-grade all + drift diff
  researchwiki grade regression --missing-only       # only un-graded papers
  researchwiki grade regression --no-persist         # diff without writing baseline
  researchwiki grade regression --threshold 0.05     # only material drift
  researchwiki grade regression --json
"""

from __future__ import annotations

import argparse
import json
import sys

from ..grade.fidelity.paper import grade_page
from ..index.pdf_chunks import build_pdf_index
from ..index import embeddings as semantic_mod
from ..log import log


# ---------- paper (single-stem) ----------


def _format_paper_report(report) -> str:
    lines = []
    lines.append(f"  page          : {report.page_path}")
    lines.append(f"  claims        : {report.n_claims} total ({report.n_graded} graded, "
                 f"{report.n_cross_refs} cross-refs skipped)")
    lines.append(f"  mean top1     : {report.mean_top1:.3f}")
    lines.append(f"  median top1   : {report.median_top1:.3f}")
    if report.semantic_available:
        lines.append(f"  mean semantic : {report.semantic_score:.3f}")
        lines.append(f"  median semant.: {report.semantic_median:.3f}")
    else:
        lines.append("  semantic      : not available (--no-semantic or model unavailable)")
    lines.append(f"  negation flag : {report.n_negation_mismatches} claims with negation mismatch")
    lines.append(f"  numeric drift : {report.n_with_numeric_drift} claims have unmatched numbers")
    if report.weakest_claim is not None:
        lines.append("")
        lines.append(f"  weakest claim ({report.weakest_score:.3f}):")
        lines.append(f"    {report.weakest_claim[:160]}")

    if report.salience is not None:
        lines.append("")
        s = report.salience
        if s.salience_score is None:
            lines.append("  salience      : no PDF anchors recoverable (skipped)")
        else:
            lines.append(
                f"  salience      : {s.salience_score:.3f} weighted recall "
                f"(matched {s.n_match}, partial {s.n_partial}, missed {s.n_miss}; n={s.n_anchors})"
            )
            for axis in ("headline_claims", "capabilities", "limitations"):
                ax = s.per_axis.get(axis)
                if not ax:
                    continue
                lines.append(
                    f"   - {axis:<17}: {ax['match']} match, "
                    f"{ax['partial']} partial, {ax['miss']} miss"
                )
            if s.missed_anchors:
                lines.append("  missed anchors (top {0}):".format(len(s.missed_anchors)))
                for m in s.missed_anchors:
                    label = f"{m['axis']:<16}/{m['importance']:<8}"
                    lines.append(f"    [{label}] {m['text'][:140]}")

    lines.append("")
    lines.append("  per-claim breakdown:")
    sem_hdr = f" {'sem':>5}" if report.semantic_available else ""
    lines.append(f"    {'sec':<18} {'pos':>3} {'top1':>7} {'top3μ':>7}{sem_hdr} "
                 f"{'neg':>3} {'#nums':>5} {'drift':>5}  text")
    for c in report.claims:
        if not c.graded:
            lines.append(f"    {c.section:<18} {c.position:>3}  (cross-ref skipped)  {c.text[:80]}")
            continue
        n_nums = len(c.numeric_tokens)
        n_drift = len(c.numeric_unmatched)
        sem_str = ""
        if report.semantic_available:
            sem_str = f" {(c.semantic_score or 0):>5.2f}"
        neg_str = "✗" if c.negation_mismatch else " "
        lines.append(
            f"    {c.section:<18} {c.position:>3} "
            f"{c.top1_score:>7.2f} {c.top3_mean:>7.2f}{sem_str} "
            f"{neg_str:>3} {n_nums:>5} {n_drift:>5}  "
            f"{c.text[:80]}"
        )
        if c.numeric_unmatched:
            lines.append(f"      ↳ unmatched numbers: {c.numeric_unmatched}")
    return "\n".join(lines)


def _run_paper(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="researchwiki grade paper",
        description="Paper-page fidelity + salience scoring against source PDF.",
    )
    p.add_argument("stem", help="Stem of the wiki paper to grade (e.g. smith-2024-...).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force-rebuild the per-PDF Tantivy chunk index")
    p.add_argument("--weakest", type=int, default=0,
                   help="In text mode, also print the N weakest claims with their top-1 chunks")
    p.add_argument("--page-path",
                   help="Override wiki-page lookup with an explicit markdown file "
                        "(used for grading test fixtures against a stem's PDF).")
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip the bi-encoder semantic scoring (BM25 + numeric "
                        "drift + negation parity only).")
    p.add_argument("--no-salience", action="store_true",
                   help="Skip the salience pass (PDF-anchor recall via "
                        "synthetic ContentFixture). Fidelity scoring only.")
    args = p.parse_args(argv)

    if args.rebuild_index:
        log(f"[grade] force-rebuilding chunk index for {args.stem}")
        build_pdf_index(args.stem, force=True)

    try:
        report = grade_page(
            args.stem,
            page_path=args.page_path,
            semantic=not args.no_semantic,
            include_salience=not args.no_salience,
        )
    except FileNotFoundError as e:
        # A missing PDF or page for the given stem — the argument was wrong.
        print(f"researchwiki grade paper: {e}", file=sys.stderr)
        return 1
    # No broad `except Exception: return 2` below it: state.db / index failures
    # raise `EnvironmentFailure` and the funnel reports those as 2, while a bug
    # in the grader now gets code 3 and a traceback instead of being mislabelled
    # an environment error.

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    log(f"{args.stem}", tag="grade")
    print(_format_paper_report(report))

    if args.weakest > 0:
        graded = [c for c in report.claims if c.graded]
        graded.sort(key=lambda c: c.top1_score)
        print()
        print(f"  weakest {args.weakest} claims (with top-1 chunk):")
        for c in graded[:args.weakest]:
            print(f"    [{c.section}#{c.position}] score={c.top1_score:.2f}")
            print(f"      claim: {c.text[:200]}")
            print(f"      chunk: {c.supporting_text[:200]}")
    return 0


# ---------- synthesis (delegates to the hidden module) ----------


def _run_synthesis(argv: list[str]) -> int:
    from . import _grade_synthesis
    return _grade_synthesis.main(argv)


# ---------- regression (sweep across the whole corpus) ----------


def _run_regression_pipeline(persist: bool, semantic: bool,
                             missing_only: bool = False,
                             include_salience: bool = True) -> list[dict]:
    """Re-grade every paper page; diff against persisted per-claim scores.

    Returns one dict per paper:
      stem, baseline_bm25, current_bm25, delta_bm25,
      baseline_sem, current_sem, delta_sem,
      baseline_model, current_model, model_changed,
      is_first_observation, last_graded_at, error

    With `missing_only=True`, restrict to papers with at least one gradable
    claim (is_cross_ref=0) whose `last_graded_at` is NULL — mirrors the
    "N papers need backfill" count in `researchwiki status`.

    `include_salience=False` skips the recall pass, which costs a synthetic
    ContentFixture plus a fresh per-page body embedding on every paper. It is
    free in tokens but not in wall-clock, and a bulk backfill of already-
    authored pages only needs the fidelity columns populated so claims become
    citable — it is not gating on PDF-anchor recall.
    """
    from ..db.connection import get_connection
    conn = get_connection()

    # Pull baseline aggregates per paper from persisted claim rows. AVG over
    # gradable (non-cross-ref) claims only; MIN/MAX(embed_model) collapses
    # since one paper grades with one model in any single run.
    baselines: dict[str, dict] = {}
    for row in conn.execute("""
        SELECT paper_stem,
               AVG(bm25_top1)          AS mean_bm25,
               AVG(semantic_score)     AS mean_sem,
               COUNT(last_graded_at)   AS n_graded,
               MAX(last_graded_at)     AS last_graded,
               MIN(embed_model)        AS embed_model
        FROM claims
        WHERE is_cross_ref = 0
        GROUP BY paper_stem
    """):
        baselines[row["paper_stem"]] = dict(row)

    if missing_only:
        stems = [r["stem"] for r in conn.execute("""
            SELECT p.stem
            FROM papers p
            WHERE p.page_type = 'paper'
              AND EXISTS (
                SELECT 1 FROM claims c
                WHERE c.paper_stem = p.stem
                  AND c.is_cross_ref = 0
                  AND c.last_graded_at IS NULL
              )
            ORDER BY p.stem
        """)]
    else:
        stems = [r["stem"] for r in conn.execute(
            "SELECT stem FROM papers WHERE page_type = 'paper' ORDER BY stem"
        )]

    current_model = semantic_mod.DEFAULT_MODEL if semantic else None
    out: list[dict] = []
    for stem in stems:
        try:
            current = grade_page(stem, persist=persist, semantic=semantic,
                                 include_salience=include_salience)
        except Exception as e:
            out.append({"stem": stem, "error": str(e)})
            continue

        b = baselines.get(stem, {})
        baseline_bm25 = b.get("mean_bm25")
        baseline_sem = b.get("mean_sem")
        baseline_model = b.get("embed_model")
        n_baseline = b.get("n_graded") or 0

        sem_comparable = (
            baseline_sem is not None
            and current.semantic_available
            and current.semantic_score is not None
            and (baseline_model is None or baseline_model == current_model)
        )
        delta_bm25 = (
            (current.mean_top1 - baseline_bm25) if baseline_bm25 is not None else None
        )
        delta_sem = (
            (current.semantic_score - baseline_sem) if sem_comparable else None
        )

        out.append({
            "stem": stem,
            "n_graded": current.n_graded,
            "baseline_bm25": baseline_bm25,
            "current_bm25": current.mean_top1,
            "delta_bm25": delta_bm25,
            "baseline_sem": baseline_sem,
            "current_sem": current.semantic_score if current.semantic_available else None,
            "delta_sem": delta_sem,
            "baseline_model": baseline_model,
            "current_model": current_model,
            "model_changed": (
                baseline_model is not None and current_model is not None
                and baseline_model != current_model
            ),
            "is_first_observation": n_baseline == 0,
            "last_graded_at": b.get("last_graded"),
        })

    return out


def _severity(r: dict) -> float:
    """Sort key — most-regressed first. Errors go last."""
    if r.get("error") or r.get("is_first_observation"):
        return 0.0
    d_bm = r.get("delta_bm25") or 0.0
    d_sem = r.get("delta_sem") or 0.0
    return -(abs(d_bm) + abs(d_sem))


def _format_regression_report(results: list[dict], threshold: float) -> str:
    new_obs = [r for r in results if r.get("is_first_observation") and not r.get("error")]
    errs = [r for r in results if r.get("error")]
    drifted = sorted(
        (r for r in results
         if not r.get("is_first_observation")
         and not r.get("error")
         and (abs(r.get("delta_bm25") or 0) >= threshold
              or abs(r.get("delta_sem") or 0) >= threshold)),
        key=_severity,
    )

    lines = []
    lines.append(f"[regression] {len(results)} papers re-graded "
                 f"({len(drifted)} drifted ≥ {threshold}, "
                 f"{len(new_obs)} first observation, {len(errs)} errored)")
    lines.append("")

    if drifted:
        lines.append(f"  {'stem':<60} {'Δbm25':>7} {'Δsem':>7}  notes")
        for r in drifted:
            d_bm = f"{r['delta_bm25']:+.3f}" if r.get("delta_bm25") is not None else "  n/a"
            d_sem = f"{r['delta_sem']:+.3f}" if r.get("delta_sem") is not None else "  n/a"
            note = ""
            if r.get("model_changed"):
                note = f"⚠ model {r['baseline_model']} → {r['current_model']}"
            lines.append(f"  {r['stem'][:60]:<60} {d_bm:>7} {d_sem:>7}  {note}")
        lines.append("")

    if new_obs:
        lines.append(f"  first observations ({len(new_obs)}; no baseline yet):")
        for r in new_obs[:10]:
            cur_bm = f"{r['current_bm25']:.3f}" if r.get("current_bm25") is not None else "n/a"
            lines.append(f"    {r['stem'][:60]:<60} bm25={cur_bm}")
        if len(new_obs) > 10:
            lines.append(f"    ... ({len(new_obs) - 10} more)")
        lines.append("")

    if errs:
        lines.append(f"  errored ({len(errs)}):")
        for r in errs:
            lines.append(f"    {r['stem']}: {r['error']}")

    return "\n".join(lines)


def _run_regression(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="researchwiki grade regression",
        description="Re-grade every paper-type page and diff against persisted scores. "
                    "Catches silent quality drift after model/embedding/prompt swaps.",
    )
    p.add_argument("--json", action="store_true",
                   help="Output JSON instead of text")
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip the bi-encoder semantic scoring (BM25 + numeric "
                        "drift + negation parity only).")
    p.add_argument("--no-persist", action="store_true",
                   help="Don't write current scores back to the claims table. "
                        "Default is to update the baseline so the next regression "
                        "run measures drift since this run.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Only show papers whose |Δbm25| or |Δsem| exceeds this. "
                        "Default 0 (show every drifted paper).")
    p.add_argument("--missing-only", action="store_true",
                   help="Only re-grade papers with at least one un-graded claim "
                        "(last_graded_at IS NULL). Matches the 'N papers need "
                        "backfill' count in `status`.")
    p.add_argument("--no-salience", action="store_true",
                   help="Skip the salience (recall) pass. Costs a synthetic "
                        "fixture + a per-page body embedding on every paper; "
                        "skip it for a bulk backfill that only needs the "
                        "fidelity columns populated.")
    args = p.parse_args(argv)

    results = _run_regression_pipeline(
        persist=not args.no_persist,
        semantic=not args.no_semantic,
        missing_only=args.missing_only,
        include_salience=not args.no_salience,
    )
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(_format_regression_report(results, threshold=args.threshold))
    return 0


# ---------- CLI ----------


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        # Bespoke help — argparse's default output for a 3-target dispatcher
        # obscures the target names.
        print(__doc__.strip())
        return 0 if argv else 2

    target, rest = argv[0], argv[1:]
    if target == "paper":
        return _run_paper(rest)
    if target == "synthesis":
        return _run_synthesis(rest)
    if target == "regression":
        return _run_regression(rest)

    print(f"researchwiki grade: unknown target '{target}'. "
          f"Available: paper, synthesis, regression", file=sys.stderr)
    return 1
