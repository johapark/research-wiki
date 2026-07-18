"""Rendering for content-fixture scoring output.

Prose-format helpers for `benchmark-fixture <stem>` (single-run and
`--repeat` replication) and `--compare-config` A/B diffs. All functions
here are pure formatters — they take a report dataclass and return a
string; nothing writes or reads state.

The CLI in `researchwiki.tasks.benchmark_fixture` calls one of these per
invocation depending on `--repeat` / `--compare-config`. The stats helper
(`paired_t`) also lives here because it's only used by
`format_compare_report`.
"""

from __future__ import annotations


def paired_t(deltas: list[float]) -> tuple[float, int]:
    """Paired t-statistic on per-slot Δs and degrees of freedom (n-1).

    Slots are matched across arms (same replicate index → same stance
    and temperature per `stance_for_slot`), so paired-t is correct.
    Returns (t, df); no p-value — we surface the raw t and let the
    reader look up the table if they care about the exact threshold.
    Handles n<2 (returns (0, 0)) and zero-variance (returns (inf, n-1)
    when |mean|>0, (0, n-1) otherwise)."""
    import math
    n = len(deltas)
    if n < 2:
        return (0.0, 0)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    if var == 0:
        return (math.inf if mean != 0 else 0.0, n - 1)
    se = math.sqrt(var / n)
    return (mean / se, n - 1)


def format_compare_report(baseline, candidate, *,
                          baseline_label: str, candidate_label: str,
                          fixture=None) -> str:
    """Render an A/B diff between two ReplicateReports for the same fixture.

    Both arms must have equal n_runs; slot i in baseline pairs with slot i
    in candidate. Prints per-axis Δmean, paired-t on per-slot Δoverall,
    and a resolved-models block per arm so the reader can tell which
    config produced which score."""
    lines: list[str] = []
    lines.append(
        f"[benchmark-fixture --compare-config] {baseline.paper_stem}  "
        f"N={baseline.n_runs} per arm  "
        f"mode={'llm' if baseline.use_llm else 'heuristic'}"
    )
    lines.append(f"  baseline:  {baseline_label}")
    lines.append(f"  candidate: {candidate_label}")
    lines.append("")

    delta_overall = candidate.mean - baseline.mean
    lines.append(
        f"  Overall (mean):  {baseline.mean:.1%} → {candidate.mean:.1%}   "
        f"Δ {delta_overall:+.1%}"
    )
    # Paired-t on per-slot Δoverall.
    per_slot_deltas = [
        c.overall - b.overall
        for b, c in zip(baseline.runs, candidate.runs)
    ]
    t, df = paired_t(per_slot_deltas)
    lines.append(
        f"  Paired t on Δoverall: t={t:+.2f} (df={df})  "
        f"n={len(per_slot_deltas)}"
    )
    lines.append("")
    lines.append(f"  {'axis':<22} {'baseline':>10} {'candidate':>10} {'Δmean':>10}")
    lines.append(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for ax in baseline.axis_mean:
        b_mean = baseline.axis_mean[ax]
        c_mean = candidate.axis_mean.get(ax, 0.0)
        lines.append(
            f"  {ax:<22} {b_mean:>9.1%} {c_mean:>9.1%} "
            f"{c_mean - b_mean:>+9.1%}"
        )
    lines.append("")

    def _model_block(header: str, resolved: dict[str, str]) -> None:
        lines.append(f"  {header}:")
        for phase, ident in resolved.items():
            lines.append(f"    {phase:<16} {ident}")

    if baseline.resolved_models:
        _model_block("baseline models", baseline.resolved_models)
    if candidate.resolved_models:
        _model_block("candidate models", candidate.resolved_models)
    return "\n".join(lines).rstrip() + "\n"


def format_report(report, *, fixture=None, llm_cutoff: str = "2026-01-01",
                  grader_report=None, style_report=None) -> str:
    """Single-run coverage report — per-axis recall + optional grader + style panes."""
    lines: list[str] = []
    lines.append(
        f"[benchmark-fixture] {report.paper_stem}  page={report.page_path}  "
        f"mode={'llm' if report.use_llm else 'heuristic'}"
    )
    if fixture is not None and fixture.published_at:
        if fixture.published_at < llm_cutoff:
            lines.append(
                f"  ⚠ contamination risk: paper published {fixture.published_at} "
                f"precedes LLM cutoff ({llm_cutoff}) — high scores may reflect "
                f"memorized training data, not pipeline quality."
            )
        else:
            lines.append(
                f"  ✓ post-cutoff: paper published {fixture.published_at} "
                f"after LLM cutoff ({llm_cutoff})."
            )
    elif fixture is not None and not fixture.published_at:
        lines.append(
            f"  ? unknown publication date — add `published_at:` to the fixture "
            f"to enable contamination tracking."
        )
    lines.append("")
    lines.append(f"  Overall weighted recall: {report.overall_weighted_recall:.1%}")
    lines.append("")
    lines.append(f"  {'axis':<22} {'recall':>8}  match / partial / miss  (n)")
    lines.append(f"  {'-' * 22} {'-' * 8}  ---------------------  ----")
    for axis_name, axis in report.axes.items():
        lines.append(
            f"  {axis_name:<22} {axis.weighted_recall:>7.1%}  "
            f"{axis.n_match:>5} / {axis.n_partial:>7} / {axis.n_miss:>4}  "
            f"({axis.n_items})"
        )
    lines.append("")

    # Per-item detail for misses + partials so the report is diagnosable.
    for axis_name, axis in report.axes.items():
        problems = [v for v in axis.items if v.verdict in ("miss", "partial")]
        if not problems:
            continue
        lines.append(f"  ── {axis_name} ──")
        for v in problems:
            badge = "MISS" if v.verdict == "miss" else "PART"
            lines.append(
                f"    [{badge}] {v.item_id}  ({v.importance})  — {v.rationale}"
            )
        lines.append("")

    # Style pane (U2) — page-level compression + extractiveness, mechanical.
    # Comes BEFORE the grader pane because it's a faster/cheaper signal:
    # a page that's outside the compression band is unlikely to score
    # well on either fixture recall or grader, but for different reasons.
    if style_report is not None:
        lines.append("  ── style (page-level, mechanical) ──")
        # Compression line.
        comp_marker = {
            "compressed": "▼ too compressed",
            "verbose":    "◀ too verbose",
            "normal":     "✓ in band",
            "unknown":    "? unknown",
        }.get(style_report.compression_verdict, "")
        lines.append(
            f"    compression: {style_report.compression_ratio:.2%} of paper "
            f"({style_report.page_tokens} page tokens / "
            f"{style_report.paper_tokens} paper tokens)   {comp_marker}"
        )
        # Extractiveness line.
        ext_marker = {
            "paraphrased": "▼ heavy paraphrase, drift risk",
            "extractive":  "◀ cargo-culted excerpts",
            "normal":      "✓ in band",
            "unknown":     "? unknown",
        }.get(style_report.extractiveness_verdict, "")
        lines.append(
            f"    extractiveness: {style_report.extractiveness_fraction:.0%} "
            f"of {style_report.n_page_sentences} eligible sentences "
            f"({style_report.n_extractive_sentences} with verbatim ≥10-word "
            f"spans from PDF)   {ext_marker}"
        )
        lines.append("")

    # Grader pane (U1) — reference-free faithfulness scores from the
    # agent's per-claim grader, run only when --with-grader. Kept after
    # the fixture pane because the fixture is the primary surface; the
    # grader is the secondary "is this drift-free?" check.
    if grader_report is not None:
        lines.append("  ── grader (reference-free; per-claim against PDF) ──")
        n = grader_report.n_graded
        if n == 0:
            lines.append(f"    no gradable claims found (n_claims={grader_report.n_claims})")
        else:
            lines.append(
                f"    n_graded={n}  "
                f"BM25 mean={grader_report.mean_top1:.2f} median={grader_report.median_top1:.2f}  "
                f"weakest={grader_report.weakest_score:.2f}"
            )
            if grader_report.semantic_available and grader_report.semantic_score is not None:
                lines.append(
                    f"    semantic (bi-encoder cos): "
                    f"mean={grader_report.semantic_score:.2f}  "
                    f"median={grader_report.semantic_median:.2f}"
                )
            flags = []
            if grader_report.n_with_numeric_drift:
                flags.append(f"numeric drift in {grader_report.n_with_numeric_drift} claim(s)")
            if grader_report.n_negation_mismatches:
                flags.append(f"negation mismatch in {grader_report.n_negation_mismatches}")
            if flags:
                lines.append(f"    ⚠ {'; '.join(flags)}")
            else:
                lines.append(f"    ✓ no numeric-drift / negation-mismatch flags")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_replicate_report(rep, *, fixture=None, llm_cutoff: str = "2026-01-01") -> str:
    """Replication report — mean ± SD across N runs, per-run detail, cost."""
    lines: list[str] = []
    lines.append(
        f"[benchmark-fixture --repeat {rep.n_runs}] {rep.paper_stem}  "
        f"mode={'llm' if rep.use_llm else 'heuristic'}"
    )
    if fixture is not None and fixture.published_at:
        if fixture.published_at < llm_cutoff:
            lines.append(
                f"  ⚠ contamination risk: paper published {fixture.published_at} "
                f"precedes LLM cutoff ({llm_cutoff})."
            )
        else:
            lines.append(
                f"  ✓ post-cutoff: paper published {fixture.published_at} "
                f"after LLM cutoff ({llm_cutoff})."
            )
    lines.append("")
    lines.append(
        f"  Overall (mean ± SD across N={rep.n_runs}):  "
        f"{rep.mean:.1%} ± {rep.sd:.1%}   range [{rep.min_overall:.1%}, {rep.max_overall:.1%}]"
    )
    lines.append("")
    lines.append(f"  {'axis':<22} {'mean ± SD':>16}   per-run")
    lines.append(f"  {'-' * 22} {'-' * 16}   {'-' * 30}")
    for axis_name in rep.axis_mean:
        per_run = " ".join(f"{r.axes[axis_name]:.0%}" for r in rep.runs)
        lines.append(
            f"  {axis_name:<22} {rep.axis_mean[axis_name]:>7.1%} ± {rep.axis_sd[axis_name]:>5.1%}   {per_run}"
        )
    lines.append("")
    lines.append(f"  Per-run detail:")
    total_in = total_out = 0
    for r in rep.runs:
        total_in += r.input_tokens
        total_out += r.output_tokens
        lines.append(
            f"    run #{r.run_index + 1}  stance={r.stance:<13} t={r.temperature:.1f}  "
            f"overall {r.overall:>5.1%}  ({r.input_tokens} in / {r.output_tokens} out)"
        )
    lines.append("")
    lines.append(
        f"  LLM cost: {total_in} input + {total_out} output tokens "
        f"(author × {rep.n_runs}; judge calls additional when --llm)"
    )
    if rep.resolved_models:
        lines.append("")
        lines.append(f"  Resolved models (this run):")
        for phase, ident in rep.resolved_models.items():
            lines.append(f"    {phase:<16} {ident}")
    return "\n".join(lines).rstrip() + "\n"
