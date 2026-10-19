"""Data-producing post-publication work shared by CLI and application callers."""

from dataclasses import dataclass, field

from .lifecycle import current_facts, maintenance, phase


@dataclass
class PostWorkReport:
    state: str = "not_requested"
    impact: dict | None = None
    hooks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _budgeted(ctx, label, fn, report):
    from ..agents.budget import BudgetExhausted, activate

    tracker = getattr(ctx, "budget_tracker", None)
    try:
        if tracker is not None:
            tracker.resume()
        with activate(tracker), phase(label):
            value = fn()
        if value is None:
            report.warnings.append(f"{label}: completion unavailable")
        elif value.get("judge_failed"):
            report.warnings.append(f"{label}: some judgments unavailable")
        return value
    except BudgetExhausted:
        report.warnings.append(f"{label}: skipped because ingest budget was exhausted")
    except Exception as exc:
        # Provider exception text can contain confidential research content.
        report.warnings.append(f"{label}: {type(exc).__name__}")
    finally:
        if tracker is not None:
            tracker.suspend()
    return None


def run_post_work(ctx, options) -> PostWorkReport:
    report = PostWorkReport()
    if options.use_stub or ctx.outcome != "promoted" or not ctx.committed_path:
        return report
    from .. import impact_review

    source_key = f"{ctx.committed_path.parent.name}/{ctx.paper_stem}"
    try:
        with phase("impact_review"):
            report.impact = impact_review.scan(source_key)
    except Exception as exc:
        try:
            report.impact = impact_review.record_unscanned(source_key, str(exc))
        except Exception:
            report.impact = {"state": "unscanned", "error": type(exc).__name__, "candidates": []}
    if report.impact.get("state") not in ("complete", "skipped"):
        report.warnings.append("impact-review: scan incomplete; review the local receipt")

    if not options.no_cross_link and ctx.paper_stem:
        if options.claim_overlap:
            from ..tasks.claim_overlap import run_after_ingest

            facts = current_facts()
            if facts and facts.publication:
                facts.components["post_hooks"] = "unknown"
            report.hooks["claim-overlap"] = _budgeted(
                ctx, "claim-overlap", lambda: run_after_ingest(
                    ctx.paper_stem, ctx.committed_path, render=False), report,
            )
            if facts and facts.publication:
                value = report.hooks["claim-overlap"]
                maintenance("post_hooks", "completed" if isinstance(value, dict) and
                            "applied" in value and not value["applied"] else "unknown")
        if options.contradiction_alert:
            from ..claim_graph.alert import alert_after_ingest

            report.hooks["contradiction-alert"] = _budgeted(
                ctx, "contradiction-alert", lambda: alert_after_ingest(
                    ctx.paper_stem, ctx.committed_path, render=False), report,
            )
    report.state = "partial" if report.warnings else "completed"
    if options.memory_evolve:
        facts = current_facts()
        memory_state = facts.memory_work if facts is not None else "unknown"
        if memory_state in ("failed", "partial"):
            report.warnings.append("memory-evolve: incomplete; inspect the local trace")
            report.state = "partial"
        elif memory_state == "unknown":
            report.state = "unknown" if not report.warnings else "partial"
    return report
