"""Analytics over the ingest telemetry log (`ingest_iterations`).

The framework writes a row after every ingest phase — author drafts, grader
scores, tournament decisions, per-call model/temperature/token cost. `status`
only surfaces a 7-day cost total; this command mines the rest: which model
produces the best drafts on *your* papers, which sections are hardest to
author, where the tokens go, and how drafts get decided. Read-only, local, no
LLM — it's a reporting view over data already captured at ingest.

Usage:
  researchwiki insights                 # all-time
  researchwiki insights --days 30       # last 30 days
  researchwiki insights --attempts      # one timing row per ingest
  researchwiki insights --attempt-id ID # exact phase breakdown for one ingest
  researchwiki insights --json

Exit codes: 0 = printed (including "no telemetry yet"); 2 = DB unreachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from types import SimpleNamespace

from ..agents import fitness
from .insights_timing import gather_attempts, latency_distribution

# Reuse the single source of truth for pricing so cost figures match `status`.
from ..agents import model_config as _mc

# Roles whose rows carry a real LLM model_used (others are deterministic phases
# with model_used NULL — reconcile/extract/grade/tournament/commit).
_NON_MODEL_SENTINELS = ("stub", "(skipped)")


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _estimate_usd(model: str, in_tok: int, out_tok: int) -> float:
    """Thin alias kept so existing call sites read unchanged; the rate table and
    the prefix matching both live in `agents.model_config`."""
    return _mc.estimate_usd(model, in_tok, out_tok)


def _score_value(scores: dict, key: str):
    if key == "semantic_score":
        return scores.get("semantic_score", scores.get("mean_semantic"))
    return scores.get(key)


def _revision_outcomes(conn, where_time: str, params: tuple,
                       stem: str | None = None) -> dict:
    """Compare revision drafts with their parent drafts when both have grades.

    Missing links and legacy score shapes are counted as incomparable. Nothing
    is imputed, which is essential for migrated corpora and interrupted runs.
    """
    stem_sql = " AND child.paper_stem = ?" if stem else ""
    rows = conn.execute(
        "SELECT child.id, child.attempt_id, child.paper_stem, child.role, "
        "child.parent_iteration_id, child.model_used, child.cost_input_tokens, "
        "child.cost_output_tokens, parent.role AS parent_role "
        "FROM ingest_iterations child "
        "LEFT JOIN ingest_iterations parent ON parent.id=child.parent_iteration_id "
        "WHERE child.parent_iteration_id IS NOT NULL "
        "AND child.role IN ('author','debug')" +
        where_time.replace("created_at", "child.created_at") + stem_sql,
        (*params, *((stem,) if stem else ())),
    ).fetchall()
    grade_rows = conn.execute(
        "SELECT parent_iteration_id, grader_scores FROM ingest_iterations "
        "WHERE role='grade' AND parent_iteration_id IS NOT NULL"
    ).fetchall()
    grades: dict[int, dict] = {}
    for row in grade_rows:
        try:
            grades[row["parent_iteration_id"]] = json.loads(row["grader_scores"] or "{}")
        except (TypeError, json.JSONDecodeError):
            pass

    counts = Counter()
    details = []
    deltas = []
    improved_tokens = 0
    for row in rows:
        intervention = "debug" if row["role"] == "debug" else "evolve"
        child_scores = grades.get(row["id"])
        parent_scores = grades.get(row["parent_iteration_id"])
        status = "incomparable"
        delta = None
        if child_scores is not None and parent_scores is not None:
            axes = []
            for key in ("semantic_score", "salience_score", "target_claim_score",
                        "coherence_score"):
                child = _score_value(child_scores, key)
                parent = _score_value(parent_scores, key)
                if isinstance(child, (int, float)) and isinstance(parent, (int, float)):
                    axes.append(float(child) - float(parent))
            if axes:
                delta = sum(axes) / len(axes)
            # Classification replays the exact operator lens used by runner,
            # rather than treating a mean across heterogeneous axes as the
            # acceptance rule. The mean delta remains descriptive only.
            child_obj = SimpleNamespace(scores=child_scores)
            parent_obj = SimpleNamespace(scores=parent_scores)
            accepts = (fitness.is_strict_improvement if intervention == "debug"
                       else fitness.is_evolve_improvement)
            if accepts(child_obj, parent_obj):
                status = "improved"
            elif accepts(parent_obj, child_obj):
                status = "regressed"
            else:
                status = "tied"
        counts[(intervention, status)] += 1
        if delta is not None:
            deltas.append(delta)
        tokens = int(row["cost_input_tokens"] or 0) + int(row["cost_output_tokens"] or 0)
        if status == "improved":
            improved_tokens += tokens
        details.append({
            "attempt_id": row["attempt_id"], "paper_stem": row["paper_stem"],
            "revision_id": row["id"], "parent_id": row["parent_iteration_id"],
            "intervention": intervention, "status": status,
            "mean_axis_delta": round(delta, 6) if delta is not None else None,
            "model": row["model_used"], "tokens": tokens,
        })
    by_intervention = {}
    for intervention in ("evolve", "debug"):
        slot = {s: counts[(intervention, s)] for s in
                ("improved", "regressed", "tied", "incomparable")}
        slot["eligible"] = sum(slot.values())
        slot["comparable"] = slot["eligible"] - slot["incomparable"]
        by_intervention[intervention] = slot
    return {
        "eligible": len(rows),
        "comparable": sum(1 for d in details if d["status"] != "incomparable"),
        "improved": sum(1 for d in details if d["status"] == "improved"),
        "regressed": sum(1 for d in details if d["status"] == "regressed"),
        "tied": sum(1 for d in details if d["status"] == "tied"),
        "incomparable": sum(1 for d in details if d["status"] == "incomparable"),
        "mean_axis_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "tokens_per_improvement": (
            round(improved_tokens / max(1, sum(1 for d in details if d["status"] == "improved")))
            if any(d["status"] == "improved" for d in details) else None
        ),
        "by_intervention": by_intervention,
        "details": details,
    }


def _gather(conn, cutoff: int | None, stem: str | None = None,
            attempt_id: str | None = None) -> dict:
    """Run the aggregation queries and fold grader_scores JSON in Python (robust
    across SQLite builds without JSON1)."""
    where_time = " AND created_at >= ?" if cutoff is not None else ""
    params = (cutoff,) if cutoff is not None else ()

    # --- draft quality: join each grade row to the author draft it scored, so
    #     the (NULL-model) grader score is attributed to the authoring model.
    quality: dict[str, dict] = {}
    q_rows = conn.execute(
        "SELECT a.model_used AS model, g.grader_scores AS scores "
        "FROM ingest_iterations g "
        "JOIN ingest_iterations a ON g.parent_iteration_id = a.id "
        "WHERE g.role = 'grade' AND a.role = 'author' "
        "  AND a.model_used IS NOT NULL" + where_time.replace("created_at", "g.created_at"),
        params,
    ).fetchall()
    for r in q_rows:
        model = r["model"]
        try:
            s = json.loads(r["scores"]) if r["scores"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        sem = s.get("mean_semantic")
        slot = quality.setdefault(model, {"drafts": 0, "sem_sum": 0.0, "sem_n": 0, "drift": 0})
        slot["drafts"] += 1
        if isinstance(sem, (int, float)):
            slot["sem_sum"] += sem
            slot["sem_n"] += 1
        slot["drift"] += int(s.get("n_drift") or 0)

    # --- token spend + cost by model (all roles that used a real model).
    by_model: dict[str, dict] = {}
    m_rows = conn.execute(
        "SELECT model_used AS model, COUNT(*) AS calls, "
        "       SUM(COALESCE(cost_input_tokens,0)) AS in_tok, "
        "       SUM(COALESCE(cost_output_tokens,0)) AS out_tok "
        "FROM ingest_iterations "
        "WHERE model_used IS NOT NULL AND model_used NOT IN (?, ?)" + where_time +
        " GROUP BY model_used",
        (*_NON_MODEL_SENTINELS, *params),
    ).fetchall()
    for r in m_rows:
        by_model[r["model"]] = {
            "calls": int(r["calls"]),
            "in_tok": int(r["in_tok"] or 0),
            "out_tok": int(r["out_tok"] or 0),
        }

    # --- token spend by role.
    by_role: dict[str, dict] = {}
    r_rows = conn.execute(
        "SELECT role, COUNT(*) AS calls, "
        "       SUM(COALESCE(cost_input_tokens,0)) AS in_tok, "
        "       SUM(COALESCE(cost_output_tokens,0)) AS out_tok "
        "FROM ingest_iterations WHERE 1=1" + where_time + " GROUP BY role",
        params,
    ).fetchall()
    for r in r_rows:
        by_role[r["role"]] = {
            "calls": int(r["calls"]),
            "in_tok": int(r["in_tok"] or 0),
            "out_tok": int(r["out_tok"] or 0),
        }

    # --- section difficulty (graded drafts).
    by_section: dict[str, dict] = {}
    s_rows = conn.execute(
        "SELECT section, grader_scores AS scores FROM ingest_iterations "
        "WHERE role = 'grade'" + where_time, params,
    ).fetchall()
    for r in s_rows:
        sec = r["section"] or "(whole page)"
        try:
            s = json.loads(r["scores"]) if r["scores"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        slot = by_section.setdefault(sec, {"graded": 0, "sem_sum": 0.0, "sem_n": 0, "drift": 0, "neg": 0})
        slot["graded"] += 1
        sem = s.get("mean_semantic")
        if isinstance(sem, (int, float)):
            slot["sem_sum"] += sem
            slot["sem_n"] += 1
        slot["drift"] += int(s.get("n_drift") or 0)
        slot["neg"] += int(s.get("n_negation_mismatches") or 0)

    # --- draft decisions.
    decisions: dict[str, int] = {}
    d_rows = conn.execute(
        "SELECT decision, COUNT(*) AS n FROM ingest_iterations "
        "WHERE decision IS NOT NULL AND decision <> ''" + where_time +
        " GROUP BY decision", params,
    ).fetchall()
    for r in d_rows:
        decisions[r["decision"]] = int(r["n"])

    n_attempts = conn.execute(
        "SELECT COUNT(DISTINCT attempt_id) AS n FROM ingest_iterations WHERE 1=1" + where_time,
        params,
    ).fetchone()["n"]

    # Attempt completion is derived conservatively. No terminal row means
    # incomplete, not failed: old/migrated databases and killed processes are
    # indistinguishable after the fact.
    attempt_rows = conn.execute(
        "SELECT attempt_id, role, decision FROM ingest_iterations WHERE 1=1" + where_time,
        params,
    ).fetchall()
    attempts: dict[str, list] = {}
    for row in attempt_rows:
        attempts.setdefault(row["attempt_id"], []).append(row)
    attempt_status = Counter()
    for rows in attempts.values():
        if any(r["role"] == "budget" and r["decision"] == "budget-exhausted" for r in rows):
            attempt_status["budget_exhausted"] += 1
        elif any(r["role"] == "commit" and str(r["decision"] or "").startswith("committed") for r in rows):
            attempt_status["completed"] += 1
        elif any(r["role"] == "commit" and r["decision"] == "promote-failed" for r in rows):
            attempt_status["promote_failed"] += 1
        else:
            attempt_status["incomplete"] += 1

    duration_rows = conn.execute(
        "SELECT role, duration_ms, gate_metrics FROM ingest_iterations "
        "WHERE 1=1" + where_time,
        params,
    ).fetchall()
    latency = latency_distribution(duration_rows)

    gate_rows = conn.execute(
        "SELECT role, gate_metrics FROM ingest_iterations "
        "WHERE role IN ('target_claims','grade','critic','claim_support','commit','budget')" +
        where_time,
        params,
    ).fetchall()
    gate_totals: Counter = Counter()
    gate_measured = 0
    for row in gate_rows:
        if not row["gate_metrics"]:
            continue
        try:
            metrics = json.loads(row["gate_metrics"])
        except (TypeError, json.JSONDecodeError):
            continue
        gate_measured += 1
        for key, value in metrics.items():
            if isinstance(value, bool):
                gate_totals[key] += int(value)
            elif isinstance(value, (int, float)):
                gate_totals[key] += value

    try:
        corpus_papers = int(conn.execute(
            "SELECT COUNT(*) AS n FROM papers WHERE page_type='paper'"
        ).fetchone()["n"] or 0)
        tracked_papers = int(conn.execute(
            "SELECT COUNT(DISTINCT i.paper_stem) AS n FROM ingest_iterations i "
            "JOIN papers p ON p.stem=i.paper_stem WHERE p.page_type='paper'"
        ).fetchone()["n"] or 0)
    except Exception:
        corpus_papers = tracked_papers = 0

    return {
        "quality": quality,
        "by_model": by_model,
        "by_role": by_role,
        "by_section": by_section,
        "decisions": decisions,
        "n_attempts": int(n_attempts or 0),
        "attempt_status": dict(attempt_status),
        "telemetry_coverage": {
            "tracked_papers": tracked_papers,
            "corpus_papers": corpus_papers,
            "untracked_papers": max(0, corpus_papers - tracked_papers),
        },
        "latency": latency,
        "attempts": gather_attempts(
            conn, where_time, params, stem=stem, attempt_id=attempt_id,
        ),
        "gate_health": {
            "samples": gate_measured,
            "eligible": len(gate_rows),
            "totals": dict(gate_totals),
        },
        "lineage": _revision_outcomes(conn, where_time, params, stem=stem),
    }


def _to_json(data: dict, days: int | None) -> dict:
    def _mean(sum_, n):
        return round(sum_ / n, 4) if n else None
    return {
        "window_days": days,
        "n_attempts": data["n_attempts"],
        "pricing_as_of": _mc.pricing_as_of(),
        "by_model": {
            m: {
                "calls": v["calls"], "input_tokens": v["in_tok"], "output_tokens": v["out_tok"],
                "estimated_usd": round(_estimate_usd(m, v["in_tok"], v["out_tok"]), 4),
                "drafts": data["quality"].get(m, {}).get("drafts", 0),
                "mean_semantic": _mean(data["quality"].get(m, {}).get("sem_sum", 0.0),
                                       data["quality"].get(m, {}).get("sem_n", 0)),
                "drift": data["quality"].get(m, {}).get("drift", 0),
            }
            for m, v in sorted(data["by_model"].items())
        },
        "by_role": {
            r: {"calls": v["calls"], "input_tokens": v["in_tok"], "output_tokens": v["out_tok"]}
            for r, v in sorted(data["by_role"].items())
        },
        "by_section": {
            s: {"graded": v["graded"], "mean_semantic": _mean(v["sem_sum"], v["sem_n"]),
                "drift": v["drift"], "negation_mismatches": v["neg"]}
            for s, v in sorted(data["by_section"].items())
        },
        "decisions": data["decisions"],
        "attempt_status": data["attempt_status"],
        "telemetry_coverage": data["telemetry_coverage"],
        "latency": data["latency"],
        "attempts": data["attempts"],
        "gate_health": data["gate_health"],
        "lineage": data["lineage"],
    }


def _minutes(ms: int | None) -> str:
    return "—" if ms is None else f"{ms / 60000:.2f}"


def _print_attempt_timings(attempts: list[dict], *, detailed: bool) -> None:
    print("Attempt timings:")
    if not attempts:
        print("  (no matching attempts)")
    for attempt in attempts:
        label = attempt["paper_stem"] or attempt["pdf_filename"]
        wall_s = f"{attempt['wall_minutes']:.2f}m"
        phase = attempt["measured_minutes"]
        phase_s = "—" if phase is None else f"{phase:.2f}m"
        fallback = "≈" if attempt["wall_source"] == "event-span-fallback" else " "
        print(f"  {attempt['attempt_id'][:8]}  wall{fallback}{wall_s:>7}  "
              f"phase={phase_s:>7}  "
              f"{attempt['timing_samples']}/{attempt['timing_eligible']} timed  "
              f"{attempt['outcome']:<22} {label}")
        if detailed:
            for step in attempt["steps"]:
                prefix = "↳" if step["nested_in_commit"] else " "
                mins = _minutes(step["duration_ms"])
                duration = "—" if mins == "—" else f"{mins}m"
                print(f"      {prefix} {step['iteration']:>3} "
                      f"{step['role']:<18} {duration:>7}  "
                      f"{step['decision'] or ''}")


def _print_report(data: dict, days: int | None, show_lineage: bool = False,
                  show_attempts: bool = False,
                  attempt_id: str | None = None) -> None:
    window = f"last {days} days" if days else "all time"
    print(f"Research Wiki — ingest insights  ({window})\n")
    if attempt_id:
        _print_attempt_timings(data["attempts"], detailed=True)
        return
    if data["n_attempts"] == 0:
        print("No ingest telemetry yet — run `researchwiki agent ingest` to populate.")
        return
    print(f"Ingest attempts: {data['n_attempts']}\n")
    cov = data["telemetry_coverage"]
    if cov["corpus_papers"]:
        print(f"Telemetry coverage: {cov['tracked_papers']}/{cov['corpus_papers']} paper pages "
              f"({cov['untracked_papers']} migrated/untracked)\n")

    # By model
    print("By model (drafts scored against the PDF; tokens across all roles):")
    print(f"  {'model':<34}{'drafts':>7}{'mean_sem':>10}{'drift':>7}{'tokens(in/out)':>18}{'est $':>10}")
    for m, v in sorted(data["by_model"].items(), key=lambda kv: -(kv[1]["in_tok"] + kv[1]["out_tok"])):
        q = data["quality"].get(m, {})
        sem = (q["sem_sum"] / q["sem_n"]) if q.get("sem_n") else None
        sem_s = f"{sem:.3f}" if sem is not None else "—"
        toks = f"{_fmt_tokens(v['in_tok'])}/{_fmt_tokens(v['out_tok'])}"
        usd = f"${_estimate_usd(m, v['in_tok'], v['out_tok']):.2f}"
        print(f"  {m:<34}{q.get('drafts', 0):>7}{sem_s:>10}{q.get('drift', 0):>7}{toks:>18}{usd:>10}")

    # Section difficulty
    if data["by_section"]:
        print("\nSection difficulty (graded drafts — lower mean_sem / more drift = harder):")
        print(f"  {'section':<20}{'graded':>8}{'mean_sem':>10}{'drift':>7}{'neg':>5}")
        for s, v in sorted(data["by_section"].items(), key=lambda kv: (kv[1]["sem_sum"] / kv[1]["sem_n"]) if kv[1]["sem_n"] else 1.0):
            sem = (v["sem_sum"] / v["sem_n"]) if v["sem_n"] else None
            sem_s = f"{sem:.3f}" if sem is not None else "—"
            print(f"  {s:<20}{v['graded']:>8}{sem_s:>10}{v['drift']:>7}{v['neg']:>5}")

    # Token spend by role
    print("\nToken spend by role:")
    print(f"  {'role':<16}{'calls':>7}{'tokens(in/out)':>18}")
    for r, v in sorted(data["by_role"].items(), key=lambda kv: -(kv[1]["in_tok"] + kv[1]["out_tok"])):
        toks = f"{_fmt_tokens(v['in_tok'])}/{_fmt_tokens(v['out_tok'])}"
        print(f"  {r:<16}{v['calls']:>7}{toks:>18}")

    # Decisions
    if data["decisions"]:
        print("\nDraft decisions:")
        for d, n in sorted(data["decisions"].items(), key=lambda kv: -kv[1]):
            print(f"  {d:<16}{n:>6}")

    lineage = data["lineage"]
    if lineage["eligible"]:
        print("\nRevision outcomes:")
        print(f"  comparable: {lineage['comparable']}/{lineage['eligible']}  "
              f"improved: {lineage['improved']}  regressed: {lineage['regressed']}  "
              f"tied: {lineage['tied']}  missing grades/parents: {lineage['incomparable']}")
        for name, slot in lineage["by_intervention"].items():
            if slot["eligible"]:
                print(f"  {name:<8} {slot['comparable']}/{slot['eligible']} comparable; "
                      f"{slot['improved']} improved, {slot['regressed']} regressed")
        if show_lineage:
            for d in lineage["details"]:
                delta = "—" if d["mean_axis_delta"] is None else f"{d['mean_axis_delta']:+.3f}"
                print(f"    {d['paper_stem'] or '(unknown)'}  {d['intervention']:<6} "
                      f"{d['status']:<12} delta={delta}  id={d['revision_id']}")

    measured = sum(v["samples"] for v in data["latency"].values())
    eligible = sum(v["eligible"] for v in data["latency"].values())
    if eligible:
        print(f"\nLatency telemetry: {measured}/{eligible} phase rows measured")
        print(f"  {'phase':<18}{'n':>7}{'avg':>8}{'min':>8}{'med':>8}{'p95':>8}{'max':>8}")
        for role, v in sorted(data["latency"].items()):
            marker = "↳ " if v["nested_in_commit"] else ""
            label = f"{marker}{role}"
            print(f"  {label:<18}{v['samples']:>3}/{v['eligible']:<3}"
                  f"{_minutes(v['mean_ms']):>8}{_minutes(v['min_ms']):>8}"
                  f"{_minutes(v['median_ms']):>8}{_minutes(v['p95_ms']):>8}"
                  f"{_minutes(v['max_ms']):>8}")
        print("  minutes; ↳ nested in commit and excluded from attempt totals")

    if show_attempts or attempt_id:
        print()
        _print_attempt_timings(data["attempts"], detailed=False)

    gh = data["gate_health"]
    if gh["eligible"]:
        print(f"\nGate-health telemetry: {gh['samples']}/{gh['eligible']} eligible rows measured")
        for key, value in sorted(gh["totals"].items()):
            print(f"  {key:<24}{value:g}")

    print(f"\n(Rates as of {_mc.pricing_as_of() or 'unknown'} from config/pricing.yaml; "
          f"local/unpriced models show $0.00. Upper bound — prompt-cache hits "
          f"cost 0.1x input and aren't recorded per-call.)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki insights",
        description="Analytics over the ingest telemetry log (read-only, no LLM).",
    )
    parser.add_argument("--days", type=int, default=None,
                        help="Restrict to the last N days (default: all time).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    parser.add_argument("--lineage", action="store_true",
                        help="Print individual revision-parent comparisons.")
    parser.add_argument("--stem", default=None,
                        help="Restrict lineage and attempt timings to one paper stem.")
    parser.add_argument("--attempts", action="store_true",
                        help="Print one timing summary row per ingest attempt.")
    parser.add_argument("--attempt-id", default=None,
                        help="Print the timing breakdown for one exact attempt ID.")
    args = parser.parse_args(argv)

    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception as e:
        # This is one of the loud paths — `insights` is a targeted analytics
        # command, so DB unavailability is a hard error rather than a silent
        # empty report (contrast with `status`, which downgrades via safe_read).
        print(f"researchwiki insights: DB unreachable ({e})", file=sys.stderr)
        return 2

    cutoff = int(time.time()) - args.days * 86400 if args.days else None
    try:
        data = _gather(
            conn, cutoff, stem=args.stem, attempt_id=args.attempt_id,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if args.json:
        print(json.dumps(_to_json(data, args.days), indent=2, ensure_ascii=False))
    else:
        _print_report(
            data, args.days, show_lineage=args.lineage,
            show_attempts=args.attempts, attempt_id=args.attempt_id,
        )
    return 0
