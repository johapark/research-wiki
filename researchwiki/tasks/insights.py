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
  researchwiki insights --json

Exit codes: 0 = printed (including "no telemetry yet"); 2 = DB unreachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Reuse the single source of truth for pricing so cost figures match `status`.
from .. import pricing

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
    the prefix matching both live in `researchwiki.pricing`."""
    return pricing.estimate_usd(model, in_tok, out_tok)


def _gather(conn, cutoff: int | None) -> dict:
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

    return {
        "quality": quality,
        "by_model": by_model,
        "by_role": by_role,
        "by_section": by_section,
        "decisions": decisions,
        "n_attempts": int(n_attempts or 0),
    }


def _to_json(data: dict, days: int | None) -> dict:
    def _mean(sum_, n):
        return round(sum_ / n, 4) if n else None
    return {
        "window_days": days,
        "n_attempts": data["n_attempts"],
        "pricing_as_of": pricing.as_of(),
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
    }


def _print_report(data: dict, days: int | None) -> None:
    window = f"last {days} days" if days else "all time"
    print(f"Research Wiki — ingest insights  ({window})\n")
    if data["n_attempts"] == 0:
        print("No ingest telemetry yet — run `researchwiki agent ingest` to populate.")
        return
    print(f"Ingest attempts: {data['n_attempts']}\n")

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

    print(f"\n(Rates as of {pricing.as_of() or 'unknown'} from config/pricing.yaml; "
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
        data = _gather(conn, cutoff)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if args.json:
        print(json.dumps(_to_json(data, args.days), indent=2, ensure_ascii=False))
    else:
        _print_report(data, args.days)
    return 0
