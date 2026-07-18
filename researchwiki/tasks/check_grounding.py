"""Flag/strip claim-bearing units that lack a [[wikilink]] anchor.

Usage:
  researchwiki check-grounding <file.md>
  researchwiki check-grounding < input.md

Modes:
  --annotate (default)   Print the markdown with ' ⚠ ungrounded' appended
                         to each ungrounded paragraph or bullet, and
                         ' ⚠ model prior' on each marker-grounded unit.
  --strip                Replace each ungrounded unit with a placeholder.
  --json                 Emit a structured report instead of markdown.

Grounding categories (per claim-unit, default mode):
  grounded     — has [[wikilink]] (including [[stem#slug]]), or footnote
                 that resolves to one. Legacy `claim_id:NNN` tokens are
                 tolerated by the grader for backward compat but new pages
                 should use `[[stem#slug]]` (durable content-addressed).
  model_prior  — has only the *(model prior)* marker (idea pages, in
                 Opportunities/Plans only — see CLAUDE.md §4)
  ungrounded   — has neither; counts as a failure (exit 1)

  --strict collapses model_prior into ungrounded — every claim must be
  wiki-grounded regardless of marker.

This is the *structural* gate: it checks a citation is present, not that it
holds. For synthesis/idea pages, pair it with `researchwiki grade synthesis`,
which grades each cited claim against the PDF(s) it cites (the fidelity gate).

Exit codes:
  0  No ungrounded claims (or input was non-claim only).
  1  ≥1 ungrounded claim found.
  2  Bad input / I/O error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..grade import grounding


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki check-grounding",
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument(
        "path", nargs="?",
        help="Markdown file to check (defaults to stdin).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--annotate", action="store_true",
                      help="(default) Append ' ⚠ ungrounded' to each ungrounded unit.")
    mode.add_argument("--strip", action="store_true",
                      help="Replace each ungrounded unit with a placeholder.")
    mode.add_argument("--json", action="store_true",
                      help="Emit a structured report instead of markdown.")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the trailing '[grounding] N/M units cited' summary on stderr.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=("Treat the *(model prior)* marker as ungrounded — every claim "
              "must carry a wiki citation regardless of section (CLAUDE.md §4)."),
    )
    args = parser.parse_args(argv)
    permissive = not args.strict

    try:
        text = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"researchwiki check-grounding: {e}", file=sys.stderr)
        return 2

    if not text.strip():
        return 0

    report = grounding.check(text, permissive=permissive)

    if report.anchor_db_unavailable:
        # Environment error, not a content failure: state.db was unreachable so
        # [[stem#slug]] anchors couldn't be resolved. Anchors were counted
        # permissively; report exit 2 rather than a spurious ungrounded failure.
        print("researchwiki check-grounding: state.db unreachable — could not "
              "resolve claim anchors; grounding not verified. Retry once the DB "
              "is available.", file=sys.stderr)
        return 2

    if args.json:
        out = {
            "total_claims": report.total_claims,
            "grounded_claims": report.grounded_claims,
            "model_prior_claims": report.model_prior_claims,
            "ungrounded_claims": len(report.ungrounded_units),
            "coverage": round(report.coverage, 3),
            "permissive": permissive,
            "units": [
                {
                    "index": u.index,
                    "line_start": u.line_start,
                    "kind": u.kind,
                    "is_claim": u.is_claim,
                    "has_citation": u.has_citation,
                    "is_model_prior": u.is_model_prior,
                    "citations": u.citations,
                    "flag_reason": u.flag_reason,
                    "preview": (u.text[:160].replace("\n", " ")
                                + ("…" if len(u.text) > 160 else "")),
                }
                for u in report.units
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.strip:
        sys.stdout.write(grounding.strip(text, permissive=permissive))
    else:
        sys.stdout.write(grounding.annotate(text, permissive=permissive))

    if not args.quiet and not args.json:
        mp = report.model_prior_claims
        mp_part = f", {mp} model prior" if mp else ""
        print(
            f"\n[grounding] {report.grounded_claims}/{report.total_claims} "
            f"claim-units wiki-cited{mp_part} "
            f"({report.coverage * 100:.0f}% acknowledged)",
            file=sys.stderr,
        )

    return 0 if not report.ungrounded_units else 1
