"""Test whether CLAUDE.md's prompt pointers fire when they should.

CLAUDE.md gates each `prompts/*.md` behind a one-line trigger. That line is
load-bearing — if it doesn't fire, the procedure may as well not exist — and
nothing tested it. This generates should-fire / should-not-fire requests per
prompt, routes each using *only* the trigger lines, and reports the misses.

    researchwiki eval-triggers                     # all pointers, 5+5 each
    researchwiki eval-triggers --slug recovery     # one prompt
    researchwiki eval-triggers -n 10 --json        # more cases, machine-readable
    researchwiki eval-triggers --dry-run           # list pointers, spend nothing

Costs tokens: one generator call per prompt plus 2N grader calls. Run it after
editing a trigger line, not on a schedule.

**The output is the named misses, not the rates.** A trigger at 7/10 tells you
less than the three requests it missed. Success here is finding the two or three
triggers that are too vague — expect to fix prompts, not this command.

Exit codes: 0 = ran; 1 = nothing to evaluate; 2 = provider unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..eval import triggers as tr


def _print_report(reports, orphans, count) -> None:
    print(f"trigger eval — {len(reports)} prompt(s), {count} case(s) each way\n")
    print(f"  {'prompt':32} {'fires':>7}  {'holds off':>10}  {'errors':>7}")
    print(f"  {'-' * 32} {'-' * 7}  {'-' * 10}  {'-' * 7}")

    for slug in sorted(reports):
        rep = reports[slug]
        rec = f"{rep.should_fire_hit}/{rep.should_fire_total}" \
            if rep.should_fire_total else "—"
        pre = f"{rep.should_not_hit}/{rep.should_not_total}" \
            if rep.should_not_total else "—"
        err = str(rep.errors) if rep.errors else ""
        print(f"  {slug:32} {rec:>7}  {pre:>10}  {err:>7}")

    flagged = [r for r in reports.values() if r.misses]
    if not flagged:
        print("\n  no misses.")
    else:
        print(f"\n  misses ({sum(len(r.misses) for r in flagged)}):")
        for rep in sorted(flagged, key=lambda r: r.slug):
            print(f"\n  {rep.slug}")
            for g in rep.misses:
                if g.case.should_fire:
                    got = g.chose or "no prompt"
                    print(f"    ✗ should have fired, routed to {got}")
                else:
                    print(f"    ✗ fired when it shouldn't")
                print(f"        {g.case.request[:150]}")

    if orphans:
        print(f"\n  ⚠ {len(orphans)} prompt file(s) with no CLAUDE.md pointer — "
              f"the agent has no condition under which it reads them:")
        for slug in orphans:
            print(f"      prompts/{slug}.md")

    total_err = sum(r.errors for r in reports.values())
    if total_err:
        print(f"\n  note: {total_err} grading(s) errored and are excluded from "
              f"the rates above, not counted as failures.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki eval-triggers",
        description="Check that CLAUDE.md's prompt triggers fire when they should.",
    )
    parser.add_argument("--slug", action="append",
                        help="Evaluate only this prompt (repeatable).")
    parser.add_argument("-n", "--count", type=int, default=tr.DEFAULT_COUNT,
                        help=f"Cases per direction (default: {tr.DEFAULT_COUNT}).")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="List the pointers and exit. Spends nothing.")
    parser.add_argument("--stub", action="store_true",
                        help="Use the offline stub provider (self-test only).")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    pointers = tr.collect_pointers()
    if args.slug:
        wanted = set(args.slug)
        pointers = [p for p in pointers if p.slug in wanted]
    if not pointers:
        print("researchwiki eval-triggers: no prompt pointers found in CLAUDE.md.",
              file=sys.stderr)
        return 1

    orphans = tr.orphan_prompts()

    if args.dry_run:
        print(f"{len(pointers)} trigger-gated prompt(s):\n")
        for p in pointers:
            print(f"  {p.slug}")
            print(f"    {p.line[:160]}")
        if orphans:
            print(f"\n  {len(orphans)} with no pointer: "
                  f"{', '.join(orphans)}")
        print(f"\n  would spend {len(pointers)} generator call(s) + "
              f"{len(pointers) * args.count * 2} grader call(s).")
        return 0

    try:
        cases: list[tr.Case] = []
        for p in pointers:
            cases.extend(tr.generate_cases(p, args.count, use_stub=args.stub))
        graded = tr.grade_all(cases, pointers, use_stub=args.stub)
    except Exception as e:
        print(f"researchwiki eval-triggers: provider error: {e}", file=sys.stderr)
        return 2

    reports = tr.summarize(graded)

    if args.as_json:
        print(json.dumps({
            "count": args.count,
            "orphan_prompts": orphans,
            "prompts": {
                slug: {
                    "recall": rep.recall,
                    "precision": rep.precision,
                    "should_fire": [rep.should_fire_hit, rep.should_fire_total],
                    "should_not_fire": [rep.should_not_hit, rep.should_not_total],
                    "errors": rep.errors,
                    "misses": [
                        {"request": g.case.request,
                         "should_fire": g.case.should_fire,
                         "routed_to": g.chose}
                        for g in rep.misses
                    ],
                }
                for slug, rep in sorted(reports.items())
            },
        }, indent=2))
        return 0

    _print_report(reports, orphans, args.count)
    return 0
