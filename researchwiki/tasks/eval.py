"""Evaluate the framework's own decision-making.

Two subcommands, both read-only over the wiki:

    researchwiki eval classifier      # where does auto-categorization put papers?
    researchwiki eval triggers        # do CLAUDE.md's prompt pointers fire?

`classifier` costs nothing — it rebuilds held-out Tantivy indexes locally.
`triggers` costs one generator call per prompt plus 2N graders, so it is
on-demand; `--dry-run` prices the run first.

Distinct from `benchmark-fixture`, which scores *model output* against fixture
papers. These score the framework's routing: which category a new paper lands
in, and which procedure the agent reads.

The free half of the trigger check — prompt files no CLAUDE.md pointer reaches,
and pointers to files that don't exist — is not here. It costs nothing and is
therefore part of `researchwiki lint` (`orphan_prompts`,
`broken_prompt_pointers`), so it runs in the health check you already run rather
than waiting for someone to remember this command.

Exit codes: 0 = ran; 1 = nothing to evaluate; 2 = provider unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_classifier(args) -> int:
    from .eval_classifier import evaluate
    return evaluate()


def _print_trigger_report(reports, count) -> None:
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
                    print(f"    ✗ should have fired, routed to {g.chose or 'no prompt'}")
                else:
                    print(f"    ✗ fired when it shouldn't")
                print(f"        {g.case.request[:150]}")

    total_err = sum(r.errors for r in reports.values())
    if total_err:
        print(f"\n  note: {total_err} grading(s) errored and are excluded from "
              f"the rates above, not counted as failures.")


def _cmd_triggers(args) -> int:
    from ..eval import pointers as ptr
    from ..eval import triggers as tr

    pointers = ptr.collect()
    if args.slug:
        wanted = set(args.slug)
        pointers = [p for p in pointers if p.slug in wanted]
    if not pointers:
        print("researchwiki eval triggers: no prompt pointers found in CLAUDE.md.",
              file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"{len(pointers)} trigger-gated prompt(s):\n")
        for p in pointers:
            print(f"  {p.slug}")
            print(f"    {p.line.splitlines()[0][:160]}")
        print(f"\n  would spend {len(pointers)} generator call(s) + "
              f"{len(pointers) * args.count * 2} grader call(s).")
        print("  (unreachable prompts are reported by `researchwiki lint`, free)")
        return 0

    try:
        cases: list = []
        for p in pointers:
            cases.extend(tr.generate_cases(p, args.count, use_stub=args.stub))
        graded = tr.grade_all(cases, pointers, use_stub=args.stub)
    except Exception as e:
        print(f"researchwiki eval triggers: provider error: {e}", file=sys.stderr)
        return 2

    reports = tr.summarize(graded)

    if args.as_json:
        print(json.dumps({
            "count": args.count,
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

    _print_trigger_report(reports, args.count)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki eval", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_cls = subs.add_parser(
        "classifier",
        help="Leave-one-out accuracy of the category auto-suggester (free).")
    p_cls.set_defaults(func=_cmd_classifier)

    p_trg = subs.add_parser(
        "triggers",
        help="Do CLAUDE.md's prompt pointers fire when they should? (costs tokens)")
    p_trg.add_argument("--slug", action="append",
                       help="Evaluate only this prompt (repeatable).")
    p_trg.add_argument("-n", "--count", type=int, default=5,
                       help="Cases per direction (default: 5).")
    p_trg.add_argument("--dry-run", dest="dry_run", action="store_true",
                       help="Price the run and exit. Spends nothing.")
    p_trg.add_argument("--stub", action="store_true",
                       help="Use the offline stub provider (self-test only).")
    p_trg.add_argument("--json", dest="as_json", action="store_true")
    p_trg.set_defaults(func=_cmd_triggers)

    args = parser.parse_args(argv)
    return args.func(args)
