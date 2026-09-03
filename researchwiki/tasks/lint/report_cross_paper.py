"""Prose rendering for the opt-in cross-paper contradiction section.

Split out of `report.py` when that module hit its size pin. This section is the
natural seam: it is the only lint surface that costs LLM calls, and consequently
the only one that has to report *coverage* as well as findings — it carries state
(`cross_paper_judgements`) that no other check has, and the reporting rules that
follow from that state have nothing in common with the rest of the report.

Why coverage is printed even when nothing was found: for every other check,
silence means "clean". Here silence used to be ambiguous between "judged the pool
and it is clean" and "never ran" — and resolving exactly that ambiguity is why the
judgements table exists. Printing a zero section with the pool size keeps the
report honest about which of the two happened.
"""

from __future__ import annotations


def _coverage_line(stats: dict) -> str:
    line = (f"pool: {stats.get('pool', 0)} pair(s) above "
            f"{stats.get('sim_threshold', 0):.2f}; "
            f"judged {stats.get('judged', 0)}, "
            f"skipped {stats.get('skipped_already_judged', 0)} already judged")
    # A partial sweep must say so, or "0 contradictions" reads as a clean bill of
    # health. Every verdict reached is already recorded, so a re-run continues.
    stopped = stats.get("stopped_early")
    if stopped:
        line += (f"; STOPPED EARLY — {stopped}. Re-run to continue from here "
                 f"(judged pairs are not re-paid)")
    return line


def print_cross_paper_section(cross_paper: list[dict], stats: dict | None) -> None:
    """Render the section. No-op when the check never ran (`stats is None`)."""
    if stats is None and not cross_paper:
        return

    if not cross_paper:
        print("## Cross-paper contradictions (0)")
        print(f"- {_coverage_line(stats or {})}")
        print()
        return

    print(f"## Cross-paper contradictions ({len(cross_paper)})")
    for c in cross_paper[:10]:
        a, b = c["pair"]
        print(f"- **{c['verdict']}** (sim={c['similarity']:.2f})")
        print(f"    A: [[{a['paper_stem']}]] ({a['section']}#{a['position']}) — {a['text']}")
        print(f"    B: [[{b['paper_stem']}]] ({b['section']}#{b['position']}) — {b['text']}")
        if c.get("rationale"):
            print(f"    Judge: {c['rationale']}")
    if len(cross_paper) > 10:
        print(f"- ... ({len(cross_paper) - 10} more)")
    print()
    if stats:
        print(f"_{_coverage_line(stats).capitalize()}._")
    print("_Cross-paper lint is opt-in (`--cross-paper`); the LLM judge classified each pair_")
    print("_as `disagree_numeric` or `disagree_direction`. Verify against the source PDFs before correcting._")
    print()
