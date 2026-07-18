"""Grade a synthesis / idea page for fidelity — does each claim hold in the
paper it cites?

The fidelity gate, sibling to `check-grounding` (the structural gate).
`check-grounding` verifies a citation is *present*; this verifies it *holds* by
retrieving each claim against the PDF(s) of the paper(s) it cites and running
the deterministic numeric + negation checks (plus advisory BM25/semantic
retrieval). Catches cross-paper misattribution — a number or assertion ascribed
to a paper that doesn't contain it — which neither check-grounding nor the
paper-page grader (`grade`) can see. The two gates are orthogonal: structural
catches a missing citation (fidelity skips it as `uncited`); fidelity catches a
present-but-wrong citation (structural passes it). Run both before a page is
done.

Usage:
  researchwiki grade synthesis wiki/synthesis/<slug>.md
  researchwiki grade synthesis wiki/ideas/<slug>.md --json
  researchwiki grade synthesis <page> --no-semantic       # BM25 + numeric + negation only
  researchwiki grade synthesis <page> --weak              # also list weak/composite claims

Verdicts: supported · weak · composite · misattributed · uncited
(see researchwiki/grade/fidelity/synthesis.py for definitions).

Exit codes:
  0  No misattributed claims (weak/composite/uncited are advisory).
  1  ≥1 misattributed claim (a number cited to a paper that lacks it).
  2  Bad input / I/O error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..grade.fidelity import grade_synthesis
from ..log import log


def _format_text_report(report, show_advisory: bool) -> str:
    lines = []
    lines.append(f"  page          : {report.page_path}")
    lines.append(
        f"  claims        : {report.n_claims} graded "
        f"({report.n_uncited} uncited skipped)"
    )
    n_anchor = getattr(report, "n_anchor_misattributed", 0)
    anchor_seg = f", {n_anchor} ANCHOR-MISATTRIBUTED" if n_anchor else ""
    lines.append(
        f"  verdicts      : {report.n_supported} supported, "
        f"{report.n_weak} weak, {report.n_composite} composite, "
        f"{report.n_misattributed} MISATTRIBUTED{anchor_seg}"
    )
    if not report.semantic_available:
        lines.append("  semantic      : not available (--no-semantic or model unavailable)")
    lines.append("")

    mis = [c for c in report.claims if c.verdict == "misattributed"]
    if mis:
        lines.append("  ✗ misattributed (number cited to a paper that lacks it):")
        for c in mis:
            cited = ", ".join(c.cited_stems) or "—"
            lines.append(f"    L{c.line_start} cites [{cited}]")
            lines.append(f"      missing numbers: {c.numeric_unmatched}")
            lines.append(f"      claim: {c.text[:160]}")
        lines.append("")

    anchor_mis = [c for c in report.claims if c.verdict == "anchor_misattributed"]
    if anchor_mis:
        lines.append("  ✗ anchor-misattributed (number in the sentence not in the cited claim):")
        for c in anchor_mis:
            for miss in c.anchor_misattributions:
                lines.append(
                    f"    L{c.line_start} [[{miss['stem']}#{miss['slug']}]]"
                )
                lines.append(f"      missing from claim: {miss['numeric_tokens_missing']}")
            lines.append(f"      sentence: {c.text[:160]}")
        lines.append("")

    if show_advisory:
        adv = [c for c in report.claims if c.verdict in ("weak", "composite")]
        if adv:
            lines.append("  advisory (weak / composite — not failures):")
            for c in adv:
                cited = ", ".join(c.cited_stems) or "—"
                sem = f" sem={c.best_semantic:.2f}" if c.best_semantic is not None else ""
                lines.append(
                    f"    [{c.verdict}] L{c.line_start} bm25={c.best_bm25:.2f}{sem} "
                    f"cites [{cited}]"
                )
                lines.append(f"      claim: {c.text[:160]}")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="researchwiki grade synthesis",
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="Synthesis or idea markdown page to grade.")
    p.add_argument("--json", action="store_true", help="Emit a structured report.")
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip bi-encoder retrieval (BM25 + numeric + negation only).")
    p.add_argument("--weak", action="store_true",
                   help="In text mode, also list weak/composite advisory claims.")
    p.add_argument("--fine-grained", action="store_true",
                   help="Verify [[stem#slug]] anchors at the specific-claim level. "
                        "A number in the sentence that's in the paper but NOT in "
                        "the cited claim's text fails as `anchor_misattributed`.")
    args = p.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"researchwiki grade synthesis: file not found: {path}", file=sys.stderr)
        return 2

    try:
        report = grade_synthesis(
            path, semantic=not args.no_semantic, fine_grained=args.fine_grained,
        )
    except Exception as e:
        print(f"researchwiki grade synthesis: error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        log(f"{path.name}", tag="grade-synthesis")
        print(_format_text_report(report, show_advisory=args.weak))
        if report.ok:
            print(f"  → OK ({report.n_supported}/{report.n_claims} supported, "
                  f"no misattribution)", file=sys.stderr)
        else:
            parts = []
            if report.n_misattributed:
                parts.append(f"{report.n_misattributed} misattributed")
            n_anchor = getattr(report, "n_anchor_misattributed", 0)
            if n_anchor:
                parts.append(f"{n_anchor} anchor-misattributed")
            print(f"  → {' + '.join(parts)} claim(s) — "
                  f"verify against the cited PDFs / claim slugs", file=sys.stderr)

    return 0 if report.ok else 1
