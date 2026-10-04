"""Human-readable synthesis contract findings for ``researchwiki lint``."""

from __future__ import annotations

from .report_proposals import print_proposal_contract_section
from .walk import page_key


def print_synthesis_contract_section(violations: list[dict]) -> None:
    if not violations:
        return
    legacy = [v for v in violations if v["kind"] == "synthesis_legacy_spine"]
    other = [v for v in violations if v["kind"] != "synthesis_legacy_spine"]
    print(f"## Synthesis-page contract violations ({len(violations)}, advisory)")
    print("Synthesis pages share one fixed H2 structure (CLAUDE.md §2). Warn-only — "
          "these don't fail lint. Neither page gate reads headings, so this is the "
          "only check that sees them.")
    if legacy:
        print(f"- {len(legacy)} page(s) predate the structure; upgrade each when next "
              "edited (prompts/synthesis-page-author.md, Mode C):")
        for v in sorted(legacy, key=lambda v: page_key(v["page"]))[:20]:
            print(f"    - {page_key(v['page'])}")
        if len(legacy) > 20:
            print(f"    - _... +{len(legacy) - 20} more_")
    by_page: dict[str, list[tuple[str, str]]] = {}
    for violation in other:
        by_page.setdefault(page_key(violation["page"]), []).append(
            (violation["kind"], violation["detail"])
        )
    for key, entries in sorted(by_page.items())[:20]:
        print(f"- **{key}**")
        for kind, detail in entries:
            print(f"    - `{kind}` — {detail}")
    print()


def print_authored_contract_sections(proposal: list[dict], synthesis: list[dict]) -> None:
    """Proposal then synthesis findings: the two page types whose shape is a
    fixed Markdown contract. One entry point keeps report.py under its cap."""
    print_proposal_contract_section(proposal)
    print_synthesis_contract_section(synthesis)
