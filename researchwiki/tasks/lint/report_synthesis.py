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


def print_authored_contract_sections(kw: dict) -> None:
    """Proposal, synthesis and cross-link-direction findings. Takes the whole
    lint keyword bundle so adding a section here costs report.py no lines —
    it is pinned by `tests/test_module_size.py` and rendering is this
    module's one job."""
    print_proposal_contract_section(kw["proposal_contract"])
    print_synthesis_contract_section(kw["synthesis_contract"])
    print_crosslink_direction_section(kw["crosslink_direction"])


def print_crosslink_direction_section(violations: list[dict]) -> None:
    if not violations:
        return
    print(f"## Cross-link citation directions the years rule out ({len(violations)}, advisory)")
    print("A `cites this paper` bullet needs the *target* to be the newer paper; "
          "`cited by this paper` needs it to be older. These assert the opposite. "
          "Most were written by `lint --fix` or `promote` from an inferred edge "
          "direction, and no other check sees a bullet's claim. Remove both "
          "directions of a false pair in one edit, or `lint --fix` re-inserts it.")
    by_page: dict[str, list[str]] = {}
    for violation in violations:
        by_page.setdefault(page_key(violation["page"]), []).append(violation["detail"])
    for key, details in sorted(by_page.items())[:20]:
        print(f"- **{key}**")
        for detail in details:
            print(f"    - {detail}")
    if len(by_page) > 20:
        print(f"_... +{len(by_page) - 20} more pages_")
    print()
