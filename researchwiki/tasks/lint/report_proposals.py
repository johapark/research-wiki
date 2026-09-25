"""Human-readable proposal contract findings for ``researchwiki lint``."""

from __future__ import annotations

from .walk import page_key


def print_proposal_contract_section(violations: list[dict]) -> None:
    if not violations:
        return
    print(f"## Proposal-page contract violations ({len(violations)}, advisory)")
    print("Proposal Markdown is canonical; fix these fields or sections before "
          "rebuilding its disposable database mirror.")
    by_page: dict[str, list[tuple[str, str]]] = {}
    for violation in violations:
        by_page.setdefault(page_key(violation["page"]), []).append(
            (violation["kind"], violation["detail"])
        )
    for key, entries in sorted(by_page.items())[:20]:
        print(f"- **{key}**")
        for kind, detail in entries:
            print(f"    - `{kind}` — {detail}")
    print()
