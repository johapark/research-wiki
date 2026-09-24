"""Human report section for author-provenance findings and acknowledgments."""


def print_author_provenance_sections(
    missing_author_model: list[str], acknowledged: list[str]
) -> None:
    if missing_author_model:
        print(
            "## Documents missing an exact `author_model:` "
            f"({len(missing_author_model)})"
        )
        print("Paper/commentary, synthesis, concept, idea, and reference pages "
              "must name the exact model that wrote their prose. Meta and "
              "dashboard pages are exempt. Generic family aliases count as "
              "missing. `lint --fix` recovers or refines these "
              "only on telemetry-backed paper/commentary pages; hand-authored "
              "reference, synthesis, concept, and idea pages need a manual value.")
        for key in missing_author_model[:20]:
            print(f"- {key}")
        if len(missing_author_model) > 20:
            print(f"- ... and {len(missing_author_model) - 20} more")
        print()

    if acknowledged:
        print(f"## Acknowledged legacy provenance ({len(acknowledged)})")
        print("These reviewed legacy pages have no recoverable exact author model. "
              "They carry `author_provenance: legacy-unrecorded` plus a dated "
              "acknowledgment, so they remain visible without repeating as "
              "actionable `missing_author_model` findings.")
        for key in acknowledged[:20]:
            print(f"- {key}")
        if len(acknowledged) > 20:
            print(f"- ... and {len(acknowledged) - 20} more")
        print()
