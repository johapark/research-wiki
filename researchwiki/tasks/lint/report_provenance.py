"""Human report section for author-provenance findings and acknowledgments."""


def print_author_provenance_sections(
    missing_author_model: list[str], acknowledged: list[str]
) -> None:
    if missing_author_model:
        print(f"## Non-idea pages missing `author_model:` ({len(missing_author_model)})")
        print("Authored paper/commentary, synthesis, concept, and reference pages "
              "must name the exact model that wrote their prose. Idea pages are "
              "intentionally exempt because they evolve over time. `lint --fix` "
              "recovers only telemetry-backed paper/commentary pages; hand-authored "
              "reference, synthesis, and concept pages need a manual value.")
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
