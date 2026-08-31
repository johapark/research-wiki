"""Tests for the concept-hub authored-prose contract lint.

Covers the three checks:
  - concept_definition_thin
  - concept_missing_bridge_section (on span ≥ 2 hubs)
  - concept_definition_paraphrases_claim
"""

from __future__ import annotations

from pathlib import Path

from researchwiki.tasks.lint.concept_contract import (
    check_hub,
    find_concept_contract_violations,
)


# ---------- fixture builders ----------


def _passing_definition() -> str:
    """A Definition body well above DEFINITION_MIN_WORDS content words after
    stopword filter, varied vocabulary, no template placeholders."""
    return (
        "Prime editing is a search-and-replace genome editing technology derived "
        "from CRISPR-Cas9 that uses a catalytically impaired Cas9 nickase fused to "
        "an engineered reverse transcriptase to install targeted insertions, "
        "deletions, and all twelve base-to-base conversions without requiring "
        "double-strand breaks or donor DNA templates in mammalian cells. Across "
        "the corpus prime editing appears as a molecular technique, a therapeutic "
        "vehicle for liver and hematopoietic indications, a benchmark surface for "
        "off-target measurement pipelines, and a testbed for lipid-nanoparticle "
        "delivery evaluation across cell types including primary hepatocytes."
    )


def _thin_definition() -> str:
    """Under the min-words threshold."""
    return "Prime editing installs targeted edits without DSBs."


def _bridge_hub_body(defn: str, corpus_section: str, bridge_section: str = "") -> str:
    """Assemble a full hub body from parts."""
    parts = [
        f"## Definition\n{defn}\n",
        "## How it appears across the corpus\n" + corpus_section + "\n",
    ]
    if bridge_section:
        parts.append(bridge_section)
    return "\n".join(parts)


def _flat_hub_body(defn: str, corpus_section: str) -> str:
    return f"## Definition\n{defn}\n\n## How it appears across the corpus\n{corpus_section}\n"


def _fm(span: int) -> dict:
    return {"type": "concept", "concept_span": span}


# ---------- check_hub ----------


def test_passing_hub_returns_no_violations():
    body = _bridge_hub_body(
        _passing_definition(),
        "- [[cgt/paper-a#kc-01]] — spoke",
        (
            "## Cross-domain connections\n\n"
            "Across cgt and compbio, prime editing shows up as a "
            "delivery-limited but PAM-flexible platform whose main tension "
            "is throughput versus off-target rate.\n"
        ),
    )
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=2))
    assert got == []


def test_thin_definition_flagged():
    body = _flat_hub_body(_thin_definition(),
                          "- [[cgt/paper-a#kc-01]] — spoke")
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=1))
    kinds = {v["kind"] for v in got}
    assert "concept_definition_thin" in kinds


def test_missing_bridge_section_flagged_on_span_ge_2():
    body = _flat_hub_body(_passing_definition(), "- [[cgt/paper-a#kc-01]]")
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=3))
    kinds = {v["kind"] for v in got}
    assert "concept_missing_bridge_section" in kinds


def test_bridge_section_not_required_on_deep_tier_hub():
    """A hub with concept_span==1 doesn't need Cross-domain connections."""
    body = _flat_hub_body(_passing_definition(), "- [[cgt/paper-a#kc-01]]")
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=1))
    kinds = {v["kind"] for v in got}
    assert "concept_missing_bridge_section" not in kinds


def test_bridge_section_variants_accepted():
    """`## Bridge` counts too — variants are OK as long as the section exists
    with real content."""
    body = _bridge_hub_body(
        _passing_definition(),
        "- [[cgt/paper-a#kc-01]]",
        "## Bridge\n\nAcross domains, this concept plays the delivery role.\n",
    )
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=2))
    kinds = {v["kind"] for v in got}
    assert "concept_missing_bridge_section" not in kinds


def test_empty_bridge_section_is_still_flagged():
    """Section exists but is empty (just the heading) — count as missing."""
    body = _bridge_hub_body(
        _passing_definition(),
        "- [[cgt/paper-a#kc-01]]",
        "## Cross-domain connections\n\n<!-- TODO -->\n",
    )
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=2))
    kinds = {v["kind"] for v in got}
    assert "concept_missing_bridge_section" in kinds


def test_definition_paraphrasing_spoke_claim_is_flagged():
    """Definition tokens are almost entirely a subset of one spoke's hint —
    that's copy-paste synthesis, flag it."""
    spoke_claim = (
        "Prime editing installs targeted insertions and deletions without "
        "requiring double-strand breaks or donor DNA templates."
    )
    # Definition is largely the same words with minimal padding.
    defn = (
        "Prime editing installs targeted insertions and deletions without "
        "requiring double-strand breaks or donor DNA templates in cells "
        "delivered to targeted tissues by injection or ex vivo pipelines."
    )
    body = _flat_hub_body(
        defn,
        f'- [[cgt/paper-a#kc-01]] — <!-- how this paper uses PE. hint: "{spoke_claim}" -->',
    )
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=1))
    kinds = {v["kind"] for v in got}
    assert "concept_definition_paraphrases_claim" in kinds


def test_definition_with_synthesis_not_flagged_as_paraphrase():
    """A Definition that pulls in words the spoke's claim doesn't have —
    real synthesis language — must not flag."""
    spoke_claim = "PE-2 improves prime editing efficiency in HEK cells."
    defn = (
        "Prime editing is a search-and-replace genome editing paradigm that "
        "spans multiple domains: delivery, mechanism, and clinical translation. "
        "Across the corpus it appears variously as a molecular technique, "
        "a therapeutic vehicle, and a benchmark for off-target measurement."
    )
    body = _flat_hub_body(
        defn,
        f'- [[cgt/paper-a#kc-01]] — <!-- how this paper uses PE. hint: "{spoke_claim}" -->',
    )
    got = check_hub(Path("wiki/concepts/pe.md"), body, _fm(span=1))
    kinds = {v["kind"] for v in got}
    assert "concept_definition_paraphrases_claim" not in kinds


def test_span_missing_from_frontmatter_skips_bridge_check():
    """If concept_span isn't parseable, don't false-flag the bridge check."""
    body = _flat_hub_body(_passing_definition(), "- [[cgt/paper-a#kc-01]]")
    got = check_hub(Path("wiki/concepts/pe.md"), body, {"type": "concept"})
    kinds = {v["kind"] for v in got}
    assert "concept_missing_bridge_section" not in kinds


# ---------- find_concept_contract_violations (page-list level) ----------


def test_only_concept_pages_are_scanned(tmp_path):
    """Non-concept pages are skipped even if they live under wiki/concepts/."""
    concepts_dir = tmp_path / "concepts"
    other_dir = tmp_path / "compbio"
    concepts_dir.mkdir()
    other_dir.mkdir()

    hub = concepts_dir / "pe.md"
    hub.write_text(_flat_hub_body(_thin_definition(), "- [[cgt/paper-a#kc-01]]"))

    # A paper-type page that happens to have a thin section shouldn't flag.
    paper = other_dir / "some-paper.md"
    paper.write_text("## Definition\nx\n")

    pages = [hub, paper]
    bodies = {p: p.read_text() for p in pages}
    fms = {
        hub: {"type": "concept", "concept_span": 1},
        paper: {"type": "paper"},
    }
    got = find_concept_contract_violations(pages, bodies, fms)
    for v in got:
        assert v["page"] == hub
