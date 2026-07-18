"""Tests for the glossary-suspect denoise on concept-hub candidates.

The `--thesis` gate (concepts.scaffold) is the creation-time backstop; this is
the detection-time filter that keeps ambient acronym/ubiquity terms from
dominating the candidate list and the `status` bridge count. See
docs/concept-vs-glossary.md. Behaviour contract: demote (label +
sort-to-bottom + exclude from bridge tier), never drop.
"""

from __future__ import annotations

from researchwiki.concepts.candidates import (
    _is_glossary_suspect,
    _label_for,
    find_candidates_from_keywords,
)


def _mk_paper(stem: str, category: str, keywords=None) -> dict:
    return {"stem": stem, "category": category, "keywords": keywords or [], "tags": []}


# ---------- the classifier ----------

def test_bare_acronyms_are_suspect():
    for t in ("SNP", "PAM", "CNN", "AUC", "WGS", "GWAS", "QTL", "RNP", "LNP", "DSB"):
        assert _is_glossary_suspect(t, pages=20, corpus_size=372), t


def test_hyphen_and_digit_codes_are_suspect():
    # Cell lines / measurements / assembly codes: caps + digits + hyphens,
    # no lowercase (a lowercase letter, e.g. GRCh38, spares it as a name).
    for t in ("LDL-C", "HEK293T", "T2T-CHM13", "GPN-MSA"):
        assert _is_glossary_suspect(t, pages=15, corpus_size=372), t


def test_mixed_case_method_names_are_not_suspect():
    # Named methods keep their casing — a lowercase letter disqualifies the
    # bare-acronym rule, and their page counts are well under the ceiling.
    for t in ("PrediXcan", "ProteinMPNN", "NanoSeq", "ATAC-seq", "HiFi"):
        assert not _is_glossary_suspect(t, pages=3, corpus_size=372), t


def test_real_concept_phrases_are_not_suspect():
    for t in ("variant effect prediction", "duplex sequencing",
              "LDL cholesterol reduction", "protein language models"):
        assert not _is_glossary_suspect(t, pages=8, corpus_size=372), t


def test_ubiquity_ceiling_catches_common_non_acronym_phrase():
    # A spelled-out phrase in >5% of a large corpus is ambient vocabulary.
    assert _is_glossary_suspect("gene expression", pages=40, corpus_size=372)
    # ...but the same phrase below the ceiling is a legitimate candidate.
    assert not _is_glossary_suspect("gene expression", pages=8, corpus_size=372)


def test_ubiquity_does_not_misfire_on_tiny_corpus():
    # Absolute floor guard: in a 3-paper corpus a shared term is 100% "ubiquitous"
    # by fraction, but must not be demoted on ubiquity grounds.
    assert not _is_glossary_suspect("prime editing", pages=3, corpus_size=3)


def test_no_corpus_size_disables_ubiquity_but_not_acronym():
    assert not _is_glossary_suspect("gene expression", pages=40, corpus_size=None)
    assert _is_glossary_suspect("SNP", pages=40, corpus_size=None)


# ---------- label + integration ----------

def test_label_demotes_suspect_over_bridge():
    # SNP would be span-3/high-page bridge, but the suspect label wins.
    assert _label_for(27, 6, term="SNP", corpus_size=372) == "glossary-suspect"
    # A real phrase at the same span stays a bridge.
    assert _label_for(8, 3, term="variant effect prediction",
                      corpus_size=372) == "concept-ready (bridge)"


def test_label_backward_compatible_without_term():
    # Old call sites (no term/corpus_size) keep the original behaviour.
    assert _label_for(3, 2) == "concept-ready (bridge)"
    assert _label_for(5, 1) == "concept-ready (deep)"
    assert _label_for(3, 1) == "candidate"


def test_suspects_demoted_but_not_dropped_and_sorted_last():
    # A bare-acronym term shared across 3 categories + a real phrase. Both
    # survive detection; the suspect is labelled and sorted after the concept.
    papers = (
        [_mk_paper(f"a{i}", c, keywords=["AUC"]) for i, c in enumerate(("cgt", "compbio", "genomics"))]
        + [_mk_paper(f"b{i}", c, keywords=["off target"]) for i, c in enumerate(("cgt", "compbio", "genomics"))]
    )
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    by_slug = {r["slug"]: r for r in got}
    # Not dropped:
    assert "auc" in by_slug and "off-target" in by_slug
    # Labelled:
    assert by_slug["auc"]["label"] == "glossary-suspect"
    assert by_slug["off-target"]["label"] == "concept-ready (bridge)"
    # Sorted last:
    assert [r["slug"] for r in got].index("auc") > [r["slug"] for r in got].index("off-target")
