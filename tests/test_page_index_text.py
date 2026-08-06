"""What gets embedded for a page, per page type.

`page_index_text` extracted `## Summary` and `## Key Contributions` — section
names that exist on paper pages and on **no other type**. So every synthesis,
idea and concept page was embedded on its title alone: measured 2026-08-06,
median embedded length was paper 2539 chars against synthesis 51, idea 64 and
concept 23 (minimum 15). The one field that did reach them, `tags:`, had been
removed hours earlier on a corpus-wide argument that only held for papers.

These tests pin the type split in both directions: the sections each type really
has, and tags counted only where they carry vocabulary rather than provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.index.pages_semantic import page_index_text
from researchwiki.wiki import Page


def _page(page_type: str, body: str, **fm) -> Page:
    return Page(
        path=Path("wiki") / "compbio" / "x-2024-a-page.md",
        stem="x-2024-a-page",
        category="compbio",
        fm={"title": "A Title", "type": page_type, **fm},
        body=body,
    )


class TestSectionsByType:
    def test_paper_uses_summary_and_key_contributions(self):
        t = page_index_text(_page(
            "paper",
            "## Summary\n\nthe summary prose\n\n## Key Contributions\n\n- a contribution\n"
            "\n## Limitations\n\nnot indexed\n",
        ))
        assert "the summary prose" in t and "a contribution" in t

    def test_concept_uses_definition_and_corpus_survey(self):
        t = page_index_text(_page(
            "concept",
            "## Definition\n\nthe definition prose\n\n"
            "## How it appears across the corpus\n\nthe survey prose\n",
        ))
        assert "the definition prose" in t
        assert "the survey prose" in t

    def test_idea_uses_verdict_and_background(self):
        t = page_index_text(_page(
            "idea",
            "## Verdict\n\nthe verdict prose\n\n## Background\n\nthe background prose\n\n"
            "## Opportunities\n\nthe opportunity prose\n",
        ))
        for expected in ("the verdict prose", "the background prose", "the opportunity prose"):
            assert expected in t

    def test_synthesis_uses_short_answer_and_question(self):
        t = page_index_text(_page(
            "synthesis",
            "## Question\n\nthe question prose\n\n## Short answer\n\nthe answer prose\n",
        ))
        assert "the answer prose" in t and "the question prose" in t

    @pytest.mark.parametrize("page_type,body", [
        ("paper", "## Summary\n\nsubstance here\n"),
        ("commentary", "## Summary\n\nsubstance here\n"),
        ("concept", "## Definition\n\nsubstance here\n"),
        ("idea", "## Verdict\n\nsubstance here\n"),
        ("synthesis", "## Short answer\n\nsubstance here\n"),
    ])
    def test_no_type_is_embedded_on_its_title_alone(self, page_type, body):
        """The regression this file exists for."""
        t = page_index_text(_page(page_type, body))
        assert "substance here" in t
        assert t.strip() != "A Title"


class TestOrderingIsTldrFirst:
    """The bi-encoder truncates at 512 tokens, so what comes first survives."""

    def test_verdict_precedes_opportunities(self):
        t = page_index_text(_page(
            "idea",
            "## Opportunities\n\nOPP\n\n## Verdict\n\nVERDICT\n\n## Background\n\nBG\n",
        ))
        assert t.index("VERDICT") < t.index("BG") < t.index("OPP")

    def test_short_answer_precedes_question(self):
        t = page_index_text(_page(
            "synthesis", "## Question\n\nQ\n\n## Short answer\n\nANSWER\n",
        ))
        assert t.index("ANSWER") < t.index("Q")


class TestBodyFallback:
    def test_a_bespoke_synthesis_still_contributes(self):
        """Synthesis has no mandated H2 contract; two corpus pages use their own."""
        t = page_index_text(_page(
            "synthesis", "## The axis\n\nthe axis prose\n\n## Positions on the axis\n\nmore\n",
        ))
        assert "the axis prose" in t

    def test_references_are_kept_out_of_the_fallback(self):
        t = page_index_text(_page(
            "synthesis",
            "## Priority 1\n\nreal prose\n\n## References\n\n[^a]: [[compbio/x-2024-y]]\n",
        ))
        assert "real prose" in t
        assert "x-2024-y" not in t

    def test_fallback_does_not_fire_when_a_named_section_matched(self):
        t = page_index_text(_page(
            "paper", "## Summary\n\nthe summary\n\n## Limitations\n\nLIMITS\n",
        ))
        assert "LIMITS" not in t


class TestTagsOnlyWhereTheyCarrySignal:
    def test_paper_tags_are_not_embedded(self):
        """On papers the field was provenance: 334 of 391 pages' only tag."""
        t = page_index_text(_page(
            "paper", "## Summary\n\ns\n", tags=["ingested-via-agent", "off-target"],
        ))
        assert "ingested-via-agent" not in t and "off-target" not in t

    @pytest.mark.parametrize("page_type,body", [
        ("concept", "## Definition\n\nd\n"),
        ("idea", "## Verdict\n\nv\n"),
        ("synthesis", "## Short answer\n\na\n"),
    ])
    def test_vocabulary_types_embed_their_tags(self, page_type, body):
        """`keywords:` is exempt for these three dirs, so tags are all they have."""
        t = page_index_text(_page(page_type, body, tags=["dna-foundation-model", "pangenome"]))
        assert "dna-foundation-model" in t and "pangenome" in t

    def test_framework_tags_are_stripped_even_there(self):
        t = page_index_text(_page(
            "concept", "## Definition\n\nd\n", tags=["concept", "migrated", "pangenome"],
        ))
        assert "pangenome" in t
        assert "migrated" not in t
        # `concept` restates `type:` and would just dilute a short vector.
        assert "\nconcept" not in t

    def test_inline_yaml_string_tags_are_parsed(self):
        t = page_index_text(_page(
            "idea", "## Verdict\n\nv\n", tags="[pangenome, off-target]",
        ))
        assert "pangenome" in t and "off-target" in t

    def test_missing_tags_field_is_harmless(self):
        assert "substance" in page_index_text(_page("concept", "## Definition\n\nsubstance\n"))
