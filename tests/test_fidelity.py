"""Pure-logic tests for the synthesis fidelity grader.

The end-to-end path (`grade_synthesis`) needs PDFs + indices; these cover the
deterministic helpers that decide verdicts: date exclusion from numeric drift,
composite (cross-paper comparison) detection, and citation→stem resolution.
"""

from researchwiki.grade.fidelity.synthesis import (
    _DATE_RE,
    _is_composite,
    _strip_for_numerics,
    _wikilink_to_stem,
    _footnote_targets,
    NUMERIC_TOKEN_RE,
)


# ---------- citation-slug year exclusion ----------

def test_footnote_ref_year_not_counted_as_number():
    # A `[^slug-YYYY]` footnote ref carries the cited paper's year; it must not
    # be scored as a claim quantity (the paper rarely restates its own year).
    text = "the population-scale successor[^lazzarotto-2025] to prior assays."
    tokens = NUMERIC_TOKEN_RE.findall(_strip_for_numerics(text))
    assert tokens == []


def test_slug_year_gone_but_prose_digit_in_name_survives():
    # Real-world shape: a method name with a trailing digit ("GUIDE-seq-2") is a
    # genuine prose token and stays; only the citation's slug year drops out.
    text = "GUIDE-seq-2[^lazzarotto-2025] is the successor."
    tokens = NUMERIC_TOKEN_RE.findall(_strip_for_numerics(text))
    assert not any("2025" in t for t in tokens)
    assert any(t.startswith("2") for t in tokens)  # the "2" in GUIDE-seq-2


def test_wikilink_slug_year_not_counted_as_number():
    text = "See [[compbio/huang-2023-personal-transcriptome-variation-is-poorly]]."
    tokens = NUMERIC_TOKEN_RE.findall(_strip_for_numerics(text))
    assert tokens == []


def test_claim_anchor_slug_digits_not_counted():
    # `[[stem#kc-2024]]`-style anchors also carry slug digits.
    text = "as shown[[cgt/foo-2024-bar#kc-9f3a2b1c]] in the assay."
    assert NUMERIC_TOKEN_RE.findall(_strip_for_numerics(text)) == []


def test_real_quantity_survives_citation_stripping():
    # Prose numbers next to citations stay checkable.
    text = "reaches 0.85 Pearson[^avsec-2021] over 196,608 bp"
    tokens = NUMERIC_TOKEN_RE.findall(_strip_for_numerics(text))
    assert any(t.startswith("0.85") for t in tokens)
    assert any(t.startswith("196") for t in tokens)


def test_iso_date_still_stripped_by_helper():
    assert NUMERIC_TOKEN_RE.findall(_strip_for_numerics("Update (2026-06-10): grew.")) == []


# ---------- date exclusion ----------

def test_iso_date_stripped_before_numeric_extraction():
    text = "Update (2026-06-10): the atlas grew."
    tokens = NUMERIC_TOKEN_RE.findall(_DATE_RE.sub(" ", text))
    assert tokens == []


def test_year_month_date_stripped():
    assert NUMERIC_TOKEN_RE.findall(_DATE_RE.sub(" ", "filed 2026-06 in review")) == []


def test_real_quantity_survives_date_stripping():
    # A bare quantity is not a date and must remain checkable. (The token
    # carries a trailing hyphen from "2048-dimensional"; check_numerics
    # matches on the digit-only form.)
    tokens = NUMERIC_TOKEN_RE.findall(_DATE_RE.sub(" ", "a 2048-dimensional embedding"))
    assert any(t.startswith("2048") for t in tokens)


def test_decimal_quantity_survives():
    tokens = NUMERIC_TOKEN_RE.findall(_DATE_RE.sub(" ", "a 95.6-million-cell atlas"))
    assert any(t.startswith("95.6") for t in tokens)


# ---------- composite detection ----------

def test_comparison_with_two_papers_is_composite():
    assert _is_composite("Method A is faster than method B", n_cited=2)


def test_comparison_with_one_paper_is_not_composite():
    # A single cited paper can't host a cross-paper comparison.
    assert not _is_composite("A is faster than the baseline", n_cited=1)


def test_non_comparative_two_paper_claim_is_not_composite():
    assert not _is_composite("Both papers use a transformer encoder", n_cited=2)


def test_outperforms_cue_detected():
    assert _is_composite("scGPT outperforms the linear baseline", n_cited=2)


# ---------- wikilink → stem ----------

def test_wikilink_category_path_stripped():
    assert _wikilink_to_stem("compbio/brixi-2026-genome-modelling") == "brixi-2026-genome-modelling"


def test_wikilink_bare_stem():
    assert _wikilink_to_stem("liao-2023-a-draft-human-pangenome") == "liao-2023-a-draft-human-pangenome"


def test_wikilink_alias_dropped():
    assert _wikilink_to_stem("compbio/brixi-2026-evo-2|Evo 2") == "brixi-2026-evo-2"


# ---------- footnote target resolution ----------

def test_footnote_definition_maps_to_wikilink():
    page = (
        "Some claim about scaling.[^evo]\n\n"
        "[^evo]: [[compbio/brixi-2026-evo-2]]\n"
    )
    targets = _footnote_targets(page)
    assert targets == {"evo": ["compbio/brixi-2026-evo-2"]}


def test_footnote_without_wikilink_is_not_mapped():
    page = "[^note]: just prose, no link\n"
    assert _footnote_targets(page) == {}


# ---------- llm cache_prefix builder (Phase 1) ----------

def test_anthropic_user_content_cache_prefix_splits_into_two_blocks():
    from researchwiki.agents.llm import _anthropic_user_content
    blocks = _anthropic_user_content("SUFFIX", cache_prefix="PREFIX", cache_prompt=False)
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["text"] == "PREFIX" and blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "SUFFIX" and "cache_control" not in blocks[1]


def test_anthropic_user_content_legacy_cache_prompt_single_block():
    from researchwiki.agents.llm import _anthropic_user_content
    blocks = _anthropic_user_content("P", cache_prefix=None, cache_prompt=True)
    assert len(blocks) == 1 and blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_user_content_bare_string_when_no_caching():
    from researchwiki.agents.llm import _anthropic_user_content
    assert _anthropic_user_content("P", cache_prefix=None, cache_prompt=False) == "P"
