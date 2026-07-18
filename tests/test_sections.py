"""Tests for caption-block extraction terminator behavior.

Covers the accelerated-preview case where Methods / Tables / Supplementary
sits between two figure-caption blocks. Without a mid-block terminator the
last main caption absorbs prose up to the per-caption cap.

Also covers the abstract extractor — explicit-header path, headerless
fallback (largest paragraph before introduction), and the empty case.
"""

from researchwiki.pdf.sections import (
    anchor_sections,
    extract_abstract,
    extract_caption_blocks,
)


_PREVIEW_BODY = """\
Fig. 1 | Workflow overview. The system takes inputs and produces outputs.
Subsequent panels show benchmarks across tasks.

Fig. 2 | Benchmark results. Performance is reported as accuracy.

488 Tables

Table 1 | Performance comparison of methods. Bold entries indicate the best.
Model Method Score
A     X      0.9
B     Y      0.8

489 Methods
490 Detailed implementation
491 We use a neural network with two layers and apply standard preprocessing.

Extended Data Fig. 1 | Ablation results.
"""


def test_table_caption_does_not_bleed_into_methods():
    main, ed = extract_caption_blocks(_PREVIEW_BODY)
    blocks = [b for b in main.split("\n\n") if b.strip()]
    assert len(blocks) == 3  # Fig 1, Fig 2, Table 1
    table_block = blocks[2]
    assert table_block.startswith("Table 1 |")
    # Methods text must not be included — terminator clips before "489 Methods".
    assert "Methods" not in table_block
    assert "neural network" not in table_block


def test_figure_caption_does_not_bleed_into_tables():
    main, _ = extract_caption_blocks(_PREVIEW_BODY)
    blocks = [b for b in main.split("\n\n") if b.strip()]
    fig2_block = blocks[1]
    assert fig2_block.startswith("Fig. 2 |")
    # The "488 Tables" header sits between Fig 2 and Table 1; with the
    # terminator, Fig 2's block ends before that header.
    assert "Tables" not in fig2_block


def test_terminator_tolerates_line_number_prefix():
    """Section headers in line-numbered preview PDFs carry a leading line
    number — `489 Methods` should still be recognized as a terminator."""
    body = (
        "Fig. 9 | Final summary panel. Showing all results.\n\n"
        "489 Methods\n"
        "490 The method is described here.\n"
    )
    main, _ = extract_caption_blocks(body)
    assert "method is described" not in main


def test_no_terminator_falls_back_to_per_caption_cap():
    """When no terminator is between two captions, the block is bounded by
    the next caption start as before."""
    body = (
        "Fig. 1 | First caption. Some descriptive text.\n\n"
        "Fig. 2 | Second caption. More descriptive text.\n"
    )
    main, _ = extract_caption_blocks(body)
    blocks = [b for b in main.split("\n\n") if b.strip()]
    assert len(blocks) == 2
    assert blocks[0].startswith("Fig. 1 |")
    assert blocks[1].startswith("Fig. 2 |")


def test_extended_data_captures_separately():
    main, ed = extract_caption_blocks(_PREVIEW_BODY)
    assert "Extended Data Fig. 1" in ed
    assert "Extended Data" not in main


# ── Abstract extractor ────────────────────────────────────────────────


def test_abstract_explicit_header_path():
    body = (
        "Title\n\nAuthors\n\n"
        "Abstract\n\n"
        "We introduce Folddisco, a structural motif search tool. "
        "It is 20× faster than pyScoMotif.\n\n"
        "Introduction\n\nIn recent years, motif search has...\n"
    )
    abs_text = extract_abstract(body)
    assert abs_text.startswith("We introduce Folddisco")
    assert "Folddisco" in abs_text
    # Header word must NOT leak into the body.
    assert not abs_text.lower().startswith("abstract")


def test_abstract_headerless_fallback_picks_largest_pre_intro_paragraph():
    """Nature-family accelerated previews skip the Abstract header; the
    abstract paragraph sits between author block and Introduction. Take
    the largest paragraph in the pre-introduction region."""
    body = (
        "An AI system to help scientists\n\n"
        "Author A, Author B, Author C\n\n"  # short — author block
        "Affiliations: Lab X, Lab Y\n\n"     # short — affiliations
        + "The cycle of scientific discovery is frequently bottlenecked. " * 12
        + "\n\n"
        "Introduction\n\nFollowing prior work...\n"
    )
    abs_text = extract_abstract(body)
    assert "scientific discovery" in abs_text
    # Author block and affiliations are too short to qualify.
    assert "Affiliations" not in abs_text
    assert "Author A" not in abs_text


def test_abstract_explicit_header_through_anchor_sections():
    """anchor_sections strips the leading 'Abstract' header from the slice
    so downstream sentence-split sees only body."""
    body = (
        "Title\n\nAbstract\n\n"
        "First sentence. Second sentence. Third sentence.\n\n"
        "Introduction\n\nMain body...\n"
    )
    sections = anchor_sections(body)
    assert "abstract" in sections
    assert sections["abstract"].startswith("First sentence")
    assert "Abstract" not in sections["abstract"]


def test_abstract_line_numbered_header_prefix():
    """Line-numbered PDFs prefix every line with `\\d+ ` — header must
    still match."""
    body = (
        "1 Title\n2 Authors\n3 Abstract\n4 First sentence of the abstract here. "
        "5 Second sentence follows nicely.\n\n"
        "Introduction\n\nBody...\n"
    )
    sections = anchor_sections(body)
    assert "abstract" in sections
    assert "First sentence" in sections["abstract"]


def test_abstract_returns_empty_when_no_candidate():
    """No introduction header AND no abstract header → empty."""
    body = "Just a single short blurb of text under 50 words."
    assert extract_abstract(body) == ""


def test_abstract_returns_empty_when_pre_intro_paragraphs_too_short():
    """Title + author block but no abstract paragraph → fallback rejects
    paragraphs under the 50-word floor and returns empty."""
    body = (
        "Title\n\nA, B, C\n\n"  # too short
        "Introduction\n\nMain body...\n"
    )
    assert extract_abstract(body) == ""
