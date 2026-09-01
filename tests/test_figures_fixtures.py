"""End-to-end figure capture against the bundled benchmark PDFs.

`tests/test_figures.py` covers the detection rules and the PNG encoder against
text and array fixtures — fast, hermetic, and where the judgement calls are
pinned. This file covers what those cannot: that real PDFs parse, that caption
detection survives the bundled venue typesetting styles, and that a render
produces a decodable PNG of the right shape.

The six PDFs under `benchmark-fixtures/pdfs/` are tracked in git and CC-BY-4.0
(see `benchmark-fixtures/LICENSES.md`), so this runs in any clone — unlike
`papers/`, which is gitignored and commonly a symlink into a personal vault.
They were added for ingest benchmarking; nothing else was using them as a test
corpus, and they happen to span exactly the caption styles that matter:

    fonseca  `Figure 1- Title` / `Figure 2 - Title`  (hyphen, ± leading space)
    li       `Fig 1. Title`                          (PLOS)
    muslu    `Fig. 1 Title`                          (BMC, no separator at all)
    zhang    `Fig. 1 Title` + `Table 1 Title`
    chuai    `Fig. 1 Title`                          (BMC)
    assa     `Figure 1. Title` / `Figure 2 Title`    (OUP, mixed separator)

Counts are pinned exactly, so this fails in both directions: a regression that
drops captions, and a widened rule that starts matching body prose. When a
detection improvement legitimately changes a count, update it here deliberately
and say why in the commit.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from researchwiki.pdf import figures as figlib
from researchwiki.pdf.render import render_page

# Resolved from this file, not `paths.benchmark_pdfs_dir()` — that one is
# relative to `Path.cwd()`, and these tests must find the fixtures regardless
# of where pytest was invoked from.
FIXTURES = Path(__file__).resolve().parent.parent / "benchmark-fixtures" / "pdfs"

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="benchmark-fixtures/pdfs not present"
)

# stem -> (figures, tables, a caption line that must be found verbatim-ish)
EXPECTED = {
    "assa-2024-quantifying-allele-specific-crispr-editing-activity":
        (6, 0, "CRISPECTOR2.0 is a tool"),
    "chuai-2018-deepcrispr-optimized-crispr-guide-rna":
        (6, 0, "Implementation details of DeepCRISPR"),
    "fonseca-2026-adjunctive-ibuprofen-in-pre-extensively-drug-resistant":
        (4, 2, "Study flow diagram"),
    "li-2026-schilda-hierarchical-integration-of-llm":
        (1, 1, "The scHilda framework"),
    "muslu-2026-variantmedium-sensitive-and-generalizable-somatic":
        (6, 1, "Overview of VariantMedium workflow"),
    "zhang-2026-mga-a-tool-for-haplotype-mixed":
        (5, 3, "The MGA workflow"),
}


def _pdf(stem: str):
    return FIXTURES / f"{stem}.pdf"


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_caption_counts_per_fixture(stem):
    n_fig, n_tab, _ = EXPECTED[stem]
    refs = figlib.locate_figures(_pdf(stem))
    assert sum(r.kind == "Figure" for r in refs) == n_fig
    assert sum(r.kind == "Table" for r in refs) == n_tab


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_known_caption_is_found(stem):
    _, _, needle = EXPECTED[stem]
    refs = figlib.locate_figures(_pdf(stem))
    assert any(needle in r.caption for r in refs), \
        f"{needle!r} not in {[r.caption[:50] for r in refs]}"


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_every_caption_lands_on_a_real_page(stem):
    pages = len(figlib.page_texts(_pdf(stem)))
    for ref in figlib.locate_figures(_pdf(stem)):
        assert 1 <= ref.page <= pages


def test_fonseca_hyphen_captions_are_detected():
    """The regression this file was written for. fonseca-2026 typesets
    `Figure 1- Study flow diagram.` — the only hyphen-separated style in the
    corpus, and invisible to the first version of the caption regex, which
    reported this 37-page trial paper as having no figures at all."""
    refs = figlib.locate_figures(
        _pdf("fonseca-2026-adjunctive-ibuprofen-in-pre-extensively-drug-resistant")
    )
    figures = [r for r in refs if r.kind == "Figure"]
    assert [r.number for r in figures] == [1, 2, 3, 4]
    assert figures[0].caption.startswith("Figure 1-")


def test_muslu_has_no_separator_at_all():
    """`Fig. 1 Overview of ...` — BMC runs the title straight on. Nothing but
    the title-shape check distinguishes this from a sentence."""
    refs = figlib.locate_figures(
        _pdf("muslu-2026-variantmedium-sensitive-and-generalizable-somatic")
    )
    fig1 = next(r for r in refs if r.label == "Figure 1")
    assert "Overview of VariantMedium" in fig1.caption


def test_resolve_then_render_round_trip(tmp_path):
    """The whole path a caller takes: list, pick, render, read the file."""
    stem = "zhang-2026-mga-a-tool-for-haplotype-mixed"
    refs = figlib.locate_figures(_pdf(stem))

    ref = figlib.resolve(refs, "2")
    assert ref is not None and ref.kind == "Figure" and ref.number == 2

    dest = tmp_path / "fig2.png"
    out = render_page(_pdf(stem), ref.page, dest, dpi=110)

    assert out.path == dest and dest.exists()
    assert out.page == ref.page
    # A 110 DPI render of a letter/A4 page: ~900-1000 x ~1200-1400.
    assert 800 < out.width < 1100
    assert 1100 < out.height < 1500

    payload = dest.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", payload[16:26])
    assert (w, h) == (out.width, out.height)
    assert depth == 8 and ctype in (0, 2)
    assert dest.stat().st_size > 10_000, "a rendered page should not be near-empty"


def test_render_dpi_scales_pixels(tmp_path):
    stem = "li-2026-schilda-hierarchical-integration-of-llm"
    low = render_page(_pdf(stem), 1, tmp_path / "lo.png", dpi=72)
    high = render_page(_pdf(stem), 1, tmp_path / "hi.png", dpi=144)
    assert high.width == pytest.approx(low.width * 2, abs=2)
    assert high.height == pytest.approx(low.height * 2, abs=2)


def test_render_rejects_out_of_range_page(tmp_path):
    stem = "li-2026-schilda-hierarchical-integration-of-llm"
    with pytest.raises(ValueError, match="out of range"):
        render_page(_pdf(stem), 9999, tmp_path / "x.png")


def test_cli_lists_then_renders_into_the_cache(tmp_path, monkeypatch, capsys):
    """CLI surface, with the cache redirected — asserts the default mode
    writes nothing, which is the property the whole design rests on."""
    from researchwiki import paths
    from researchwiki.tasks import figures as cli

    stem = "zhang-2026-mga-a-tool-for-haplotype-mixed"
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "figures_cache_dir", lambda: tmp_path / ".figures-cache")
    monkeypatch.setattr(cli, "resolve_pdf", lambda s: _pdf(s))

    assert cli.main([stem]) == 0
    out = capsys.readouterr().out
    assert "Figure 1" in out and "caption(s)" in out
    assert not (tmp_path / ".figures-cache").exists(), \
        "listing must not render anything"

    assert cli.main([stem, "--figure", "1"]) == 0
    written = list((tmp_path / ".figures-cache" / stem).glob("*.png"))
    assert len(written) == 1
    assert written[0].name.endswith("@110.png")


def test_cli_unknown_figure_exits_1_and_lists_what_exists(tmp_path, monkeypatch, capsys):
    from researchwiki.tasks import figures as cli
    stem = "li-2026-schilda-hierarchical-integration-of-llm"
    monkeypatch.setattr(cli, "figures_cache_dir", lambda: tmp_path / ".figures-cache")
    monkeypatch.setattr(cli, "resolve_pdf", lambda s: _pdf(s))

    assert cli.main([stem, "--figure", "42"]) == 1
    err = capsys.readouterr().err
    assert "no caption matching" in err
    assert "found: " in err
