"""On-demand figure capture: caption location, PNG encoding, and the CLI.

Two things here have real judgement in them and are pinned accordingly:

  - **Caption vs. body prose.** A line opening "Table 2 summarizes the various
    datasets..." is not a caption. The discriminator is title shape — a caption
    continues with an uppercase letter, digit, or quote; a sentence continues
    with a lowercase verb.
  - **Channel order.** pypdfium2 renders BGR. Slicing three channels and
    calling them RGB swaps red and blue, which mostly-grayscale pages hide and
    colour figures — the whole point of the feature — expose.

Hermetic: detection runs against text fixtures, encoding against synthetic
arrays. No PDF, no rendering, no network.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from researchwiki.pdf import figures as figlib
from researchwiki.pdf.render import encode_png, _to_rgb


# ---------- caption detection ----------

def test_locates_nature_style_pipe_captions():
    refs = figlib.locate_in_texts([
        "body text\n",
        "Fig. 1 | Construction of the pangenome. A Distribution of depth\n",
        "Fig. 2 | Contents of the pangenome. A Length of pangenome\n",
    ])
    assert [(r.label, r.page) for r in refs] == [("Figure 1", 2), ("Figure 2", 3)]


@pytest.mark.parametrize("line,label", [
    ("Fig. 1 | Construction of the pangenome.", "Figure 1"),      # Nature
    ("Figure 1: Tree construction process.", "Figure 1"),         # most preprints
    ("Fig 1. The scHilda framework, divided in two.", "Figure 1"),  # PLOS
    ("Fig. 1 Overview of VariantMedium workflow.", "Figure 1"),   # BMC, no separator
    ("Figure 1- Study flow diagram.", "Figure 1"),                # hyphen, no space
    ("Figure 2 - Primary and secondary outcomes.", "Figure 2"),   # hyphen, spaced
])
def test_separator_styles_seen_in_the_corpus(line, label):
    """Each of these is a real caption from a corpus paper or benchmark
    fixture. The two hyphen forms are `fonseca-2026`, which the first version
    of this regex missed entirely — it accepted only `|`, `.` and `:`."""
    refs = figlib.locate_in_texts([line + "\n"])
    assert [r.label for r in refs] == [label]


def test_cross_reference_range_is_not_a_caption():
    """`Figure 1-3 show ...` must not read as a hyphen-separated caption. The
    separator has to be followed by whitespace, and a range has a digit."""
    assert figlib.locate_in_texts(["Figure 1-3 show the recall curves.\n"]) == []


def test_locates_colon_and_period_styles():
    """Most non-Nature venues typeset `Figure 1:` or `Figure 1.` — the style
    `sections.py`'s pipe-only CAPTION_START_RE deliberately skips."""
    refs = figlib.locate_in_texts([
        "Figure 1: Tree construction process: RAPTOR clusters chunks\n"
        "Figure 2. Illustration of the tree traversal\n",
    ])
    assert [r.label for r in refs] == ["Figure 1", "Figure 2"]


@pytest.mark.parametrize("line", [
    "Table 2 summarizes the various datasets used in our work.",
    "Table 11 shows the prompt used for summarization.",
    "Table 14 reports Boltz-2's results on the CASP16 blind challenge.",
    "Figure 3 illustrates why this matters.",
])
def test_body_sentence_starting_with_a_label_is_not_a_caption(line):
    """All four are real lines from corpus papers that the first draft of the
    regex accepted."""
    assert figlib.locate_in_texts([line + "\n"]) == []


def test_inline_mention_is_not_a_caption():
    assert figlib.locate_in_texts([
        "as shown in (Fig. 3) the recall improves, see Figure 4 for detail\n"
    ]) == []


def test_extended_data_is_a_separate_series():
    refs = figlib.locate_in_texts([
        "Fig. 1 | Main result. Something\n",
        "Extended Data Fig. 1 | Supplementary detail. Something else\n",
    ])
    assert [(r.label, r.page) for r in refs] == [
        ("Figure 1", 1), ("Extended Data Figure 1", 2),
    ]


def test_tables_and_figures_do_not_collide():
    refs = figlib.locate_in_texts([
        "Figure 1: A thing\nTable 1: Another thing\n",
    ])
    assert {r.label for r in refs} == {"Figure 1", "Table 1"}


def test_first_occurrence_wins():
    refs = figlib.locate_in_texts([
        "Figure 3: Original caption here\n",
        "Figure 3: Reprinted in the appendix\n",
    ])
    assert len(refs) == 1
    assert refs[0].page == 1
    assert "Original" in refs[0].caption


def test_paper_with_no_captions_is_empty_not_an_error():
    assert figlib.locate_in_texts(["Just prose.\n", "More prose.\n"]) == []


# ---------- --figure spec resolution ----------

@pytest.fixture
def refs():
    return figlib.locate_in_texts([
        "Figure 1: First figure\nTable 1: First table\n",
        "Extended Data Fig. 2 | Supplementary figure\n",
    ])


@pytest.mark.parametrize("spec,expected", [
    ("3", None),
    ("1", "Figure 1"),
    ("fig 1", "Figure 1"),
    ("Figure 1", "Figure 1"),
    ("table 1", "Table 1"),
    ("tab 1", "Table 1"),
    ("ed 2", "Extended Data Figure 2"),
    ("extended data fig 2", "Extended Data Figure 2"),
    ("nonsense", None),
])
def test_resolve_spec(refs, spec, expected):
    got = figlib.resolve(refs, spec)
    assert (got.label if got else None) == expected


def test_extended_spec_does_not_match_the_main_series(refs):
    """`ed 1` must not resolve to `Figure 1` — different series, and silently
    showing the wrong figure is worse than saying it isn't there."""
    assert figlib.resolve(refs, "ed 1") is None


# ---------- PNG encoding ----------

def _png_header(payload: bytes) -> tuple[int, int, int]:
    """Return (width, height, colour_type) parsed from IHDR."""
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    length = struct.unpack(">I", payload[8:12])[0]
    assert payload[12:16] == b"IHDR" and length == 13
    w, h, depth, ctype = struct.unpack(">IIBB", payload[16:26])
    assert depth == 8
    return w, h, ctype


def _png_pixels(payload: bytes, width: int, height: int, channels: int) -> np.ndarray:
    """Decode our own filter-0 scanlines back to an array."""
    idat = b""
    pos = 8
    while pos < len(payload):
        length = struct.unpack(">I", payload[pos:pos + 4])[0]
        tag = payload[pos + 4:pos + 8]
        if tag == b"IDAT":
            idat += payload[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels + 1
    assert len(raw) == stride * height
    rows = [raw[i * stride + 1:(i + 1) * stride] for i in range(height)]
    assert all(raw[i * stride] == 0 for i in range(height)), "filter byte must be 0"
    return np.frombuffer(b"".join(rows), np.uint8).reshape(height, width, channels)


def test_encode_png_round_trips_colour():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[:, :, 0] = 200   # distinct per channel so a swap would show
    rgb[:, :, 1] = 100
    rgb[:, :, 2] = 50
    payload, grayscale = encode_png(rgb)

    assert grayscale is False
    w, h, ctype = _png_header(payload)
    assert (w, h, ctype) == (6, 4, 2)
    np.testing.assert_array_equal(_png_pixels(payload, 6, 4, 3), rgb)


def test_encode_png_emits_grayscale_when_channels_match():
    gray = np.tile(np.arange(8, dtype=np.uint8)[None, :, None], (3, 1, 3))
    payload, grayscale = encode_png(gray)

    assert grayscale is True
    w, h, ctype = _png_header(payload)
    assert (w, h, ctype) == (8, 3, 0)
    np.testing.assert_array_equal(
        _png_pixels(payload, 8, 3, 1)[:, :, 0], gray[:, :, 0]
    )


def test_grayscale_encoding_is_smaller():
    gray = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (64, 1, 3))
    as_gray, _ = encode_png(gray)
    coloured = gray.copy()
    coloured[0, 0, 0] = 1  # break channel equality by one pixel
    as_colour, _ = encode_png(coloured)
    assert len(as_gray) < len(as_colour)


def test_encode_png_rejects_wrong_shape():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), np.float32))


# ---------- channel order ----------

class _Bitmap:
    def __init__(self, arr, mode):
        self._arr, self.mode = arr, mode

    def to_numpy(self):
        return self._arr


def test_bgr_is_converted_to_rgb():
    """The silent-corruption case: pypdfium2's default mode is BGR, so a naive
    three-channel slice would return this array unchanged and swap red/blue."""
    bgr = np.zeros((1, 1, 3), np.uint8)
    bgr[0, 0] = [10, 20, 30]          # B=10, G=20, R=30
    out = _to_rgb(_Bitmap(bgr, "BGR"))
    assert list(out[0, 0]) == [30, 20, 10]


def test_bgra_drops_alpha_and_reorders():
    bgra = np.zeros((1, 1, 4), np.uint8)
    bgra[0, 0] = [10, 20, 30, 255]
    assert list(_to_rgb(_Bitmap(bgra, "BGRA"))[0, 0]) == [30, 20, 10]


def test_rgb_is_passed_through():
    rgb = np.zeros((1, 1, 3), np.uint8)
    rgb[0, 0] = [30, 20, 10]
    assert list(_to_rgb(_Bitmap(rgb, "RGB"))[0, 0]) == [30, 20, 10]


def test_unknown_mode_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unsupported"):
        _to_rgb(_Bitmap(np.zeros((1, 1, 3), np.uint8), "CMYK"))


# ---------- CLI ----------

def test_missing_pdf_exits_2(tmp_path, monkeypatch, capsys):
    """2 = environment error, matching `pdf-search`'s contract for the same
    condition — the stem names nothing this machine has."""
    from researchwiki import paths
    from researchwiki.tasks import figures as cli
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    assert cli.main(["no-such-paper-2026-nothing-here"]) == 2
    err = capsys.readouterr().err
    assert "researchwiki figures:" in err
    assert "no-such-paper-2026-nothing-here" in err


def test_conflicting_selectors_exit_1(tmp_path, monkeypatch, capsys):
    from researchwiki.tasks import figures as cli
    monkeypatch.setattr(cli, "resolve_pdf", lambda stem: tmp_path / "x.pdf")
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4\n")
    assert cli.main(["stem", "--figure", "1", "--page", "2"]) == 1
    assert "only one of" in capsys.readouterr().err


@pytest.mark.parametrize("spec,expected", [
    ("3,7,9", [3, 7, 9]),
    (" 4 , 5 ", [4, 5]),
])
def test_parse_pages(spec, expected):
    from researchwiki.tasks.figures import _parse_pages
    assert _parse_pages(spec) == expected


@pytest.mark.parametrize("spec", ["0", "-1", "", "abc"])
def test_parse_pages_rejects_junk(spec):
    from researchwiki.tasks.figures import _parse_pages
    with pytest.raises(ValueError):
        _parse_pages(spec)


# ---------- caption page vs. artwork page ----------
#
# Accepted manuscripts collect every caption onto one page and put the plates
# several pages later (`fonseca-2026`: captions p29-30, artwork p31-37). A
# caption-page render then shows text and no figure, silently.

def test_artwork_candidates_are_nearest_first():
    """Nearest, not densest: plates follow the caption block roughly in order,
    so sorting by object count put fonseca's 963-object last plate ahead of
    the first one."""
    # fonseca-2026's real per-page counts: captions on 29-30, plates after.
    densities = [0] * 28 + [1, 1, 287, 1, 230, 32, 231, 61, 963]
    assert figlib.artwork_candidates(densities, caption_page=29) == \
        [31, 33, 34, 35, 36, 37]


def test_artwork_candidates_ignores_sparse_pages():
    """A rule or a logo is not artwork."""
    densities = [0, 2, 1, 300, 3]
    assert figlib.artwork_candidates(densities, caption_page=1) == [4]


def test_artwork_candidates_empty_when_nothing_follows():
    assert figlib.artwork_candidates([0, 500, 1, 1], caption_page=2) == []


def test_artwork_candidates_respects_window():
    densities = [0] * 5 + [900]
    assert figlib.artwork_candidates(densities, caption_page=1, window=3) == []
    assert figlib.artwork_candidates(densities, caption_page=1, window=10) == [6]
