"""Rasterize a PDF page to PNG — pypdfium2 in, bytes out, no new dependency.

Why rendering rather than extracting image objects: a PDF image object is a
*placed raster*. Vector art — which is what every matplotlib / R / Illustrator
plot in a typeset paper is — is a path object and is never returned by an
image-object walk, in any library. Measured over 12 random corpus papers: 498
image objects against 73,294 path objects. `buenrostro-2015` p. 5 is the worked
case: Figure 21.29.2 is a vector line plot (A), two raster gel strips (B), and a
vector density plot (C), so object extraction yields two disembodied gel strips
and nothing else. Rendering the page captures all three plus the caption.

PNG encoding is done here rather than via Pillow because `PdfBitmap.to_pil()`
is the only thing that would have needed it, and PNG is a simple enough
container that ~30 lines of `zlib` + `struct` avoids a dependency entirely:
signature, IHDR, one zlib-compressed IDAT of filter-prefixed scanlines, IEND.

Rendering is local compute and costs no tokens (~0.03 s/page). The cost that
matters is context, and it is paid by whoever `Read`s the file — which is why
the caller reports pixel dimensions and why `tasks/figures.py` renders one page
per invocation by default.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2

# 110 DPI renders a US-letter page to 935x1210 and stays legible down to axis
# tick labels (checked by eye against buenrostro-2015 p. 5). 150 DPI is
# 1275x1651 — ~1.9x the pixel area, hence ~1.9x the context when a model reads
# it — and is worth passing only for a small inset.
DEFAULT_DPI = 110
PDF_UNITS_PER_INCH = 72.0

# PNG colour types (IHDR byte 9).
_PNG_GRAY = 0
_PNG_RGB = 2


@dataclass(frozen=True)
class RenderedPage:
    path: Path
    page: int          # 1-based, as printed
    width: int         # pixels
    height: int
    dpi: int
    grayscale: bool


def _to_rgb(bitmap) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB array from a pypdfium2 bitmap.

    pypdfium2 renders **BGR** by default (`bitmap.mode == "BGR"`), so slicing
    the first three channels and treating them as RGB silently swaps red and
    blue. Mostly-grayscale pages hide it — the bug shows up on exactly the
    colour figures this module exists to capture. Branch on the declared mode
    rather than assuming, so a pypdfium2 that changes its default is handled
    rather than mis-decoded.
    """
    arr = bitmap.to_numpy()
    mode = bitmap.mode
    if mode in ("BGRA", "BGR"):
        return np.ascontiguousarray(arr[:, :, 2::-1])
    if mode in ("RGBA", "RGB"):
        return np.ascontiguousarray(arr[:, :, :3])
    if mode == "L":
        return np.repeat(arr[:, :, None] if arr.ndim == 2 else arr, 3, axis=2)
    raise ValueError(f"unsupported pypdfium2 bitmap mode: {mode!r}")


def encode_png(rgb: np.ndarray, *, level: int = 9) -> tuple[bytes, bool]:
    """Encode an (H, W, 3) uint8 RGB array as PNG. Returns (bytes, grayscale).

    Emits colour type 0 (grayscale) when all three channels are identical,
    which most paper pages are — about a third off the file for free, and a
    third off the pixels a reader has to decode.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")

    grayscale = bool(
        np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
    )
    planes = rgb[:, :, 0] if grayscale else rgb
    colour_type = _PNG_GRAY if grayscale else _PNG_RGB
    height, width = rgb.shape[:2]
    per_pixel = 1 if grayscale else 3

    # Each scanline is prefixed with its filter byte; 0 = None. Filtering
    # would compress better but costs a per-row pass in Python, and zlib
    # already gets these pages down to ~215 KB at 110 DPI.
    rows = planes.reshape(height, width * per_pixel)
    raw = np.hstack([np.zeros((height, 1), np.uint8), rows]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level))
        + chunk(b"IEND", b"")
    )
    return png, grayscale


def render_page(
    pdf_path: Path | str,
    page: int,
    dest: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
) -> RenderedPage:
    """Render one 1-based `page` of `pdf_path` to `dest` as PNG.

    Raises FileNotFoundError if the PDF is missing, ValueError if `page` is
    out of range. The caller owns the destination directory's lifetime — this
    writes a cache artifact, and re-rendering costs ~0.03 s, so deleting it is
    always safe.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pypdfium2.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page <= len(doc):
            raise ValueError(
                f"page {page} out of range for {pdf_path.name} (1-{len(doc)})"
            )
        pdf_page = doc[page - 1]
        try:
            bitmap = pdf_page.render(scale=dpi / PDF_UNITS_PER_INCH)
            rgb = _to_rgb(bitmap)
        finally:
            pdf_page.close()
    finally:
        doc.close()

    payload, grayscale = encode_png(rgb)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return RenderedPage(
        path=dest,
        page=page,
        width=rgb.shape[1],
        height=rgb.shape[0],
        dpi=dpi,
        grayscale=grayscale,
    )
