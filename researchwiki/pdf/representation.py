"""Page-aware extraction that preserves the existing joined-text transform."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pypdfium2

from .repair import repair_text
from .text import PAGE_SEPARATOR, _extract_all_pages, _slash_prefix_metadata, page_offsets


@dataclass(frozen=True)
class PDFRepresentation:
    text: str
    metadata: dict
    record: dict


def extract_representation(path: Path, max_pages: int = 80) -> PDFRepresentation:
    """Read once, retaining raw pages and the exact legacy repaired join.

    Repaired page offsets are asserted only when independent per-page repair
    agrees with the legacy joined transform. Cross-page repair remains visible
    without fabricating a mapping or changing the text supplied to ingestion.
    """
    if type(max_pages) is not int or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        total = len(pdf)
        metadata = _slash_prefix_metadata(pdf.get_metadata_dict() or {})
        raw = _extract_all_pages(pdf, max_pages)
    finally:
        pdf.close()
    repaired = [repair_text(text) for text in raw]
    text = repair_text(PAGE_SEPARATOR.join(raw))
    mapped = PAGE_SEPARATOR.join(repaired) == text
    offsets = page_offsets(repaired) if mapped else [None] * len(raw)
    return PDFRepresentation(text, metadata, {
        "schema_version": 1,
        "transform": "pdfium-joined-repair-v1",
        "extractor": {"name": "pypdfium2", "version": version("pypdfium2")},
        "page_separator": PAGE_SEPARATOR,
        "page_count": total,
        "page_limit": max_pages,
        "pages_read": len(raw),
        "unread_pages": list(range(len(raw) + 1, total + 1)),
        "offset_unit": "unicode_codepoint",
        "page_mapping": "exact" if mapped else "unavailable_cross_page_repair",
        "text": text,
        "metadata": metadata,
        "pages": [{"page": i + 1, "raw_text": before, "repaired_text": after,
                   "start": start, "end": None if start is None else start + len(after)}
                  for i, (before, after, start) in enumerate(zip(raw, repaired, offsets))],
    })
