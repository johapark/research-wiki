"""PDF text and structure extraction — the pypdfium2-backed primitives.

Two focused sub-modules, imported explicitly by callers:

- `text` — page-text extraction, DOI detection, reference-list DOI harvesting.
- `sections` — section-anchor detection from the extracted text (structure).

Nothing is re-exported here; callers write `from ..pdf.text import extract_pdf`
or `from ..pdf.sections import anchor_sections`. Making the split visible in
the import site makes it easy to spot which sub-module a function belongs to.
"""
