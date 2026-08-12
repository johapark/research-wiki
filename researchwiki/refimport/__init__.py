"""Import a reference-manager corpus (Zotero / Paperpile / Mendeley / ReadCube).

The manager's own export carries curated DOI, title, authors and year — exactly
the fields `agent ingest` otherwise rediscovers through its most failure-prone
stretch (PDF extract → DOI hunt → S2 lookup → LLM reconcile → `metadata_sanity`),
and the stretch that produces every `unknown-` stem and wrong-but-resolving DOI.
Supplying them turns that stretch into a lookup.

Named `refimport` rather than `import` (a keyword) or `importlib` (shadows the
stdlib on some paths).
"""

from .latex import delatex
from .parse import ExportItem, clean_doi, parse_export, sniff_format

__all__ = ["ExportItem", "clean_doi", "parse_export", "sniff_format", "delatex"]
