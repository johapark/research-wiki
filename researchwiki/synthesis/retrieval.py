"""Claims-first bounded retrieval, with passages checked against fresh PDF extraction."""
from __future__ import annotations

from pathlib import Path
import re

from ..errors import EnvironmentFailure
from ..index.pages_bm25 import TantivySearchBackend
from ..index.pdf_chunks import MAX_PDF_PAGES
from ..pdf.text import extract_pdf_page_texts
from ..search import claim_lookup, pdf_section_search
from .evidence_records import Evidence


def _normalized(text):
    return " ".join(text.split())


def _overview(pages):
    """Include a bounded verbatim abstract when the PDF exposes its heading.

    Keyword ranking alone can select narrow methods passages for a question
    about the overall task. Keep source context alongside those specific hits.
    """
    for i, page in enumerate(pages[:10], 1):
        heading = re.search(r'(?im)^\s*abstract\s*[:\r\n]', page)
        if heading is None:
            continue
        text = page[heading.end():]
        end = re.search(r'(?im)^\s*(?:keywords|background|introduction|author summary)\s*[:\r\n]', text)
        if end:
            text = text[:end.start()]
        text = _normalized(text[:3000])
        if text:
            return text, i, i
    return None


def _passages(stem, query, scope, pages, k):
    hits = pdf_section_search(stem, query, k=k, scope=scope)
    if any("note" in hit for hit in hits):
        raise EnvironmentFailure(f"cannot retrieve PDF evidence for {stem}; rebuild/check its PDF index")
    passages = []
    for hit in hits[:k]:
        start, end = hit.get("page_start"), hit.get("page_end")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(pages):
            raise EnvironmentFailure(f"PDF evidence has no valid page range: {stem}")
        text = hit["text"]
        if not text.strip() or len(text) > 6000 or _normalized(text) not in _normalized(" ".join(pages[start-1:end])):
            raise EnvironmentFailure(f"retrieved passage does not match current PDF pages: {stem}")
        passages.append((text, start, end))
    return passages


def retrieve(question, scope):
    """At most four papers/twelve passages per card retrieval, never a corpus review.

    Claims nominate papers first. Wiki lexical hits extend that selection. When
    those are thin, inspect selected PDFs in stem order up to the same bound.
    No retrieved summary or cached support text is treated as PDF evidence.
    """
    if not scope.stems:
        return ()
    claims = claim_lookup(question, k=8, scope=scope)
    stems = list(dict.fromkeys(hit["paper_stem"] for hit in claims))[:4]
    if len(stems) < 4:
        hits = TantivySearchBackend().more_like_text(question, limit=4, page_type="paper", scope=scope)
        stems = list(dict.fromkeys([*stems, *(hit.stem for hit in hits)]))[:4]
    stems = list(dict.fromkeys([*stems, *sorted(scope.stems)]))[:4]
    evidence = []
    for stem in stems:
        if not scope.allows(stem):
            raise ValueError("retriever returned an out-of-scope paper")
        try:
            pages = extract_pdf_page_texts(Path("papers") / f"{stem}.pdf", max_pages=MAX_PDF_PAGES)
        except Exception as exc:
            raise EnvironmentFailure(f"cannot extract synthesis source: {stem}") from exc
        if not any(page.strip() for page in pages):
            raise EnvironmentFailure(f"PDF has no extractable evidence: {stem}")
        overview = _overview(pages)
        candidates = [(hit["text"], f'[[{stem}#{hit["claim_slug"]}]]', hit["text"])
                      for hit in claims if hit["paper_stem"] == stem][:1 if overview else 2]
        # Keep the existing three-passages-per-paper limit. With no graded
        # claims, one best chunk often misses a second part of a question;
        # use the remaining slots for independently validated question hits.
        candidates = [(q, citation, claim, 1) for q, citation, claim in candidates]
        candidates.append((question, f"[[{stem}]]", "", 3 - len(candidates) - bool(overview)))
        seen = set()
        for query, citation, claim, k in candidates:
            for text, start, end in _passages(stem, query, scope, pages, k):
                if (citation, text) in seen:
                    continue
                seen.add((citation, text))
                evidence.append(Evidence(evidence_id=f"e{len(evidence)}", stem=stem,
                                         citation=citation, claim_text=claim, passage=text,
                                         page_start=start, page_end=end))
        if overview and (f"[[{stem}]]", overview[0]) not in seen:
            text, start, end = overview
            evidence.append(Evidence(evidence_id=f"e{len(evidence)}", stem=stem,
                                     citation=f"[[{stem}]]", claim_text="", passage=text,
                                     page_start=start, page_end=end))
    return tuple(evidence)
