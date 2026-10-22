"""Claims-first bounded retrieval, with passages checked against fresh PDF extraction."""
from __future__ import annotations

from pathlib import Path

from ..errors import EnvironmentFailure
from ..index.pages_bm25 import TantivySearchBackend
from ..index.pdf_chunks import MAX_PDF_PAGES
from ..pdf.text import extract_pdf_page_texts
from ..search import claim_lookup, pdf_section_search
from .contracts import Evidence


def _normalized(text):
    return " ".join(text.split())


def _passage(stem, query, scope, pages):
    hits = pdf_section_search(stem, query, k=1, scope=scope)
    if any("note" in hit for hit in hits):
        raise EnvironmentFailure(f"cannot retrieve PDF evidence for {stem}; rebuild/check its PDF index")
    if not hits:
        return None
    hit = hits[0]
    start, end = hit.get("page_start"), hit.get("page_end")
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(pages):
        raise EnvironmentFailure(f"PDF evidence has no valid page range: {stem}")
    text = hit["text"]
    if not text.strip() or len(text) > 6000 or _normalized(text) not in _normalized(" ".join(pages[start-1:end])):
        raise EnvironmentFailure(f"retrieved passage does not match current PDF pages: {stem}")
    return text, start, end


def retrieve(question, scope):
    """At most four papers/twelve passages; a bounded answer, never a corpus review.

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
            raise EnvironmentFailure(f"cannot extract answer source: {stem}") from exc
        if not any(page.strip() for page in pages):
            raise EnvironmentFailure(f"PDF has no extractable evidence: {stem}")
        candidates = [(hit["text"], f'[[{stem}#{hit["claim_slug"]}]]', hit["text"])
                      for hit in claims if hit["paper_stem"] == stem][:2]
        candidates.append((question, f"[[{stem}]]", ""))
        seen = set()
        for query, citation, claim in candidates:
            result = _passage(stem, query, scope, pages)
            if result is None or (citation, result[0]) in seen:
                continue
            seen.add((citation, result[0]))
            text, start, end = result
            evidence.append(Evidence(evidence_id=f"e{len(evidence)}", stem=stem,
                                     citation=citation, claim_text=claim, passage=text,
                                     page_start=start, page_end=end))
    return tuple(evidence)
