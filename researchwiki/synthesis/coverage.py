"""Scoped advisory recall; excluded papers never enter a strict selection's report."""
import re

from ..index.pages_bm25 import TantivySearchBackend
from ..search import claim_lookup


def coverage(text, question, scope, top_n=20):
    cited = set(re.findall(r'\[\[(?:[^\]/]+/)?([a-z0-9-]+)(?:#[^\]]+)?\]\]', text))
    # Retrieve selected claims and pages before applying top-K; no global
    # candidate list, semantic download, or out-of-scope titles/snippets.
    claims = claim_lookup(question, k=max(top_n, len(scope.stems)), scope=scope)
    by_stem = {hit['paper_stem']: hit for hit in reversed(claims)}
    pages = TantivySearchBackend().more_like_text(question, limit=max(top_n, len(scope.stems)),
                                                 page_type='paper', scope=scope)
    candidates = {hit.stem: {'stem': hit.stem, 'key': hit.key, 'score': hit.score,
                             'sources': ['page-bm25']} for hit in pages if hit.stem not in cited}
    for stem, hit in by_stem.items():
        if stem in cited:
            continue
        item = candidates.setdefault(stem, {'stem': stem, 'key': stem, 'score': 0, 'sources': []})
        item['sources'].append('claim-bm25')
        item.update(claim_slug=hit['claim_slug'], claim_text=hit['text'], claim_section=hit['section'])
    return sorted(candidates.values(), key=lambda h: ('claim-bm25' not in h['sources'], -h['score'], h['stem']))[:top_n]
