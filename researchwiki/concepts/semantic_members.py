"""Semantic recall tier for concept-hub member discovery.

`find_members` matches a concept term against claim text lexically (plus the
LLM-normalized keyword list). That works until the term is a *bridge* — and a
term is a bridge precisely when different fields name the same thing
differently, so lexical matching fails hardest on exactly the candidates worth
the most. Building `wiki/concepts/mixture-model.md` lost three of seven spokes
that way, including both of its cross-domain anchors: "Gaussian Mixture Models",
"three-component normal mixture", and "normal-mixture mean estimation" all name
the concept the hub is about, and none contains the string "mixture model".

This module closes that gap by ranking contribution claims against the embedded
term, and reporting the papers lexical matching missed.

**It proposes; it never decides.** Cosine alone cannot carry membership. On the
calibration case every true member outranked the first false positive — by
0.003 (`xu-2019` at 0.736 vs a "pretraining data mixture" claim at 0.733), and
a floor low enough to admit the true members admits 31 distinct papers for a
term like "ATAC-seq". So the output is a *candidate list* for the author, whose
confirmation path is the existing, already-trusted one: re-run the scaffold with
`--aliases`, and the candidates become members through `find_members` unchanged.

To make that cheap, each candidate carries a suggested alias mined from the
wording that actually matched — the claim text is what reveals the vocabulary.

Cache-only by construction: reads `load_cached_claim_embeddings` and never
`get_claim_embeddings`. The latter rewrites the shared claim cache to whatever
row set it was handed, so calling it with a filtered set (contribution sections
only) silently evicts the rest and makes the next `claim-overlap` run re-embed
them. A cold or thin cache therefore degrades to "no candidates reported" rather
than to a surprise model load or a clobbered cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..log import log
from .term_claims import _CONTRIBUTION_SECTIONS

# Cosine floor for reporting a candidate. Calibrated on the mixture-model case,
# where the three lexically-missed true members scored 0.744-0.774; 0.70 clears
# them with margin while keeping the reviewable-volume ceiling in sight (5-31
# distinct papers per term across the terms measured).
DEFAULT_FLOOR = 0.70

# Max candidates reported. A hub is an entry note, not a category listing, so a
# wall of 31 proposals is worse than the top handful.
DEFAULT_LIMIT = 10

# Minimum fraction of contribution claims the embedding cache must cover before
# the pass is trusted. Below this the ranking is over an arbitrary subset and a
# short candidate list would read as "nothing found" rather than "not checked".
MIN_CACHE_COVERAGE = 0.5

# Words that never make a useful alias qualifier on their own.
_ALIAS_STOPWORDS = frozenset("""
a an the of for with and or in on at to from by as this that these those its it
is are was were be been being using used use uses via than then when where which
each per both all any some new novel our we they he she them their his her
""".split())


@dataclass(frozen=True)
class SemanticCandidate:
    """A paper the semantic pass proposes as a hub member.

    `score` is cosine against the embedded term; `text` is the claim that
    matched, which is both the evidence a reviewer judges and the source of
    `suggested_alias`.
    """
    stem: str
    category: str
    claim_slug: str
    section: str
    text: str
    score: float
    suggested_alias: str | None


def _head_token(term: str) -> str:
    """Last alphanumeric word of `term` — the noun the concept hangs on.

    "mixture model" -> "model" is useless; the *distinguishing* token is
    "mixture". Prefer the longest word instead, which picks the specific one
    over the generic head in the cases that matter ("mixture model",
    "attention mechanism", "chromatin accessibility").
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", term)
    if not words:
        return ""
    return max(words, key=len)


def _suggest_alias(term: str, text: str, known: set[str]) -> str | None:
    """Mine a vocabulary variant for `term` out of the claim that matched.

    Two shapes, both observed in the calibration corpus:

      - a parenthesized acronym — "Gaussian Mixture Models (GMMs)" -> "GMM";
      - a qualifier bigram on the concept's distinguishing token —
        "three-component normal mixture" -> "normal mixture".

    Returns None when nothing better than what the caller already has turns up.
    `known` holds the term and its existing aliases, lowercased.
    """
    head = _head_token(term)
    if not head:
        return None

    # Acronym in parentheses, allowing a trailing plural ("(GMMs)"). The
    # expansion immediately preceding it must itself mention the concept's
    # distinguishing token, or any parenthetical in the claim qualifies:
    # "Hilbert-Schmidt Independence Criterion (HSIC)" is not an alias for
    # "mixture model" merely by sharing a sentence with one.
    for m in re.finditer(r"\(([A-Z][A-Za-z]{1,7}?)s?\)", text):
        acr = m.group(1)
        if len(acr) < 2 or not acr.isupper() or acr.lower() in known:
            continue
        expansion = text[max(0, m.start() - 80):m.start()]
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(head)}", expansion, re.IGNORECASE):
            return acr

    # Qualifier + head bigram.
    for m in re.finditer(rf"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9-]*)\s+{re.escape(head)}",
                         text, re.IGNORECASE):
        qualifier = m.group(1)
        if qualifier.lower() in _ALIAS_STOPWORDS:
            continue
        # A qualifier ending in a digit-hyphen ("three-component") is a count,
        # not vocabulary — the useful variant is the word after it.
        if re.fullmatch(r"[A-Za-z]+-[A-Za-z]+", qualifier) and "-" in qualifier:
            continue
        candidate = f"{qualifier} {head}".lower()
        if candidate not in known and candidate != term.lower():
            return candidate
    return None


def _contribution_claims() -> list[dict]:
    """All graded contribution claims, with their paper's category."""
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT c.paper_stem, c.claim_slug, c.section, c.position, c.text, "
            "       p.category, p.page_type "
            "  FROM claims c JOIN papers p ON p.stem = c.paper_stem "
            " WHERE c.section IN (?, ?, ?) "
            "   AND c.claim_slug IS NOT NULL "
            "   AND c.is_cross_ref = 0 "
            "   AND p.page_type = 'paper'",
            _CONTRIBUTION_SECTIONS,
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def semantic_member_candidates(
    term: str,
    aliases: list[str] | None = None,
    exclude_stems: set[str] | None = None,
    *,
    floor: float = DEFAULT_FLOOR,
    limit: int = DEFAULT_LIMIT,
) -> list[SemanticCandidate]:
    """Papers whose contribution claims are semantically near `term`.

    `exclude_stems` are the members lexical matching already found — the whole
    point is to report what it missed, so passing them keeps the list to the
    delta. Returns at most `limit` candidates above `floor`, best claim per
    paper, ranked by score. Empty on any failure: no numpy, no cached
    embeddings, thin cache coverage, or no hits.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    rows = _contribution_claims()
    if not rows:
        return []

    from ..index import embeddings as semantic
    from ..index.claim_embeddings import load_cached_claim_embeddings

    loaded = load_cached_claim_embeddings(rows)
    if loaded is None:
        log("semantic member pass skipped: claim-embedding cache cold "
            "(warm it with any `claim-overlap` run)", tag="concepts")
        return []
    vecs, row_indices = loaded

    coverage = len(row_indices) / len(rows)
    if coverage < MIN_CACHE_COVERAGE:
        log(f"semantic member pass skipped: claim-embedding cache covers only "
            f"{coverage:.0%} of contribution claims", tag="concepts")
        return []

    queries = [term, *(aliases or [])]
    q = semantic.embed_texts(queries)
    if q is None or len(q) == 0:
        return []
    # Score against the term and every alias, keeping each claim's best — an
    # alias the author already supplied should pull in its own neighborhood.
    sims = (vecs @ np.asarray(q).T).max(axis=1)

    excluded = exclude_stems or set()
    known = {term.lower(), *(a.lower() for a in (aliases or []))}

    best: dict[str, tuple[float, dict]] = {}
    for local_idx, row_idx in enumerate(row_indices):
        score = float(sims[local_idx])
        if score < floor:
            continue
        row = rows[row_idx]
        stem = row["paper_stem"]
        if stem in excluded:
            continue
        if stem not in best or score > best[stem][0]:
            best[stem] = (score, row)

    out = [
        SemanticCandidate(
            stem=stem,
            category=row["category"],
            claim_slug=row["claim_slug"],
            section=row["section"],
            text=row["text"],
            score=score,
            suggested_alias=_suggest_alias(term, row["text"], known),
        )
        for stem, (score, row) in best.items()
    ]
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def suggested_alias_set(
    candidates: list[SemanticCandidate], *, top: int = 5,
) -> list[str]:
    """De-duplicated alias suggestions across `candidates`, best-score first.

    What the author pastes into `--aliases` to convert candidates into members
    through the normal lexical path.

    Drawn from the `top` highest-scoring candidates only. An alias mined from a
    false-positive paper is itself a false alias — "candidate mixture", from a
    claim about retraining cost for pretraining data mixtures, is real English
    from a real claim and still worthless as a concept alias. The per-candidate
    `suggested_alias` stays attached to its paper, where a reviewer rejecting
    the paper rejects the alias with it; this paste-ready list is where tail
    noise would do damage unreviewed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates[:top]:
        if c.suggested_alias and c.suggested_alias.lower() not in seen:
            seen.add(c.suggested_alias.lower())
            out.append(c.suggested_alias)
    return out
