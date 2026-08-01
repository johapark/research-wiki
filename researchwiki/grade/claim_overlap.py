"""Claim-overlap compute core.

Given a newly-ingested paper, find existing wiki papers whose claims are
near-paraphrases of the new paper's claims — the cheap, LLM-free half of the
proactive cross-linker. Cosine over cached bi-encoder embeddings; no grading
required (we compare claim *text*). The `tasks/claim_overlap.py` command judges
the survivors and applies cross-links.

Output is collapsed to at most one candidate per existing paper — the
highest-cosine claim pair — because the downstream action is a page-to-page
`[[wikilink]]`, not a per-claim link.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ..log import log


@dataclass
class OverlapCandidate:
    existing_stem: str
    cosine: float
    new_claim: dict          # {section, position, text}
    existing_claim: dict     # {section, position, text}


def _load_claims(conn, stem: str, *, other: bool):
    op = "<>" if other else "="
    rows = conn.execute(
        f"SELECT paper_stem, section, position, text FROM claims "
        f"WHERE paper_stem {op} ? AND is_cross_ref = 0",
        (stem,),
    ).fetchall()
    return [
        {"paper_stem": r["paper_stem"], "section": r["section"],
         "position": r["position"], "text": r["text"]}
        for r in rows
    ]


def find_claim_overlaps(
    new_stem: str,
    *,
    new_claims: list[dict] | None = None,
    sim_threshold: float = 0.83,
    top_papers: int = 10,
    conn=None,
) -> list[OverlapCandidate]:
    """Existing papers whose claims most resemble `new_stem`'s claims.

    `new_claims` (dicts with paper_stem/section/position/text) can be supplied
    directly — used by the ingest hook, where the new page's claims exist in the
    committed markdown but aren't in the DB yet (claim rows are INSERTed only by
    `db rebuild`). When None, the new side is loaded from the DB (post-rebuild
    standalone use). The *existing* side is always the DB.

    Returns up to `top_papers` candidates (one per existing paper, highest-cosine
    pair) with cosine ≥ `sim_threshold`, sorted by cosine descending. Returns []
    when numpy/the bi-encoder is unavailable, either side has no claims, or
    nothing clears the threshold.
    """
    if not _NUMPY:
        log("numpy unavailable — skipping.", tag="claim-overlap")
        return []

    from ..index.claim_embeddings import get_claim_embeddings

    close_conn = False
    if conn is None:
        from ..db.connection import get_connection
        conn = get_connection()
        close_conn = True
    try:
        if new_claims is None:
            new_claims = _load_claims(conn, new_stem, other=False)
        existing_claims = _load_claims(conn, new_stem, other=True)
    finally:
        if close_conn:
            conn.close()

    if not new_claims or not existing_claims:
        return []

    # One embedding call over the union keeps the cache coherent (each call
    # rewrites the cache to its row set); slice back into the two groups.
    combined = new_claims + existing_claims
    embs = get_claim_embeddings(combined)
    if embs is None or embs.size == 0:
        return []
    n = len(new_claims)
    new_embs, existing_embs = embs[:n], embs[n:]

    sims = new_embs @ existing_embs.T  # (n_new, n_existing), cosine (unit rows)

    # Best (new, existing) pair per existing paper.
    best: dict[str, OverlapCandidate] = {}
    for ej, ec in enumerate(existing_claims):
        col = sims[:, ej]
        i = int(col.argmax())
        s = float(col[i])
        if s < sim_threshold:
            continue
        stem = ec["paper_stem"]
        prev = best.get(stem)
        if prev is None or s > prev.cosine:
            best[stem] = OverlapCandidate(
                existing_stem=stem,
                cosine=round(s, 4),
                new_claim={k: new_claims[i][k] for k in ("section", "position", "text")},
                existing_claim={k: ec[k] for k in ("section", "position", "text")},
            )

    ranked = sorted(best.values(), key=lambda c: -c.cosine)
    return ranked[:top_papers]
