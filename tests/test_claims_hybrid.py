from __future__ import annotations

import numpy as np

from researchwiki.search import claims_hybrid


def _hit(stem, position, score=1.0):
    return {
        "claim_slug": f"kc-{stem}-{position}",
        "paper_stem": stem,
        "section": "key_contributions",
        "position": position,
        "text": stem,
        "semantic_score": 0.8,
        "bm25_top1": 10.0,
        "pdf_anchor_chunk_id": 1,
        "graded": True,
        "match_score": score,
    }


def test_semantic_claim_lookup_uses_query_similarity(monkeypatch):
    rows = [_hit("a", 0), _hit("b", 0)]
    monkeypatch.setattr(claims_hybrid, "_rows", lambda **kw: rows)
    monkeypatch.setattr(
        claims_hybrid.claim_embeddings, "get_claim_embeddings",
        lambda values: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        claims_hybrid, "embed_texts",
        lambda values: np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    hits = claims_hybrid.semantic_claim_lookup("query", k=2)
    assert [hit["paper_stem"] for hit in hits] == ["b", "a"]
    assert hits[0]["query_semantic_score"] == 1.0


def test_hybrid_claim_lookup_fuses_both_rankings(monkeypatch):
    lexical = [_hit("a", 0), _hit("b", 0)]
    semantic = [_hit("b", 0), _hit("c", 0)]
    for rank, hit in enumerate(semantic):
        hit["query_semantic_score"] = 0.9 - rank * 0.1
    monkeypatch.setattr(claims_hybrid, "claim_lookup", lambda *a, **kw: lexical)
    monkeypatch.setattr(
        claims_hybrid, "semantic_claim_lookup", lambda *a, **kw: semantic,
    )
    hits = claims_hybrid.hybrid_claim_lookup("query", k=3)
    assert hits[0]["paper_stem"] == "b"
    assert hits[0]["lexical_rank"] == 2
    assert hits[0]["semantic_rank"] == 1


def test_hybrid_claim_lookup_falls_back_to_lexical(monkeypatch):
    lexical = [_hit("a", 0)]
    monkeypatch.setattr(claims_hybrid, "claim_lookup", lambda *a, **kw: lexical)
    monkeypatch.setattr(claims_hybrid, "semantic_claim_lookup", lambda *a, **kw: [])
    assert claims_hybrid.hybrid_claim_lookup("query") == lexical


def test_default_stays_lexical_until_hybrid_beats_the_local_baseline(monkeypatch):
    lexical = [_hit("a", 0)]
    monkeypatch.setattr(claims_hybrid, "claim_lookup", lambda *a, **kw: lexical)
    monkeypatch.setattr(
        claims_hybrid, "hybrid_claim_lookup",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("hybrid called")),
    )
    assert claims_hybrid.claim_query("query") == lexical
