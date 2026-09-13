from __future__ import annotations

from researchwiki.search import rerank


class _Hit:
    def __init__(self, key):
        self.key = key


def test_reranker_accepts_only_candidate_ids_and_appends_omissions(monkeypatch):
    hits = [_Hit("ai/a"), _Hit("ai/b"), _Hit("ai/c")]
    response = type("R", (), {
        "text": '{"order": ["ai/c", "invented/x", "ai/a"]}',
        "model": "test", "input_tokens": 10, "output_tokens": 4,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
    })()
    monkeypatch.setattr(rerank, "_prompt", lambda query, values: "prompt")
    monkeypatch.setattr("researchwiki.agents.llm.call", lambda **kw: response)
    result = rerank.rerank_hits("query", hits)
    assert result.applied is True
    assert [hit.key for hit in result.hits] == ["ai/c", "ai/a", "ai/b"]
    assert result.usage["input_tokens"] == 10


def test_reranker_falls_back_on_invalid_output(monkeypatch):
    hits = [_Hit("ai/a"), _Hit("ai/b")]
    response = type("R", (), {
        "text": "not json", "model": "test", "input_tokens": 1,
        "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0,
    })()
    monkeypatch.setattr(rerank, "_prompt", lambda query, values: "prompt")
    monkeypatch.setattr("researchwiki.agents.llm.call", lambda **kw: response)
    result = rerank.rerank_hits("query", hits)
    assert result.applied is False
    assert result.hits == hits


def test_reranker_caps_candidates_at_twelve(monkeypatch):
    hits = [_Hit(f"ai/{i}") for i in range(20)]
    monkeypatch.setattr(rerank, "_prompt", lambda query, values: (_ for _ in ()).throw(RuntimeError()))
    result = rerank.rerank_hits("query", hits)
    assert len(result.hits) == 12
