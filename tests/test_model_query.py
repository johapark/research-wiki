"""Optional Harrier queries use isolated retrieval and preserve existing defaults."""

from types import SimpleNamespace

from researchwiki.search import model_query
from researchwiki.tasks import search


def test_live_model_query_preserves_metadata_and_selects_the_requested_encoder(monkeypatch):
    configured = []
    monkeypatch.setattr(model_query, "configure_model", lambda model, **kw: configured.append((model, kw)))
    monkeypatch.setattr(model_query, "_page_inputs", lambda:
                        ([{"key": "ai/a", "stem": "a", "category": "ai", "page_type": "paper", "title": "A"}], []))
    called = []

    def retrieve(fixture, model, mode):
        called.append((fixture.query, model, mode, fixture.k))
        return [SimpleNamespace(paper_stem="ai/a", score=.8)]

    monkeypatch.setattr(model_query, "retrieve_pages", retrieve)
    hits = model_query.query_with_model("query", "harrier-270m", mode="semantic", limit=3)
    assert configured == [("harrier-270m", {"max_seq_length": 2048})]
    assert called == [("query", "harrier-270m", "semantic", 3)]
    assert hits[0].key == "ai/a" and hits[0].title == "A"


def test_search_embedding_option_uses_the_model_path_without_changing_default_routing(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(model_query, "query_with_model", lambda *a, **kw: calls.append((a, kw)) or [])
    monkeypatch.setattr(search, "_resolve_mode", lambda *a: (_ for _ in ()).throw(AssertionError("default path used")))
    assert search.main(["query", "--embedding", "harrier-270m", "--mode", "semantic", "--json"]) == 1
    assert calls[0][0] == ("query", "harrier-270m")
    assert capsys.readouterr().out.strip() == "[]"
