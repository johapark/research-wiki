"""Harrier instructions, corpus cache freshness, and honest failure reporting."""

from types import SimpleNamespace

import numpy as np
import pytest

from researchwiki.benchmark import retrieval_embeddings as encoders
from researchwiki.benchmark import retrieval
from researchwiki.benchmark.fixture import ExpectedClaim, RetrievalFixture


@pytest.fixture(autouse=True)
def isolated_models(monkeypatch, tmp_path):
    monkeypatch.setattr(encoders, "_OPTIONS", {})
    monkeypatch.setattr(encoders, "_MODEL_CACHE", {})
    monkeypatch.setattr(encoders, "_LOAD_MS", {})
    monkeypatch.setattr(encoders, "_CACHE_EVENTS", [])
    monkeypatch.setattr(encoders, "wiki_root", lambda: tmp_path)


@pytest.mark.parametrize("model", ["harrier-270m", "harrier-0.6b"])
def test_harrier_query_instruction_is_applied_once_only_to_queries(monkeypatch, model):
    calls = []

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 2))

    monkeypatch.setattr(encoders, "_load_model", lambda *a, **kw: Encoder())
    encoders._embed_with(["question"], model, role="query")
    encoders._embed_with(["passage"], model, role="document")
    encoders._embed_with(["question"], model, role="query", prefix="custom: ")
    assert calls[0][1]["prompt_name"] == "web_search_query"
    assert calls[1][1]["prompt"] == ""
    assert "prompt_name" not in calls[1][1]
    assert calls[2][0] == ["custom: question"]
    assert calls[2][1]["prompt"] == ""
    assert "prompt_name" not in calls[2][1]
    assert calls[0][1]["normalize_embeddings"] is True


def test_model_namespaces_and_aliases_have_correct_cache_identity():
    assert encoders.cache_dir_for("harrier-270m") == encoders.cache_dir_for(
        "microsoft/harrier-oss-v1-270m")
    assert encoders.cache_dir_for("org-one/same") != encoders.cache_dir_for("org-two/same")


def _cache_setup(monkeypatch):
    rows = [{"paper_stem": "a", "section": "kc", "position": 0, "text": "original"}]
    calls = []
    details = {"model": "fake", "revision": "rev1", "dimension": 2,
               "max_seq_length": 512, "dtype": "torch.float32"}
    monkeypatch.setattr(encoders, "_all_db_claims", lambda: [dict(row) for row in rows])
    monkeypatch.setattr(encoders, "model_details", lambda *a, **kw: dict(details))

    def embed(texts, *args, **kwargs):
        calls.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(encoders, "_embed_with", embed)
    return rows, calls, details


def test_claim_cache_rebuilds_when_text_changes_without_row_count_change(monkeypatch):
    rows, calls, _ = _cache_setup(monkeypatch)
    encoders._load_or_build_claim_cache("fake")
    encoders._load_or_build_claim_cache("fake")
    assert len(calls) == 1
    rows[0]["text"] = "edited"
    _, current = encoders._load_or_build_claim_cache("fake")
    assert calls == [["original"], ["edited"]]
    assert current[0]["text"] == "edited"


def test_claim_cache_rebuilds_after_addition_and_deletion(monkeypatch):
    rows, calls, _ = _cache_setup(monkeypatch)
    encoders._load_or_build_claim_cache("fake")
    rows.append({**rows[0], "paper_stem": "b"})
    vectors, current = encoders._load_or_build_claim_cache("fake")
    assert len(vectors) == len(current) == 2
    rows.pop(0)
    _, current = encoders._load_or_build_claim_cache("fake")
    assert len(calls) == 3
    assert [row["paper_stem"] for row in current] == ["b"]


@pytest.mark.parametrize("setting,value", [("revision", "rev2"), ("max_seq_length", 1024),
                                           ("dtype", "torch.bfloat16")])
def test_model_setting_changes_invalidate_cache(monkeypatch, setting, value):
    _, calls, details = _cache_setup(monkeypatch)
    encoders._load_or_build_claim_cache("fake")
    details[setting] = value
    encoders._load_or_build_claim_cache("fake")
    assert len(calls) == 2


def test_document_prefix_changes_invalidate_cache(monkeypatch):
    _, calls, _ = _cache_setup(monkeypatch)
    encoders._load_or_build_claim_cache("fake")
    encoders._load_or_build_claim_cache("fake", doc_prefix="passage: ")
    assert len(calls) == 2


def test_page_body_edit_invalidates_cache(monkeypatch):
    _, calls, _ = _cache_setup(monkeypatch)
    inputs = ([{"key": "a", "stem": "a", "content_hash": "old"}], ["original"])
    monkeypatch.setattr(encoders, "_page_inputs", lambda: inputs)
    encoders._load_or_build_page_cache("fake")
    inputs[0][0]["content_hash"] = "new"
    inputs[1][0] = "edited"
    encoders._load_or_build_page_cache("fake")
    assert calls == [["original"], ["edited"]]


def test_page_inputs_exclude_bookkeeping_documents(monkeypatch):
    from researchwiki.index import pages_semantic
    from researchwiki import wiki
    pages = [SimpleNamespace(key=f"wiki/{kind}", stem=kind, category="wiki",
                             page_type=kind, fm={"title": kind})
             for kind in ("paper", "meta", "dashboard")]
    monkeypatch.setattr(wiki, "read_pages", lambda: pages)
    monkeypatch.setattr(pages_semantic, "page_index_text", lambda page: "substantive text")
    rows, texts = encoders._page_inputs()
    assert [row["page_type"] for row in rows] == ["paper"]
    assert texts == ["substantive text"]


@pytest.mark.parametrize("text,prefix,reuse", [("a", "", True), ("1234567", "", False),
                                              ("a", "123456", False)])
def test_context_cache_reuse_requires_all_inputs_to_fit_both_limits(monkeypatch, text, prefix, reuse):
    rows = [{"key": "a"}]
    signature = {"model": "microsoft/harrier-oss-v1-270m", "max_seq_length": 8,
                 "doc_prefix": prefix}
    cached = (np.ones((1, 2)), {"rows": rows, "signature": {**signature, "max_seq_length": 10}})
    model = SimpleNamespace(tokenizer=lambda inputs, **kw:
                            {"input_ids": [[0] * (len(value) + 2) for value in inputs]})
    monkeypatch.setattr(encoders, "_load_model", lambda *a, **kw: model)
    assert encoders._compatible_context(cached, rows, [text], signature, "harrier-270m", False) is reuse


def test_loader_keys_include_options_and_trust_remote_code(monkeypatch):
    import sys
    made = []

    class Encoder:
        max_seq_length = 32768
        prompts = {"web_search_query": "configured prompt"}

        def __init__(self, *args, **kwargs):
            made.append(kwargs)

        def float(self):
            return self

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    original = encoders._load_model("harrier-270m")
    assert encoders._load_model("microsoft/harrier-oss-v1-270m") is original
    encoders.configure_model("harrier-270m", max_seq_length=512)
    capped = encoders._load_model("harrier-270m")
    assert capped is not original and capped.max_seq_length == 512
    assert encoders._load_model("harrier-270m", trust_remote_code=True) is not capped
    assert len(made) == 3


def test_failed_encoder_is_not_scored_as_zero_recall(monkeypatch):
    fixture = RetrievalFixture("test", "claims", "query", 5,
                               expected_claims=[ExpectedClaim("a", "kc", 0, "critical")],
                               expected_pages=[], must_not_appear=[])
    monkeypatch.setattr(retrieval, "_load_or_build_claim_cache", lambda *a, **kw: None)
    with pytest.raises(encoders.EmbeddingUnavailable):
        retrieval.retrieve_claims(fixture, "broken")


def test_missing_bm25_index_is_not_scored_as_zero_recall(monkeypatch):
    from researchwiki.index import pages_bm25
    from researchwiki.index.types import SearchBackendUnavailable

    class MissingIndex:
        def query(self, *args, **kwargs):
            raise SearchBackendUnavailable("index missing")

    monkeypatch.setattr(pages_bm25, "TantivySearchBackend", MissingIndex)
    with pytest.raises(SearchBackendUnavailable, match="index missing"):
        retrieval._bm25_pages("query", 5)


def test_bm25_comparison_excludes_the_same_bookkeeping_pages(monkeypatch):
    from researchwiki.index import pages_bm25
    from researchwiki import wiki
    monkeypatch.setattr(wiki, "read_pages", lambda: [SimpleNamespace(key="wiki/log", page_type="meta")])

    class Index:
        def query(self, query, limit):
            assert limit == 2
            return [SimpleNamespace(key="wiki/log", score=2), SimpleNamespace(key="ai/paper", score=1)]

    monkeypatch.setattr(pages_bm25, "TantivySearchBackend", Index)
    assert retrieval._bm25_pages("query", 1)[0].paper_stem == "ai/paper"


def test_invalid_vectors_are_rejected(monkeypatch):
    model = SimpleNamespace(encode=lambda *a, **kw: np.array([[np.nan, 1]]))
    monkeypatch.setattr(encoders, "_load_model", lambda *a, **kw: model)
    with pytest.raises(encoders.EmbeddingUnavailable, match="invalid embeddings"):
        encoders._embed_with(["query"], "fake")


@pytest.mark.parametrize("automatic", [True, False])
def test_automatic_bge_offline_mode_does_not_block_candidates_but_explicit_offline_does(monkeypatch, automatic):
    import os
    from huggingface_hub import constants
    from researchwiki.index import embeddings
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(embeddings, "_AUTOMATIC_OFFLINE_MODE", automatic)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with encoders._model_download_mode("microsoft/harrier-oss-v1-270m"):
        assert constants.HF_HUB_OFFLINE is (not automatic)
        assert os.environ["HF_HUB_OFFLINE"] == ("0" if automatic else "1")
    assert constants.HF_HUB_OFFLINE is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_retriever_uses_query_encoding_role(monkeypatch):
    fixture = RetrievalFixture("test", "claims", "query", 5,
                               expected_claims=[ExpectedClaim("a", "kc", 0, "critical")],
                               expected_pages=[], must_not_appear=[])
    row = {"paper_stem": "a", "section": "kc", "position": 0}
    monkeypatch.setattr(retrieval, "_load_or_build_claim_cache",
                        lambda *a, **kw: (np.array([[1., 0.]]), [row]))
    roles = []

    def embed(*args, **kwargs):
        roles.append(kwargs["role"])
        return np.array([[1., 0.]])

    monkeypatch.setattr(retrieval, "_embed_with", embed)
    assert retrieval.retrieve_claims(fixture, "harrier-270m")[0].paper_stem == "a"
    assert roles == ["query"]
