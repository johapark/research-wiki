"""Cache publication invariants with deterministic local embeddings."""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from researchwiki.index import bundle, claim_embeddings as claims


def row(stem):
    return {"paper_stem": stem, "section": "results", "position": 0, "text": stem}


def test_interrupted_publication_preserves_previous_bundle(tmp_path, monkeypatch):
    path = tmp_path / "cache.npz"
    metadata = {"model": "test", "dim": 2, "rows": [{"text": "alpha"}]}
    bundle.save_bundle(path, np.asarray([[1., 0.]]), metadata)
    before = path.read_bytes()
    def fail(*args):
        raise OSError("interrupted publication")
    monkeypatch.setattr(bundle, "_replace_with_retry", fail)
    with pytest.raises(OSError, match="interrupted"):
        bundle.save_bundle(path, np.asarray([[0., 1.]]), dict(metadata, rows=[{"text": "beta"}]))
    assert path.read_bytes() == before
    vectors, meta = bundle.load_bundle(path, "test")
    assert meta["rows"][0]["text"] == "alpha"
    assert vectors[0].tolist() == [1., 0.]
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_warming_merges_without_misaligning_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "semantic_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(claims.embeddings, "is_available", lambda: True)
    barrier = threading.Barrier(2)
    def embed(texts):
        barrier.wait(timeout=5)
        return np.asarray([[1., 0.] if t == "alpha" else [0., 1.] for t in texts])
    monkeypatch.setattr(claims.embeddings, "embed_texts", embed)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(claims.warm_claim_embeddings, [row(s)]) for s in ("alpha", "beta")]
        for future in futures:
            assert future.result(timeout=10) is not None
    vectors, covered = claims.load_cached_claim_embeddings([row("alpha"), row("beta")])
    assert covered == [0, 1]
    assert vectors.tolist() == [[1., 0.], [0., 1.]]


def test_same_dimension_model_change_rejects_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "semantic_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(claims.embeddings, "is_available", lambda: True)
    monkeypatch.setattr(claims.embeddings, "embed_texts", lambda texts: np.ones((len(texts), 2)))
    claims.warm_claim_embeddings([row("alpha")])
    monkeypatch.setattr(claims.embeddings, "DEFAULT_MODEL", "different-model")
    assert claims.load_cached_claim_embeddings([row("alpha")]) is None


def test_legacy_pair_is_a_miss_even_when_row_counts_match(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "semantic_cache_dir", lambda: tmp_path)
    np.save(tmp_path / "claims.npy", np.ones((1, 2)))
    (tmp_path / "claims_meta.json").write_text('[{"id":"alpha·results·0","hash":"x"}]')
    assert claims.load_cached_claim_embeddings([row("alpha")]) is None


@pytest.mark.parametrize("metadata", [
    [], {"model": "test", "dim": 2, "rows": [None]},
    {"model": "test", "dim": 2, "rows": [{"id": 12}]},
    {"model": "test", "dim": 3, "rows": [{"id": "alpha"}]},
])
def test_malformed_bundle_is_a_cache_miss(tmp_path, metadata):
    path = tmp_path / "cache.npz"
    bundle.save_bundle(path, np.ones((1, 2)), metadata)
    assert bundle.load_bundle(path, "test", row_fields=("id",)) is None


def test_corrupt_zip_is_a_cache_miss(tmp_path):
    path = tmp_path / "cache.npz"
    path.write_bytes(b"PK\x03\x04broken zip")
    assert bundle.load_bundle(path, "test") is None
