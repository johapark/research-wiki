"""Bound replay work without weakening archive verification or repair."""

from dataclasses import replace

import pytest

from researchwiki.records import RecordCorrupt
from researchwiki.records.jsonio import canonical_bytes
from tests.test_ingest_records_store import artifact, spec as spec, store as store


@pytest.mark.parametrize("size", [4, 12])
def test_current_checkpoint_projects_each_history_once(store, spec, monkeypatch, size):
    import researchwiki.records.store as module

    store.start_attempt(spec)
    project = module.project
    counts = []
    def counted(events):
        counts.append(len(events))
        return project(events)
    monkeypatch.setattr(module, "project", counted)
    for index in range(size):
        store.record_artifact("target.response", artifact(store, spec, data=str(index).encode()))
    store.verify(spec.attempt_id)
    assert counts == list(range(1, size + 2))


def test_stale_checkpoint_still_checks_its_exact_prefix(store, spec, monkeypatch):
    import researchwiki.records.store as module

    first = store.start_attempt(spec)
    store.record_artifact("target.response", artifact(store, spec))
    path = store.root / "attempts" / spec.attempt_id / "manifest.json"
    path.write_bytes(canonical_bytes(first.to_dict()))
    original, counts = module.project, []
    def counted(events):
        counts.append(len(events))
        return original(events)
    monkeypatch.setattr(module, "project", counted)
    assert store.verify(spec.attempt_id).checkpoint == "stale"
    assert counts == [2, 1]
    path.write_bytes(canonical_bytes(replace(first, spec=replace(spec, request_id="changed")).to_dict()))
    with pytest.raises(RecordCorrupt, match="prefix"):
        store.repair_checkpoint(spec.attempt_id)


def test_reused_projection_never_caches_blob_or_checkpoint_integrity(store, spec):
    store.start_attempt(spec)
    item = artifact(store, spec, data=b"important payload")
    store.record_artifact("target.response", item)
    report = store.verify(spec.attempt_id)
    blob = store.root / item.payload.relative_path
    blob.write_bytes(b"x" * item.payload.size_bytes)
    with pytest.raises(RecordCorrupt, match="digest"):
        store.record_artifact("target.response", artifact(store, spec, data=b"next"))
    blob.write_bytes(b"important payload")
    path = store.root / "attempts" / spec.attempt_id / "manifest.json"
    changed = replace(report.manifest, spec=replace(spec, request_id="changed"))
    path.write_bytes(canonical_bytes(changed.to_dict()))
    with pytest.raises(RecordCorrupt, match="prefix"):
        store.verify(spec.attempt_id)
