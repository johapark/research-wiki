"""Durable archive behavior against disposable synthetic content, never the corpus."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest

from researchwiki.operations._wire import ContractError
from researchwiki.records import (
    Artifact, AttemptFinished, AttemptSpec, CapturePolicy, CheckpointStale,
    RecordConflict, RecordCorrupt, RecordStore, RecordStoreUnavailable, SourceRef,
)
from researchwiki.records.jsonio import canonical_bytes, decode_json

FIXTURE = Path(__file__).with_name('record-fixtures') / 'attempt-spec-v1.json'


@pytest.fixture
def spec():
    return AttemptSpec.from_dict(decode_json(FIXTURE.read_bytes()))


@pytest.fixture
def store(tmp_path):
    return RecordStore.create(tmp_path / 'archive')


def artifact(store, spec, slot='target.response', *, data=b'{"claims":[', parents=()):
    kind = next(req.kind for req in spec.requirements if req.slot == slot)
    if kind == 'source_snapshot':
        payload = store.put_blob(data, media_type='application/pdf')
        sources = (SourceRef(document=payload, role='original'),)
    elif kind == 'representation':
        document = store.put_blob(b'%PDF-synthetic', media_type='application/pdf')
        payload = store.put_blob(data, media_type='text/plain', encoding='utf-8')
        sources = (SourceRef(document=document, role='effective', representation=payload),)
    else:
        payload, sources = store.put_blob(data), ()
    return Artifact(artifact_id=uuid.uuid4().hex, attempt_id=spec.attempt_id, kind=kind,
                    payload=payload, producer=spec.producer, created_at=spec.created_at,
                    origin='synthetic', sources=sources, parents=parents,
                    phase='target_claims' if slot.startswith('target.') else 'extract')


def finish():
    return AttemptFinished(execution='finished', outcome='draft_ready',
                           publication='not_published', projections='not_applicable')


def test_reads_never_initialize_and_create_is_explicit_idempotent_and_private(tmp_path):
    root = tmp_path / 'archive'
    unopened = RecordStore(root)
    with pytest.raises(RecordStoreUnavailable, match='not initialized'):
        unopened.list_attempts()
    assert not root.exists()
    store = RecordStore.create(root)
    before = (root / 'store.json').read_bytes()
    assert RecordStore.create(root).metadata() == store.metadata()
    assert (root / 'store.json').read_bytes() == before and store.list_attempts() == ()
    if sys.platform != 'win32':
        assert (root / 'store.json').stat().st_mode & 0o777 == 0o600
    unknown = tmp_path / 'unrecognized'
    unknown.mkdir()
    (unknown / 'preserve.json').write_text('important')
    with pytest.raises(RecordConflict, match='nonempty'):
        RecordStore.create(unknown)
    assert not (unknown / 'store.json').exists()


def test_payload_identity_deduplication_and_source_copy_are_byte_exact(store, tmp_path):
    raw = b'  ```json\n{"claims":[{"unselected":true}]}\n```  '
    first = store.put_blob(raw)
    path = store.root / first.relative_path
    before = path.stat().st_mtime_ns
    assert store.put_blob(raw) == first and path.stat().st_mtime_ns == before
    assert store.read_blob(first) == raw
    assert first.sha256 == hashlib.sha256(raw).hexdigest()
    assert store.put_json({'b': 2, 'a': 1}) == store.put_json({'a': 1, 'b': 2})
    source = tmp_path / 'input.pdf'
    source.write_bytes(b'%PDF-original')
    snap = store.put_file(source)
    source.write_bytes(b'%PDF-replacement')
    newer = store.put_file(source)
    assert snap != newer and store.read_blob(snap) == b'%PDF-original'
    assert source.stat().st_ino != (store.root / newer.relative_path).stat().st_ino


def test_complete_attempt_keeps_every_artifact_and_parent_without_reinterpreting_output(store, spec):
    store.start_attempt(spec)
    prior = ()
    for req in spec.requirements:
        value = artifact(store, spec, req.slot, parents=prior)
        manifest = store.record_artifact(req.slot, value)
        prior = (value.artifact_id,)
        assert store.record_artifact(req.slot, value) == manifest
    manifest = store.finish_attempt(spec.attempt_id, finish())
    assert manifest.capture_state == 'complete' and len(manifest.artifacts) == 6
    assert store.start_attempt(spec) == manifest
    assert store.finish_attempt(spec.attempt_id, finish()) == manifest
    assert store.verify(spec.attempt_id).event_count == 8
    assert len(store.list_attempts()) == 1
    response = next(a for a in manifest.artifacts if a.kind == 'model_response')
    assert store.read_blob(response.payload) == b'{"claims":['
    with pytest.raises(RecordConflict, match='closed'):
        store.record_artifact('target.response', artifact(store, spec))


def test_incomplete_disabled_and_not_run_states_do_not_turn_into_empty_success(store, spec):
    store.start_attempt(spec)
    value = artifact(store, spec)
    store.record_artifact('target.response', value)
    manifest = store.finish_attempt(spec.attempt_id, AttemptFinished(
        execution='interrupted', outcome=None, publication='unknown', projections='unknown'))
    assert manifest.capture_state == 'incomplete'
    assert all(item.reason == 'not_recorded_before_terminal' for item in manifest.inventory if item.status == 'unknown')
    with pytest.raises(ContractError):
        replace(manifest, capture_state='complete')
    skipped = replace(spec, attempt_id=uuid.uuid4().hex)
    store.start_attempt(skipped)
    for req in skipped.requirements:
        store.resolve_slot(skipped.attempt_id, req.slot, 'not_run', 'synthetic path invoked no phase')
    assert store.finish_attempt(skipped.attempt_id, finish()).capture_state == 'complete'
    disabled = replace(spec, attempt_id=uuid.uuid4().hex, policy=CapturePolicy(mode='off'))
    opened = store.start_attempt(disabled)
    assert opened.capture_state == 'disabled'
    assert all(item.status == 'omitted_by_policy' for item in opened.inventory)
    with pytest.raises(RecordConflict):
        store.record_artifact('target.response', artifact(store, disabled))


def test_identity_parent_kind_and_inventory_conflicts_do_not_write_history(store, spec):
    store.start_attempt(spec)
    value = artifact(store, spec)
    initial = store.verify(spec.attempt_id).event_count
    with pytest.raises(RecordConflict):
        store.start_attempt(replace(spec, request_id='different request'))
    with pytest.raises(RecordConflict, match='parent'):
        store.record_artifact('target.response', replace(value, parents=(uuid.uuid4().hex,)))
    with pytest.raises(RecordConflict, match='kind'):
        store.record_artifact('target.request', value)
    assert store.verify(spec.attempt_id).event_count == initial
    store.record_artifact('target.response', value)
    with pytest.raises(RecordConflict, match='duplicate'):
        store.record_artifact('target.response', replace(value, created_at='2026-10-24T00:00:00+00:00'))
    store.resolve_slot(spec.attempt_id, 'target.response', 'failed', 'schema rejected the retained response')
    assert store.read_attempt(spec.attempt_id).inventory[3].artifact_ids == (value.artifact_id,)
    with pytest.raises(RecordConflict, match='resolved'):
        store.record_artifact('target.response', artifact(store, spec))


def test_checkpoint_failure_leaves_durable_response_and_requires_explicit_repair(store, spec, monkeypatch):
    store.start_attempt(spec)
    checkpoint = store.root / 'attempts' / spec.attempt_id / 'manifest.json'
    original = checkpoint.read_bytes()
    value = artifact(store, spec)
    with monkeypatch.context() as patch:
        patch.setattr(store, '_write_checkpoint', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(RecordStoreUnavailable):
            store.record_artifact('target.response', value)
    assert checkpoint.read_bytes() == original
    report = store.verify(spec.attempt_id)
    assert report.checkpoint == 'stale' and report.manifest.artifacts == (value,)
    with pytest.raises(CheckpointStale):
        store.read_attempt(spec.attempt_id)
    with pytest.raises(CheckpointStale):
        store.record_artifact('target.response', value)
    repaired = store.repair_checkpoint(spec.attempt_id)
    assert store.record_artifact('target.response', value) == repaired
    assert store.verify(spec.attempt_id).event_count == 2
    before = checkpoint.stat().st_mtime_ns
    assert store.repair_checkpoint(spec.attempt_id) == repaired and checkpoint.stat().st_mtime_ns == before
    checkpoint.unlink()
    assert store.verify(spec.attempt_id).checkpoint == 'missing'
    assert store.repair_checkpoint(spec.attempt_id) == repaired


def test_start_failure_after_event_is_repairable_without_replacing_attempt(store, spec, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(store, '_write_checkpoint', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(RecordStoreUnavailable):
            store.start_attempt(spec)
    assert store.verify(spec.attempt_id).checkpoint == 'missing'
    with pytest.raises(CheckpointStale):
        store.start_attempt(spec)
    store.repair_checkpoint(spec.attempt_id)
    assert store.start_attempt(spec).last_sequence == 1


@pytest.mark.parametrize('damage', ['future', 'invalid_json', 'wrong_head', 'false_inventory'])
def test_unrecognized_checkpoint_is_preserved_and_never_repaired(store, spec, damage):
    manifest = store.start_attempt(spec)
    path = store.root / 'attempts' / spec.attempt_id / 'manifest.json'
    data = manifest.to_dict()
    if damage == 'future':
        data['schema_version'] = 99
    elif damage == 'wrong_head':
        data['last_event_sha256'] = '0' * 64
    elif damage == 'false_inventory':
        data['inventory'][0]['status'] = 'unknown'
        data['inventory'][0]['reason'] = 'not an observed event'
    path.write_bytes(b'{broken' if damage == 'invalid_json' else canonical_bytes(data))
    before = path.read_bytes()
    with pytest.raises(RecordCorrupt):
        store.repair_checkpoint(spec.attempt_id)
    assert path.read_bytes() == before


@pytest.mark.parametrize('damage', ['gap', 'same_sequence', 'duplicate_id', 'foreign_archive', 'future', 'wrong_parent'])
def test_event_chain_corruption_is_not_hidden_by_a_checkpoint(store, spec, damage):
    store.start_attempt(spec)
    store.record_artifact('target.response', artifact(store, spec))
    directory = store.root / 'attempts' / spec.attempt_id / 'events'
    paths = sorted(directory.glob('*.json'))
    first, second = [json.loads(p.read_text()) for p in paths]
    if damage == 'gap':
        paths[0].unlink()
    else:
        if damage == 'same_sequence':
            second['sequence'] = 1
            second['previous_sha256'] = None
        elif damage == 'duplicate_id':
            second['event_id'] = first['event_id']
        elif damage == 'foreign_archive':
            second['archive_id'] = uuid.uuid4().hex
        elif damage == 'future':
            second['schema_version'] = 2
        else:
            second['previous_sha256'] = '0' * 64
        paths[1].unlink()
        path = directory / f"{second['sequence']:08d}-{second['event_id']}.json"
        path.write_bytes(canonical_bytes(second))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(RecordCorrupt):
        store.repair_checkpoint(spec.attempt_id)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_corrupt_missing_and_source_blobs_block_verification_and_never_get_overwritten(store, spec):
    store.start_attempt(spec)
    value = artifact(store, spec, 'source.representation')
    store.record_artifact('source.representation', value)
    document = value.sources[0].document
    path = store.root / document.relative_path
    original = path.read_bytes()
    path.write_bytes(b'x' * len(original))
    with pytest.raises(RecordCorrupt):
        store.verify(spec.attempt_id)
    with pytest.raises(RecordCorrupt):
        store.put_blob(original, media_type='application/pdf')
    assert path.read_bytes() == b'x' * len(original)
    path.unlink()
    with pytest.raises(RecordCorrupt, match='missing'):
        store.verify(spec.attempt_id)


def test_directory_symlink_root_is_allowed_but_internal_escape_is_rejected(tmp_path, spec):
    target = tmp_path / 'synced'
    store = RecordStore.create(target)
    link = tmp_path / 'archive-link'
    link.symlink_to(target, target_is_directory=True)
    through_link = RecordStore(link)
    assert through_link.metadata() == store.metadata()
    through_link.start_attempt(spec)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (target / 'blobs').symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecordCorrupt, match='symlink'):
        through_link.put_blob(b'do not write outside the selected root')
    assert list(outside.iterdir()) == []
    with pytest.raises(ContractError):
        through_link.read_attempt('../outside')


def test_relocation_keeps_blobs_identity_and_verified_lineage_without_a_database(store, spec, tmp_path):
    store.start_attempt(spec)
    value = artifact(store, spec)
    store.record_artifact('target.response', value)
    before = store.read_attempt(spec.attempt_id)
    target = tmp_path / 'relocated'
    shutil.move(str(store.root), target)
    restored = RecordStore(target)
    assert restored.metadata().archive_id == before.archive_id
    assert restored.read_attempt(spec.attempt_id) == before
    assert restored.read_blob(value.payload) == b'{"claims":['
    assert not any(path.suffix == '.db' for path in target.rglob('*'))


def test_process_writers_serialize_and_retry_without_lost_events(store, spec):
    store.start_attempt(spec)
    script = '''import sys, uuid
from researchwiki.records import RecordStore, AttemptSpec, Artifact
from researchwiki.records.jsonio import decode_json
store=RecordStore(sys.argv[1]); spec=AttemptSpec.from_dict(decode_json(open(sys.argv[2], 'rb').read()))
blob=store.put_blob(sys.argv[3].encode())
item=Artifact(artifact_id=uuid.uuid4().hex, attempt_id=spec.attempt_id, kind='model_response', payload=blob,
              producer=spec.producer, created_at=spec.created_at, origin='synthetic')
store.record_artifact('target.response', item)
store.record_artifact('target.response', item)
'''
    def append(index):
        return subprocess.run([sys.executable, '-c', script, str(store.root), str(FIXTURE), str(index)],
                              capture_output=True, text=True, timeout=30)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(append, range(8)))
    assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
    manifest = store.read_attempt(spec.attempt_id)
    assert len(manifest.artifacts) == 8 and manifest.last_sequence == 9
    assert {store.read_blob(a.payload).decode() for a in manifest.artifacts} == {str(i) for i in range(8)}


def test_duplicate_blob_writers_publish_one_immutable_payload(store):
    data = b'one shared payload' * 100000
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: store.put_blob(data), range(16)))
    assert len(set(refs)) == 1
    assert store.read_blob(refs[0]) == data
    assert len(list((store.root / 'blobs/sha256').glob('*/*'))) == 1
    assert not list(store.root.rglob('.write-*'))


def test_failure_before_first_event_can_retry_without_fabricating_or_resetting_history(store, spec, monkeypatch):
    import researchwiki.records.storage as storage
    replace_file = storage.os.replace
    def fail_event(source, target):
        if target.name.startswith('00000001-'):
            raise OSError('event publish interrupted')
        return replace_file(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, 'replace', fail_event)
        with pytest.raises(RecordStoreUnavailable):
            store.start_attempt(spec)
    directory = store.root / 'attempts' / spec.attempt_id
    assert not (directory / 'manifest.json').exists()
    assert list((directory / 'events').iterdir()) == []
    assert store.start_attempt(spec).last_sequence == 1


def test_failure_before_later_event_preserves_checkpoint_and_unreferenced_payload(store, spec, monkeypatch):
    import researchwiki.records.storage as storage
    before = store.start_attempt(spec)
    value = artifact(store, spec)
    replace_file = storage.os.replace
    def fail_event(source, target):
        if target.name.startswith('00000002-'):
            raise OSError('event publish interrupted')
        return replace_file(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, 'replace', fail_event)
        with pytest.raises(RecordStoreUnavailable):
            store.record_artifact('target.response', value)
    assert store.read_attempt(spec.attempt_id) == before
    assert store.read_blob(value.payload) == b'{"claims":['
    assert store.record_artifact('target.response', value).last_sequence == 2


def test_future_store_metadata_is_not_reinitialized_and_identity_cannot_switch(store):
    path = store.root / 'store.json'
    value = json.loads(path.read_text())
    path.write_bytes(canonical_bytes({**value, 'schema_version': 2}))
    before = path.read_bytes()
    with pytest.raises(RecordCorrupt):
        RecordStore.create(store.root)
    assert path.read_bytes() == before
    path.write_bytes(canonical_bytes({**value, 'archive_id': uuid.uuid4().hex}))
    with pytest.raises(RecordCorrupt, match='identity changed'):
        store.metadata()


def test_declared_text_encoding_does_not_silently_repair_invalid_response_bytes(store):
    with pytest.raises(ContractError, match='UTF-8'):
        store.put_blob(b'\xff', media_type='text/plain', encoding='utf-8')
    raw = store.put_blob(b'\xff')
    assert store.read_blob(raw) == b'\xff'
    with pytest.raises(RecordCorrupt, match='encoding'):
        store.read_blob(replace(raw, encoding='utf-8'))
    # A multibyte character straddles the 64 KiB streaming boundary.
    text = b'a' * 65535 + 'α'.encode() + b' rest'
    assert store.read_blob(store.put_blob(text, media_type='text/plain', encoding='utf-8')) == text


def test_verification_checks_encoding_even_when_blob_digest_was_already_verified(store, spec):
    from researchwiki.records.contracts import AttemptEvent
    from researchwiki.records.state import project

    store.start_attempt(spec)
    raw = artifact(store, spec, data=b'\xff')
    store.record_artifact('target.response', raw)
    second = replace(raw, artifact_id=uuid.uuid4().hex)
    store.record_artifact('target.response', second)
    base = store.root / 'attempts' / spec.attempt_id
    paths = sorted((base / 'events').glob('*.json'))
    # A self-consistent history/checkpoint can still make a false encoding claim
    # on the second reference to the same immutable binary payload.
    value = decode_json(paths[-1].read_bytes())
    value['payload']['artifact']['payload']['encoding'] = 'utf-8'
    paths[-1].write_bytes(canonical_bytes(value))
    events = tuple(AttemptEvent.from_dict(decode_json(path.read_bytes())) for path in paths)
    (base / 'manifest.json').write_bytes(canonical_bytes(project(events).to_dict()))
    assert store.read_blob(raw.payload) == b'\xff'
    with pytest.raises(RecordCorrupt, match='encoding'):
        store.verify(spec.attempt_id)
