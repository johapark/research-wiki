"""Frozen archive contract, finite JSON, and truthful declared completeness."""

from dataclasses import replace
from pathlib import Path

import pytest

from researchwiki.operations._wire import ContractError
from researchwiki.records import AttemptSpec, BlobRef, CapturePolicy, Producer, Requirement, SourceRef
from researchwiki.records.contracts import AttemptEvent, ArtifactRecorded
from researchwiki.records.jsonio import canonical_bytes, decode_json

FIXTURE = Path(__file__).with_name("record-fixtures") / "attempt-spec-v1.json"


def spec():
    return AttemptSpec.from_dict(decode_json(FIXTURE.read_bytes()))


def test_frozen_wire_roundtrip_preserves_existing_hyphenated_attempt_identity():
    value = spec()
    assert value.attempt_id == "a132b9fa-90e8-4b73-8040-e67eec62fb29"
    assert AttemptSpec.from_dict(decode_json(canonical_bytes(value.to_dict()))) == value


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("attempt_id", "../escape"),
    ("attempt_id", "A132B9FA-90E8-4B73-8040-E67EEC62FB29"), ("created_at", "2026-10-23"),
    ("generation", True), ("generation", 1), ("job_id", "not-an-id"),
])
def test_invalid_identity_or_future_version_is_rejected(field, value):
    data = spec().to_dict()
    data[field] = value
    with pytest.raises(ContractError):
        AttemptSpec.from_dict(data)


@pytest.mark.parametrize("change", ["extra", "missing_version", "missing_optional", "missing_slot", "optional_required", "wrong_kind"])
def test_wire_fields_and_profile_inventory_cannot_silently_disappear(change):
    data = spec().to_dict()
    if change == "extra":
        data['ignored_future_field'] = True
    elif change == "missing_version":
        del data['schema_version']
    elif change == "missing_optional":
        del data['job_id']
    elif change == "missing_slot":
        data['requirements'].pop()
    elif change == "optional_required":
        data['requirements'][0]['required'] = False
    else:
        data['requirements'][0]['kind'] = 'model_response'
    with pytest.raises(ContractError):
        AttemptSpec.from_dict(data)


@pytest.mark.parametrize("raw", [
    b'{"schema_version":1,"schema_version":99}', b'{"value":{"a":1,"a":2}}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}', b'"\\ud800"', b'\xff',
])
def test_archive_json_rejects_ambiguous_or_nonfinite_content(raw):
    with pytest.raises(ContractError):
        decode_json(raw)


def test_api_temperature_sentinel_requires_an_explicit_normalized_projection():
    with pytest.raises(ContractError):
        canonical_bytes({'applied_temperature': float('nan')})
    value = {'requested_temperature': .2, 'applied_temperature': None, 'status': 'provider_controlled'}
    assert decode_json(canonical_bytes(value)) == value
    with pytest.raises(ContractError):
        canonical_bytes({1: 'silently-coerced-key'})


def test_source_and_policy_validation_is_strict_without_opening_any_pdf():
    blob = BlobRef(sha256='0' * 64, size_bytes=0, media_type='text/plain', encoding='utf-8')
    with pytest.raises(ContractError):
        SourceRef(document=blob, role='original')
    with pytest.raises(ContractError):
        replace(blob, size_bytes=True)
    with pytest.raises(ContractError):
        CapturePolicy(mode='best_effort')
    with pytest.raises(ContractError):
        Requirement(slot='../file', kind='model_request')
    with pytest.raises(ContractError):
        Producer(name='fixture', version='1', revision='main')
    with pytest.raises(ContractError):
        replace(spec(), requirements=spec().requirements + (spec().requirements[0],))


def test_event_cannot_rename_the_payload_type_or_fabricate_a_predecessor():
    original = AttemptEvent(archive_id='a' * 32, attempt_id=spec().attempt_id,
                            event_id='b' * 32, sequence=1, previous_sha256=None,
                            created_at=spec().created_at, kind='attempt_started', payload=spec())
    assert AttemptEvent.from_dict(original.to_dict()) == original
    with pytest.raises(ContractError):
        replace(original, kind='attempt_finished')
    with pytest.raises(ContractError):
        replace(original, sequence=2)
    with pytest.raises(ContractError):
        ArtifactRecorded.from_dict({'slot': 'source.original', 'artifact': {}})
