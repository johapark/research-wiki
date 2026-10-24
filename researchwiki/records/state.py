"""Deterministic projection of retained events; no publication or model execution."""

from dataclasses import replace

from ..operations._wire import ContractError
from .contracts import AttemptManifest, InventoryItem, capture_state
from .errors import RecordConflict, RecordCorrupt


def project(events):
    if not events or events[0].kind != "attempt_started":
        raise RecordCorrupt("attempt history must start with its specification")
    first = events[0]
    spec = first.payload
    if first.attempt_id != spec.attempt_id:
        raise RecordCorrupt("attempt specification identity mismatch")
    inventory = tuple(InventoryItem(slot=req.slot,
                       status="omitted_by_policy" if spec.policy.mode == "off" else "pending",
                       reason="capture_disabled" if spec.policy.mode == "off" else None)
                      for req in spec.requirements)
    manifest = AttemptManifest(archive_id=first.archive_id, spec=spec, inventory=inventory,
                               artifacts=(), terminal=None, last_sequence=1,
                               last_event_sha256=first.digest,
                               capture_state=capture_state(spec, inventory, None))
    ids = set()
    for sequence, event in enumerate(events, 1):
        if event.event_id in ids or event.sequence != sequence or event.archive_id != first.archive_id or event.attempt_id != spec.attempt_id:
            raise RecordCorrupt("attempt event identity/sequence mismatch")
        ids.add(event.event_id)
        if sequence == 1:
            if event.previous_sha256 is not None:
                raise RecordCorrupt("initial event cannot have a predecessor")
            continue
        if event.previous_sha256 != manifest.last_event_sha256:
            raise RecordCorrupt("attempt event predecessor mismatch")
        try:
            manifest = advance(manifest, event)
        except (RecordConflict, ContractError) as exc:
            raise RecordCorrupt("attempt history contains an invalid transition") from exc
    return manifest


def advance(manifest, event):
    if manifest.terminal is not None or event.kind == "attempt_started":
        raise RecordConflict("attempt is closed or already started")
    inventory = list(manifest.inventory)
    artifacts = manifest.artifacts
    terminal = manifest.terminal
    if event.kind == "attempt_finished":
        terminal = event.payload
        inventory = [replace(item, status="unknown", reason="not_recorded_before_terminal")
                     if item.status == "pending" else item for item in inventory]
    else:
        slots = {item.slot: i for i, item in enumerate(inventory)}
        slot = event.payload.slot
        if slot not in slots:
            raise RecordConflict("artifact slot is not in the declared inventory")
        index = slots[slot]
        item = inventory[index]
        if item.status not in ("pending", "retained"):
            raise RecordConflict("inventory slot is already resolved")
        if event.kind == "artifact_recorded":
            artifact = event.payload.artifact
            known = {a.artifact_id for a in artifacts}
            if manifest.spec.policy.mode == "off" or artifact.attempt_id != manifest.spec.attempt_id:
                raise RecordConflict("capture disabled or artifact belongs to another attempt")
            if artifact.artifact_id in known or not set(artifact.parents).issubset(known):
                raise RecordConflict("duplicate artifact or missing earlier parent")
            if artifact.kind != manifest.spec.requirements[index].kind:
                raise RecordConflict("artifact kind differs from its inventory slot")
            artifacts += (artifact,)
            inventory[index] = replace(item, status="retained", artifact_ids=item.artifact_ids + (artifact.artifact_id,))
        elif event.kind == "inventory_resolved":
            inventory[index] = replace(item, status=event.payload.status, reason=event.payload.reason)
        else:
            raise RecordConflict("unsupported archive transition")
    inventory = tuple(inventory)
    return replace(manifest, inventory=inventory, artifacts=artifacts, terminal=terminal,
                   last_sequence=event.sequence, last_event_sha256=event.digest,
                   capture_state=capture_state(manifest.spec, inventory, terminal))
