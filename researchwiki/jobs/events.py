"""Private append-only event observations; manifests own terminal job state."""

import json
import os

from ..operations import IngestEvent
from .files import JobStoreUnavailable


class EventWriter:
    def __init__(self, path):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.stream = os.fdopen(fd, "w", encoding="utf-8")

    def __call__(self, event):
        self.stream.write(json.dumps(event.to_dict(), allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def read_events(store, job_id, *, after_sequence=0):
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("after_sequence must be nonnegative")
    record = store.get_job(job_id)
    path = store.path(job_id) / "events.jsonl"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise JobStoreUnavailable("cannot read job events") from exc
    events = []
    sequence = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # only an incomplete final line is ignored
        try:
            event = IngestEvent.from_dict(json.loads(line))
        except (ValueError, UnicodeError) as exc:
            raise JobStoreUnavailable("invalid complete job event") from exc
        if (event.request_id != record.request.request_id or event.job_id != job_id
                or event.sequence != sequence + 1):
            raise JobStoreUnavailable("job event identity or sequence mismatch")
        sequence = event.sequence
        if sequence > after_sequence:
            events.append(event)
    return tuple(events)


def pending_relays(store, job_id):
    if store.get_job(job_id).execution != "running":
        return ()
    pending = {}
    for event in read_events(store, job_id):
        if event.kind in ("model_call_finished", "model_call_failed"):
            pending = {key: value for key, value in pending.items() if key[0] != event.call_id}
        if event.relay is not None:
            key = (event.call_id, event.relay.op_id)
            if event.kind == "relay_waiting":
                pending[key] = event.relay
            else:
                pending.pop(key, None)
    return tuple(pending.values())
