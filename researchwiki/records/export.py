"""Explicit self-contained export of one verified attempt; no import or pruning."""

from pathlib import Path
import shutil

from ..fsatomic import exclusive_lock
from .errors import CheckpointStale, RecordConflict
from .storage import Storage, storage_errors
from .store import RecordStore


def export_attempt(store, attempt_id, destination):
    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(store.root) or store.root.is_relative_to(destination):
        raise RecordConflict("export destination must be separate from the source archive")
    if destination.exists():
        raise RecordConflict("export destination already exists")
    with storage_errors(), exclusive_lock(store._attempt_path(attempt_id)):
        report, _ = store._inspect(attempt_id)
        if report.checkpoint != "current":
            raise CheckpointStale("repair the source checkpoint explicitly before export")
        destination.mkdir(mode=0o700, parents=True)
        try:
            target = Storage(destination)
            target.publish(target.path("store.json"), store._storage.path("store.json").read_bytes())
            copied = set()
            for artifact in report.manifest.artifacts:
                for ref in artifact.blobs():
                    if ref.sha256 in copied:
                        continue
                    with store._storage.path(*ref.relative_path.split("/")).open("rb") as stream:
                        actual = target.put_stream(stream, media_type=ref.media_type, encoding=ref.encoding)
                    if actual != ref:
                        raise RecordConflict("source payload changed while exporting")
                    copied.add(ref.sha256)
            base = store._attempt_path(attempt_id)
            for path in (base / "events").iterdir():
                if path.name.startswith(".write-"):
                    continue
                target.publish(target.path("attempts", attempt_id, "events", path.name), path.read_bytes())
            target.publish(target.path("attempts", attempt_id, "manifest.json"), (base / "manifest.json").read_bytes())
            RecordStore(destination).verify(attempt_id)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination
