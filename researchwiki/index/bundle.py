"""One atomic file for a matrix and its row-aligned JSON metadata."""

import json
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ..fsatomic import _open_unique_sibling, _replace_with_retry


def load_bundle(path: Path, model: str, *, row_fields=()):
    try:
        with np.load(path, allow_pickle=False) as bundle:
            vectors = bundle["vectors"].copy()
            metadata = json.loads(str(bundle["metadata"].item()))
        if not isinstance(metadata, dict) or metadata["model"] != model or vectors.ndim != 2:
            return None
        if not isinstance(metadata["rows"], list) or any(
            not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in row_fields)
            for row in metadata["rows"]
        ):
            return None
        if vectors.shape != (len(metadata["rows"]), metadata["dim"]):
            return None
        if not np.issubdtype(vectors.dtype, np.number) or not np.isfinite(vectors).all():
            return None
        return vectors, metadata
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None


def save_bundle(path: Path, vectors, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = _open_unique_sibling(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, vectors=vectors, metadata=np.asarray(json.dumps(metadata)))
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
