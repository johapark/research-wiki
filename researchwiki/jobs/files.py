"""Private, atomic job files. State is independent of derived DBs and indexes."""

import hashlib
import json

from ..errors import EnvironmentFailure
from ..operations import ContractError
from ..fsatomic import write_private_json as write_private_json

__all__ = ["write_private_json"]  # Shared primitive, retained at the existing import path.


class JobStoreUnavailable(EnvironmentFailure):
    pass


class JobConflict(ContractError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JobStoreUnavailable(f"Cannot read job record: {path}") from exc


def outside_library(path, library_root):
    roots = [library_root, *(library_root / name for name in ("wiki", "papers", "inbox"))]
    for root in roots:
        root = root.resolve()
        if path == root or root in path.parents:
            raise JobConflict("job storage must be outside the checkout and canonical content")
