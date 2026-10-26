"""Explicit, immutable paper selection for retrieval; never an authorization grant.

None means the existing whole-library behavior. An empty scope means no evidence.
Selection snapshots carry source hashes so consumers can detect stale packets.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


def validate_stem(stem: str) -> str:
    if not isinstance(stem, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", stem):
        raise ValueError("expected a canonical bare paper stem")
    return stem


@dataclass(frozen=True)
class EvidenceScope:
    stems: frozenset[str]
    library_root: str | None = None
    library_id: str | None = None
    selection_id: str | None = None
    selection_revision: int | None = None
    # (stem, wiki-page SHA256, PDF SHA256), sorted by stem.
    source_revisions: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self):
        if isinstance(self.stems, str):
            raise ValueError("stems must be a collection of canonical paper stems")
        object.__setattr__(self, "stems", frozenset(validate_stem(s) for s in self.stems))
        if any(not isinstance(r, (tuple, list)) or len(r) != 3 or
               not all(isinstance(value, str) for value in r) for r in self.source_revisions):
            raise ValueError("invalid source revisions in evidence scope")
        object.__setattr__(self, "source_revisions", tuple(sorted(tuple(r) for r in self.source_revisions)))
        for row in self.source_revisions:
            if len(row) != 3 or row[0] not in self.stems or any(
                not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in row[1:]
            ):
                raise ValueError("invalid source revision in evidence scope")
        if self.source_revisions and {r[0] for r in self.source_revisions} != self.stems:
            raise ValueError("source revisions must cover every scoped paper")
        if len({r[0] for r in self.source_revisions}) != len(self.source_revisions):
            raise ValueError("duplicate source revision")
        if self.library_root is not None:
            if not isinstance(self.library_root, str):
                raise ValueError("library_root must be text")
            object.__setattr__(self, "library_root", str(Path(self.library_root).resolve()))
        for value in (self.library_id, self.selection_id):
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value)):
                raise ValueError("invalid library/selection ID in scope")
        if self.selection_revision is not None and (type(self.selection_revision) is not int or self.selection_revision < 1):
            raise ValueError("invalid selection revision in scope")

    def check_root(self):
        if self.library_root is not None and Path.cwd().resolve() != Path(self.library_root):
            raise ValueError("evidence scope belongs to a different library; run from its root")

    def allows(self, stem: str, page_type: str = "paper") -> bool:
        self.check_root()
        # Cross-paper pages/commentaries have unproven dependencies. Never use
        # their text as a shortcut into strict scoped evidence.
        return stem in self.stems and page_type == "paper"

    def sql(self, column: str = "paper_stem") -> tuple[str, list[str]]:
        """Internal SQL clause; values are parameters, including an empty set."""
        self.check_root()
        if column not in {"paper_stem", "c.paper_stem"}:
            raise ValueError("unsupported scope column")
        # One JSON parameter avoids SQLite's host-parameter limit for large
        # libraries. JSON functions ship with supported Python SQLite builds.
        return f" AND {column} IN (SELECT value FROM json_each(?))", [json.dumps(sorted(self.stems))]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1, "stems": sorted(self.stems),
            "library_root": self.library_root, "library_id": self.library_id,
            "selection_id": self.selection_id, "selection_revision": self.selection_revision,
            "source_revisions": [list(r) for r in self.source_revisions],
        }

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "stems", "library_root", "library_id", "selection_id",
                  "selection_revision", "source_revisions"}
        if not isinstance(data, dict) or set(data) - {"fingerprint"} != fields:
            raise ValueError("invalid evidence scope fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported evidence scope version")
        if not isinstance(data["stems"], list) or not isinstance(data["source_revisions"], list):
            raise ValueError("scope stems/source_revisions must be arrays")
        scope = cls(**{k: v for k, v in data.items() if k in fields - {"schema_version"}})
        if "fingerprint" in data and data["fingerprint"] != scope.fingerprint:
            raise ValueError("scope fingerprint does not match its snapshot")
        return scope

    @property
    def fingerprint(self) -> str:
        # Location is not content identity: a portable backup can be relocated.
        data = self.to_dict()
        data.pop("library_root")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
