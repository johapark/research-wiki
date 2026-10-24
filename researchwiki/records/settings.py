"""Selected-profile retention settings; resolution never initializes storage."""

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .errors import RecordStoreUnavailable

_frozen = ContextVar("researchwiki_retention_settings", default=None)


def validate(value):
    if not isinstance(value, dict) or set(value) - {"mode", "profile", "root"}:
        raise ValueError("retention must contain only mode, profile and root")
    if value.get("mode", "off") not in ("off", "required"):
        raise ValueError("retention.mode must be off or required")
    if value.get("profile", "extraction-v1") != "extraction-v1":
        raise ValueError("retention.profile must be extraction-v1")
    root = value.get("root")
    if root is not None and (not isinstance(root, str) or not root.strip() or "\0" in root):
        raise ValueError("retention.root must be a nonempty path")
    if value.get("mode") == "required" and root is None:
        raise ValueError("required retention needs an explicit root")


def validate_root(root, library):
    root, library = Path(root).resolve(), Path(library).resolve()
    if root == library or library.is_relative_to(root):
        raise RecordStoreUnavailable("archive must not contain the library")
    for name in ("wiki", "papers", "inbox", ".mutation"):
        content = (library / name).resolve()
        if root.is_relative_to(content) or content.is_relative_to(root):
            raise RecordStoreUnavailable("archive must be separate from canonical content and journals")
    return root


@dataclass(frozen=True)
class Settings:
    mode: str = "off"
    profile: str = "extraction-v1"
    root: Path | None = None

    def identity(self):
        return {"mode": self.mode, "profile": self.profile, "root": str(self.root) if self.root else None}


def selected():
    if _frozen.get() is not None:
        return _frozen.get()
    from ..agents import model_config

    value = model_config._ingest_settings().get("retention", {})
    validate(value)
    mode = value.get("mode", "off")
    root = validate_root(Path(value["root"]).expanduser(), Path.cwd()) if mode == "required" else None
    return Settings(mode, value.get("profile", "extraction-v1"), root)


@contextmanager
def freeze_selected():
    value = selected()
    token = _frozen.set(value)
    try:
        yield value
    finally:
        _frozen.reset(token)
