"""Machine-local repository identity shared by databases and application jobs."""

import hashlib
from pathlib import Path


def repository_key(root: Path | str) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    return f"{root.name}-{digest}"


def repository_runtime_dir(root: Path | str, *, base: Path | str | None = None) -> Path:
    base = Path(base) if base is not None else Path.home() / ".local/share/researchwiki"
    return base.expanduser().resolve() / "repos" / repository_key(root)
