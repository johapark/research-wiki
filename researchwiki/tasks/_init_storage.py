"""Storage-layout decision for the first-run wizard.

Content directories are durable user data and may live in a synced folder, so
the wizard must settle their layout before the scaffold creates anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def confirm_storage_layout(
    root: Path,
    *,
    confirm: Callable[..., bool],
) -> bool:
    """Validate an existing layout or ask whether a fresh one should sync.

    The wizard deliberately does not create cross-platform sync links itself:
    the destination is a consequential user choice, and Windows may require a
    documented no-symlink fallback. A sync choice therefore pauses setup with
    concrete instructions. The agent-guided path makes the same choice before
    invoking non-interactive ``--scaffold-only``.
    """
    paths = [root / name for name in ("wiki", "papers", "inbox")]
    present = [path for path in paths if path.exists() or path.is_symlink()]

    if present:
        # Only a path that cannot hold content blocks the rerun: a dangling
        # sync link, or a file where a directory belongs. Replacing either would
        # strand the real content, which is `ensure_scaffold`'s refusal too.
        invalid = [path for path in present if not path.is_dir()]
        if invalid:
            names = ", ".join(path.name for path in invalid)
            print(
                f"Storage layout needs attention: {names} are not usable "
                "directories (a sync target may be missing). No scaffold changes "
                "were made."
            )
            return False
        # Anything else is a layout the user already chose, and rerunning init
        # must not refuse it. The wizard says it is safe to rerun. That covers
        # a deleted empty `inbox/`, which the scaffold simply recreates, and
        # synced `wiki/` + `papers/` beside a local `inbox/`, where only the two
        # content dirs need to sit together in the vault.
        for path in paths:
            if path not in present:
                print(f"  {path.name}/: missing; the scaffold creates it as an "
                      "ordinary directory in this checkout.")
            elif path.is_symlink():
                print(f"  {path.name}/: synced link → {path.resolve()}")
            else:
                print(f"  {path.name}/: ordinary directory")
        linked_missing = any(p.is_symlink() for p in present) and len(present) < len(paths)
        if linked_missing:
            print(
                "To sync a missing directory as well, link it into your synced "
                "folder before continuing (README.md § Sync across computers)."
            )
        print("Using the existing storage layout.")
        return True

    if not confirm("Sync wiki/, papers/, and inbox/ across devices?", default=False):
        print("Using ordinary directories in this checkout.")
        return True

    print(
        "No content directories were created. First create them in your synced "
        "folder and link all three into this checkout:\n\n"
        "  SYNC_ROOT=/absolute/path/to/your/synced/research-wiki\n"
        "  mkdir -p \"$SYNC_ROOT\"/wiki \"$SYNC_ROOT\"/papers \"$SYNC_ROOT\"/inbox\n"
        "  ln -s \"$SYNC_ROOT/wiki\" wiki\n"
        "  ln -s \"$SYNC_ROOT/papers\" papers\n"
        "  ln -s \"$SYNC_ROOT/inbox\" inbox\n\n"
        "Then rerun `researchwiki init`. See README.md § Sync across computers "
        "for Windows and no-symlink alternatives."
    )
    return False
