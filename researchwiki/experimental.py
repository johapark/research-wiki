"""Maturity markers for surfaces published ahead of their design settling.

The counterpart to `concepts.warn_deprecated`, from the other end of a feature's
life. A deprecated surface is frozen until removal; an experimental one is
exempt from the freeze while it still moves. Both announce themselves the same
way, and for the same reason: `COMPATIBILITY.md` requires machine-readable
stdout — `--json`, a `--result-file` payload — to stay parseable, so a notice
goes to **stderr** and nowhere else.

The marker is not decoration. `COMPATIBILITY.md` applies "from the first beta
release onward" to CLI spellings, JSON keys and versioned artifact schemas, and
changing one of those normally costs a ledger row, a 90-day window and dual
emission. An entry in `data/deprecations.yaml`'s `experimental:` section is what
buys the exemption, so the notice text and the ledger must agree — hence the
loader here rather than a hardcoded string in each call site, and a test that
compares the two.

Each entry names what is exempt *and* what is not. Nothing here can retract a
persisted phase-role identifier: `ingest_iterations.role` values are
append-only permanently, because `db rebuild` cannot reconstruct historical
telemetry from a role nobody writes any more.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import resources
from typing import Any

VISUAL_INSPECTION = "visual-inspection"


@lru_cache(maxsize=1)
def load_experimental() -> dict[str, dict[str, Any]]:
    """`id -> entry` for every active experimental surface.

    Reads the packaged ledger through `importlib.resources`, the way the file's
    own comment promises ("ships in the wheel so automation can inspect the
    same ledger as a source checkout"). An unreadable or malformed ledger
    yields `{}`: a missing maturity marker must not break a working command,
    and the notice degrades to its own wording rather than vanishing.
    """
    try:
        import yaml

        text = (
            resources.files("researchwiki")
            .joinpath("data/deprecations.yaml")
            .read_text(encoding="utf-8")
        )
        data = yaml.safe_load(text)
    except Exception:
        return {}
    rows = (data or {}).get("experimental")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("id") and row.get("status") == "active"
    }


def announced_in(entry_id: str) -> str:
    """The release that published the surface, or `''` when unrecorded."""
    row = load_experimental().get(entry_id) or {}
    return str(row.get("announced_in") or "")


def _since(entry_id: str) -> str:
    version = announced_in(entry_id)
    return f" (since {version})" if version else ""


#: Printed by every command whose behaviour depends on the visual-inspection
#: surface. One constant so the two call sites cannot drift and the
#: compatibility test can match it against the ledger row.
VISUAL_INSPECTION_NOTICE = (
    f"note: visual inspection is experimental{_since(VISUAL_INSPECTION)} — its flags, "
    "receipt shape and JSON may change without a deprecation window, and a figure "
    "reading you have checked yourself still cannot be recorded as approved "
    "(see WORKFLOW.md)."
)


def warn_visual_inspection() -> None:
    """Announce the experimental status on stderr, never stdout.

    Called before argparse at each site, so `--help` and a mistyped flag
    announce it too — the same ordering `concepts.warn_deprecated` relies on.
    """
    print(VISUAL_INSPECTION_NOTICE, file=sys.stderr)
