"""Shared back-link helper.

A "back-link" is a bullet on page B's `## Related Papers` section that says
B was cited by paper A. Forward links — what A says about B — are editorial
and live on A's page. Back-links are structural and live on B's page.

Two callers use this:
  - `agents/promote.py` at ingest commit, for every verified citation-graph
    or topical candidate the new paper has.
  - `tasks/lint.py` `--fix` for the symmetric `missing_backlinks` finding.

Both pass through the same `append_related_paper` to keep the on-disk
convention consistent: same bullet shape, same `(auto-added; refine)` marker,
same idempotency rule (skip if `[[source_key]]` already appears anywhere in
the target body).
"""

from __future__ import annotations

import re
from pathlib import Path

from .fsatomic import update_locked
from .wiki import commit_page


_RELATED_HEADING_RE = re.compile(r"^##\s+Related Papers\s*$", re.MULTILINE | re.IGNORECASE)
_NEXT_HEADING_RE = re.compile(r"^##?\s+", re.MULTILINE)


def append_related_paper(
    target_path: Path, source_key: str,
    note: str = "cites this paper (auto-added; refine)",
) -> bool:
    """Append a back-link bullet to `target_path`'s Related Papers section.

    `source_key` is the `category/stem` of the paper that cites the target.
    `note` is the trailing explanation on the bullet; it defaults to the
    citation-graph phrasing so existing callers (promote, `lint --fix`) are
    unchanged. The claim-overlap cross-linker passes its own marker.

    Idempotent: returns False (no write) if `[[source_key]]` already appears
    anywhere in the target's body. Creates the section if missing. Otherwise
    inserts the bullet at the section's tail (before the next heading).

    Returns True iff the file was modified.
    """
    if not target_path.exists():
        return False

    source_link = f"[[{source_key}]]"
    bullet = f"- {source_link} — {note}"

    def _splice(text: str) -> str:
        # Idempotent: returning `text` unchanged signals update_locked to skip
        # the write. Concurrent ingests editing the same target page are
        # serialized by the flock, so this read-modify-write can't be clobbered.
        if source_link in text:
            return text
        m = _RELATED_HEADING_RE.search(text)
        if not m:
            sep = "\n\n" if not text.endswith("\n") else ("\n" if not text.endswith("\n\n") else "")
            return text + f"{sep}## Related Papers\n\n{bullet}\n"
        section_start = m.end()
        next_m = _NEXT_HEADING_RE.search(text, section_start + 1)
        section_end = next_m.start() if next_m else len(text)
        section_body = text[section_start:section_end].rstrip() + "\n"
        new_body = section_body + bullet + "\n\n"
        return text[:section_start] + "\n" + new_body + text[section_end:]

    changed = update_locked(target_path, _splice, missing_ok=False)
    if changed:
        commit_page(target_path)
    return changed
