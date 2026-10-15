"""Supplementary file staging and page YAML updates shared by ingest and attach."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .fsatomic import write_text_atomic
from .paths import inbox_dir, supp_dir
from .wiki import commit_page

VALID_KINDS = ("methods", "data", "figures", "other")
SAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+")


def _normalize_filename(name: str) -> str:
    """Lowercase, collapse whitespace to `-`, strip unsafe chars.

    Returns a slug ending in the original extension (or empty if the input
    has no usable characters).
    """
    base = name.strip().lower().replace(" ", "-")
    base = SAFE_NAME_RE.sub("", base)
    base = re.sub(r"-{2,}", "-", base).strip("-_.")
    return base


def _consume_inbox_source(src: Path) -> bool:
    """Delete an already-staged `src` when it lives under `inbox/`. Returns
    whether it was removed.

    Staging copies rather than moves, which is right for the common
    `attach ~/Downloads/Table_S4.xlsx` case — the user's own file is not ours
    to consume. An `inbox/` source is the exception: `inbox/` *is* the ingest
    backlog, so a leftover copy there is not inert. A supplementary `.pdf` gets
    swept up by the next `agent ingest inbox/*.pdf` and ingested as a standalone
    paper (different text → different stem → no collision guard fires), and a
    non-PDF is worse than harmless in the other direction — `status` only globs
    `*.pdf`, so it accumulates unseen forever. Neither is visible to
    `find_supplementary_issues`, which only compares YAML against
    `papers/*.supp/`.

    `src` must already be resolved by the caller, and `inbox_dir()` is resolved
    here, so this holds when `inbox/` is a directory symlink into a synced
    folder (a supported layout). A *file* symlink inside `inbox/` resolves to
    its target outside the inbox and is therefore left alone — the conservative
    outcome, since deleting it would reach into a location the user only linked.

    Never fatal: the file is already staged under `papers/{stem}.supp/`, so a
    failed unlink costs a stale inbox entry, not data.
    """
    try:
        inbox = inbox_dir().resolve()
    except OSError:
        return False
    if not src.is_relative_to(inbox):
        return False
    try:
        src.unlink()
        return True
    except OSError:
        return False


def _existing_supp_block_span(text: str, yaml_start: int, yaml_end: int) -> tuple[int, int] | None:
    """Locate the `supplementary:` block inside YAML by line span [start, end).

    Returns None when the field is absent. Indentation-based: the block ends
    at the next line that's same-or-less indented than the `supplementary:`
    key (or the YAML closing `---`).
    """
    lines = text[yaml_start:yaml_end].splitlines(keepends=True)
    line_starts: list[int] = []
    pos = yaml_start
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)
    line_starts.append(yaml_end)

    for i, ln in enumerate(lines):
        if ln.startswith("supplementary:"):
            for j in range(i + 1, len(lines)):
                stripped = lines[j].lstrip(" ")
                if not stripped or stripped == "\n":
                    continue
                indent = len(lines[j]) - len(stripped)
                if indent == 0:
                    return line_starts[i], line_starts[j]
            return line_starts[i], line_starts[len(lines)]
    return None


def _render_entry(filename: str, kind: str) -> str:
    return (
        f"  - file: {filename}\n"
        f"    kind: {kind}\n"
    )


def _insert_supplementary(
    page_path: Path, filename: str, kind: str,
) -> None:
    """Insert or extend the `supplementary:` YAML field in place.

    Uses text manipulation rather than YAML round-trip so existing
    indentation, quoting, and comments are preserved. Idempotent on the
    *filename* — if a file with this name is already listed, raise so the
    caller can prompt for `--replace`.
    """
    text = page_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{page_path}: missing YAML frontmatter")
    yaml_end = text.find("\n---\n", 4)
    if yaml_end < 0:
        raise ValueError(f"{page_path}: malformed YAML frontmatter (no closing ---)")
    yaml_start = 4
    yaml_end_inclusive = yaml_end  # position of the trailing \n before ---

    block = _existing_supp_block_span(text, yaml_start, yaml_end_inclusive)
    new_entry = _render_entry(filename, kind)

    if block is None:
        insertion = f"supplementary:\n{new_entry}"
        prefix = text[:yaml_end_inclusive]
        suffix = text[yaml_end_inclusive:]
        if not prefix.endswith("\n"):
            prefix += "\n"
        write_text_atomic(page_path, prefix + insertion + suffix)
        commit_page(page_path)
        return

    block_text = text[block[0]:block[1]]
    if re.search(rf"^\s*- file:\s*{re.escape(filename)}\b", block_text, re.MULTILINE):
        raise FileExistsError(
            f"`{filename}` is already listed under supplementary: in {page_path.name}; "
            f"pass --replace to overwrite the file on disk (YAML stays unchanged)."
        )
    updated = block_text.rstrip("\n") + "\n" + new_entry
    write_text_atomic(page_path, text[:block[0]] + updated + text[block[1]:])
    # `supplementary:` is part of the frontmatter the DB mirrors in
    # `raw_frontmatter` — reconcile rather than wait for the next rebuild.
    commit_page(page_path)


def stage_supplementary(
    stem: str,
    src: Path,
    *,
    kind: str | None = None,
    name: str | None = None,
    replace: bool = False,
) -> dict:
    """Copy a file into `papers/{stem}.supp/` without touching any wiki page.

    Used by ingest paths (digest + agent) to stage supplementary files at
    ingest time when the wiki page may not exist yet (digest path) or
    hasn't been written yet (agent, pre-promote). The caller is
    responsible for inserting the resulting entry into the page's YAML
    via `insert_supplementary_entry()` once the page is on disk.

    Returns `{filename, kind, consumed}` describing what landed on disk;
    `consumed` is True when `src` was an `inbox/` entry and was removed
    afterwards (see `_consume_inbox_source`).
    Raises FileNotFoundError if `src` doesn't exist, FileExistsError if
    the target exists and `replace` is False, ValueError on a degenerate
    normalized filename.
    """
    src = src.expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"source not found or not a file: {src}")

    target_name = _normalize_filename(name or src.name)
    if not target_name or target_name.startswith("."):
        raise ValueError(f"normalized filename is empty or hidden: {target_name!r}")

    suffix = Path(target_name).suffix.lower()
    if kind is None:
        kind = "methods" if suffix == ".pdf" else (
            "data" if suffix in (".xlsx", ".csv", ".tsv") else "other"
        )

    sd = supp_dir(stem)
    sd.mkdir(parents=True, exist_ok=True)
    target = sd / target_name
    if target.exists() and not replace:
        raise FileExistsError(
            f"papers/{stem}.supp/{target_name} already exists. "
            f"Pass replace=True to overwrite."
        )
    shutil.copy2(str(src), str(target))
    return {
        "filename": target_name,
        "kind": kind,
        "consumed": _consume_inbox_source(src),
    }


def insert_supplementary_entry(
    page_path: Path, filename: str, kind: str,
) -> None:
    """Public alias for the in-place YAML inserter. See `_insert_supplementary`."""
    _insert_supplementary(page_path, filename, kind)
