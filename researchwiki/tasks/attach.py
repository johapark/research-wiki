"""Attach a supplementary file to an existing wiki paper page.

Use this for retroactive adds — when you have a published paper page in
the wiki and want to drop the methods PDF, an extended-data supplement,
or a benchmarks spreadsheet alongside it. New ingests get the same
behavior via the `--supplementary` flag (S3, not yet shipped).

The file is copied into `papers/{stem}.supp/{normalized-name}` and the
target page's YAML is updated to declare it under `supplementary:`. The
wiki page body is never touched.

Usage:
    researchwiki attach {category}/smith-2024-... \\
        ~/Downloads/Brixi_Methods.pdf
    researchwiki attach {category}/smith-2024-... \\
        ~/Downloads/Table_S4.xlsx --kind data
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from ..fsatomic import write_text_atomic
from ..log import append_log_md, log
from ..paths import supp_dir, wiki_dir


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


def _resolve_page(identifier: str) -> Path | None:
    """Find a wiki page by `category/stem` or bare stem. Returns the .md path."""
    root = wiki_dir()
    if not root.exists():
        return None
    if "/" in identifier:
        cat, stem = identifier.split("/", 1)
        p = root / cat / f"{stem}.md"
        return p if p.exists() else None
    for md in root.rglob(f"{identifier}.md"):
        return md
    return None


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
        return

    block_text = text[block[0]:block[1]]
    if re.search(rf"^\s*- file:\s*{re.escape(filename)}\b", block_text, re.MULTILINE):
        raise FileExistsError(
            f"`{filename}` is already listed under supplementary: in {page_path.name}; "
            f"pass --replace to overwrite the file on disk (YAML stays unchanged)."
        )
    updated = block_text.rstrip("\n") + "\n" + new_entry
    write_text_atomic(page_path, text[:block[0]] + updated + text[block[1]:])


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

    Returns `{filename, kind}` describing what landed on disk.
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
    return {"filename": target_name, "kind": kind}


def insert_supplementary_entry(
    page_path: Path, filename: str, kind: str,
) -> None:
    """Public alias for the in-place YAML inserter. See `_insert_supplementary`."""
    _insert_supplementary(page_path, filename, kind)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki attach",
        description="Attach a supplementary file (PDF, xlsx, etc.) to an "
                    "existing wiki paper page. Copies the file into "
                    "papers/{stem}.supp/ and updates the page's YAML.",
    )
    parser.add_argument("page", help="Page identifier — `category/stem` or bare stem")
    parser.add_argument("file", type=Path, help="Path to the source file")
    parser.add_argument("--kind", choices=VALID_KINDS, default=None,
                        help="Type of attachment. Default: 'methods' for *.pdf, 'data' "
                             "for *.xlsx/*.csv/*.tsv, 'other' otherwise.")
    parser.add_argument("--name", default=None,
                        help="Override the on-disk filename (lowercased, "
                             "[a-z0-9._-]+ enforced). Default: derive from source basename.")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite an existing file at the target name. "
                             "YAML is unchanged when the entry already exists.")
    args = parser.parse_args(argv)

    src: Path = args.file.expanduser().resolve()
    if not src.exists() or not src.is_file():
        log(f"source not found or not a file: {src}", tag="attach")
        return 1

    page_path = _resolve_page(args.page)
    if page_path is None:
        log(f"page not found: `{args.page}` (try `category/stem`)", tag="attach")
        return 1
    stem = page_path.stem

    if args.name:
        target_name = _normalize_filename(args.name)
    else:
        target_name = _normalize_filename(src.name)
    if not target_name or target_name.startswith("."):
        log(f"normalized filename is empty or hidden: {target_name!r}", tag="attach")
        return 1
    if target_name != src.name:
        log(f"renamed `{src.name}` → `{target_name}` to match safe-char rule", tag="attach")

    suffix = Path(target_name).suffix.lower()
    if args.kind is None:
        kind = "methods" if suffix == ".pdf" else (
            "data" if suffix in (".xlsx", ".csv", ".tsv") else "other"
        )
    else:
        kind = args.kind

    sd = supp_dir(stem)
    sd.mkdir(parents=True, exist_ok=True)
    target = sd / target_name
    if target.exists() and not args.replace:
        log(f"target exists: {target}. Pass --replace to overwrite.", tag="attach")
        return 1

    shutil.copy2(str(src), str(target))
    log(f"copied {src.name} → papers/{stem}.supp/{target_name}", tag="attach")

    try:
        _insert_supplementary(page_path, target_name, kind)
        log(f"updated YAML supplementary: in {page_path.relative_to(wiki_dir().parent)}",
            tag="attach")
    except FileExistsError as e:
        log(str(e), tag="attach")
    except Exception as e:
        log(f"YAML update failed: {e}", tag="attach")
        return 1

    append_log_md(
        kind="attach",
        headline=f"{stem} — attached `{target_name}` ({kind})",
        details=f"From: `{src}`\nTo: `papers/{stem}.supp/{target_name}`\n"
                f"Page: `{page_path.relative_to(wiki_dir().parent)}`",
    )
    return 0
