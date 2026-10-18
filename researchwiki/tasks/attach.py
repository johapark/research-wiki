"""Attach a supplementary file to an existing wiki paper page.

Use this for retroactive adds — when you have a published paper page in
the wiki and want to drop the methods PDF, an extended-data supplement,
or a benchmarks spreadsheet alongside it. New ingests get the same
behavior via `agent ingest --supplementary` (single-PDF invocations only).

The file is copied into `papers/{stem}.supp/{normalized-name}` and the
target page's YAML is updated to declare it under `supplementary:`. The
wiki page body is never touched. A source under `inbox/` is *moved*
rather than copied — see `_consume_inbox_source`.

Usage:
    researchwiki attach {category}/smith-2024-... \\
        ~/Downloads/Brixi_Methods.pdf
    researchwiki attach {category}/smith-2024-... \\
        ~/Downloads/Table_S4.xlsx --kind data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..log import append_log_md, log
from ..paths import supp_dir, wiki_dir
from ..supplementary import (
    VALID_KINDS, _normalize_filename, _consume_inbox_source,
    _insert_supplementary, stage_supplementary, insert_supplementary_entry,
)

# Compatibility exports for callers of the original adapter.
__all__ = ["stage_supplementary", "insert_supplementary_entry", "_insert_supplementary"]


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
    verb = "moved" if _consume_inbox_source(src) else "copied"
    log(f"{verb} {src.name} → papers/{stem}.supp/{target_name}", tag="attach")

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
