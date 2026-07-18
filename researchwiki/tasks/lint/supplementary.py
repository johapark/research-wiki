"""Supplementary-file consistency between page YAML and `papers/*.supp/`.

Catches two failure modes:
  - YAML lists a file under `supplementary:` that's not on disk.
  - A file exists under `papers/{stem}.supp/` that no page references.
"""

from __future__ import annotations

from pathlib import Path

from ...paths import papers_dir, supp_dir
from .walk import page_key


def _yaml_supplementary_files(fm: dict) -> list[str]:
    """Extract supplementary filenames from a page's YAML, tolerating the
    three observed shapes:

      - `supplementary: [{file: name.pdf, kind: ..., ...}, ...]` (canonical)
      - `supplementary: [name.pdf, ...]` (legacy hand-authored)
      - `supplementary: name.pdf` (single string)
    """
    raw = fm.get("supplementary")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.strip()]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            f = item.get("file") or item.get("filename")
            if isinstance(f, str) and f.strip():
                out.append(f.strip())
    return out


def find_supplementary_issues(
    pages: list[Path], pages_fm: dict[Path, dict],
) -> tuple[list[dict], list[dict]]:
    """Return (yaml_missing_on_disk, orphaned_on_disk).

    - `yaml_missing_on_disk`: YAML lists a file under `supplementary:` that
      is not present in `papers/{stem}.supp/`.
    - `orphaned_on_disk`: a file exists under `papers/*.supp/` that no
      page's YAML references.
    """
    yaml_missing: list[dict] = []
    yaml_listed_per_stem: dict[str, set[str]] = {}

    for md in pages:
        fm = pages_fm.get(md, {})
        files = _yaml_supplementary_files(fm)
        if not files:
            continue
        stem = md.stem
        sd = supp_dir(stem)
        missing = [f for f in files if not (sd / f).exists()]
        if missing:
            yaml_missing.append({"page": page_key(md), "missing": missing})
        yaml_listed_per_stem[stem] = set(files)

    orphans: list[dict] = []
    pdir = papers_dir()
    if pdir.exists():
        for d in sorted(pdir.glob("*.supp")):
            if not d.is_dir():
                continue
            stem = d.name[:-len(".supp")]
            on_disk = {f.name for f in d.iterdir() if f.is_file()}
            listed = yaml_listed_per_stem.get(stem, set())
            unlisted = sorted(on_disk - listed)
            if unlisted:
                orphans.append({"stem": stem, "files": unlisted})

    return yaml_missing, orphans
