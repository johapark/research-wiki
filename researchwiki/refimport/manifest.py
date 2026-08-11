"""Run directory and manifest for an import.

    .ingest/import-{stamp}/
        manifest.json    what `inspect` decided, one record per export item
        report.md        the human-readable version, including the fetch list

Deliberately smaller than `migrate`'s equivalent. There is **no journal** and no
staged copy, because this command's only mutation is copying a PDF into
`inbox/`; everything after that is `_ingest_batch`, which already keeps a
crash-safe `checkpoint.json` and is covered by tests. Adding a second progress
record here would be untested crash-safety layered on tested crash-safety, and
the two would drift.

There is also **no `latest_run_dir()`**. `apply` requires `--run` explicitly: a
bare `apply` that silently picks the most recent of several `inspect` runs is a
footgun, and `inspect` prints the exact command with the path filled in anyway.

`inspect` is the only phase that reads the export or the PDFs. Every later phase
is driven by the manifest, so pairing cannot change between phases — with one
deliberate exception, documented on `apply`: whether a stem is *already in the
wiki* is re-checked at copy time, since that is a fact about the world now
rather than a decision made at inspect time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..fsatomic import read_json, write_json_atomic
from ..paths import ingest_dir

MANIFEST_VERSION = 1


@dataclass
class RunDir:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    def write_manifest(self, records: list[dict], *, export_path: Path,
                       export_format: str, pdf_root: Path | None,
                       category: str | None, created_at: str,
                       summary: dict, unclaimed_pdfs: list[str]) -> None:
        write_json_atomic(self.manifest_path, {
            "version": MANIFEST_VERSION,
            "created_at": created_at,
            "export_path": str(export_path),
            "export_format": export_format,
            "pdf_root": str(pdf_root) if pdf_root else None,
            "category": category,
            "summary": summary,
            "unclaimed_pdfs": unclaimed_pdfs,
            "items": records,
        })

    def read_manifest(self) -> dict:
        data = read_json(self.manifest_path, default=None)
        if not isinstance(data, dict) or "items" not in data:
            raise FileNotFoundError(
                f"no usable manifest at {self.manifest_path} — run "
                f"`researchwiki import inspect <export>` first"
            )
        return data


def new_run_dir(stamp: str, *, base: Path | None = None) -> RunDir:
    """Create `.ingest/import-{stamp}/`. Fails if it exists, so two runs can
    never share a directory."""
    root = (base or ingest_dir()) / f"import-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    return RunDir(root)


def open_run_dir(path: Path) -> RunDir:
    if not path.is_dir():
        raise FileNotFoundError(f"no such run directory: {path}")
    return RunDir(path)
