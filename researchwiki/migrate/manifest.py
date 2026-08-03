"""Run directory, manifest and journal for a migration.

Layout, mirroring `_ingest_batch`'s batch dir so the two read alike:

    .ingest/migrate-{stamp}/
        manifest.json    what `inspect` decided, one record per source page
        journal.json     which steps `apply` has completed, per page
        staged/          post-rewrite, pre-landing copy of every page
        pre-apply.tar.gz targets `apply` was about to overwrite
        report.md        the human-readable inspect report

`inspect` is the only phase that reads the source directory; every later phase is
driven by `manifest.json`. That's deliberate — it means the section-alias table
cannot change between inspect and apply, so a page can't be assessed under one
mapping and rewritten under another.

**This directory is the rollback path.** It lives under `.ingest/`, which is
gitignored, and `wiki/` + `papers/` are gitignored too — so git cannot undo a
migration. Deleting the run dir destroys the staged copies and the pre-apply
tarball. `--run-dir` puts it somewhere durable for a large run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..fsatomic import read_json, write_json_atomic
from ..paths import ingest_dir

MANIFEST_VERSION = 1
#: Steps `apply` performs per page, in order. The journal records each as it
#: completes, so `--resume` is a set difference.
STEPS: tuple[str, ...] = (
    "stage", "headings", "frontmatter", "relink", "pdf", "page", "commit",
)


@dataclass
class RunDir:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.json"

    @property
    def staged_dir(self) -> Path:
        return self.root / "staged"

    @property
    def backup_path(self) -> Path:
        return self.root / "pre-apply.tar.gz"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    def ensure(self) -> None:
        self.staged_dir.mkdir(parents=True, exist_ok=True)

    # --- manifest ---

    def write_manifest(self, records: list[dict], *, src_dir: Path, pdf_dir: Path,
                       category: str, created_at: str) -> None:
        write_json_atomic(self.manifest_path, {
            "version": MANIFEST_VERSION,
            "created_at": created_at,
            "src_dir": str(src_dir),
            "pdf_dir": str(pdf_dir),
            "category": category,
            "items": records,
        })

    def read_manifest(self) -> dict:
        data = read_json(self.manifest_path, default=None)
        if not isinstance(data, dict) or "items" not in data:
            raise FileNotFoundError(
                f"no usable manifest at {self.manifest_path} — run "
                f"`researchwiki migrate inspect` first"
            )
        return data

    # --- journal ---

    def read_journal(self) -> dict:
        data = read_json(self.journal_path, default=None)
        return data if isinstance(data, dict) else {"version": MANIFEST_VERSION, "items": {}}

    def write_journal(self, journal: dict) -> None:
        """Atomic (tmp + os.replace), so a crash can't truncate it. Called after
        every completed step, matching `_ingest_batch`'s per-completion write."""
        write_json_atomic(self.journal_path, journal)

    def record_step(self, journal: dict, src_key: str, step: str, value: str = "done",
                    **extra) -> None:
        item = journal.setdefault("items", {}).setdefault(src_key, {"steps": {}})
        item["steps"][step] = value
        item.update(extra)
        self.write_journal(journal)

    @staticmethod
    def steps_done(journal: dict, src_key: str) -> dict:
        return (journal.get("items", {}).get(src_key) or {}).get("steps", {})


def new_run_dir(stamp: str, *, base: Path | None = None) -> RunDir:
    """Create `.ingest/migrate-{stamp}/`. Fails if it exists, so two runs can
    never share a directory (same guard as `_ingest_batch._batch_dir_for_new_run`)."""
    root = (base or ingest_dir()) / f"migrate-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    rd = RunDir(root)
    rd.ensure()
    return rd


def open_run_dir(path: Path) -> RunDir:
    if not path.is_dir():
        raise FileNotFoundError(f"no such run directory: {path}")
    return RunDir(path)


def latest_run_dir(base: Path | None = None) -> RunDir | None:
    """Most recent `migrate-*` run dir, or None. Lets `apply` default to the run
    `inspect` just wrote instead of making the user paste a timestamp."""
    root = base or ingest_dir()
    if not root.is_dir():
        return None
    runs = sorted((d for d in root.glob("migrate-*") if d.is_dir()),
                  key=lambda d: d.name)
    return RunDir(runs[-1]) if runs else None
