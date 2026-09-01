"""Reviewed migration for legacy pages missing exact author provenance.

Planning never edits a wiki page. It inventories every actionable page, records
its content hash, and preselects only models recoverable from ingest telemetry.
Everything else stays ``pending`` until a maintainer chooses an exact model,
acknowledges that the model was never recorded, or explicitly skips the page.

Apply is deliberately two-phase: it validates every decision and every page
hash before creating a pre-apply tarball or writing anything. Re-running an
already applied manifest is safe and reports the pages as already applied.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..fsatomic import file_sha256, read_json, write_json_atomic, write_text_atomic
from ..paths import ingest_dir, wiki_dir
from ..provenance import (
    LEGACY_AUTHOR_PROVENANCE,
    author_provenance_required,
    has_usable_author_model,
    is_acknowledged_legacy,
    normalized_author_model,
)
from ..wiki import Page, commit_page, read_pages


MANIFEST_VERSION = 1
DECISIONS = frozenset(
    {"pending", "recover", "set-author-model", "acknowledge", "skip"}
)
_TOP_LEVEL_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")


class ProvenanceMigrationError(ValueError):
    """A review manifest or current page state is unsafe to apply."""


def _now() -> dt.datetime:
    return dt.datetime.now()


def _page_hash(page: Page) -> str:
    return file_sha256(page.path)


def _candidate_pages() -> list[Page]:
    return [
        page
        for page in read_pages()
        if author_provenance_required(page.fm)
        and not has_usable_author_model(page.fm)
        and not is_acknowledged_legacy(page.fm)
    ]


def _new_run_dir(base: Path | None = None) -> Path:
    root = base or ingest_dir()
    stamp = _now().strftime("%Y%m%dT%H%M%S%f")
    run = root / f"provenance-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def _manifest_item(page: Page, telemetry: dict[str, str]) -> dict[str, Any]:
    model = normalized_author_model(telemetry.get(page.stem))
    return {
        "page": page.key,
        "page_type": page.page_type,
        "sha256": _page_hash(page),
        "telemetry_author_model": model or None,
        "decision": "recover" if model else "pending",
        "author_model": None,
        "review_note": "",
    }


def create_plan(*, run_base: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Create a review manifest and return ``(run_dir, document)``."""
    from ..tasks.lint.provenance import telemetry_author_models

    pages = _candidate_pages()
    telemetry = telemetry_author_models()
    run = _new_run_dir(run_base)
    items = [_manifest_item(page, telemetry) for page in pages]
    document = {
        "schema_version": MANIFEST_VERSION,
        "created_at": _now().isoformat(timespec="seconds"),
        "status": "review-required" if items else "nothing-to-migrate",
        "instructions": {
            "pending": "Choose set-author-model, acknowledge, or skip.",
            "recover": "Use telemetry_author_model; do not edit that value.",
            "set-author-model": "Set author_model to an exact maintainer-attested ID.",
            "acknowledge": (
                "Use only when no exact model can be recovered; apply writes "
                "author_provenance: legacy-unrecorded and a dated acknowledgment."
            ),
            "skip": "Leave the page unchanged and actionable in lint.",
        },
        "items": items,
    }
    write_json_atomic(run / "manifest.json", document)
    return run, document


def load_manifest(run: Path) -> dict[str, Any]:
    if not run.is_dir():
        raise ProvenanceMigrationError(f"no such provenance run directory: {run}")
    path = run / "manifest.json"
    data = read_json(path, default=None)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ProvenanceMigrationError(f"no usable provenance manifest at {path}")
    if data.get("schema_version") != MANIFEST_VERSION:
        raise ProvenanceMigrationError(
            f"unsupported provenance manifest schema {data.get('schema_version')!r}"
        )
    if any(not isinstance(item, dict) for item in data["items"]):
        raise ProvenanceMigrationError("every provenance manifest item must be an object")
    keys = [str(item.get("page") or "") for item in data["items"]]
    if len(keys) != len(set(keys)):
        raise ProvenanceMigrationError("provenance manifest contains duplicate pages")
    return data


def _path_for_key(key: object) -> Path:
    raw = str(key or "")
    rel = Path(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 2:
        raise ProvenanceMigrationError(f"unsafe page key in manifest: {raw!r}")
    path = wiki_dir().joinpath(*rel.parts[:-1], rel.name + ".md")
    try:
        path.relative_to(wiki_dir())
    except ValueError as exc:  # defensive for unusual Path implementations
        raise ProvenanceMigrationError(f"page escapes wiki/: {raw!r}") from exc
    return path


def _decision(item: dict[str, Any]) -> str:
    value = str(item.get("decision") or "").strip()
    if value not in DECISIONS:
        raise ProvenanceMigrationError(
            f"{item.get('page')}: unknown decision {value!r}; choose {sorted(DECISIONS)}"
        )
    return value


def _desired_model(item: dict[str, Any], decision: str) -> str:
    if decision == "recover":
        model = normalized_author_model(item.get("telemetry_author_model"))
        if not model:
            raise ProvenanceMigrationError(
                f"{item.get('page')}: recover requires telemetry_author_model"
            )
        supplied = normalized_author_model(item.get("author_model"))
        if supplied and supplied != model:
            raise ProvenanceMigrationError(
                f"{item.get('page')}: recover cannot replace telemetry model {model!r}"
            )
        return model
    if decision == "set-author-model":
        model = normalized_author_model(item.get("author_model"))
        if not model:
            raise ProvenanceMigrationError(
                f"{item.get('page')}: set-author-model requires an exact author_model"
            )
        return model
    return ""


def _split_frontmatter(text: str, key: str) -> tuple[list[str], str]:
    if not text.startswith("---\n"):
        raise ProvenanceMigrationError(f"{key}: page has no YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ProvenanceMigrationError(f"{key}: page has unterminated frontmatter")
    return text[4:end].splitlines(), text[end + 5 :]


def _field_positions(lines: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _TOP_LEVEL_FIELD.match(line)
        if match:
            out[match.group(1)] = index
    return out


def _drop_fields(lines: list[str], fields: set[str]) -> list[str]:
    """Drop scalar top-level migration fields, refusing complex YAML values."""
    out: list[str] = []
    dropping = False
    for line in lines:
        match = _TOP_LEVEL_FIELD.match(line)
        if match:
            dropping = match.group(1) in fields
            if dropping:
                continue
        elif dropping and (line.startswith(" ") or line.startswith("\t")):
            raise ProvenanceMigrationError(
                "provenance migration refuses a nested/multiline migration field"
            )
        else:
            dropping = False
        if not dropping:
            out.append(line)
    return out


def _insert_after(lines: list[str], rendered: list[str], anchors: tuple[str, ...]) -> None:
    positions = _field_positions(lines)
    for anchor in anchors:
        if anchor in positions:
            lines[positions[anchor] + 1 : positions[anchor] + 1] = rendered
            return
    lines.extend(rendered)


def _render_page(
    text: str,
    *,
    key: str,
    decision: str,
    model: str,
    acknowledged_at: dt.date,
) -> str:
    lines, body = _split_frontmatter(text, key)
    lines = _drop_fields(
        lines,
        {"author_model", "author_provenance", "provenance_acknowledged_at"},
    )
    if decision in {"recover", "set-author-model"}:
        _insert_after(
            lines,
            [f"author_model: {json.dumps(model, ensure_ascii=False)}"],
            ("keywords", "tags", "year"),
        )
    elif decision == "acknowledge":
        _insert_after(
            lines,
            [
                f"author_provenance: {LEGACY_AUTHOR_PROVENANCE}",
                f"provenance_acknowledged_at: {acknowledged_at.isoformat()}",
            ],
            ("author_model", "keywords", "tags", "year"),
        )
    rendered = "---\n" + "\n".join(lines) + "\n---\n" + body
    try:
        parsed = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as exc:
        raise ProvenanceMigrationError(f"{key}: migration produced invalid YAML") from exc
    if not isinstance(parsed, dict):
        raise ProvenanceMigrationError(f"{key}: migration produced non-mapping YAML")
    return rendered


def _already_desired(
    frontmatter: dict[str, Any],
    *,
    decision: str,
    model: str,
) -> bool:
    if decision in {"recover", "set-author-model"}:
        return normalized_author_model(frontmatter.get("author_model")) == model
    if decision == "acknowledge":
        return is_acknowledged_legacy(frontmatter)
    return decision == "skip"


def _validate_item(
    item: dict[str, Any], acknowledged_at: dt.date, telemetry: dict[str, str]
) -> tuple[Path, str, str, str | None]:
    key = str(item.get("page") or "")
    decision = _decision(item)
    if decision == "pending":
        raise ProvenanceMigrationError(f"{key}: decision is still pending")
    path = _path_for_key(key)
    if decision == "skip":
        return path, decision, "", None
    if not path.is_file():
        raise ProvenanceMigrationError(f"{key}: page no longer exists")
    model = _desired_model(item, decision)
    exact_model = normalized_author_model(telemetry.get(path.stem))
    planned_model = normalized_author_model(item.get("telemetry_author_model"))
    if decision == "recover" and (not exact_model or planned_model != exact_model):
        raise ProvenanceMigrationError(
            f"{key}: telemetry no longer verifies planned recovery {planned_model!r}"
        )
    if decision == "set-author-model" and exact_model and model != exact_model:
        raise ProvenanceMigrationError(
            f"{key}: attested model {model!r} conflicts with exact telemetry "
            f"{exact_model!r}"
        )
    text = path.read_text(encoding="utf-8")
    page = _parse_current_page(path)
    current_model = normalized_author_model(page.fm.get("author_model"))
    if current_model and current_model != model:
        raise ProvenanceMigrationError(
            f"{key}: recorded author_model {current_model!r} conflicts with review"
        )
    if _already_desired(page.fm, decision=decision, model=model):
        return path, decision, model, None
    if file_sha256(path) != str(item.get("sha256") or ""):
        raise ProvenanceMigrationError(
            f"{key}: page changed since planning; create and review a new plan"
        )
    if decision == "acknowledge" and (planned_model or exact_model):
        raise ProvenanceMigrationError(
            f"{key}: exact telemetry exists; use recover instead of acknowledge"
        )
    return path, decision, model, _render_page(
        text,
        key=key,
        decision=decision,
        model=model,
        acknowledged_at=acknowledged_at,
    )


def _parse_current_page(path: Path) -> Page:
    from ..wiki import read_page

    page = read_page(path)
    if page is None:
        raise ProvenanceMigrationError(f"{path}: page has no usable frontmatter")
    return page


def _write_backup(run: Path, paths: list[Path]) -> Path:
    backup = run / "pre-apply.tar.gz"
    if backup.exists():
        raise ProvenanceMigrationError(f"backup already exists: {backup}")
    fd, temp_name = tempfile.mkstemp(prefix=".pre-apply-", suffix=".tar.gz", dir=run)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with tarfile.open(temp, "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(wiki_dir().parent))
        os.replace(temp, backup)
    finally:
        temp.unlink(missing_ok=True)
    return backup


def apply_plan(run: Path) -> dict[str, Any]:
    """Validate and apply one reviewed manifest atomically per page."""
    from ..tasks.lint.provenance import telemetry_author_models

    document = load_manifest(run)
    acknowledged_at = _now().date()
    telemetry = telemetry_author_models()
    prepared = [
        _validate_item(item, acknowledged_at, telemetry)
        for item in document["items"]
    ]
    changed = [row for row in prepared if row[3] is not None]
    backup: Path | None = None
    if changed:
        backup = _write_backup(run, [row[0] for row in changed])
        for path, _decision_name, _model, rendered in changed:
            assert rendered is not None
            write_text_atomic(path, rendered)
            commit_page(path)
    skipped = sum(1 for row in prepared if row[1] == "skip")
    already = len(prepared) - len(changed) - skipped
    result = {
        "run_dir": str(run),
        "changed": len(changed),
        "already_applied": already,
        "skipped": skipped,
        "backup": str(backup) if backup else None,
    }
    document["status"] = "applied"
    document["applied_at"] = _now().isoformat(timespec="seconds")
    document["result"] = result
    write_json_atomic(run / "manifest.json", document)
    return result
