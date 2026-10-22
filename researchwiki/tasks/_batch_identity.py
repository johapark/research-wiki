"""Local, credential-free identity of an ingest batch.

The plan owns the original identity; resume compares it before touching a
checkpoint or dispatching workers. Completed PDFs may have moved into papers/,
so only inputs about to be retried are rehashed. Missing inputs remain the
batch driver's existing unresumable/recovery case.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

from .. import __version__, paths
from ..fsatomic import file_sha256


IDENTITY_VERSION = 1
PLAN_VERSION = 2
_FILE_FLAGS = {"--author-prompt-file": False, "--supplementary": True}


def normalize_file_args(args: list[str]) -> list[str]:
    """Freeze file-valued arguments as absolute paths, including --flag=value."""
    result = list(args)
    index = 0
    while index < len(result):
        flag, separator, value = result[index].partition("=")
        if flag not in _FILE_FLAGS:
            index += 1
            continue
        if separator:
            result[index] = f"{flag}={Path(value).expanduser().resolve()}"
            index += 1
            continue
        index += 1
        while index < len(result) and not result[index].startswith("-"):
            result[index] = str(Path(result[index]).expanduser().resolve())
            index += 1
            if not _FILE_FLAGS[flag]:
                break
    return result


def _file_paths(args: list[str]) -> list[str]:
    """Read the same file-valued flags without treating other values as paths."""
    result = []
    index = 0
    while index < len(args):
        flag, separator, value = args[index].partition("=")
        index += 1
        if flag not in _FILE_FLAGS:
            continue
        if separator:
            result.append(value)
            continue
        while index < len(args) and not args[index].startswith("-"):
            result.append(args[index])
            index += 1
            if not _FILE_FLAGS[flag]:
                break
    return result


def _hash(path: Path, *, allow_missing: bool = False) -> str | None:
    try:
        return file_sha256(path)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise


def _files(args: list[str], *, allow_missing: bool) -> dict:
    return {name: _hash(Path(name), allow_missing=allow_missing)
            for name in _file_paths(args)}


def _framework() -> dict:
    """Fingerprint actual installed source, including uncommitted changes."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\0")
    return {"version": __version__, "source_sha256": digest.hexdigest()}


def _prompts() -> dict:
    root = paths.wiki_root()
    result = {path.relative_to(root).as_posix(): file_sha256(path)
              for path in sorted((root / "prompts").glob("*.md"))}
    # AGENTS.md may be a symlink; hash the instructions it actually exposes.
    for name in ("CLAUDE.md", "AGENTS.md"):
        result[name] = _hash(root / name, allow_missing=True)
    return result


def _models() -> dict:
    """Resolve dispatch settings without probing or recording credentials."""
    from ..agents import llm, model_config

    # A resume in the same process must observe edits to the selected YAML.
    model_config.clear_caches()
    model_config.validate_config()
    phases = {name: asdict(model_config.for_phase(name))
              for name in model_config.list_phases()}
    providers = {cfg["provider"] for cfg in phases.values()}
    endpoints = {}
    if providers & model_config.OPENAI_COMPAT_PROVIDERS:
        endpoints["openai"] = llm.resolve_openai_endpoint().url
    if "anthropic" in providers:
        endpoints["anthropic"] = llm._validated_anthropic_base_url() or "https://api.anthropic.com"
    config = model_config.config_path().resolve()
    pricing = model_config.pricing_path().resolve()
    return {
        "config_path": str(config),
        "config_sha256": _hash(config, allow_missing=True),
        "phases": phases,
        "endpoints": endpoints,
        "ingest_defaults": {"n_drafts": model_config.default_n_drafts(),
                            "target_claims_max_chars": model_config.target_claims_max_chars()},
        "pricing_path": str(pricing),
        "pricing_sha256": _hash(pricing, allow_missing=True),
    }


def _command(plan: dict) -> dict:
    return deepcopy({key: plan.get(key, default) for key, default in (
        ("subcommand", ["agent", "ingest"]), ("inputs", []),
        ("extra_args", []), ("per_input_args", {}),
    )})


def capture(plan: dict, *, pending: list[str] | None = None) -> dict:
    """Snapshot at creation, or re-read the workflow and pending inputs on resume."""
    allow_missing = pending is not None
    inputs = plan["inputs"] if pending is None else pending
    return {
        "version": IDENTITY_VERSION,
        "framework": _framework(),
        "prompts": _prompts(),
        "models": _models(),
        "command": _command(plan),
        "shared_files": _files(plan.get("extra_args", []), allow_missing=allow_missing),
        "inputs": {name: {
            "pdf_sha256": _hash(Path(name), allow_missing=allow_missing),
            "files": _files(plan.get("per_input_args", {}).get(name, []),
                            allow_missing=allow_missing),
        } for name in inputs},
    }


def check_readable(inputs: list[str], extra_args: list[str],
                   per_input_args: dict[str, list[str]]) -> None:
    """Read everything `capture` will, before a caller commits to side effects.

    `import apply` copies its wave into inbox/ before `new_batch` records the
    identity. A read failure there strands the copies, and the next apply skips
    them as already staged instead of dispatching their frozen metadata.
    """
    capture({
        "inputs": inputs,
        "extra_args": normalize_file_args(extra_args),
        "per_input_args": {name: normalize_file_args(args)
                           for name, args in per_input_args.items()},
    })


def _changes(before, after, prefix: str) -> list[str]:
    """Report field names only; never put prompt text or settings in diagnostics."""
    if isinstance(before, dict) and isinstance(after, dict):
        return [change for key in sorted(before.keys() | after.keys())
                for change in _changes(before.get(key), after.get(key), f"{prefix}.{key}")]
    return [prefix] if before != after else []


def check_resume(plan: dict, pending: list[str]) -> list[str] | None:
    """Changed fields; None for a legacy plan with no recoverable identity.

    Read failures propagate as environment errors. A modern plan missing its
    identity fails closed rather than masquerading as a legacy checkpoint.
    """
    before = plan.get("identity")
    if before is None and "version" not in plan:
        return None
    if (plan.get("version") != PLAN_VERSION or not isinstance(before, dict)
            or before.get("version") != IDENTITY_VERSION):
        return ["unsupported or missing batch identity"]
    required = {"framework", "prompts", "models", "command", "shared_files", "inputs"}
    if (not required <= before.keys()
            or any(not isinstance(before[area], dict) for area in required)):
        return ["incomplete batch identity"]
    command_changes = _changes(before["command"], _command(plan), "command")
    if command_changes:
        return command_changes
    after = capture(plan, pending=pending)
    changes = [change for area in sorted(required - {"inputs", "command"})
               for change in _changes(before[area], after[area], area)]
    for name, current in after["inputs"].items():
        if current["pdf_sha256"] is not None:
            changes.extend(_changes(before["inputs"].get(name), current, f"inputs.{name}"))
    return changes
