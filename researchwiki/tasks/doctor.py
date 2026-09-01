"""Check whether this checkout is ready to ingest a paper.

The default check is local and free: it does not contact an LLM provider or
download the semantic model.  Pass ``--probe`` to make one explicit, minimal
call through the configured classifier route.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import paths


@dataclass(frozen=True)
class Check:
    level: str                 # ok | warn | block
    label: str
    detail: str
    fix: str | None = None


def _content_checks() -> list[Check]:
    checks: list[Check] = []
    required = (paths.wiki_dir(), paths.papers_dir(), paths.inbox_dir())
    dangling = [p for p in required if p.is_symlink() and not p.exists()]
    if dangling:
        names = ", ".join(p.name for p in dangling)
        checks.append(Check(
            "block", "Content paths", f"dangling symlink(s): {names}",
            "mount the synced folder or repair the symlink target",
        ))

    missing = [p for p in required if not p.is_dir() and p not in dangling]
    if missing:
        names = ", ".join(p.name for p in missing)
        checks.append(Check(
            "block", "Content paths", f"missing director{'y' if len(missing) == 1 else 'ies'}: {names}",
            "run `researchwiki init --scaffold-only` from the repository root",
        ))

    present = [p for p in required if p.is_dir()]
    unwritable = [p for p in present if not os.access(p, os.W_OK)]
    if unwritable:
        names = ", ".join(p.name for p in unwritable)
        checks.append(Check(
            "block", "Content paths", f"not writable: {names}",
            "repair permissions or the synced-folder mount",
        ))
    elif not dangling and not missing:
        checks.append(Check("ok", "Content paths", "wiki/, papers/, and inbox/ are writable"))
    return checks


def _dependency_checks() -> list[Check]:
    if sys.version_info < (3, 10):
        return [Check(
            "block", "Python", f"{sys.version.split()[0]} is unsupported",
            "install Python 3.10 or newer and reinstall researchwiki",
        )]

    checks = [Check("ok", "Python", sys.version.split()[0])]
    required = {
        "pypdfium2": "pypdfium2",
        "tantivy": "tantivy",
        "yaml": "PyYAML",
        "numpy": "numpy",
    }
    missing = [display for module, display in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        checks.append(Check(
            "block", "Dependencies", f"missing: {', '.join(missing)}",
            "reinstall with `pip install -e .`",
        ))
    else:
        checks.append(Check("ok", "Dependencies", "PDF, YAML, and search packages are installed"))
    return checks


def _provider_checks() -> list[Check]:
    try:
        from ..agents import model_config
        from ..agents.llm import missing_provider_credentials

        phases = model_config.list_phases()
        for phase in phases:
            model_config.for_phase(phase)
        config = model_config.config_path()
        source = str(config) if config.exists() else "built-in OpenAI defaults"
        checks = [Check("ok", "Model routing", f"valid ({source})")]
        problems = missing_provider_credentials()
    except Exception as e:
        return [Check(
            "block", "Model routing", str(e),
            "repair the active models config, then rerun `researchwiki doctor`",
        )]

    if problems:
        checks.append(Check(
            "block", "Provider", "\n".join(problems),
            "set the required key in .env or rerun `researchwiki init`",
        ))
    else:
        checks.append(Check("ok", "Provider", "credentials and local routing look usable"))
    return checks


def _semantic_check() -> Check:
    if importlib.util.find_spec("sentence_transformers") is None:
        return Check(
            "warn", "Semantic model", "sentence-transformers is not installed",
            "reinstall the project, or ingest with `--no-semantic`",
        )

    try:
        from ..index.embeddings import DEFAULT_MODEL
    except ImportError as e:
        return Check(
            "warn", "Semantic model", f"semantic runtime dependency is missing: {e}",
            "reinstall the project, or ingest with `--no-semantic`",
        )

    hf_home = Path(os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface"),
    )).expanduser()
    model_dir = hf_home / "hub" / f"models--{DEFAULT_MODEL.replace('/', '--')}"
    if model_dir.is_dir():
        return Check("ok", "Semantic model", f"cached ({DEFAULT_MODEL})")
    return Check(
        "warn", "Semantic model",
        f"{DEFAULT_MODEL} is not cached; first semantic ingest will download it",
        "use `researchwiki add ... --no-semantic` when offline",
    )


def _database_check() -> Check:
    try:
        from ..db.connection import get_connection
        conn = get_connection()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except Exception as e:
        return Check(
            "block", "State DB", str(e),
            "repair the reported path or set RESEARCHWIKI_DB_PATH to a writable location",
        )
    return Check("ok", "State DB", "opens successfully")


def _index_check() -> Check:
    pages = list(paths.wiki_dir().glob("*/*.md")) if paths.wiki_dir().is_dir() else []
    if not pages:
        return Check("ok", "Search index", "will be created after the first paper")
    if paths.search_index_dir().is_dir():
        return Check("ok", "Search index", ".tantivy-index/ exists")
    return Check(
        "warn", "Search index", f"missing for {len(pages)} wiki page(s)",
        "run `researchwiki reindex`",
    )


def _curl_check() -> Check:
    if shutil.which("curl"):
        return Check("ok", "curl", "available for structured-metadata lookups")
    return Check(
        "warn", "curl", "not found; metadata lookups may be unavailable",
        "install curl, or ingest with explicit metadata overrides when needed",
    )


def _probe_check() -> Check:
    try:
        from ..agents import llm, model_config

        cfg = model_config.for_phase("classifier")
        provider = model_config.canonical_provider_id(cfg.provider)
        if provider == "chat-relay":
            return Check(
                "warn", "Provider probe",
                "skipped: chat-relay is interactive and cannot be probed synchronously",
            )
        response = llm.call(
            phase="classifier",
            prompt="Reply with exactly OK.",
            max_tokens=8,
            disable_thinking=True,
        )
        return Check("ok", "Provider probe", f"responded via {response.model}")
    except Exception as e:
        return Check(
            "block", "Provider probe", str(e),
            "check the key, endpoint, model name, quota, and network connection",
        )


def run_checks(*, probe: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks.extend(_dependency_checks())
    checks.extend(_content_checks())
    checks.extend(_provider_checks())
    checks.append(_semantic_check())
    checks.append(_database_check())
    checks.append(_index_check())
    checks.append(_curl_check())
    if probe:
        checks.append(_probe_check())
    return checks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki doctor",
        description="Run local readiness checks without spending tokens.",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="make one minimal classifier call (may use the network and spend tokens)",
    )
    args = parser.parse_args(argv)

    print("Research Wiki doctor")
    checks = run_checks(probe=args.probe)
    icons = {"ok": "OK", "warn": "WARN", "block": "BLOCK"}
    for check in checks:
        detail = check.detail.replace("\n", "\n         ")
        print(f"[{icons[check.level]}] {check.label}: {detail}")

    blocked = [c for c in checks if c.level == "block"]
    warnings = [c for c in checks if c.level == "warn"]
    print()
    if blocked:
        print("BLOCKED")
        for check in blocked:
            if check.fix:
                print(f"Fix: {check.fix}")
        return 1

    print("READY TO INGEST")
    if warnings:
        print(f"{len(warnings)} warning(s) above do not block ingestion.")
    if not args.probe:
        print("Provider connectivity was not tested; use `researchwiki doctor --probe` explicitly.")
    return 0
