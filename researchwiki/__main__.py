"""CLI entry point for `researchwiki` command.

Subcommands are auto-discovered from `researchwiki.tasks.*`. Each task module
must expose a `main(argv: list[str]) -> int` callable; its docstring's first
line becomes the subcommand help string. Module names use underscores; the
CLI name converts those to dashes (e.g. `tasks/eval_classifier.py` →
`researchwiki eval-classifier`).
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from pathlib import Path

from . import __version__, tasks as _tasks


def _load_dotenv() -> None:
    """Load .env from the wiki root into os.environ (explicit vars take precedence)."""
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _discover_tasks() -> dict[str, str]:
    """Map CLI command name → module name, by walking `researchwiki.tasks`.

    Sorted alphabetically so help output is stable. Modules whose name starts
    with `_` are skipped (private helpers, not CLI entry points).
    """
    out: dict[str, str] = {}
    for info in pkgutil.iter_modules(_tasks.__path__):
        if info.name.startswith("_"):
            continue
        cli_name = info.name.replace("_", "-")
        out[cli_name] = info.name
    return dict(sorted(out.items()))


def _build_parser(tasks: dict[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchwiki",
        description="A markdown-first wiki for research papers — LLM-authored, "
                    "claim-graded against source PDFs, refines synthesis pages "
                    "as related work arrives. See CLAUDE.md for the full workflow.",
    )
    parser.add_argument("--version", action="version", version=f"researchwiki {__version__}")
    subs = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for cli_name, module_name in tasks.items():
        try:
            module = importlib.import_module(f"researchwiki.tasks.{module_name}")
        except ImportError as e:
            print(f"Warning: could not load task '{cli_name}': {e}", file=sys.stderr)
            continue
        help_text = (module.__doc__ or "").strip().split("\n", 1)[0]
        subs.add_parser(cli_name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    tasks = _discover_tasks()
    if not argv:
        _build_parser(tasks).print_help(sys.stderr)
        return 2
    command = argv[0]
    if command in ("-h", "--help"):
        _build_parser(tasks).print_help()
        return 0
    if command == "--version":
        print(f"researchwiki {__version__}")
        return 0
    if command not in tasks:
        print(f"researchwiki: unknown command '{command}'. "
              f"Available: {', '.join(tasks)}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"researchwiki.tasks.{tasks[command]}")
    return int(module.main(argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
