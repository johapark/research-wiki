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
import re
import sys
import traceback
from pathlib import Path

from . import __version__, tasks as _tasks
from .errors import EnvironmentFailure


_ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _load_dotenv() -> None:
    """Load .env from the wiki root into os.environ (explicit vars take precedence).

    Accept the common ``export NAME=value`` form as well as ``NAME=value``.
    A value that consists solely of ``$NAME`` or ``${NAME}`` reuses an already
    exported variable without copying its secret into the repository.
    """
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        reference = _ENV_REFERENCE.fullmatch(val)
        if reference:
            source = reference.group(1) or reference.group(2)
            if source not in os.environ:
                continue
            val = os.environ[source]
        os.environ.setdefault(key, val)


def _discover_tasks() -> dict[str, str]:
    """Map CLI command name → module name, by walking `researchwiki.tasks`.

    Sorted alphabetically so help output is stable. Modules whose name starts
    with `_` are skipped (private helpers, not CLI entry points).

    Note this cannot be the only filter, and deliberately does not import
    anything: `main()`'s fast path imports exactly one module, and importing all
    41 to inspect them would be paid on every invocation. The complementary
    check — does the module actually expose `main()` — lives at the two places
    the module object already exists (`_build_parser`, and `main()` itself), via
    `_is_entry_point`.
    """
    out: dict[str, str] = {}
    for info in pkgutil.iter_modules(_tasks.__path__):
        if info.name.startswith("_"):
            continue
        cli_name = info.name.replace("_", "-")
        out[cli_name] = info.name
    return dict(sorted(out.items()))


def _is_entry_point(module) -> bool:
    """Whether a task module is actually a CLI command.

    `_discover_tasks` registers every non-underscore module under
    `researchwiki.tasks`, which silently swept up two library modules —
    `claim_discover` (whose `discover_pairs()` backs `candidates pairs`) and
    `pair_dismissals` (the dismissal store behind `--decline`). Both were
    advertised in `--help` and both crashed on dispatch with an AttributeError
    reported as exit 3, "internal bug" — which it was, just not the kind a caller
    could act on.

    The leading-underscore convention is the intended signal and stays valid;
    this is the invariant behind it, checked rather than trusted, so a future
    helper dropped into `tasks/` cannot become a broken command by accident.
    """
    return callable(getattr(module, "main", None))


def _entry_point_names(tasks: dict[str, str]) -> list[str]:
    """Command names that actually dispatch — for error messages and tests."""
    out = []
    for cli_name, module_name in tasks.items():
        try:
            module = importlib.import_module(f"researchwiki.tasks.{module_name}")
        except ImportError:
            continue
        if _is_entry_point(module):
            out.append(cli_name)
    return out


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
        if not _is_entry_point(module):
            continue        # library module under tasks/, not a command
        help_text = (module.__doc__ or "").strip().split("\n", 1)[0]
        subs.add_parser(cli_name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a task module, mapping failures onto the exit-code contract.

    Codes are the ones CLAUDE.md publishes (and `AGENTS.md` symlinks to every
    other agent tool): 0 success, 1 user-input error, 2 environment error,
    3 internal bug. Note this is *not* argparse's convention, where 2 means a
    usage error — a caller uses these codes to decide whether to fix its own
    command line or go inspect the environment, so "you typo'd a flag" must not
    arrive as "the disk is unreadable."

    2 vs 3 is decided by exception *type*, not by where the failure was caught:
    anything deriving from `errors.EnvironmentFailure` is 2, everything else is
    3. Task modules should therefore *not* wrap their work in
    `except Exception: return 2` — that reports genuine bugs as environment
    errors and swallows the traceback. Raise a typed failure at the boundary
    that touched the DB / index / provider instead.
    """
    _load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    tasks = _discover_tasks()
    if not argv:
        _build_parser(tasks).print_help(sys.stderr)
        return 1
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
        return 1
    # Paths resolve relative to cwd (paths.wiki_root), so running from any
    # other directory used to silently operate on a phantom empty repo —
    # `db rebuild` reported 0 pages, `status` an empty wiki, and connection.py
    # minted a fresh per-repo state.db keyed on the wrong cwd. Environment
    # error (exit 2): the fix is `cd` to the wiki root, not editing flags.
    # `init` is exempt — it's the one command meant for a not-yet-a-wiki dir.
    if command != "init" and not (Path.cwd() / "wiki").is_dir():
        print(f"researchwiki {command}: no wiki/ directory under {Path.cwd()} — "
              f"run from the wiki root (or `researchwiki init` to create one here).",
              file=sys.stderr)
        return 2
    try:
        module = importlib.import_module(f"researchwiki.tasks.{tasks[command]}")
    except ImportError as e:
        # A task module that won't import means a missing dependency or a
        # broken install, not a bad command line.
        print(f"researchwiki {command}: cannot load task module — {e}",
              file=sys.stderr)
        return 2
    if not _is_entry_point(module):
        # Reachable because `_discover_tasks` names modules without importing
        # them. A library module under tasks/ is not a command, so this is a bad
        # command line (1), not an internal bug (3).
        print(f"researchwiki: unknown command '{command}'. "
              f"Available: {', '.join(_entry_point_names(tasks))}", file=sys.stderr)
        return 1
    try:
        return int(module.main(argv[1:]) or 0)
    except SystemExit as e:
        # argparse exits 2 for every usage error — bad flag, missing required
        # argument, unparseable type — and all 34 task modules build their own
        # ArgumentParser, so this funnel is the only place the code can be
        # brought back onto the contract. Left alone, `researchwiki db --nope`
        # reported 2 = "environment error" and sent a caller inspecting the disk
        # over a typo. Nothing in the package raises SystemExit(2) of its own
        # (grepped), so the remap is unambiguous; every other code passes
        # through, including the 0 from `--help` inside a subparser.
        code = e.code
        if code is None:
            return 0
        if not isinstance(code, int):
            return 1
        return 1 if code == 2 else code
    except KeyboardInterrupt:
        print(f"\nresearchwiki {command}: interrupted", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # `researchwiki ... | head` closes the pipe early; not a failure.
        return 0
    except EnvironmentFailure as e:
        # Code 2's only reliable source. Whether a failure is environmental is
        # not decidable here — a locked state.db and a KeyError in the grader
        # both arrive as `Exception` — so the classification is made at the
        # boundary that touched the unreliable resource and carried up in the
        # exception type. See researchwiki/errors.py.
        #
        # No traceback: the message is the diagnostic, and these are conditions
        # the user fixes on their machine rather than reports as bugs.
        print(f"researchwiki {command}: {e}", file=sys.stderr)
        return 2
    except Exception:
        # Code 3 is reserved for exactly this. Without the handler an uncaught
        # exception surfaced as Python's own exit 1, which the contract labels
        # a user-input error and would send a caller off editing its flags.
        # The traceback still goes to stderr so debugging is unaffected.
        traceback.print_exc()
        print(f"researchwiki {command}: internal error (see traceback above)",
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
