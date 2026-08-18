"""Interactive first-time setup wizard (provider + categories).

Human-runnable cold-start for a fresh clone. Walks the user through the two
settings a new wiki actually needs — an LLM **provider** and an initial
**category** taxonomy — then scaffolds the Dataview dashboard and confirms via
`status`. Everything it writes is reversible; each step prints how to change it
later.

This is the scripted complement to `prompts/init.md` (the LLM-guided
conversational path). Neither supersedes the other: use this when you want to
drive setup yourself from the terminal; use the prompt when an agent is
driving.

Usage:
  researchwiki init

Interactive only — if stdin is not a TTY it exits 2 with a pointer to
`prompts/init.md`. Idempotent: it detects already-configured steps and offers
to skip or reconfigure each.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from ..categories import PAGE_TYPE_DIRS, content_categories
from ..fsatomic import write_text_atomic
from ..paths import ensure_scaffold, inbox_dir, wiki_dir, wiki_root

# ── Provider wiring ──────────────────────────────────────────────────────────

# Menu order → internal provider id. Mirrors README's *Providers* table, in
# its order, because the two disagreeing is what this menu got wrong before:
# it labelled Anthropic "default, ~$0.10/paper" and offered it as entry 1,
# while README and `model_config._FALLBACK_ROLES` both make OpenAI the default
# at ~$0.01/paper. A user who ran the wizard instead of reading the README was
# steered onto a 10x-dearer setup and told it was the default.
_PROVIDER_MENU: list[tuple[str, str, str]] = [
    ("openai", "OpenAI / ChatGPT", "RECOMMENDED — ~$0.01/paper; needs OPENAI_API_KEY and no config file"),
    ("openai-compatible", "Other OpenAI-compatible cloud", "Gemini / Groq / OpenRouter / Together; needs that provider's key + base URL"),
    ("anthropic", "Anthropic cloud", "highest fidelity, ~$0.10/paper; needs ANTHROPIC_API_KEY"),
    ("local", "Local LLM", "LM Studio / vLLM / llama.cpp / ollama; no key, ~free after the download"),
    ("chat-relay", "Chat-relay", "no API key or server; the chat agent fills each prompt"),
]

# Which config/ template each provider copies to config/models.yaml.
#
# `openai` maps to None on purpose: with no config file the loader falls back
# to `_FALLBACK_ROLES`, which already routes every role to OpenAI — so the
# default path is genuinely zero-config, exactly as README documents it.
# Copying `models.chatgpt.yaml` here would NOT be equivalent: that template
# puts author/critic/judge on gpt-5.6-terra (~$0.071/paper by its own header)
# where the fallback uses gpt-5.6-luna (~$0.009/paper). Writing a file that
# silently costs ~7x more than writing nothing is not a sane default.
#
# Chat-relay keeps the anthropic template — it is an env override
# (RW_LLM_PROVIDER) rather than a distinct role config, but the config still
# supplies the model *names* the relay reports per phase.
_TEMPLATE_BY_PROVIDER: dict[str, str | None] = {
    "openai": None,
    "anthropic": "models.anthropic.yaml",
    "openai-compatible": "models.openai-compatible.yaml",
    "local": "models.lmstudio.yaml",
    "chat-relay": "models.anthropic.yaml",
}

_LOCAL_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

VIEWS_MD_TEMPLATE = """\
---
title: "Wiki Dashboard — Recent Additions"
type: dashboard
tags: [dashboard, dataview]
---

# Wiki Dashboard

Live views of recent additions across the wiki. Rendered by the [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) community plugin — **install and enable Dataview in Obsidian** for the tables below to render. On GitHub (or without the plugin) they show as inert code blocks.

Sort keys are the YAML stamps (`ingested_at` / `generated_at`) only, and a page without one is excluded rather than ranked by a filesystem time. Birthtime is reset by back-link splicing and mtime moves on any edit, so either fallback ranks *recently touched* pages as *recently added* ones — one ingest splicing 12 reciprocal links would send 12 unrelated papers to the top. `researchwiki lint --fix` recovers real stamps from the ingest log where a run exists.

## Recent papers (top 15)

```dataview
TABLE WITHOUT ID
  link(file.link, default(short_name, title)) AS "Paper",
  join(category, ", ") AS "Category",
  year AS "Year",
  dateformat(ingested_at, "yyyy-MM-dd") AS "Added"
FROM ""
WHERE type = "paper" AND ingested_at
SORT ingested_at DESC
LIMIT 15
```

## Recent synthesis pages (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Synthesis",
  topic_seed AS "Topic seed",
  dateformat(generated_at, "yyyy-MM-dd") AS "Updated"
FROM ""
WHERE type = "synthesis" AND generated_at
SORT generated_at DESC
LIMIT 10
```

## Recent ideas (top 5)

```dataview
TABLE WITHOUT ID
  file.link AS "Idea",
  verdict AS "Verdict",
  status AS "Status",
  dateformat(generated_at, "yyyy-MM-dd") AS "Filed"
FROM ""
WHERE type = "idea" AND generated_at
SORT generated_at DESC
LIMIT 5
```
"""


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────

def _template_for_provider(provider: str) -> str | None:
    """config/ template filename for a provider id, or None when the provider
    needs no config file (see `_TEMPLATE_BY_PROVIDER`)."""
    return _TEMPLATE_BY_PROVIDER[provider]


def _env_updates_for_provider(
    provider: str, *, api_key: str | None = None, base_url: str | None = None,
) -> dict[str, str]:
    """The `.env` KEY→value map for a provider, given collected values.

    Only non-empty values are included, so callers can pass whatever the user
    supplied and get back exactly the vars worth writing. Chat-relay carries a
    fixed `RW_LLM_PROVIDER=chat-relay` and no secret."""
    u: dict[str, str] = {}
    if provider == "anthropic":
        if api_key:
            u["ANTHROPIC_API_KEY"] = api_key
    elif provider == "openai":
        # Deliberately no RW_LLM_BASE_URL: the built-in fallback already points
        # at api.openai.com, and README asks users to keep that var out of .env
        # so swapping backends stays a one-line shell export.
        if api_key:
            u["OPENAI_API_KEY"] = api_key
    elif provider == "openai-compatible":
        if api_key:
            u["OPENAI_API_KEY"] = api_key
        if base_url:
            u["RW_LLM_BASE_URL"] = base_url
    elif provider == "local":
        if base_url:
            u["RW_LLM_BASE_URL"] = base_url
    elif provider == "chat-relay":
        u["RW_LLM_PROVIDER"] = "chat-relay"
    return u


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Insert or replace `KEY="val"` lines in a `.env`, preserving every other
    line (comments, blanks, unrelated vars). Creates the file if absent and
    restricts it to mode 0600 since it may hold secrets. No-op on empty
    updates."""
    if not updates:
        return
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f'{key}="{remaining.pop(key)}"')
                continue
        out.append(raw)
    for key, val in remaining.items():
        out.append(f'{key}="{val}"')
    write_text_atomic(path, "\n".join(out).rstrip("\n") + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _valid_slug(s: str) -> bool:
    """A syntactically valid content-category slug: lowercase alnum words
    joined by single hyphens, and not a reserved page-type dir name."""
    return bool(_SLUG_RE.match(s)) and s not in PAGE_TYPE_DIRS


# ── Interactive I/O ──────────────────────────────────────────────────────────

def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return ans or (default or "")


def _ask_choice(n_options: int, default: str = "1") -> int:
    """Prompt until the answer is one of 1..n_options; return a 0-based index.

    Re-prompts on bad input rather than falling back to a default. The old
    behavior resolved anything unparseable to menu entry 1 with a printed
    "defaulting to Anthropic" — so a slipped keystroke silently chose the
    dearest provider on the list. A menu that costs money per wrong answer
    should ask again.

    `_ask` returns the default on EOF, and the default is always a valid
    choice, so this terminates on a closed stdin instead of spinning.
    """
    while True:
        raw = _ask(f"Choose 1-{n_options}", default=default).strip()
        try:
            i = int(raw)
        except ValueError:
            print(f"  '{raw}' isn't a number — enter 1-{n_options}.")
            continue
        if 1 <= i <= n_options:
            return i - 1
        print(f"  {i} is out of range — enter 1-{n_options}.")


def _confirm(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    ans = _ask(f"{prompt} [{d}]").strip().lower()
    if not ans:
        return default
    return ans.startswith("y")


def _header(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ── Steps ────────────────────────────────────────────────────────────────────

def _current_provider(models_yaml: Path) -> str | None:
    """Best-effort read of the active provider from config/models.yaml — the
    first `provider:` value. Returns None if the file is absent/unreadable."""
    if not models_yaml.exists():
        return None
    for raw in models_yaml.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("provider:"):
            return s.split(":", 1)[1].strip().strip("\"'")
    return None


def _step_provider(root: Path) -> None:
    _header("Step 1 — LLM provider")
    config_dir = root / "config"
    models_yaml = config_dir / "models.yaml"

    current = _current_provider(models_yaml)
    if current:
        print(f"config/models.yaml already routes to `{current}`.")
        if not _confirm("Reconfigure the provider?", default=False):
            print("Keeping the current provider.")
            return
    elif os.environ.get("OPENAI_API_KEY"):
        # No config file *is* the configured state for the default provider, so
        # a re-run has to recognize it — otherwise the wizard re-asks a user who
        # is already set up and reads as though nothing took.
        print("No config/models.yaml — the built-in defaults route every role to "
              "OpenAI, and OPENAI_API_KEY is set. You're already configured.")
        if not _confirm("Reconfigure the provider?", default=False):
            print("Keeping the built-in OpenAI defaults.")
            return

    print("Which LLM provider will you use?")
    for i, (_pid, label, blurb) in enumerate(_PROVIDER_MENU, 1):
        print(f"  {i}. {label} — {blurb}")
    provider = _PROVIDER_MENU[_ask_choice(len(_PROVIDER_MENU))][0]

    _write_models_config(config_dir, models_yaml, provider)

    # Collect + persist required env vars (skip any already set in the shell).
    api_key = base_url = None
    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY already set in your shell — leaving it.")
        else:
            api_key = _ask("Anthropic API key (blank to set later)") or None
    elif provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY already set in your shell — leaving it.")
        else:
            api_key = _ask("OpenAI API key (blank to set later)") or None
    elif provider == "openai-compatible":
        if os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY already set in your shell — leaving it.")
        else:
            api_key = _ask("Provider API key, forwarded as Bearer (blank to set later)") or None
        base_url = _ask("Provider base URL", default=os.environ.get("RW_LLM_BASE_URL") or _OPENAI_DEFAULT_BASE_URL)
    elif provider == "local":
        base_url = _ask("Local server base URL", default=os.environ.get("RW_LLM_BASE_URL") or _LOCAL_DEFAULT_BASE_URL)
    elif provider == "chat-relay":
        print("Chat-relay needs no key. Read prompts/chat-relay.md for the relay protocol before your first ingest.")

    updates = _env_updates_for_provider(provider, api_key=api_key, base_url=base_url)
    if updates:
        env_path = root / ".env"
        _upsert_env(env_path, updates)
        os.environ.update(updates)  # so the readiness check below sees them
        print(f"Wrote {', '.join(updates)} to .env (mode 600).")
        _warn_gitignore(root)
        if "RW_LLM_PROVIDER" in updates:
            print("Note: RW_LLM_PROVIDER in .env is a GLOBAL override — it forces every "
                  "role to this provider and defeats per-role mixing. Comment it out "
                  "later if you want to mix providers via config/models.yaml.")
        if "RW_LLM_BASE_URL" in updates:
            print("Note: RW_LLM_BASE_URL is routing config — you can move it to a shell "
                  "export later so switching backends doesn't touch .env.")

    _report_readiness(provider)


def _write_models_config(config_dir: Path, models_yaml: Path, provider: str) -> None:
    """Put `config/models.yaml` into the state the chosen provider needs.

    For every provider but OpenAI that means copying a template. For OpenAI it
    means the *absence* of the file, since the built-in fallback already routes
    every role there — so an existing `models.yaml` left over from a previous
    run has to go, or it silently overrides the choice just made and the wizard
    reports success for a provider the user didn't pick.
    """
    template_name = _template_for_provider(provider)

    if template_name is None:
        if not models_yaml.exists():
            print("No config/models.yaml needed — the built-in defaults already "
                  "route every role to OpenAI.")
            return
        if _confirm("Remove the existing config/models.yaml so the built-in "
                    "OpenAI defaults apply?", default=True):
            models_yaml.unlink()
            print("Removed config/models.yaml — built-in OpenAI defaults now apply.")
        else:
            print("⚠ Left config/models.yaml in place. It overrides this choice — "
                  "whatever providers it names are what will actually run.")
        return

    template = config_dir / template_name
    if not template.exists():
        print(f"⚠ template {template} not found — skipping config copy. "
              f"You'll need to create config/models.yaml by hand.")
        return
    if models_yaml.exists() and not _confirm(
        f"Overwrite existing config/models.yaml with the {provider} template?", default=True
    ):
        print("Left config/models.yaml untouched.")
        return
    shutil.copyfile(template, models_yaml)
    print(f"Wrote config/models.yaml from {template.name}.")


def _warn_gitignore(root: Path) -> None:
    gi = root / ".gitignore"
    if gi.exists() and ".env" in gi.read_text(encoding="utf-8"):
        return
    print("⚠ .env does not appear to be gitignored — add it before committing so your "
          "key doesn't reach GitHub.")


def _report_readiness(provider: str) -> None:
    """Report whether the provider just configured can actually run.

    Uses the same provider-aware check `agent ingest` preflights with, so the
    wizard's verdict and the first ingest's outcome can't disagree. The check
    this replaced (`has_synchronous_llm`) answered "is any key set anywhere",
    and so printed a ✓ for an Anthropic key against an OpenAI-routed config —
    the precise mix-up this step exists to catch.
    """
    try:
        from ..agents import model_config as _mc
        from ..agents.llm import missing_provider_credentials
    except Exception:  # pragma: no cover - defensive; llm deps optional
        return
    # config/models.yaml was just written or removed, and each of these reads
    # it behind an lru_cache — without clearing, the verdict describes the
    # config as it was when the process started.
    for fn in (_mc._config, _mc.base_url, _mc._ingest_settings):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear:
            cache_clear()

    problems = missing_provider_credentials()
    if not problems:
        if provider == "chat-relay":
            print("✓ Chat-relay configured — no key needed; a chat agent answers "
                  "each prompt from .llm-relay/pending/.")
        else:
            print("✓ Provider configured — every role has the credentials it needs.")
        return
    for p in problems:
        print(f"… Not ready yet — {p}")


def _step_categories(root: Path) -> None:
    _header("Step 2 — Initial categories")
    existing = sorted(content_categories())
    print(f"Current content categories: {existing}")
    print("A category is just a `wiki/<slug>/` directory. Classify by METHOD, not topic; "
          "`other` is always present as the abstention bucket.")
    if existing != ["other"]:
        if not _confirm("Add or propose more categories now?", default=False):
            _print_category_help()
            return

    from .bootstrap_categories import MIN_INBOX_FOR_BOOTSTRAP
    print("\nTwo ways to seed categories:")
    print(f"  1. Bootstrap — drop ≥{MIN_INBOX_FOR_BOOTSTRAP} PDFs in inbox/ and let the "
          f"classifier propose a taxonomy from your actual papers.")
    print("  2. Manual — type the category slugs yourself.")

    if _ask_choice(2, default="2") == 0:
        _bootstrap_categories()
    else:
        _manual_categories(root)
    _print_category_help()


def _bootstrap_categories() -> None:
    # Import the threshold rather than restating it: this used to hardcode 5
    # against the real value of 3, so users with 3-4 PDFs were told bootstrap
    # was unavailable when it would have worked.
    from .bootstrap_categories import MIN_INBOX_FOR_BOOTSTRAP
    n_pdfs = len(list(inbox_dir().glob("*.pdf")))
    if n_pdfs < MIN_INBOX_FOR_BOOTSTRAP:
        print(f"Only {n_pdfs} PDF(s) in inbox/ — bootstrap needs "
              f"≥{MIN_INBOX_FOR_BOOTSTRAP}. Drop more PDFs and "
              f"re-run `researchwiki bootstrap-categories --apply`, or set categories "
              f"manually now.")
        if _confirm("Set categories manually instead?", default=True):
            _manual_categories(wiki_root())
        return
    print("Running the taxonomy proposer (this calls your provider)…")
    from . import bootstrap_categories
    bootstrap_categories.main(["--apply"])


def _manual_categories(root: Path) -> None:
    print("\nNaming rules: lowercase, hyphen-separated, ASCII. Pick DURABLE cuts:")
    print("  • methods/techniques — e.g. prime-editing, transformer-models, differential-privacy")
    print("  • fields/disciplines — e.g. immunology, rna-biology, computer-vision")
    print("  Avoid transient topic slugs (alphafold-papers, chatgpt-papers) that age when "
          "the field moves. `other` is added automatically.")
    raw = _ask("Category slugs, comma-separated (blank to skip)")
    slugs = [s.strip() for s in raw.split(",") if s.strip()]
    if not slugs:
        print("No categories entered — skipping.")
        return
    created, rejected = [], []
    for s in slugs:
        if not _valid_slug(s):
            rejected.append(s)
            continue
        d = wiki_dir() / s
        d.mkdir(parents=True, exist_ok=True)
        created.append(s)
    if created:
        print(f"Created: {', '.join(sorted(set(created)))}")
        print("Run `researchwiki reindex` to pick up the new directories.")
    if rejected:
        print(f"⚠ Skipped invalid or reserved names: {', '.join(rejected)} "
              f"(must be lowercase-hyphenated and not a page-type dir).")


def _print_category_help() -> None:
    print("\nHow to change categories later:")
    print("  • Add one:    mkdir wiki/<slug>/  (then `researchwiki reindex`) — or ask your LLM.")
    print("  • Propose from papers: `researchwiki bootstrap-categories` (print-only) / `--apply`.")
    print("  • When `other` grows: `researchwiki suggest-splits` proposes promotions.")
    print("  • Move a paper: follow prompts/recategorize.md.")


def _step_dashboard(root: Path) -> None:
    _header("Step 3 — Dashboard")
    views = wiki_dir() / "views.md"
    if views.exists():
        print("wiki/views.md already exists — leaving it.")
        return
    wiki_dir().mkdir(parents=True, exist_ok=True)
    write_text_atomic(views, VIEWS_MD_TEMPLATE)
    print("Wrote wiki/views.md (recent papers / synthesis / ideas).")
    print("It renders only inside Obsidian with the Dataview plugin enabled; on GitHub "
          "the blocks show as inert code.")


def _step_confirm() -> None:
    _header("Step 4 — Confirm")
    try:
        from . import status
        status.main([])
    except Exception as e:  # pragma: no cover - status is best-effort here
        print(f"(couldn't run status: {e})")
    print("\nNext: drop a PDF in inbox/ and run")
    print("    researchwiki agent ingest inbox/<file>.pdf")
    print("or just tell your LLM \"ingest the PDFs in inbox/\".")
    print("\nChange settings later: swap providers by copying another "
          "config/models.*.yaml template over config/models.yaml — or delete that "
          "file to fall back to the built-in OpenAI defaults; keys live in .env; "
          "manage categories per the tips above; re-run `researchwiki init` any "
          "time (it's idempotent).")


def _scaffold(quiet: bool = False) -> int:
    """Create the gitignored content dirs. Shared by the wizard and
    `--scaffold-only`, which is the non-interactive entry point (the LLM-guided
    setup in prompts/init.md has no TTY, so it cannot run the wizard)."""
    try:
        created = ensure_scaffold()
    except FileExistsError as e:
        print(f"researchwiki init: {e}", file=sys.stderr)
        return 2
    if quiet:
        return 0
    if created:
        rel = sorted(str(p.relative_to(wiki_root())) for p in created)
        print(f"Created: {', '.join(rel)}")
    else:
        print("Content directories already present — nothing to create.")
    return 0


def main(argv: list[str]) -> int:
    if "--scaffold-only" in argv:
        return _scaffold()

    if not sys.stdin.isatty():
        print("`researchwiki init` is interactive — run it in a terminal, or follow the "
              "conversational setup in prompts/init.md. To only create the content "
              "directories (no prompts), use `researchwiki init --scaffold-only`.",
              file=sys.stderr)
        return 2

    root = wiki_root()
    _header("Research Wiki — setup")
    print("This wizard configures your LLM provider and initial categories, scaffolds the "
          "dashboard, and confirms the install. Every choice is reversible.")

    rc = _scaffold(quiet=True)
    if rc:
        return rc

    _step_provider(root)
    _step_categories(root)
    _step_dashboard(root)
    _step_confirm()
    return 0
