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
from ..paths import inbox_dir, wiki_dir, wiki_root

# ── Provider wiring ──────────────────────────────────────────────────────────

# Menu order → internal provider id.
_PROVIDER_MENU: list[tuple[str, str, str]] = [
    ("anthropic", "Anthropic cloud", "default, ~$0.10/paper; needs ANTHROPIC_API_KEY"),
    ("openai-compatible", "OpenAI-compatible cloud", "OpenAI / Gemini / Groq / OpenRouter; needs OPENAI_API_KEY + base URL"),
    ("local", "Local LLM", "LM Studio / vLLM / llama.cpp / ollama; free per paper"),
    ("chat-relay", "Chat-relay", "no API key or server; the chat agent fills each prompt"),
]

# Which config/ template each provider copies to config/models.yaml. Chat-relay
# keeps the anthropic template — it is an env override (RW_LLM_PROVIDER), not a
# distinct role config.
_TEMPLATE_BY_PROVIDER: dict[str, str] = {
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

Sort keys fall back to the file's creation/modification time when a page predates the `ingested_at` / `generated_at` stamp, so every page ranks even if it was ingested before those fields existed.

## Recent papers (top 15)

```dataview
TABLE WITHOUT ID
  link(file.link, default(short_name, title)) AS "Paper",
  join(category, ", ") AS "Category",
  year AS "Year",
  dateformat(default(ingested_at, file.ctime), "yyyy-MM-dd") AS "Added"
WHERE type = "paper"
SORT default(ingested_at, file.ctime) DESC
LIMIT 15
```

## Recent synthesis pages (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Synthesis",
  length(referenced_papers) AS "Papers",
  dateformat(default(generated_at, file.mtime), "yyyy-MM-dd") AS "Updated"
WHERE type = "synthesis"
SORT default(generated_at, file.mtime) DESC
LIMIT 10
```

## Recent ideas (top 5)

```dataview
TABLE WITHOUT ID
  file.link AS "Idea",
  verdict AS "Verdict",
  status AS "Status",
  length(referenced_papers) AS "Papers",
  dateformat(default(generated_at, file.mtime), "yyyy-MM-dd") AS "Filed"
WHERE type = "idea"
SORT default(generated_at, file.mtime) DESC
LIMIT 5
```
"""


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────

def _template_for_provider(provider: str) -> str:
    """config/ template filename for a provider id."""
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
    lines = path.read_text().splitlines() if path.exists() else []
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
    for raw in models_yaml.read_text().splitlines():
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

    print("Which LLM provider will you use?")
    for i, (_pid, label, blurb) in enumerate(_PROVIDER_MENU, 1):
        print(f"  {i}. {label} — {blurb}")
    choice = _ask("Choose 1-4", default="1")
    try:
        provider = _PROVIDER_MENU[int(choice) - 1][0]
    except (ValueError, IndexError):
        print(f"'{choice}' isn't a valid choice — defaulting to Anthropic.")
        provider = "anthropic"

    # Copy the matching template → config/models.yaml.
    template = config_dir / _template_for_provider(provider)
    if template.exists():
        if models_yaml.exists() and not _confirm(
            f"Overwrite existing config/models.yaml with the {provider} template?", default=True
        ):
            print("Left config/models.yaml untouched.")
        else:
            shutil.copyfile(template, models_yaml)
            print(f"Wrote config/models.yaml from {template.name}.")
    else:
        print(f"⚠ template {template} not found — skipping config copy. "
              f"You'll need to create config/models.yaml by hand.")

    # Collect + persist required env vars (skip any already set in the shell).
    api_key = base_url = None
    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY already set in your shell — leaving it.")
        else:
            api_key = _ask("Anthropic API key (blank to set later)") or None
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


def _warn_gitignore(root: Path) -> None:
    gi = root / ".gitignore"
    if gi.exists() and ".env" in gi.read_text():
        return
    print("⚠ .env does not appear to be gitignored — add it before committing so your "
          "key doesn't reach GitHub.")


def _report_readiness(provider: str) -> None:
    try:
        from ..agents.llm import has_any_llm, has_synchronous_llm
    except Exception:  # pragma: no cover - defensive; llm deps optional
        return
    ok = has_any_llm() if provider == "chat-relay" else has_synchronous_llm()
    if ok:
        print("✓ Provider looks reachable.")
    else:
        print("… Provider not reachable yet — set the missing key/URL (in .env or your "
              "shell) before the first ingest.")


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

    print("\nTwo ways to seed categories:")
    print("  1. Bootstrap — drop ≥5 PDFs in inbox/ and let the classifier propose a "
          "taxonomy from your actual papers.")
    print("  2. Manual — type the category slugs yourself.")
    choice = _ask("Choose 1-2", default="2")

    if choice == "1":
        _bootstrap_categories()
    else:
        _manual_categories(root)
    _print_category_help()


def _bootstrap_categories() -> None:
    n_pdfs = len(list(inbox_dir().glob("*.pdf")))
    if n_pdfs < 5:
        print(f"Only {n_pdfs} PDF(s) in inbox/ — bootstrap needs ≥5. Drop more PDFs and "
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
    print("\nChange settings later: swap providers by copying another config/models.*.yaml "
          "template (or editing .env); manage categories per the tips above; re-run "
          "`researchwiki init` any time (it's idempotent).")


def main(argv: list[str]) -> int:
    if not sys.stdin.isatty():
        print("`researchwiki init` is interactive — run it in a terminal, or follow the "
              "conversational setup in prompts/init.md.", file=sys.stderr)
        return 2

    root = wiki_root()
    _header("Research Wiki — setup")
    print("This wizard configures your LLM provider and initial categories, scaffolds the "
          "dashboard, and confirms the install. Every choice is reversible.")

    _step_provider(root)
    _step_categories(root)
    _step_dashboard(root)
    _step_confirm()
    return 0
