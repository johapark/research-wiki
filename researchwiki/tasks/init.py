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

import ipaddress
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.parse
from pathlib import Path

from ..categories import PAGE_TYPE_DIRS, content_categories
from ..env_profiles import (
    ACTIVE_ENV_FILE_VAR,
    commit_profile_and_config,
    credential_keys,
    credential_keys_in_text,
    edit_profile_text,
    effective_assignment_value,
    parse_assignment,
    require_private_credentials,
    snapshot_profile,
    write_profile_atomic,
)
from ..errors import EnvironmentFailure
from ..fsatomic import write_text_atomic
from ..package_resources import model_template_text
from ..paths import ensure_scaffold, inbox_dir, wiki_dir, wiki_root
from ._provider_setup import (
    choose_endpoint_api_key as _choose_endpoint_api_key,
    profile_controls_env_key as _profile_controls_env_key,
    same_endpoint as _same_endpoint,
)

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
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ACTIVE_ENV_FILE_VAR = ACTIVE_ENV_FILE_VAR

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
        # The wizard writes this endpoint into config/models.yaml. Keeping a
        # second global override in .env would unexpectedly defeat the next
        # named config the user selects.
    elif provider == "local":
        if base_url:
            u["RW_LLM_BASE_URL"] = base_url
    elif provider == "chat-relay":
        u["RW_LLM_PROVIDER"] = "chat-relay"
    return u


def _dotenv_assignment(raw: str) -> tuple[str, str, bool] | None:
    """Parse one simple dotenv assignment as ``(key, value, exported)``.

    This deliberately mirrors ``__main__._load_dotenv`` rather than growing a
    second, more permissive dotenv dialect in the setup writer.
    """
    try:
        assignment = parse_assignment(raw, path=Path("<env>"), line_no=1)
    except EnvironmentFailure:
        return None
    if assignment is None:
        return None
    return assignment.key, assignment.value, assignment.exported


def _dotenv_value(path: Path, wanted: str) -> str | None:
    """The effective value for ``wanted`` using the current ambient environment."""
    return effective_assignment_value(snapshot_profile(path), wanted)


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Insert or replace `KEY="val"` lines in a `.env`, preserving every other
    line (comments, blanks, unrelated vars). Creates the file if absent and
    restricts it to mode 0600 since it may hold secrets. No-op on empty
    updates."""
    if not updates:
        return
    snapshot = snapshot_profile(path)
    text, _removed = edit_profile_text(snapshot, updates=updates)
    write_profile_atomic(path, text)


def _remove_env_keys(path: Path, keys: set[str]) -> set[str]:
    """Remove routing keys from a dotenv file without exposing their values."""
    if not keys:
        return set()
    snapshot = snapshot_profile(path)
    text, removed = edit_profile_text(snapshot, removals=keys)
    if removed:
        write_profile_atomic(path, text)
    return removed


def _stale_routing_keys(provider: str) -> set[str]:
    """Global overrides that would defeat the provider just selected."""
    # The wizard configures the canonical `config/models.yaml` (or the built-in
    # no-file OpenAI fallback), so a named-config override always defeats it.
    stale: set[str] = {"RW_MODELS_CONFIG"}
    if provider != "chat-relay":
        stale.add("RW_LLM_PROVIDER")
    # Local setup intentionally writes its chosen endpoint to this override.
    # Every cloud config either owns its endpoint or does not use this variable.
    if provider != "local":
        stale.add("RW_LLM_BASE_URL")
    # The Anthropic SDK reads this independently of models.yaml. The menu's
    # `anthropic` choice means Anthropic's official cloud, so a profile-owned
    # compatibility endpoint must not survive and receive that provider's key.
    if provider == "anthropic":
        stale.add("ANTHROPIC_BASE_URL")
    return stale


def _active_env_path(root: Path) -> Path:
    """The explicit profile loaded by the CLI, else the root ``.env``."""
    selected = os.environ.get(_ACTIVE_ENV_FILE_VAR)
    return Path(selected) if selected else root / ".env"


def _external_routing_keys(env_path: Path) -> set[str]:
    """Routing overrides that came from outside the selected dotenv profile."""
    routing = {
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
    }
    return {
        key for key in routing
        if key in os.environ and not _profile_controls_env_key(env_path, key)
    }


def _loopback_host(hostname: str) -> bool:
    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _valid_provider_base_url(value: str, *, local_only: bool = False) -> bool:
    """True for a safe request-compatible HTTP(S) base endpoint.

    Bearer credentials require HTTPS off-host.  The local provider is stricter:
    it may target loopback only, so choosing the friendly "local" label cannot
    silently send prompts or a retained token to another machine on the LAN.
    """
    try:
        parsed = urllib.parse.urlsplit(value)
        # Accessing `.port` performs the range/type validation that urlsplit
        # itself defers (``:bad`` and ``:99999`` otherwise look valid).
        port = parsed.port
    except ValueError:
        return False
    structurally_valid = (
        value == value.strip()
        and not any(
            char.isspace() or unicodedata.category(char).startswith("C")
            for char in value
        )
        and parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.netloc.endswith(":")
        and (port is None or port > 0)
        and not parsed.query
        and not parsed.fragment
    )
    if not structurally_valid:
        return False
    is_loopback = _loopback_host(parsed.hostname or "")
    return (not local_only or is_loopback) and (
        parsed.scheme == "https" or is_loopback
    )


def _effective_openai_base_url() -> str | None:
    """Best-effort endpoint before the wizard changes any routing state."""
    if os.environ.get("RW_LLM_BASE_URL"):
        return os.environ["RW_LLM_BASE_URL"]
    try:
        from ..agents import model_config
        return model_config.base_url()
    except Exception:
        # An unavailable explicit config is precisely something `init` should
        # be able to repair, so do not block the wizard while describing it.
        return None


def _effective_anthropic_base_url() -> str:
    """SDK endpoint before the wizard changes Anthropic routing state."""
    return os.environ.get("ANTHROPIC_BASE_URL") or _ANTHROPIC_DEFAULT_BASE_URL


def _effective_provider(models_yaml: Path) -> tuple[str, str] | None:
    """Provider and source that currently win after environment precedence."""
    forced = (os.environ.get("RW_LLM_PROVIDER") or "").strip()
    if forced:
        return forced, "RW_LLM_PROVIDER"

    if "RW_MODELS_CONFIG" in os.environ:
        selected = (os.environ.get("RW_MODELS_CONFIG") or "").strip()
        if not selected:
            return "unavailable", "RW_MODELS_CONFIG=''"
        try:
            from ..agents import model_config
            selected_path = model_config.config_path()
        except Exception:
            selected_path = models_yaml.parent / selected
        provider = _current_provider(selected_path)
        return provider or "unavailable", f"RW_MODELS_CONFIG={selected!r}"

    provider = _current_provider(models_yaml)
    if provider:
        return provider, "config/models.yaml"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "built-in defaults"
    return None


def _choose_openai_api_key(
    env_path: Path,
    *,
    previous_endpoint: str | None,
    selected_endpoint: str,
    prompt: str,
) -> tuple[bool, str | None]:
    """Choose whether/how to use ``OPENAI_API_KEY`` at a selected endpoint.

    Returns ``(proceed, replacement)``. ``replacement=None`` means retain the
    existing value (or leave the key unset); a false ``proceed`` leaves both
    config and env files untouched.
    """
    return _choose_endpoint_api_key(
        env_path,
        env_key="OPENAI_API_KEY",
        previous_endpoint=previous_endpoint,
        selected_endpoint=selected_endpoint,
        prompt=prompt,
        ask=_ask,
        confirm=_confirm,
    )


def _choose_anthropic_api_key(
    env_path: Path,
    *,
    previous_endpoint: str | None,
    prompt: str,
) -> tuple[bool, str | None]:
    """Move an Anthropic credential to the official endpoint only by consent."""
    return _choose_endpoint_api_key(
        env_path,
        env_key="ANTHROPIC_API_KEY",
        previous_endpoint=previous_endpoint,
        selected_endpoint=_ANTHROPIC_DEFAULT_BASE_URL,
        prompt=prompt,
        ask=_ask,
        confirm=_confirm,
    )


def _prepare_local_api_key(
    env_path: Path,
    *,
    previous_endpoint: str | None,
    selected_endpoint: str,
) -> tuple[bool, bool]:
    """Decide whether a pre-existing OpenAI key may reach a local endpoint.

    The OpenAI-compatible transport forwards ``OPENAI_API_KEY`` as a Bearer
    token even to loopback servers.  A cloud credential must therefore never
    follow an endpoint change merely because it happens to be present in the
    process.  The second return value asks the caller to remove a profile-owned
    key only after the new models config has been accepted.
    """
    existing = os.environ.get("OPENAI_API_KEY")
    if not existing or _same_endpoint(previous_endpoint, selected_endpoint):
        return True, False

    if _confirm(
        "The endpoint changed. Reuse the currently set OPENAI_API_KEY at this "
        "local endpoint? It will be sent as a Bearer token", default=False
    ):
        shadowed = _dotenv_value(env_path, "OPENAI_API_KEY")
        if shadowed is not None and shadowed != existing:
            print(
                "… Provider setup cancelled — the selected profile contains a "
                "different OPENAI_API_KEY shadowed by the parent shell. Unset the "
                "shell key and rerun init before changing endpoints."
            )
            return False, False
        print("Reusing OPENAI_API_KEY at the newly selected local endpoint.")
        return True, False

    if not _profile_controls_env_key(env_path, "OPENAI_API_KEY"):
        print(
            "… Provider setup cancelled — OPENAI_API_KEY comes from the parent "
            "shell (or another higher-precedence source), so this wizard cannot "
            "stop it reaching the local endpoint. Unset it there, then rerun init."
        )
        return False, False

    if not _confirm(
        f"Remove OPENAI_API_KEY from {env_path.name} for the local endpoint?",
        default=True,
    ):
        print("… Provider setup cancelled — the existing routing was left unchanged.")
        return False, False
    return True, True


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


def _customize_openai_compatible_text(
    template_text: str,
    *,
    base_url: str,
    quality_model: str,
    utility_model: str,
) -> str:
    """Return a validated generic template customized for one real provider.

    The checked-in template is a documented Gemini example. Copying it while
    changing only the endpoint creates invalid pairs such as Groq + Gemini
    model IDs, so the wizard must rewrite both halves of the routing decision.
    Comments and per-role budgets stay intact; JSON strings are valid YAML
    scalars and safely preserve model IDs containing punctuation. Validation is
    deliberately in-memory so template drift cannot destroy an active config.
    """
    lines = template_text.splitlines()
    out: list[str] = []
    in_roles = False
    current_role: str | None = None
    replaced_url = False
    replaced_roles: set[str] = set()
    quality_roles = {"author", "critic", "judge"}
    expected_roles = quality_roles | {"classifier", "proposer", "extractor"}

    for raw in lines:
        stripped = raw.strip()
        if raw.startswith("base_url:"):
            out.append(f"base_url: {json.dumps(base_url)}")
            replaced_url = True
            continue
        if raw == "roles:":
            in_roles = True
            current_role = None
            out.append(raw)
            continue
        if raw == "phases:":
            in_roles = False
            current_role = None
            out.append(raw)
            continue
        if in_roles:
            role_match = re.fullmatch(r"  ([a-z][a-z0-9_-]*):", raw)
            if role_match:
                current_role = role_match.group(1)
            elif current_role and raw.startswith("    model:"):
                model = quality_model if current_role in quality_roles else utility_model
                out.append(f"    model: {json.dumps(model)}")
                replaced_roles.add(current_role)
                continue
        out.append(raw)

    if not replaced_url or not expected_roles.issubset(replaced_roles):
        raise RuntimeError(
            "models.openai-compatible.yaml is missing its base_url or expected role model fields"
        )
    return "\n".join(out).rstrip("\n") + "\n"


def _customize_openai_compatible_config(
    models_yaml: Path,
    *,
    base_url: str,
    quality_model: str,
    utility_model: str,
) -> None:
    """Atomically customize an existing template path (test/maintenance API)."""
    contents = _customize_openai_compatible_text(
        models_yaml.read_text(encoding="utf-8"),
        base_url=base_url,
        quality_model=quality_model,
        utility_model=utility_model,
    )
    write_text_atomic(models_yaml, contents)


def _step_provider(root: Path) -> None:
    _header("Step 1 — LLM provider")
    config_dir = root / "config"
    models_yaml = config_dir / "models.yaml"
    env_path = _active_env_path(root)
    profile_before = snapshot_profile(env_path)
    require_private_credentials(profile_before)
    previous_base_url = _effective_openai_base_url()
    previous_anthropic_base_url = _effective_anthropic_base_url()

    current = _effective_provider(models_yaml)
    if current:
        current_provider, source = current
        print(f"Effective provider is `{current_provider}` via {source}.")
        if not _confirm("Reconfigure the provider?", default=False):
            if credential_keys(profile_before):
                _warn_gitignore(root, env_path)
            print("Keeping the effective provider routing.")
            return

    external_routing = _external_routing_keys(env_path)
    if external_routing:
        print(
            "… Provider setup cancelled — routing override(s) come from the "
            "parent shell or another higher-precedence source: "
            f"{', '.join(sorted(external_routing))}. Unset them there and rerun "
            "init; changing only this child process would be undone by the next "
            "researchwiki command."
        )
        return

    print("Which LLM provider will you use?")
    for i, (_pid, label, blurb) in enumerate(_PROVIDER_MENU, 1):
        print(f"  {i}. {label} — {blurb}")
    provider = _PROVIDER_MENU[_ask_choice(len(_PROVIDER_MENU))][0]

    # Collect + persist required env vars (skip any already set in the shell).
    api_key = base_url = None
    quality_model = utility_model = None
    remove_openai_key = False
    anthropic_endpoint_changed = False
    if provider == "anthropic":
        anthropic_endpoint_changed = not _same_endpoint(
            previous_anthropic_base_url, _ANTHROPIC_DEFAULT_BASE_URL,
        )
        proceed, api_key = _choose_anthropic_api_key(
            env_path,
            previous_endpoint=previous_anthropic_base_url,
            prompt="Anthropic API key (blank to set later)",
        )
        if not proceed:
            return
    elif provider == "openai":
        proceed, api_key = _choose_openai_api_key(
            env_path,
            previous_endpoint=previous_base_url,
            selected_endpoint=_OPENAI_DEFAULT_BASE_URL,
            prompt="OpenAI API key (blank to set later)",
        )
        if not proceed:
            return
    elif provider == "openai-compatible":
        base_url = _ask(
            "Provider base URL (required)",
            default=previous_base_url,
        ) or None
        if not base_url or not _valid_provider_base_url(base_url):
            print("… Provider setup cancelled — an absolute HTTPS base URL "
                  "without credentials, query, or fragment is required "
                  "(plain HTTP is allowed only on loopback); the existing "
                  "routing was left unchanged.")
            return
        proceed, api_key = _choose_openai_api_key(
            env_path,
            previous_endpoint=previous_base_url,
            selected_endpoint=base_url,
            prompt="Provider API key, forwarded as Bearer (blank to set later)",
        )
        if not proceed:
            return
        quality_model = _ask(
            "Exact model ID for author/critic/judge (required)"
        ) or None
        if quality_model:
            utility_model = _ask(
                "Exact model ID for classifier/proposer/extractor",
                default=quality_model,
            ) or quality_model
        if not quality_model:
            print("… Provider setup cancelled — model IDs are required; "
                  "the existing routing was left unchanged.")
            return
    elif provider == "local":
        local_default = (
            previous_base_url
            if current and current[0] in {"local", "lmstudio"}
            and previous_base_url
            and _valid_provider_base_url(previous_base_url, local_only=True)
            else _LOCAL_DEFAULT_BASE_URL
        )
        base_url = _ask("Local server base URL", default=local_default)
        if not _valid_provider_base_url(base_url, local_only=True):
            print("… Provider setup cancelled — an absolute loopback HTTP(S) "
                  "base URL without credentials, query, or fragment is required; "
                  "the existing routing was left unchanged.")
            return
        proceed, remove_openai_key = _prepare_local_api_key(
            env_path,
            previous_endpoint=previous_base_url,
            selected_endpoint=base_url,
        )
        if not proceed:
            return
    elif provider == "chat-relay":
        print("Chat-relay needs no key. Read prompts/chat-relay.md for the relay protocol before your first ingest.")

    prepared_template: str | None = None
    if provider == "openai-compatible":
        assert base_url and quality_model and utility_model
        template_name = _template_for_provider(provider)
        assert template_name is not None
        template_text = model_template_text(config_dir, template_name)
        if template_text is None:
            print(f"⚠ template {config_dir / template_name} not found locally or "
                  "in the installed package — the existing routing was left unchanged.")
            return
        try:
            prepared_template = _customize_openai_compatible_text(
                template_text,
                base_url=base_url,
                quality_model=quality_model,
                utility_model=utility_model,
            )
        except RuntimeError as e:
            print(f"… Provider setup cancelled — {e}; the existing routing was "
                  "left unchanged.")
            return

    stale_keys = _stale_routing_keys(provider)
    updates = _env_updates_for_provider(provider, api_key=api_key, base_url=base_url)
    preview_text, _preview_removed = edit_profile_text(
        profile_before,
        updates=updates,
        removals=stale_keys | ({"OPENAI_API_KEY"} if remove_openai_key else set()),
    )
    existing_credentials = credential_keys(profile_before)
    final_credentials = credential_keys_in_text(preview_text, path=env_path)
    if existing_credentials or final_credentials:
        ignored = _warn_gitignore(root, env_path)
        if ignored is False and final_credentials and not _confirm(
            f"Store credentials in unignored profile {env_path.name} anyway?",
            default=False,
        ):
            print("… Provider setup cancelled before any config or credential change.")
            return

    committed, removed_file, removed_process = commit_profile_and_config(
        config_path=models_yaml,
        apply_config=lambda: _write_models_config(
            config_dir,
            models_yaml,
            provider,
            template_contents=prepared_template,
        ),
        env_path=env_path,
        updates=updates,
        removals=stale_keys,
        remove_openai_key=remove_openai_key,
        replace_openai_key=bool(api_key and os.environ.get("OPENAI_API_KEY")),
        protected_credential_keys=(
            {"ANTHROPIC_API_KEY"}
            if provider == "anthropic" and anthropic_endpoint_changed
            else set()
        ),
    )
    if not committed:
        print("… Provider setup stopped — the existing routing was left unchanged.")
        return
    if provider == "openai-compatible":
        print("Wrote the selected endpoint and model IDs to config/models.yaml.")

    if remove_openai_key:
        print(f"Removed OPENAI_API_KEY from {env_path.name} for the local endpoint.")

    removed = (removed_file | removed_process) & stale_keys
    if removed:
        print(f"Removed stale routing override(s): {', '.join(sorted(removed))}.")

    if updates:
        print(f"Wrote {', '.join(updates)} to {env_path.name} (mode 600).")
        if "RW_LLM_PROVIDER" in updates:
            print("Note: RW_LLM_PROVIDER in .env is a GLOBAL override — it forces every "
                  "role to this provider and defeats per-role mixing. Comment it out "
                  "later if you want to mix providers via config/models.yaml.")
        if "RW_LLM_BASE_URL" in updates:
            print("Note: RW_LLM_BASE_URL is routing config — you can move it to a shell "
                  "export later so switching backends doesn't touch .env.")

    _report_readiness(provider)


def _write_models_config(
    config_dir: Path,
    models_yaml: Path,
    provider: str,
    *,
    template_contents: str | None = None,
) -> bool:
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
            return True
        if _confirm("Remove the existing config/models.yaml so the built-in "
                    "OpenAI defaults apply?", default=True):
            models_yaml.unlink()
            print("Removed config/models.yaml — built-in OpenAI defaults now apply.")
            return True
        else:
            print("⚠ Left config/models.yaml in place. It overrides this choice — "
                  "whatever providers it names are what will actually run.")
            return False

    template = config_dir / template_name
    contents = (
        template_contents
        if template_contents is not None
        else model_template_text(config_dir, template_name)
    )
    if contents is None:
        print(f"⚠ template {template} not found locally or in the installed "
              "package — skipping config copy. "
              f"You'll need to create config/models.yaml by hand.")
        return False
    if models_yaml.exists() and not _confirm(
        f"Overwrite existing config/models.yaml with the {provider} template?", default=True
    ):
        print("Left config/models.yaml untouched.")
        return False
    write_text_atomic(models_yaml, contents)
    print(f"Wrote config/models.yaml from {template_name}.")
    return True


def _warn_gitignore(root: Path, env_path: Path) -> bool:
    """Warn unless git confirms the exact credential file is ignored.

    Looking for the substring ``.env`` in ``.gitignore`` produced false safety
    for explicitly selected files such as ``secrets.prod`` and for tracked
    exceptions. ``git check-ignore`` applies the repository's complete rule
    stack to the actual path and intentionally returns non-zero for tracked
    files even when a broad pattern would otherwise match them.
    """
    try:
        display = env_path.resolve().relative_to(root.resolve())
    except ValueError:
        # A profile outside this repository cannot be committed by it.
        return True
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--", str(env_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        return True
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(display)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        tracked = False
    if tracked:
        print(
            f"⚠ {display} is tracked by git — run `git rm --cached -- "
            f"{display}` and add an exact ignore rule before storing credentials."
        )
        return False
    print(
        f"⚠ {display} is not confirmed gitignored — add an exact ignore rule "
        "before committing so your key doesn't reach version control."
    )
    return False


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
    # Use the public reset so warning latches and every present/future routing
    # cache move together; reaching into three private cached functions drifted
    # as soon as model_config gained another piece of cached state.
    _mc.clear_caches()

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
