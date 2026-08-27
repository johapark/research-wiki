"""Credential ownership and mutable-config decisions for the provider wizard.

Kept separate from ``tasks.init`` because endpoint transitions are a security
boundary of their own: an API key may follow a new host only after explicit
consent, and a shell-owned value cannot be replaced by editing a dotenv file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from ..env_profiles import effective_assignment_value, loaded_from_profile, snapshot_profile
from ..fsatomic import write_text_atomic


def customize_openai_compatible_text(
    template_text: str,
    *,
    base_url: str,
    quality_model: str,
    utility_model: str,
) -> str:
    """Customize and structurally validate the generic compatible template."""
    lines = template_text.splitlines()
    out: list[str] = []
    in_roles = False
    current_role: str | None = None
    replaced_url = False
    replaced_roles: set[str] = set()
    quality_roles = {"author", "critic", "judge"}
    expected_roles = quality_roles | {"classifier", "proposer", "extractor"}

    for raw in lines:
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


def customize_openai_compatible_config(
    models_yaml: Path,
    *,
    base_url: str,
    quality_model: str,
    utility_model: str,
) -> None:
    """Atomically customize an existing template path."""
    contents = customize_openai_compatible_text(
        models_yaml.read_text(encoding="utf-8"),
        base_url=base_url,
        quality_model=quality_model,
        utility_model=utility_model,
    )
    write_text_atomic(models_yaml, contents)


def profile_controls_env_key(env_path: Path, key: str) -> bool:
    """Whether the selected profile inserted and still supplies ``key``."""
    return (
        loaded_from_profile(env_path, key)
        and key in os.environ
        and effective_assignment_value(snapshot_profile(env_path), key) == os.environ[key]
    )


def provider_config_target(
    root: Path,
    selected_path: Path | None,
    *,
    provider: str,
    named_profile: bool,
    named_template: str | None = None,
) -> tuple[Path, bool, str | None, bool]:
    """Choose a mutable config target without overwriting tracked templates.

    Returns ``(path, preserve_selector, replacement_selector, write_config)``.
    A named profile selects an immutable provider template directly; it does
    not derive a second filename from ``.env.NAME``. A custom compatible
    backend must instead select an explicit writable config path. A
    user-selected mutable config remains authoritative.
    """
    default = root / "config" / "models.yaml"
    if not named_profile:
        return default, False, None, True

    config_dir = root / "config"
    tracked_template = bool(
        selected_path
        and selected_path.parent.resolve() == config_dir.resolve()
        and selected_path.name.startswith("models.")
        and selected_path.name.endswith(".yaml")
        and selected_path.name != "models.yaml"
    )
    if provider == "openai-compatible":
        if selected_path is None or tracked_template:
            raise ValueError(
                "a named profile using a custom OpenAI-compatible backend needs "
                "an explicit writable model-config path"
            )
        return selected_path, True, None, True

    if named_template is None:
        raise ValueError(f"no named-profile template is defined for {provider}")
    target = config_dir / named_template
    return target, True, named_template, False


def stale_routing_keys(
    provider: str, *, preserve_models_config: bool = False,
) -> set[str]:
    """Environment overrides that would defeat the provider just selected."""
    stale = set() if preserve_models_config else {"RW_MODELS_CONFIG"}
    if provider != "chat-relay":
        stale.add("RW_LLM_PROVIDER")
    if provider != "local":
        stale.add("RW_LLM_BASE_URL")
    if provider == "anthropic":
        stale.add("ANTHROPIC_BASE_URL")
    return stale


def same_endpoint(left: str | None, right: str | None) -> bool:
    """Endpoint equality that ignores only an inconsequential trailing slash."""
    return bool(left and right and left.rstrip("/") == right.rstrip("/"))


def choose_endpoint_api_key(
    env_path: Path,
    *,
    env_key: str,
    previous_endpoint: str | None,
    selected_endpoint: str,
    prompt: str,
    ask: Callable[[str], str],
    confirm: Callable[[str, bool], bool],
) -> tuple[bool, str | None]:
    """Choose whether/how one Bearer credential follows an endpoint change."""
    existing = os.environ.get(env_key)
    if not existing:
        return True, ask(prompt) or None
    if same_endpoint(previous_endpoint, selected_endpoint):
        print(f"{env_key} is already set for this endpoint — leaving it.")
        return True, None

    if confirm(
        f"The endpoint changed. Reuse the currently set {env_key} at the new "
        "endpoint?",
        False,
    ):
        shadowed = effective_assignment_value(snapshot_profile(env_path), env_key)
        if shadowed is not None and shadowed != existing:
            print(
                "… Provider setup cancelled — the selected profile contains a "
                f"different {env_key} shadowed by the parent shell. Unset the "
                "shell key and rerun init so both future and current invocations "
                "agree on the credential."
            )
            return False, None
        print(f"Reusing {env_key} at the newly selected endpoint.")
        return True, None

    if not profile_controls_env_key(env_path, env_key):
        print(
            f"… Provider setup cancelled — {env_key} comes from the parent "
            "shell (or another higher-precedence source). Unset it there, then "
            "rerun init to store a key for the new endpoint."
        )
        return False, None

    replacement = ask("New API key for the selected endpoint (blank to cancel)") or None
    if replacement is None:
        print("… Provider setup cancelled — the existing routing was left unchanged.")
        return False, None
    return True, replacement
