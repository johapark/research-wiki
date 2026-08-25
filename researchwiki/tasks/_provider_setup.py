"""Credential ownership decisions shared by the interactive provider wizard.

Kept separate from ``tasks.init`` because endpoint transitions are a security
boundary of their own: an API key may follow a new host only after explicit
consent, and a shell-owned value cannot be replaced by editing a dotenv file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..env_profiles import effective_assignment_value, loaded_from_profile, snapshot_profile


def profile_controls_env_key(env_path: Path, key: str) -> bool:
    """Whether the selected profile inserted and still supplies ``key``."""
    return (
        loaded_from_profile(env_path, key)
        and key in os.environ
        and effective_assignment_value(snapshot_profile(env_path), key) == os.environ[key]
    )


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
