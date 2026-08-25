"""Read setup resources that must survive a non-editable installation.

The repository-root copies of ``.env.template`` and ``config/models.*.yaml``
are convenient in a clone, but they are not importable package data.  The
wheel therefore carries byte-for-byte mirrors under ``researchwiki/resources``.
Callers prefer a checkout's local model template when it exists and fall back
to that bundled mirror when only the installed package is available.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path


_MODEL_TEMPLATE_RE = re.compile(r"models\.[a-z0-9][a-z0-9-]*\.yaml\Z")


def bundled_env_template_text() -> str:
    """Return the packaged ``.env.template`` as UTF-8 text."""
    resource = files("researchwiki").joinpath("resources", ".env.template")
    return resource.read_text(encoding="utf-8")


def bundled_model_template_text(template_name: str) -> str | None:
    """Return one packaged ``models.<profile>.yaml`` template if it exists.

    ``template_name`` comes from the setup wizard's fixed provider mapping,
    but validating the basename here keeps this helper safe for future callers
    and prevents resource traversal from becoming an accidental public API.
    """
    if not _MODEL_TEMPLATE_RE.fullmatch(template_name):
        raise ValueError(f"invalid models template name: {template_name!r}")
    resource = files("researchwiki").joinpath(
        "resources", "config", template_name
    )
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def model_template_text(config_dir: Path, template_name: str) -> str | None:
    """Read a checkout-local template, or its installed-package fallback."""
    local_template = config_dir / template_name
    if local_template.is_file():
        return local_template.read_text(encoding="utf-8")
    return bundled_model_template_text(template_name)
