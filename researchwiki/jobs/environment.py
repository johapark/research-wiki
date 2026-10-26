"""Per-child profile selection and non-secret execution identity."""

from dataclasses import replace
import json
import os
from pathlib import Path

from ..env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR
from ..fsatomic import file_sha256
from .contracts import InputIdentity
from .files import JobConflict, digest


def child_environment(parent=None):
    """Remove profile-owned state; retain genuinely exported shell credentials.

    An explicit profile still rejects genuinely exported routing variables,
    matching the CLI loader's rule. The child's own library profile may select
    its DB override; an unrelated parent's DB override is never inherited.
    """
    env = dict(os.environ if parent is None else parent)
    provenance = env.get(DOTENV_PROVENANCE_VAR)
    if provenance:
        try:
            keys = json.loads(provenance)["keys"]
            if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
                raise ValueError("invalid profile keys")
        except (KeyError, ValueError, TypeError) as exc:
            raise JobConflict("cannot isolate inherited profile metadata") from exc
        for key in keys:
            env.pop(key, None)
    for key in (ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR, "RESEARCHWIKI_DB_PATH"):
        env.pop(key, None)
    return env


def configure_worker(request):
    from ..env_profiles import load_profile

    if Path.cwd().resolve() != Path(request.library_root).resolve():
        raise JobConflict("worker was not started in its selected library")
    profile = Path(request.provider_profile) if request.provider_profile else Path(request.library_root) / ".env"
    load_profile(profile, required=request.provider_profile is not None)
    # An implicit root .env can become the active profile without changing the
    # immutable queued request. The synchronous service validates this resolved
    # worker selection rather than selecting another environment itself.
    return replace(request, provider_profile=os.environ.get(ACTIVE_ENV_FILE_VAR))


def execution_identity(request):
    from .. import __version__
    from ..agents import model_config

    model_config.validate_config()
    config = model_config.config_path()
    pricing = model_config.pricing_path()
    routing = {name: os.environ.get(name) for name in (
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL", "ANTHROPIC_BASE_URL", "RW_RELAY_TIMEOUT", "RW_PRICING_CONFIG",
    )}
    configuration = {
        "profile": request.provider_profile,
        "config_path": str(config.resolve()),
        "config_sha256": file_sha256(config) if config.is_file() else None,
        "framework_version": __version__,
        "pricing_sha256": file_sha256(pricing) if pricing.is_file() else None,
        "prompts": [(str(path.relative_to(request.library_root)), file_sha256(path)) for path in
                    sorted((Path(request.library_root) / "prompts").rglob("*.md"))],
        "routing": routing,
    }
    # No raw profile bytes, API keys, prompt text, or provider config values
    # appear in the persisted identity. Credentials are not part of its hash.
    auxiliary = [(path, file_sha256(Path(path))) for path in (
        *request.options.supplementary, request.options.author_prompt_file,
    ) if path is not None]
    return InputIdentity(pdf_sha256=file_sha256(Path(request.source_pdf)),
                         auxiliary_sha256=digest(auxiliary), configuration_sha256=digest(configuration))
