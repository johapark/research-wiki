"""Central model configuration. Stage 1 of multi-model support.

Maps each LLM-using *phase* of the framework to a *role* (author / critic /
judge / classifier / proposer), and each role to a (provider, model,
temperature, max_tokens) tuple. Source of truth: `config/models.yaml` at
the repo root.

Also owns per-token **pricing** (`config/pricing.yaml`), the other half of
"facts about a model" — see the Pricing section at the bottom.

Public API:
  for_phase(name) -> ModelConfig
  for_role(name)  -> ModelConfig
  rate_for(model) -> Rate | None
  estimate_usd(model, in_tok, out_tok) -> float
  pricing_as_of() -> str
  list_phases()   -> list[str]
  validate_config() -> None
  clear_caches()  -> None

If the implicit `config/models.yaml` is missing, the loader falls back to the
hardcoded defaults below. That makes the framework usable on a fresh checkout.
Any selected file that *exists* fails closed when unreadable, malformed, or
schema-invalid; only absence of the implicit file enables zero-config fallback.
This prevents a damaged profile from silently changing both model and endpoint.

`provider` is forward-compatible. Today only "anthropic" is implemented in
researchwiki/agents/llm.py. When a second provider is added (OpenAI,
Ollama, etc.), this config schema doesn't change — just edit a role's
`provider:` to point at it.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import stat
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from ..errors import EnvironmentFailure
from ..paths import wiki_root


class PhaseNotRegistered(KeyError):
    """A phase (or its bound role) is absent from the model config.

    This is a *programming* error, not a runtime one: an unregistered phase
    will never resolve on retry, so callers that tolerate transient LLM
    failures (`run_llm_judge` returns None on any exception) must not swallow
    it — doing so silently no-ops the whole phase. Subclasses `KeyError` so
    existing `except KeyError` handlers keep working.

    Adding a new LLM phase? Register it in `_FALLBACK_PHASES` below; the
    `test_phase_registration` suite fails CI if a `phase=` literal in the
    package has no entry.
    """


class ModelConfigUnavailable(EnvironmentFailure):
    """The selected model-routing file cannot be used safely."""


def _explicit_config_error(path: Path, detail: str) -> ModelConfigUnavailable:
    override = os.environ.get("RW_MODELS_CONFIG")
    return ModelConfigUnavailable(
        f"RW_MODELS_CONFIG={override!r} resolves to {path}, {detail}; "
        "fix the path/config or unset RW_MODELS_CONFIG to use the implicit "
        "config/models.yaml (or built-in OpenAI defaults when that file is absent)"
    )


def _config_error(path: Path, detail: str) -> ModelConfigUnavailable:
    """Describe a selected-file failure without mislabeling implicit config."""
    if _has_explicit_config():
        return _explicit_config_error(path, detail)
    return ModelConfigUnavailable(
        f"implicit model config {path} is present but {detail}; fix that file "
        "or remove it to use the built-in OpenAI defaults"
    )


def _has_explicit_config() -> bool:
    """Whether the override exists, including an unsafe empty value."""
    return "RW_MODELS_CONFIG" in os.environ


def config_path() -> Path:
    """Resolve which models-config file to load.

    `RW_MODELS_CONFIG` overrides the default `config/models.yaml`, so you can
    switch backends without copying files over your active config (e.g.
    `RW_MODELS_CONFIG=models.glm.yaml researchwiki agent ingest …`). A bare
    filename resolves under `config/`; an absolute or path-separated value is
    used verbatim. This is cleaner than `RW_LLM_PROVIDER` for whole-backend
    swaps — the selected file keeps its per-role provider mixing, whereas
    `RW_LLM_PROVIDER` forces one provider across every phase.
    """
    if not _has_explicit_config():
        return wiki_root() / "config" / "models.yaml"
    override = os.environ["RW_MODELS_CONFIG"]
    if not override.strip():
        raise ModelConfigUnavailable(
            f"RW_MODELS_CONFIG={override!r} is empty; set it to an existing "
            "models YAML file or unset it to use the implicit config"
        )
    p = Path(override).expanduser()
    sep_in = os.sep in override or bool(os.altsep and os.altsep in override)
    if p.is_absolute() or sep_in:
        return p
    return wiki_root() / "config" / override


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Reasoning / thinking budget hint. Honored by openai-compatible providers
    # that recognize the OpenAI `reasoning_effort` field — notably Gemini 2.5
    # Flash/Pro via Google's /v1beta/openai/ shim, and OpenAI o-series. Passed
    # through verbatim when set; omitted from the request when None so the
    # provider default applies. Anthropic / chat-relay / stub silently ignore.
    # Valid values per OpenAI spec: "minimal" | "low" | "medium" | "high".
    reasoning_effort: str | None = None
    # Client-side requests-per-minute cap for this model. When set, llm.call
    # blocks before dispatch so calls to this *model* stay >= 60/rpm seconds
    # apart (keyed by model, so roles sharing a model share one budget). Lets
    # free-tier configs (Gemini/Gemma) stay under their RPM ceiling instead of
    # bursting into 429s. None = unthrottled. See llm._throttle.
    rpm: int | None = None


# Hardcoded zero-config defaults — used only when the implicit
# `config/models.yaml` is absent. Every role is Luna so a fresh checkout with
# OPENAI_API_KEY works with no copy. A present file always validates strictly,
# including when PyYAML is unavailable. This deliberately does
# *not* mirror opt-in `models.chatgpt.yaml`, which upgrades three roles to
# Terra. The paired endpoint is _FALLBACK_BASE_URL below; keep the two in sync.
_FALLBACK_ROLES: dict[str, ModelConfig] = {
    "author":     ModelConfig("openai-compatible", "gpt-5.6-luna", 0.5, 6000),
    "critic":     ModelConfig("openai-compatible", "gpt-5.6-luna", 0.3, 2500),
    "judge":      ModelConfig("openai-compatible", "gpt-5.6-luna", 0.2, 1500),
    # Every role is gpt-5.6-luna. gpt-5.4-mini used to hold classifier and
    # extractor on the assumption it was the cheap one; it is not. mini is a
    # 5.4-generation model and the 5.6 line cut prices, so it costs 3.75x luna
    # per token ($0.75/$4.50 against $0.20/$1.20 — config/pricing.yaml) with
    # no benefit on either role's work.
    "classifier": ModelConfig("openai-compatible", "gpt-5.6-luna", 0.1, 200),
    "proposer":   ModelConfig("openai-compatible", "gpt-5.6-luna", 0.3, 200),
    "extractor":  ModelConfig("openai-compatible", "gpt-5.6-luna", 0.0, 800),
}

# Endpoint for the openai-compatible fallback roles above. Only consulted on
# the no-config-file path — a present config supplies its own `base_url:`, and
# RW_LLM_BASE_URL still overrides at the call site (see llm.call).
_FALLBACK_BASE_URL = "https://api.openai.com/v1"

_FALLBACK_PHASES: dict[str, dict] = {
    "author":            {"role": "author"},
    "evolve":            {"role": "author"},
    "debug":             {"role": "author", "temperature": 0.2},
    "critic":            {"role": "critic"},
    "classifier":        {"role": "classifier"},
    # Concept-hub candidate triage. Same cheap low-temp classifier role, but
    # the role's 200-token cap is sized for the category auto-suggester's
    # single verdict — triage returns one verdict per term for a whole
    # CHUNK_SIZE batch. Registered here (not only in config/models.yaml) so it
    # resolves under every RW_MODELS_CONFIG, not just the default one.
    "concept_triage":    {"role": "classifier", "max_tokens": 3500},
    # 220, not 32: this phase emits a HANDLE line plus a <=400-char HOOK.
    "short_name":        {"role": "proposer", "max_tokens": 220},
    "keywords":          {"role": "proposer"},
    "link_generation":   {"role": "judge", "max_tokens": 1200},
    "memory_evolution":  {"role": "judge", "max_tokens": 3000},
    "synthesis_judge":   {"role": "judge"},
    "eval_judge":        {"role": "judge"},
    "cross_paper_judge": {"role": "judge", "temperature": 0.0, "max_tokens": 400},
    "claim_overlap_judge": {"role": "judge", "temperature": 0.0, "max_tokens": 400},
    "reconcile":         {"role": "extractor"},
    "target_claims":     {"role": "extractor", "max_tokens": 2500},
    "claim_support":     {"role": "judge"},
}


_MODEL_TOP_LEVEL_KEYS = frozenset({"base_url", "roles", "phases", "ingest"})
_ROLE_FIELDS = frozenset({
    "provider", "model", "temperature", "max_tokens", "reasoning_effort", "rpm",
})
_PHASE_FIELDS = _ROLE_FIELDS | {"role"}
_INGEST_FIELDS = frozenset({"n_drafts", "target_claims_max_chars"})


def _schema_error(path: Path, detail: str) -> ModelConfigUnavailable:
    return _config_error(path, f"its schema is invalid ({detail})")


def _valid_base_url(value: str) -> bool:
    """Accept a structurally safe absolute HTTP(S) endpoint."""
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
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


def validate_env_base_url(
    value: str, *, variable: str = "RW_LLM_BASE_URL",
) -> str:
    """Validate an env endpoint without reflecting possible secrets in errors."""
    if not _valid_base_url(value):
        raise ModelConfigUnavailable(
            f"{variable} is not a safe absolute endpoint; use HTTP(S) and omit "
            "credentials, whitespace, query, and fragment"
        )
    return value


def canonical_provider_id(value: str) -> str:
    """Canonical identifier used by routing analysis and provider dispatch."""
    return value.strip().lower()


def _env_provider_override() -> str:
    """One normalized value for endpoint analysis, warnings, and application."""
    return canonical_provider_id(os.environ.get("RW_LLM_PROVIDER") or "")


def _validate_positive_int(path: Path, label: str, value) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise _schema_error(path, f"{label} must be a positive integer") from e
    if isinstance(value, bool) or parsed < 1 or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise _schema_error(path, f"{label} must be a positive integer")


def _validate_routing_fields(
    path: Path, label: str, fields: dict, *, require_model: bool,
) -> None:
    allowed = _ROLE_FIELDS if require_model else _PHASE_FIELDS
    unknown = sorted(str(k) for k in set(fields) - allowed)
    if unknown:
        raise _schema_error(path, f"{label} has unknown field(s): {', '.join(unknown)}")
    if require_model and "model" not in fields:
        raise _schema_error(path, f"{label} is missing required field 'model'")
    for name in ("provider", "model"):
        if name in fields and (
            not isinstance(fields[name], str) or not fields[name].strip()
        ):
            raise _schema_error(path, f"{label}.{name} must be a non-empty string")
    if "temperature" in fields:
        try:
            temperature = float(fields["temperature"])
        except (TypeError, ValueError) as e:
            raise _schema_error(path, f"{label}.temperature must be a finite number") from e
        if isinstance(fields["temperature"], bool) or not math.isfinite(temperature):
            raise _schema_error(path, f"{label}.temperature must be a finite number")
    if "max_tokens" in fields:
        _validate_positive_int(path, f"{label}.max_tokens", fields["max_tokens"])
    if fields.get("rpm") is not None:
        _validate_positive_int(path, f"{label}.rpm", fields["rpm"])
    effort = fields.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise _schema_error(path, f"{label}.reasoning_effort must be a string or null")


def _validate_document(path: Path, data: dict) -> None:
    """Reject any present file that cannot safely drive provider routing."""
    keys = set(data)
    unknown = sorted(str(k) for k in keys - _MODEL_TOP_LEVEL_KEYS)
    if unknown:
        raise _schema_error(path, f"unknown top-level key(s): {', '.join(unknown)}")
    if not keys:
        raise _schema_error(path, "the document is empty")

    if "base_url" in data:
        value = data["base_url"]
        if not isinstance(value, str) or not _valid_base_url(value):
            raise _schema_error(
                path,
                "base_url must be a safe absolute endpoint using HTTP(S), "
                "without credentials, query, or fragment",
            )

    roles = data.get("roles", {})
    if not isinstance(roles, dict):
        raise _schema_error(path, "roles must be a mapping")
    for name, fields in roles.items():
        if not isinstance(name, str) or not name.strip():
            raise _schema_error(path, "every role name must be a non-empty string")
        if not isinstance(fields, dict):
            raise _schema_error(path, f"role {name!r} must be a mapping")
        _validate_routing_fields(path, f"role {name!r}", fields, require_model=True)

    phases = data.get("phases", {})
    if not isinstance(phases, dict):
        raise _schema_error(path, "phases must be a mapping")
    known_roles = set(_FALLBACK_ROLES) | set(roles)
    for name, spec in phases.items():
        if not isinstance(name, str) or not name.strip():
            raise _schema_error(path, "every phase name must be a non-empty string")
        if isinstance(spec, str):
            role = spec
        elif isinstance(spec, dict):
            _validate_routing_fields(path, f"phase {name!r}", spec, require_model=False)
            role = spec.get("role")
        else:
            raise _schema_error(path, f"phase {name!r} must be a role string or mapping")
        if not isinstance(role, str) or not role.strip():
            raise _schema_error(path, f"phase {name!r} needs a non-empty role")
        if role not in known_roles:
            raise _schema_error(path, f"phase {name!r} references unknown role {role!r}")

    ingest = data.get("ingest", {})
    if not isinstance(ingest, dict):
        raise _schema_error(path, "ingest must be a mapping")
    unknown_ingest = sorted(str(k) for k in set(ingest) - _INGEST_FIELDS)
    if unknown_ingest:
        raise _schema_error(
            path, f"ingest has unknown field(s): {', '.join(unknown_ingest)}",
        )
    for name in _INGEST_FIELDS & set(ingest):
        _validate_positive_int(path, f"ingest.{name}", ingest[name])


@dataclass(frozen=True)
class _RoutingSnapshot:
    """One coherent view of every field read from the selected YAML file."""

    file_present: bool
    roles: dict[str, ModelConfig]
    phases: dict[str, dict]
    ingest: dict
    base_url: str | None


@lru_cache(maxsize=8)
def _load_routing_snapshot(
    path: Path,
    explicit_override: str | None,
    env_base_url: str,
    env_provider: str,
) -> _RoutingSnapshot:
    """Parse, validate, and resolve one selected config atomically.

    ``path`` and the routing-relevant environment are cache keys so all three
    consumers (models, ingest defaults, endpoint) observe the same document.
    Public :func:`clear_caches` is still required when a file is edited in
    place, as it is for the benchmark profile switch.
    """
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as e:
        # ``Path.exists()`` treats a broken symlink as absent. It is instead a
        # deliberately present but unusable selection and must not enable the
        # implicit zero-config fallback.
        if path.is_symlink():
            raise _config_error(path, "cannot be read (broken symbolic link)") from e
        if explicit_override is not None:
            raise _config_error(path, "does not exist")
        return _RoutingSnapshot(
            file_present=False,
            roles=dict(_FALLBACK_ROLES),
            phases=dict(_FALLBACK_PHASES),
            ingest={},
            base_url=_FALLBACK_BASE_URL,
        )
    except OSError as e:
        raise _config_error(path, f"cannot be inspected ({e})") from e
    if not stat.S_ISREG(mode):
        raise _config_error(path, "is not a regular file")

    try:
        import yaml
    except ImportError as e:
        raise _config_error(path, "cannot be read because PyYAML is not installed") from e
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise _config_error(path, f"cannot be read ({e})") from e
    try:
        data = yaml.safe_load(source)
    except Exception as e:
        raise _config_error(path, f"cannot be parsed ({e})") from e
    if not isinstance(data, dict):
        raise _config_error(path, "must contain a YAML mapping")
    _validate_document(path, data)

    yaml_roles: dict[str, ModelConfig] = {}
    for name, fields in data.get("roles", {}).items():
        re_raw = fields.get("reasoning_effort")
        rpm_raw = fields.get("rpm")
        yaml_roles[name] = ModelConfig(
            provider=canonical_provider_id(
                str(fields.get("provider", "anthropic")),
            ),
            model=str(fields["model"]),
            temperature=float(fields.get("temperature", 0.5)),
            max_tokens=int(fields.get("max_tokens", 2000)),
            reasoning_effort=str(re_raw) if re_raw is not None else None,
            rpm=int(rpm_raw) if rpm_raw is not None else None,
        )

    yaml_phases: dict[str, dict] = {}
    for phase, spec in data.get("phases", {}).items():
        if isinstance(spec, str):
            yaml_phases[phase] = {"role": spec}
        else:
            yaml_phases[phase] = dict(spec)

    roles = {**_FALLBACK_ROLES, **yaml_roles}
    phases = {**_FALLBACK_PHASES, **yaml_phases}
    configured_base_url = data.get("base_url")

    # Provider env override is applied last by for_phase(), so include it when
    # deciding whether any *effective* role needs an OpenAI-compatible endpoint.
    from .llm import _OPENAI_COMPAT_PROVIDERS
    forced_provider = canonical_provider_id(env_provider)
    if forced_provider:
        compat = (
            [f"role {name!r}" for name in sorted(roles)]
            if forced_provider in _OPENAI_COMPAT_PROVIDERS else []
        )
    else:
        compat_targets = {
            f"role {name!r}" for name, cfg in roles.items()
            if canonical_provider_id(cfg.provider) in _OPENAI_COMPAT_PROVIDERS
        }
        for phase, spec in phases.items():
            role = roles[spec["role"]]
            provider = canonical_provider_id(
                str(spec.get("provider", role.provider)),
            )
            if provider in _OPENAI_COMPAT_PROVIDERS:
                compat_targets.add(f"phase {phase!r}")
        compat = sorted(compat_targets)
    if compat and configured_base_url is None and not env_base_url:
        shown = ", ".join(compat[:4]) + (", …" if len(compat) > 4 else "")
        raise _config_error(
            path,
            "does not declare a top-level base_url and RW_LLM_BASE_URL is unset, "
            f"although effective OpenAI-compatible role(s) are {shown}; add an "
            "endpoint to the config or environment",
        )

    return _RoutingSnapshot(
        file_present=True,
        roles=roles,
        phases=phases,
        ingest=dict(data.get("ingest", {})),
        base_url=str(configured_base_url) if configured_base_url is not None else None,
    )


def _routing_snapshot() -> _RoutingSnapshot:
    override = os.environ.get("RW_MODELS_CONFIG") if _has_explicit_config() else None
    env_base_url = os.environ.get("RW_LLM_BASE_URL") or ""
    if env_base_url:
        validate_env_base_url(env_base_url)
    env_provider = _env_provider_override()
    return _load_routing_snapshot(
        config_path(), override, env_base_url, env_provider,
    )


def _ingest_settings() -> dict:
    """Top-level `ingest:` block from the models config — pipeline defaults
    that aren't per-role (as opposed to `roles:` / `phases:`). Optional; an
    empty dict when the implicit file is absent or has no `ingest:` key.

    Currently recognizes `n_drafts` (default author-draft count when the CLI
    `-n` flag is omitted). The shared routing snapshot supplies the caching.
    """
    return dict(_routing_snapshot().ingest)


def default_n_drafts() -> int | None:
    """Config default for author drafts (`ingest.n_drafts`), or None if unset.

    Invalid values are rejected while loading a present config. The CLI `-n`
    flag, when passed, overrides this setting.
    """
    val = _ingest_settings().get("n_drafts")
    if val is None:
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


_DEFAULT_TARGET_CLAIMS_MAX_CHARS = 120_000


def target_claims_max_chars() -> int:
    """Char budget for the target-claims extraction prompt (`ingest.
    target_claims_max_chars`). The extractor is fed the whole substantive
    paper (references excluded) up to this many chars — high enough that a
    normal-length paper is sent in full, but bounded so a huge paper or a
    small-context local model doesn't overflow. Defaults generous (cloud
    context is ample); a 32K-context local config should lower it.
    """
    val = _ingest_settings().get("target_claims_max_chars")
    if val is None:
        return _DEFAULT_TARGET_CLAIMS_MAX_CHARS
    try:
        n = int(val)
    except (TypeError, ValueError):
        return _DEFAULT_TARGET_CLAIMS_MAX_CHARS
    return n if n > 0 else _DEFAULT_TARGET_CLAIMS_MAX_CHARS


def base_url() -> str | None:
    """Top-level `base_url:` from the models config — the OpenAI-compatible
    endpoint for this config's `openai-compatible`/`lmstudio`/`openai` roles.

    Folding the endpoint into the config file means switching backends needs
    only `RW_MODELS_CONFIG` to change, not a paired `RW_LLM_BASE_URL` edit.
    Precedence at the call site (see llm.call): the `RW_LLM_BASE_URL` env var,
    when set, still wins as an ad-hoc override; then this; then the built-in
    LM Studio localhost default. Returns None only when a present file has no
    `base_url:` and no effective role needs one. On the no-config-file path returns
    `_FALLBACK_BASE_URL` — the OpenAI endpoint paired with the openai-compatible
    fallback roles — so a fresh checkout reaches OpenAI, not localhost.
    `anthropic` roles ignore it (they use `ANTHROPIC_BASE_URL`).
    """
    return _routing_snapshot().base_url


def config_file_present() -> bool:
    """Whether the cached routing snapshot came from a selected YAML file."""
    return _routing_snapshot().file_present


def _config() -> tuple[dict[str, ModelConfig], dict[str, dict]]:
    """Resolve the merged config — YAML overrides, then fallback fills gaps.

    The shared routing snapshot caches the YAML once per selected path and
    routing environment. Use :func:`clear_caches` after editing a file in place.
    """
    snapshot = _routing_snapshot()
    return snapshot.roles, snapshot.phases


_env_override_warned = False


def _maybe_warn_env_override_defeats_mixing() -> None:
    """Fire once per process when `RW_LLM_PROVIDER` silently defeats per-role
    mixing declared in the resolved models config.

    Documented scar (CLAUDE.md § *Model providers*): `RW_LLM_PROVIDER` in
    `.env` overrides every phase's provider, so a mixed `models.yaml` (some
    roles anthropic, some openai-compatible) reads as if mixing works but
    doesn't. Print a stderr banner when both conditions hold so the footgun
    is visible without users having to inspect their `.env`.

    Uses `print` on `sys.stderr` directly rather than `log()`/`logging` —
    this must fire before any logging config runs and must not respect
    verbosity filters. No color codes (output may be piped/redirected).
    """
    global _env_override_warned
    if _env_override_warned:
        return
    env_provider = _env_provider_override()
    if not env_provider:
        return
    roles, _ = _config()
    providers = {cfg.provider for cfg in roles.values()}
    if len(providers) < 2:
        # Uniform config — the env override doesn't hide any mixing.
        return
    print(
        f"⚠  RW_LLM_PROVIDER={env_provider!r} overrides per-role providers "
        f"in {config_path()}.\n"
        f"    All roles will use {env_provider!r}. "
        f"Unset RW_LLM_PROVIDER to enable mixing.",
        file=sys.stderr,
    )
    _env_override_warned = True


_env_model_mismatch_warned = False


def _maybe_warn_env_override_model_mismatch() -> None:
    """Fire once per process when `RW_LLM_PROVIDER` forces a provider the
    resolved config did not choose its *models* for.

    `for_phase` overrides the provider and nothing else, so the two halves of a
    routing decision arrive from different layers and can contradict each other:
    `RW_MODELS_CONFIG=models.chatgpt.yaml RW_LLM_PROVIDER=anthropic` resolves to
    `anthropic/gpt-5.6-terra`, which no API serves. The call then fails on an
    unknown model, naming neither the env var nor the config that produced it.

    Distinct from `_maybe_warn_env_override_defeats_mixing`, which returns early
    unless the config declares ≥2 providers — and so stays silent on exactly
    this case, a *uniform* config whose provider the env var replaces wholesale.

    `chat-relay` is exempt: it hands prompts to a chat agent and treats the
    model string as a label, so a "mismatch" there is the documented workflow.
    """
    global _env_model_mismatch_warned
    if _env_model_mismatch_warned:
        return
    env_provider = _env_provider_override()
    if not env_provider or env_provider.lower() == "chat-relay":
        return
    roles, _ = _config()
    clashing = {n: c for n, c in roles.items()
                if canonical_provider_id(c.provider) != env_provider}
    if not clashing:
        return
    role_name, cfg = sorted(clashing.items())[0]
    chosen_for = ", ".join(sorted({c.provider for c in clashing.values()}))
    print(
        f"⚠  RW_LLM_PROVIDER={env_provider!r} replaces the provider but NOT the "
        f"model.\n"
        f"    {config_path().name} picked its models for {chosen_for} — e.g. role "
        f"{role_name!r} now resolves to {env_provider}/{cfg.model}.\n"
        f"    If that pair isn't real, unset RW_LLM_PROVIDER and pick a config "
        f"with RW_MODELS_CONFIG instead.",
        file=sys.stderr,
    )
    _env_model_mismatch_warned = True


_missing_base_url_warned = False


def clear_caches() -> None:
    """Forget every process-scoped model-routing decision and warning.

    ``RW_MODELS_CONFIG`` is normally fixed for one CLI process, but the benchmark
    A/B harness deliberately switches it between arms. Keeping cache invalidation
    in one public helper prevents the model, endpoint, and ingest settings from
    coming from different profiles.
    """
    global _env_override_warned, _env_model_mismatch_warned, _missing_base_url_warned
    _load_routing_snapshot.cache_clear()
    _env_override_warned = False
    _env_model_mismatch_warned = False
    _missing_base_url_warned = False


def validate_config() -> None:
    """Load every routing section, raising on any unusable selected config."""
    _config()
    _ingest_settings()
    base_url()
    anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if anthropic_base_url:
        validate_env_base_url(
            anthropic_base_url, variable="ANTHROPIC_BASE_URL",
        )


def _maybe_warn_missing_base_url() -> None:
    """Compatibility no-op for the former warning-only endpoint check.

    The shared snapshot now rejects this state before returning any routing
    data, so warning and continuing would reintroduce the unsafe localhost
    fallback. The helper remains temporarily to avoid a needless call-path
    change while downstream code migrates.
    """


def for_phase(name: str) -> ModelConfig:
    """Resolve a phase to its effective ModelConfig.

    Looks up the phase's bound role, then applies any per-phase overrides
    (provider / model / temperature / max_tokens). Raises KeyError when
    the phase isn't registered or its target role is missing.

    Final step: if the `RW_LLM_PROVIDER` env var is set, override the
    resolved provider for every phase. This is the one-shot opt-in for
    subscription users (`RW_LLM_PROVIDER=chat-relay <cmd>`) — mirrors the
    existing `RW_LLM_BASE_URL` env-var pattern for OpenAI-compatible
    base-URL overrides.
    """
    roles, phases = _config()
    spec = phases.get(name)
    if spec is None:
        raise PhaseNotRegistered(
            f"phase {name!r} not in model config (known: {sorted(phases)}). "
            f"Register it in _FALLBACK_PHASES in agents/model_config.py."
        )
    role_name = spec["role"]
    base = roles.get(role_name)
    if base is None:
        raise PhaseNotRegistered(
            f"role {role_name!r} (for phase {name!r}) not in model config"
        )
    # Apply per-phase overrides.
    # reasoning_effort: only override if the phase spec explicitly includes
    # the key (so a phase can both *set* a value and *clear* it to None).
    if "reasoning_effort" in spec:
        re_raw = spec["reasoning_effort"]
        re_val: str | None = str(re_raw) if re_raw is not None else None
    else:
        re_val = base.reasoning_effort
    rpm_val = int(spec["rpm"]) if spec.get("rpm") is not None else base.rpm
    cfg = ModelConfig(
        provider=canonical_provider_id(
            str(spec.get("provider", base.provider)),
        ),
        model=str(spec.get("model", base.model)),
        temperature=float(spec.get("temperature", base.temperature)),
        max_tokens=int(spec.get("max_tokens", base.max_tokens)),
        reasoning_effort=re_val,
        rpm=rpm_val,
    )
    # Env-var override applied last — wins over config/models.yaml. Used by
    # subscription users to flip every phase to chat-relay without editing
    # the YAML.
    _maybe_warn_missing_base_url()
    env_provider = _env_provider_override()
    if env_provider:
        _maybe_warn_env_override_defeats_mixing()
        _maybe_warn_env_override_model_mismatch()
        cfg = replace(cfg, provider=env_provider)
    return cfg


def for_role(name: str) -> ModelConfig:
    """Resolve a role directly. For ad-hoc calls that don't fit a phase."""
    roles, _ = _config()
    cfg = roles.get(name)
    if cfg is None:
        raise PhaseNotRegistered(f"role {name!r} not in model config")
    env_provider = _env_provider_override()
    if env_provider:
        cfg = replace(cfg, provider=env_provider)
    return cfg


def uses_chat_relay() -> bool:
    """True when any registered phase resolves to the `chat-relay` provider.

    Two ways it can be on, and a caller that checks only the first will miss the
    second: `RW_LLM_PROVIDER=chat-relay` forces every phase (the usual route for a
    subscription user), but `config/models.yaml` can also route individual phases
    to it while the rest stay on an API provider. The env check is just a fast
    path — `for_phase` applies that override itself.

    Used to warn before a batch run, where chat-relay's handoff notice would be
    redirected into a per-worker log file nobody is watching.
    """
    # Always materialize the strict snapshot first. The env fast path used to
    # return before a malformed selected file was read, letting batch workers
    # start only to fail after spawning.
    _routing_snapshot()
    if _env_provider_override() == "chat-relay":
        return True
    for name in list_phases():
        try:
            if for_phase(name).provider == "chat-relay":
                return True
        except PhaseNotRegistered:
            continue
    return False


def list_phases() -> list[str]:
    """Sorted list of all registered phase names. For introspection."""
    _, phases = _config()
    return sorted(phases.keys())


def list_roles() -> list[str]:
    """Sorted list of all registered role names."""
    roles, _ = _config()
    return sorted(roles.keys())


# ---------------------------------------------------------------------------
# Pricing
#
# Per-token rates are the other half of "facts about a model", so they live here
# beside the routing rather than in their own module — same `config/*.yaml`
# source, same lru_cache convention, and `tasks/status.py` already imports
# `config_path` from here.
#
# Data lives in `config/pricing.yaml` with an `as_of:` date. Every caller that
# prints a dollar figure prints that date too: a rate table nobody has
# re-checked is the normal state, and showing the date makes it visible rather
# than silently wrong.
# ---------------------------------------------------------------------------

_PRICING_FILENAME = "pricing.yaml"

#: Values `ingest_iterations.model_used` uses for "no API call happened".
NON_MODEL_SENTINELS = frozenset({
    "(local)", "(skipped)", "(no calls)", "(failed)", "(missing)", "stub", "",
})


@dataclass(frozen=True)
class Rate:
    """USD per million tokens for one model."""
    model_key: str
    input_per_mtok: float
    output_per_mtok: float
    provider: str = ""
    note: str = ""

    def usd(self, in_tok: int, out_tok: int) -> float:
        return (in_tok / 1_000_000) * self.input_per_mtok + \
               (out_tok / 1_000_000) * self.output_per_mtok


def pricing_path() -> Path:
    """`$RW_PRICING_CONFIG` if set, else `config/pricing.yaml` beside the package.

    Unlike `config_path()`, this resolves relative to the *package*, not the
    cwd: the rate table is shipped data, not per-wiki state, so `insights` must
    find it from any working directory.
    """
    override = os.environ.get("RW_PRICING_CONFIG")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() or p.parent != Path(".") else Path("config") / p
    return Path(__file__).resolve().parent.parent.parent / "config" / _PRICING_FILENAME


@lru_cache(maxsize=1)
def _pricing() -> dict:
    """Parsed `pricing.yaml`. Cached like `_config`; tests clear with
    `_pricing.cache_clear()`.

    A missing or malformed table yields `{}` rather than raising — `status`
    should still print its report, just without a cost estimate.
    """
    try:
        import yaml
        with pricing_path().open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def pricing_as_of() -> str:
    """Date the rates were last verified (`YYYY-MM-DD`), or '' if absent."""
    return str(_pricing().get("as_of") or "")


def pricing_sources() -> dict:
    """Vendor pricing URLs the rates were read from."""
    return dict(_pricing().get("sources") or {})


def pricing_modifiers() -> dict:
    """Cache / batch multipliers, for reference only.

    Never applied automatically: `ingest_iterations` records just input and
    output totals, so there is no cache-read/write split to multiply.
    """
    return dict(_pricing().get("modifiers") or {})


def _pick_dated_rate(entries: list, today: _dt.date) -> dict | None:
    """Choose among time-boxed entries for one model.

    `until: YYYY-MM-DD` applies through that date inclusive; the entry with no
    `until` is the fallback. Sonnet 5's introductory rate is the live case — it
    lapses 2026-08-31, and a table that couldn't express that would start
    misreporting on a specific calendar day with nothing to explain why.
    """
    fallback: dict | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        until = e.get("until")
        if not until:
            fallback = fallback or e
            continue
        try:
            end = _dt.date.fromisoformat(str(until))
        except ValueError:
            continue
        if today <= end:
            return e
    return fallback


def rate_for(model: str, *, today: _dt.date | None = None) -> Rate | None:
    """Rate for `model`, or None when it isn't priced.

    Exact key first, then **longest matching prefix**. The prefix match is not
    cosmetic: the recorded `model_used` is whatever the SDK echoed back, usually
    a dated build ID. The dict this replaced was keyed on `claude-haiku-4-5`,
    the API returns `claude-haiku-4-5-20251001`, `dict.get` missed, and 429
    calls over 2.7M input tokens priced at $0.00 in every report. Longest —
    rather than first — match keeps `claude-haiku-3-5` from being shadowed.

    None for a local backend (LM Studio, Ollama, gemma-*, qwen-*), which has no
    per-token price. See `unpriced_models` to tell that apart from a cloud model
    missing from the table.
    """
    if not model or model in NON_MODEL_SENTINELS:
        return None
    table = _pricing().get("models") or {}
    if not isinstance(table, dict):
        return None
    today = today or _dt.date.today()

    key: str | None = None
    if model in table:
        key = model
    else:
        candidates = [k for k in table if model.startswith(k)]
        if candidates:
            key = max(candidates, key=len)
    if key is None:
        return None

    entry = table[key]
    if isinstance(entry, list):
        entry = _pick_dated_rate(entry, today)
    if not isinstance(entry, dict):
        return None
    try:
        return Rate(
            model_key=key,
            input_per_mtok=float(entry["in"]),
            output_per_mtok=float(entry["out"]),
            provider=str(entry.get("provider") or ""),
            note=str(entry.get("note") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def estimate_usd(model: str, in_tok: int, out_tok: int, *,
                 today: _dt.date | None = None) -> float:
    """Estimated USD for one model's token totals; 0.0 when unpriced.

    An **upper bound**: a prompt-cache hit costs 0.1x base input and the author
    phase passes `cache_prompt=True`, but the schema records no cache split, so
    a cached run really cost less than this returns.
    """
    rate = rate_for(model, today=today)
    return rate.usd(in_tok, out_tok) if rate else 0.0


def unpriced_models(models) -> list[str]:
    """Which of `models` have no rate, excluding the not-a-model sentinels.

    Separates "local model, $0.00 is correct" from "cloud model missing from the
    table, $0.00 is a lie" — `status` surfaces the second so a stale table gets
    noticed instead of quietly understating the bill.
    """
    return sorted({
        m for m in models
        if m and m not in NON_MODEL_SENTINELS and rate_for(m) is None
    })
