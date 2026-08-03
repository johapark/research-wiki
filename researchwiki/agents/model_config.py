"""Central model configuration. Stage 1 of multi-model support.

Maps each LLM-using *phase* of the framework to a *role* (author / critic /
judge / classifier / proposer), and each role to a (provider, model,
temperature, max_tokens) tuple. Source of truth: `config/models.yaml` at
the repo root.

Public API:
  for_phase(name) -> ModelConfig
  for_role(name)  -> ModelConfig
  list_phases()   -> list[str]

If `config/models.yaml` is missing, malformed, or PyYAML isn't installed,
the loader silently falls back to the hardcoded defaults below. That makes
the framework usable on a fresh checkout without any new dependency.

`provider` is forward-compatible. Today only "anthropic" is implemented in
researchwiki/agents/llm.py. When a second provider is added (OpenAI,
Ollama, etc.), this config schema doesn't change — just edit a role's
`provider:` to point at it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from ..paths import wiki_root
from ..log import log


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
    override = os.environ.get("RW_MODELS_CONFIG")
    if not override:
        return wiki_root() / "config" / "models.yaml"
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


# Hardcoded defaults — used if `config/models.yaml` is absent / malformed
# or PyYAML isn't installed. Mirrors `config/models.chatgpt.yaml` (the
# recommended default) so a fresh checkout with OPENAI_API_KEY set works with
# no copy. The paired endpoint is _FALLBACK_BASE_URL below; keep the two in
# sync, and update both whenever the default config's roles change.
_FALLBACK_ROLES: dict[str, ModelConfig] = {
    "author":     ModelConfig("openai-compatible", "gpt-5.6-luna", 0.5, 6000),
    "critic":     ModelConfig("openai-compatible", "gpt-5.6-luna", 0.3, 2500),
    "judge":      ModelConfig("openai-compatible", "gpt-5.6-luna", 0.2, 1500),
    "classifier": ModelConfig("openai-compatible", "gpt-5.4-mini", 0.1, 200),
    "proposer":   ModelConfig("openai-compatible", "gpt-5.6-luna", 0.3, 200),
    "extractor":  ModelConfig("openai-compatible", "gpt-5.4-mini", 0.0, 800),
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


def _load_yaml() -> tuple[dict[str, ModelConfig], dict[str, dict]] | None:
    """Read config/models.yaml. Returns (roles, phase-overrides) on success.
    None if file missing, PyYAML absent, or schema malformed — caller falls
    back to hardcoded defaults."""
    path = config_path()
    if not path.exists():
        # Silent fallback is fine for the default path (fresh clone), but an
        # explicit RW_MODELS_CONFIG that points nowhere is a misconfig worth
        # surfacing rather than silently routing to the hardcoded defaults.
        if os.environ.get("RW_MODELS_CONFIG"):
            log(f"RW_MODELS_CONFIG={os.environ['RW_MODELS_CONFIG']!r} resolves to "
                f"{path}, which does not exist — using hardcoded defaults", tag="model-config")
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"could not parse {path}: {e}", tag="model-config")
        return None
    if not isinstance(data, dict):
        return None

    roles: dict[str, ModelConfig] = {}
    for name, fields in (data.get("roles") or {}).items():
        if not isinstance(fields, dict) or "model" not in fields:
            continue
        try:
            re_raw = fields.get("reasoning_effort")
            rpm_raw = fields.get("rpm")
            roles[name] = ModelConfig(
                provider=str(fields.get("provider", "anthropic")),
                model=str(fields["model"]),
                temperature=float(fields.get("temperature", 0.5)),
                max_tokens=int(fields.get("max_tokens", 2000)),
                reasoning_effort=str(re_raw) if re_raw is not None else None,
                rpm=int(rpm_raw) if rpm_raw is not None else None,
            )
        except (ValueError, TypeError) as e:
            log(f"bad role {name!r}: {e}", tag="model-config")
            continue

    phases: dict[str, dict] = {}
    for phase, spec in (data.get("phases") or {}).items():
        if not isinstance(phase, str):
            continue
        if isinstance(spec, str):
            phases[phase] = {"role": spec}
        elif isinstance(spec, dict) and "role" in spec:
            phases[phase] = {k: v for k, v in spec.items()
                             if k in ("role", "model", "provider",
                                      "temperature", "max_tokens",
                                      "reasoning_effort", "rpm")}
    return roles, phases


@lru_cache(maxsize=1)
def _ingest_settings() -> dict:
    """Top-level `ingest:` block from the models config — pipeline defaults
    that aren't per-role (as opposed to `roles:` / `phases:`). Optional; an
    empty dict when the file is absent, unparsable, or has no `ingest:` key.

    Currently recognizes `n_drafts` (default author-draft count when the CLI
    `-n` flag is omitted). Cached like `_config`; test resets clear both.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    ing = data.get("ingest")
    return dict(ing) if isinstance(ing, dict) else {}


def default_n_drafts() -> int | None:
    """Config default for author drafts (`ingest.n_drafts`), or None when
    unset/invalid so the caller falls back to its hardcoded default. A value
    < 1 is treated as unset (a tournament needs >= 1 draft). The CLI `-n`
    flag, when passed, overrides this.
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


@lru_cache(maxsize=1)
def base_url() -> str | None:
    """Top-level `base_url:` from the models config — the OpenAI-compatible
    endpoint for this config's `openai-compatible`/`lmstudio`/`openai` roles.

    Folding the endpoint into the config file means switching backends needs
    only `RW_MODELS_CONFIG` to change, not a paired `RW_LLM_BASE_URL` edit.
    Precedence at the call site (see llm.call): the `RW_LLM_BASE_URL` env var,
    when set, still wins as an ad-hoc override; then this; then the built-in
    LM Studio localhost default. Returns None when the file is present but
    declares no `base_url:`. On the no-config-file path returns
    `_FALLBACK_BASE_URL` — the OpenAI endpoint paired with the openai-compatible
    fallback roles — so a fresh checkout reaches OpenAI, not localhost.
    `anthropic` roles ignore it (they use `ANTHROPIC_BASE_URL`).
    """
    path = config_path()
    if not path.exists():
        return _FALLBACK_BASE_URL
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    url = data.get("base_url")
    return str(url) if url else None


@lru_cache(maxsize=1)
def _config() -> tuple[dict[str, ModelConfig], dict[str, dict]]:
    """Resolve the merged config — YAML overrides, then fallback fills gaps.

    Cached so the YAML is read once per process. Tests that need to
    reset can clear with `_config.cache_clear()`.
    """
    loaded = _load_yaml()
    if loaded is None:
        return dict(_FALLBACK_ROLES), dict(_FALLBACK_PHASES)
    yaml_roles, yaml_phases = loaded
    # Merge: YAML wins, fallback fills gaps so missing roles still resolve.
    roles = {**_FALLBACK_ROLES, **yaml_roles}
    phases = {**_FALLBACK_PHASES, **yaml_phases}
    return roles, phases


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
    env_provider = os.environ.get("RW_LLM_PROVIDER")
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
        provider=str(spec.get("provider", base.provider)),
        model=str(spec.get("model", base.model)),
        temperature=float(spec.get("temperature", base.temperature)),
        max_tokens=int(spec.get("max_tokens", base.max_tokens)),
        reasoning_effort=re_val,
        rpm=rpm_val,
    )
    # Env-var override applied last — wins over config/models.yaml. Used by
    # subscription users to flip every phase to chat-relay without editing
    # the YAML.
    env_provider = os.environ.get("RW_LLM_PROVIDER")
    if env_provider:
        _maybe_warn_env_override_defeats_mixing()
        cfg = replace(cfg, provider=env_provider)
    return cfg


def for_role(name: str) -> ModelConfig:
    """Resolve a role directly. For ad-hoc calls that don't fit a phase."""
    roles, _ = _config()
    cfg = roles.get(name)
    if cfg is None:
        raise PhaseNotRegistered(f"role {name!r} not in model config")
    return cfg


def list_phases() -> list[str]:
    """Sorted list of all registered phase names. For introspection."""
    _, phases = _config()
    return sorted(phases.keys())


def list_roles() -> list[str]:
    """Sorted list of all registered role names."""
    roles, _ = _config()
    return sorted(roles.keys())
