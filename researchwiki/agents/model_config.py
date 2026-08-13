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

If `config/models.yaml` is missing, malformed, or PyYAML isn't installed,
the loader silently falls back to the hardcoded defaults below. That makes
the framework usable on a fresh checkout without any new dependency.

`provider` is forward-compatible. Today only "anthropic" is implemented in
researchwiki/agents/llm.py. When a second provider is added (OpenAI,
Ollama, etc.), this config schema doesn't change — just edit a role's
`provider:` to point at it.
"""

from __future__ import annotations

import datetime as _dt
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
    env_provider = (os.environ.get("RW_LLM_PROVIDER") or "").strip()
    if not env_provider or env_provider.lower() == "chat-relay":
        return
    roles, _ = _config()
    clashing = {n: c for n, c in roles.items()
                if c.provider.lower() != env_provider.lower()}
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


def _maybe_warn_missing_base_url() -> None:
    """Fire once per process when a config declares OpenAI-compatible roles but
    no top-level `base_url:`.

    `base_url()` returns None there and `call_openai_compatible` reads None as
    "use the LM Studio default", so a cloud config missing one key silently
    becomes a localhost one. The asymmetry is what makes it hard to spot: a
    *missing* config file falls back to OpenAI (`_FALLBACK_BASE_URL`), while a
    *present* file with no `base_url:` falls back to http://localhost:1234/v1.
    Same "unspecified endpoint", two different answers.

    No shipped template trips this — the two without `base_url:`
    (models.anthropic.yaml, models.glm.yaml) both route to the `anthropic`
    provider, which ignores it — so this is aimed at hand-edited configs.
    Silent when RW_LLM_BASE_URL is set, since that supplies the endpoint.
    """
    global _missing_base_url_warned
    if _missing_base_url_warned:
        return
    if os.environ.get("RW_LLM_BASE_URL"):
        return
    path = config_path()
    if not path.exists() or base_url() is not None:
        return
    # Lazy import: llm imports this module at module scope, so the reverse
    # edge has to be deferred to call time.
    from .llm import _DEFAULT_LOCAL_BASE_URL, _OPENAI_COMPAT_PROVIDERS
    roles, _ = _config()
    compat = sorted(n for n, c in roles.items()
                    if c.provider.lower() in _OPENAI_COMPAT_PROVIDERS)
    if not compat:
        return
    # "resolve to", not "declares": the fallback fills roles the file omits, so
    # some of these are inherited rather than written down in it.
    shown = ", ".join(compat[:4]) + (", …" if len(compat) > 4 else "")
    print(
        f"⚠  With {path.name} in force, role(s) {shown} resolve to an "
        f"OpenAI-compatible provider, but no top-level `base_url:` is set.\n"
        f"    Those calls will go to {_DEFAULT_LOCAL_BASE_URL} (the LM Studio "
        f"default), not to a cloud endpoint.\n"
        f"    Add `base_url:` to the config, or set RW_LLM_BASE_URL.",
        file=sys.stderr,
    )
    _missing_base_url_warned = True


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
    _maybe_warn_missing_base_url()
    env_provider = os.environ.get("RW_LLM_PROVIDER")
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
    if (os.environ.get("RW_LLM_PROVIDER") or "").strip().lower() == "chat-relay":
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
