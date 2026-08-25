"""LLM wrapper for the ingest agent.

Four call modes:
  - **Anthropic** (`provider: anthropic`): cloud API via the official SDK.
    Requires ANTHROPIC_API_KEY in the environment. Supports prompt caching
    via the cache_control beta header.
  - **OpenAI-compatible** (`provider: openai-compatible` or `lmstudio`):
    POST to a `/v1/chat/completions` endpoint that speaks the OpenAI shape.
    Covers LM Studio, vLLM, llama.cpp server, ollama-in-OpenAI-mode, and
    real OpenAI. Default base URL is LM Studio's localhost
    (`http://localhost:1234/v1`); override with `RW_LLM_BASE_URL` env var.
    No prompt caching (the spec doesn't expose it). Forwards
    `reasoning_effort` when set in the role config — recognized by Gemini
    2.5 Flash/Pro via Google's `/v1beta/openai/` shim and by OpenAI
    o-series; cleanly ignored by most local servers.
  - **Chat-relay** (`provider: chat-relay`): delegate the prompt to a
    chat-platform agent (Claude Code, Codex, Cursor, …) running in the
    same terminal session via a filesystem protocol. Implementation lives
    in `agents/relay.py` (split out at phase 2 when the validation logic
    started outgrowing this file). No API key required — designed for
    users on a chat subscription with no Anthropic/OpenAI credentials and
    no local LM. Supports JSON Schema validation + retry-with-feedback
    when the caller passes `schema=`.
  - **Stub** (`use_stub=True` at the call site, regardless of provider):
    deterministic placeholder responses keyed off the prompt + temperature.
    Lets the framework be tested offline; reproducible snapshots for tests.

The mode is selected by the caller (stub) and the role's `provider:` config
(real). Explicit choice keeps test/dev/prod boundaries unambiguous.

Model selection is centralized in `config/models.yaml` via the
`model_config` module. Call sites pass `phase="author"` (or another phase
name) and inherit (provider, model, temperature, max_tokens) from the config.
Explicit `model` / `temperature` / `max_tokens` kwargs at the call site
override the config — useful for the author phase where each parallel draft
varies temperature.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import model_config
from ..errors import EnvironmentFailure


# --- Client-side rate limiting -------------------------------------------
# Keyed by model string so roles that share a model (e.g. every role in
# models.gemini-flash-lite.yaml) share one budget rather than each getting
# its own. Process-local: the lock serializes threads (batch mode uses a
# thread pool), but does NOT coordinate across multiprocessing workers.
_throttle_lock = threading.Lock()
_last_call_at: dict[str, float] = {}


def _throttle(model: str, rpm: int | None) -> None:
    """Block until at least 60/rpm seconds have elapsed since the last call
    to `model`. No-op when `rpm` is None or <= 0.

    The lock is intentionally held across the sleep: concurrent workers then
    queue and each computes its wait against the previous caller's updated
    timestamp, so N threads calling one model serialize onto a steady
    ~rpm/minute cadence instead of bursting past the provider's ceiling.
    """
    if not rpm or rpm <= 0:
        return
    interval = 60.0 / rpm
    with _throttle_lock:
        last = _last_call_at.get(model)
        if last is not None:
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        _last_call_at[model] = time.monotonic()


@dataclass
class LLMResponse:
    text: str
    model: str
    temperature: float
    input_tokens: int
    output_tokens: int


def has_synchronous_llm() -> bool:
    """True if a fast-turnaround LLM provider is reachable.

    Means: an Anthropic API key is set, OR an OpenAI key is set (for real
    OpenAI), OR a local OpenAI-compatible endpoint is plausibly reachable.
    We don't probe the network: an explicit RW_LLM_BASE_URL is the user's
    signal that they have a server, while a loopback ``base_url:`` in the
    selected models YAML is the equivalent config-file signal.

    Chat-relay does NOT count as synchronous. Each call blocks polling for
    a chat agent's response, with a default 10-minute timeout. Callers
    that need millisecond-to-second turnaround (interactive REPL
    completions, hot-loop validators) check this signal; callers that
    just need *some* LLM reachable use `has_any_llm()` instead.
    """
    _validated_anthropic_base_url()
    env_endpoint = os.environ.get("RW_LLM_BASE_URL")
    if env_endpoint:
        model_config.validate_env_base_url(env_endpoint)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if os.environ.get("OPENAI_API_KEY"):
        return True
    # Local LM-server signal: either explicitly configured or implied.
    if env_endpoint:
        return True
    configured_url = model_config.base_url()
    return bool(configured_url and _is_local_endpoint(configured_url))


def has_any_llm() -> bool:
    """True if any provider — synchronous or relay — is available.

    Includes everything `has_synchronous_llm()` covers, plus chat-relay
    (signaled by RW_LLM_PROVIDER=chat-relay or by the `.llm-relay/`
    directory existing in the wiki root). Use this for "can we do LLM work
    at all?" gates; use `has_synchronous_llm()` when latency matters.
    """
    if has_synchronous_llm():
        return True
    if (os.environ.get("RW_LLM_PROVIDER") or "").strip().lower() == "chat-relay":
        return True
    # Lazy import to dodge the llm.py ↔ relay.py cycle.
    from .relay import _relay_dir
    if _relay_dir().exists():
        return True
    return False


# Backward-compat alias. Existing call sites used `is_real_mode_available`
# to gate "should I attempt a real call vs fall back to stub" — the
# semantically-equivalent check today is `has_synchronous_llm` (the relay
# path can take minutes; treating it as "real mode available" would
# silently introduce long blocks where callers expected fast paths).
def is_real_mode_available() -> bool:
    return has_synchronous_llm()


# Default base URL for OpenAI-compatible local endpoints. LM Studio ships
# with this URL; vLLM / llama.cpp / ollama users can override via env.
_DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"

_OPENAI_COMPAT_PROVIDERS = model_config.OPENAI_COMPAT_PROVIDERS


def _validated_anthropic_base_url() -> str | None:
    """Validate the SDK-controlled endpoint without echoing its raw value."""
    value = os.environ.get("ANTHROPIC_BASE_URL")
    if not value:
        return None
    return model_config.validate_env_base_url(
        value, variable="ANTHROPIC_BASE_URL",
    )


@dataclass(frozen=True)
class EndpointResolution:
    """Effective OpenAI-compatible endpoint and the layer that selected it."""

    url: str
    source: str  # "env" | "config" | "fallback"

    @property
    def display_url(self) -> str:
        """URL safe for diagnostics: omit user-info, query, and fragment."""
        try:
            parsed = urllib.parse.urlsplit(self.url)
            host = parsed.hostname or ""
            port = parsed.port
        except ValueError:
            return "(invalid endpoint URL)"
        if not host:
            return "(invalid endpoint URL)"
        netloc = f"[{host}]" if ":" in host else host
        if port is not None:
            netloc += f":{port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


class ProviderUnavailable(EnvironmentFailure):
    """The configured LLM provider has no usable credentials — exit code 2.

    Raised by `preflight_providers()` before a run spends anything, not by the
    call sites: a provider that 401s mid-pipeline is the same condition
    discovered too late.
    """


def resolve_openai_endpoint() -> EndpointResolution:
    """The endpoint an `openai-compatible` role will actually POST to.

    One resolver serves the request path, provider preflight, and ``status`` so
    their precedence and provenance cannot drift apart.
    """
    override = os.environ.get("RW_LLM_BASE_URL")
    if override:
        return EndpointResolution(model_config.validate_env_base_url(override), "env")
    configured = model_config.base_url()
    if configured:
        source = "config" if model_config.config_file_present() else "fallback"
        return EndpointResolution(configured, source)
    return EndpointResolution(_DEFAULT_LOCAL_BASE_URL, "fallback")


def _resolved_openai_base_url() -> str:
    """Backward-compatible URL-only view of :func:`resolve_openai_endpoint`."""
    return resolve_openai_endpoint().url


def _is_local_endpoint(url: str) -> bool:
    """True for a loopback endpoint, which needs no API key — LM Studio, vLLM,
    llama.cpp and ollama all ignore the Bearer token."""
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def missing_provider_credentials() -> list[str]:
    """Every provider this config routes to but has no usable credentials for.

    Returns one human-readable problem per unusable provider, empty when the
    config is runnable. Pure environment + config inspection: no network, no
    tokens, microseconds.

    Checks *all* registered phases rather than a curated ingest subset. An
    ingest already reaches nearly every phase — author, critic, debug, evolve,
    keywords, link_generation, memory_evolution, reconcile, short_name,
    target_claims, and classifier via the category auto-suggest — so a
    hand-listed subset would be a near-superset that silently rots as phases
    are added, which is how the gap this closes appeared in the first place.

    Deliberately stricter than `has_synchronous_llm()`, which answers "is any
    key set anywhere" and so returns True for the failure README and
    prompts/init.md both call the one that actually happens: an Anthropic key
    set, the config copy skipped, and every role therefore still routed to
    OpenAI.
    """
    _validated_anthropic_base_url()
    providers: set[str] = set()
    for name in model_config.list_phases():
        try:
            providers.add(
                model_config.canonical_provider_id(
                    model_config.for_phase(name).provider,
                ),
            )
        except model_config.PhaseNotRegistered:
            continue

    problems: list[str] = []
    if "anthropic" in providers and not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append(
            "anthropic: ANTHROPIC_API_KEY is not set — put it in .env at the "
            "wiki root (loaded on every invocation) or export it."
        )
    if providers & _OPENAI_COMPAT_PROVIDERS:
        url = _resolved_openai_base_url()
        if not _is_local_endpoint(url) and not os.environ.get("OPENAI_API_KEY"):
            msg = (
                f"openai-compatible: OPENAI_API_KEY is not set and the endpoint "
                f"is remote ({url}) — set that provider's key in .env, or point "
                f"RW_LLM_BASE_URL at a local server."
            )
            if os.environ.get("ANTHROPIC_API_KEY"):
                # The documented trap: an Anthropic key is present, so the user
                # believes they are configured, but no config file means every
                # role still resolves to the OpenAI-compatible fallback.
                msg += (
                    "\n    You have ANTHROPIC_API_KEY set but this config routes "
                    "to OpenAI — Anthropic needs its own config file: "
                    "`cp config/models.anthropic.yaml config/models.yaml`."
                )
            problems.append(msg)
    # chat-relay needs no credentials; the stub path never reaches a provider.
    return problems


def preflight_providers() -> None:
    """Raise `ProviderUnavailable` before a run spends anything, if it can't run.

    `agent ingest` used to discover a missing key only in the author phase —
    after PDF extraction and metadata reconcile had already run — and let it
    escape as an uncaught RuntimeError, so a plain configuration error exited 3
    ("internal bug — file a report") with a traceback. Worse, the
    OpenAI-compatible path defaults an unset `OPENAI_API_KEY` to the literal
    string `lm-studio`, so the diagnostic a new user actually got was a 401
    quoting a value they had never typed.
    """
    problems = missing_provider_credentials()
    if not problems:
        return
    cfg = model_config.config_path()
    where = str(cfg) if cfg.exists() else f"{cfg} (missing — using built-in defaults)"
    raise ProviderUnavailable(
        "no usable LLM provider for this run.\n"
        + "\n".join(f"  - {p}" for p in problems)
        + f"\n  active model config: {where}"
        + "\n  fix: see README.md § Providers, or run `researchwiki init`."
    )


# Retry policy for transient OpenAI-compatible failures. Gemini's free tier
# returns 503 ("model experiencing high demand") and 429 (rate limit) under
# load; these are transient, so we retry with exponential backoff + jitter
# rather than failing the whole ingest. Local servers (LM Studio) rarely hit
# these but inherit the same harmless policy. 401 is included because OpenAI
# intermittently returns spurious 401 "insufficient permissions" on valid keys
# (observed 2026-07-18: some calls in a run 401 while others on the same key +
# model succeed; the retry cleared it). A genuinely bad key still fails — just
# after the bounded backoff budget rather than instantly. Other 4xx (400 bad
# request, 403, 404) are NOT retried — they won't succeed on a retry.
_RETRY_STATUS = frozenset({401, 429, 500, 502, 503, 504})
_RETRY_MAX_ATTEMPTS = 5          # total tries, including the first
_RETRY_BASE_DELAY = 2.0          # seconds; delay = base * 2**(attempt-1) + jitter
_RETRY_MAX_DELAY = 60.0          # cap any single backoff wait

# OpenAI's GPT-5+/o-series reject the legacy `max_tokens` field and require
# `max_completion_tokens` instead (HTTP 400 otherwise). Everything else on this
# code path — older OpenAI models, Gemini's OpenAI shim, Groq llama, Upstage
# solar, LM Studio/vLLM locals — still takes `max_tokens`, so the swap is gated
# on the model name. Matches gpt-5..gpt-99 and o1..o9 (optionally provider-
# prefixed, e.g. OpenRouter's `openai/gpt-5`); leaves gpt-4/gpt-4o untouched.
_MAX_COMPLETION_TOKENS_RE = re.compile(
    r"(?:^|/)(?:gpt-(?:[5-9]|\d{2,})|o[1-9])\b", re.IGNORECASE
)


def _needs_max_completion_tokens(model: str) -> bool:
    """True for OpenAI models that reject `max_tokens` (GPT-5+/o-series)."""
    return bool(_MAX_COMPLETION_TOKENS_RE.search(model or ""))


# --- reasoning_effort negotiation ----------------------------------------
# Models disagree about this field's vocabulary and the disagreement is a hard
# 400, not a warning. OpenAI o-series/GPT-5 take "minimal"; some newer models
# take "none" and reject "minimal"; several take "xhigh"; local servers ignore
# the field or reject it outright. A static per-model table would go stale with
# every release, so instead we ask the server: on a 400 that names the field we
# step to the nearest value it will accept, and remember the answer.
#
# Ascending order of thinking. Negotiation walks outward from the requested
# value, preferring the *lower* side on ties — a caller that asked for less
# thinking (disable_thinking maps here) should not be nudged upward into paying
# for reasoning tokens it explicitly declined.
_EFFORT_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh")

# Per (endpoint, model): values the server rejected, and the vocabulary it
# advertised in an error body (when it bothered to). Process-local; the lock
# serializes the thread pool used by batch mode. Worst case on a race is a
# duplicate negotiation, not a wrong answer.
_EFFORT_REJECTED: dict[tuple[str, str], set[str]] = {}
_EFFORT_SUPPORTED: dict[tuple[str, str], set[str]] = {}
_EFFORT_LOCK = threading.Lock()

# Bounded so a server that 400s on every value can't spin. Three steps is
# enough to cross the whole scale from any starting point and still land on
# "omit the field", which every server accepts.
_EFFORT_MAX_RENEGOTIATIONS = 3

# "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
_EFFORT_SUPPORTED_RE = re.compile(
    r"supported\s+values?\s+(?:are|is)\s*:?\s*(?P<list>[^.]+)", re.IGNORECASE
)
_QUOTED_RE = re.compile(r"['\"`]([a-z_]+)['\"`]", re.IGNORECASE)


def _mentions_reasoning_effort(body_text: str) -> bool:
    """True when an error body implicates the reasoning-effort field.

    LiteLLM/Bedrock can wrap an upstream 400 as HTTP 500 and spell the nested
    parameter ``reasoning.effort``. Keep this deliberately narrow: an error
    that names neither spelling is a real request/transient failure and must
    retain the normal retry behavior.
    """
    text = body_text or ""
    return "reasoning_effort" in text or "reasoning.effort" in text


def _parse_supported_efforts(body_text: str) -> set[str]:
    """Pull the advertised vocabulary out of an error body, if present.

    Only reads the span after "Supported values are", so the rejected value
    quoted earlier in the same sentence isn't mistaken for a supported one.
    Returns an empty set when the server didn't say — callers then fall back
    to walking `_EFFORT_SCALE`.
    """
    m = _EFFORT_SUPPORTED_RE.search(body_text or "")
    if not m:
        return set()
    return {v.lower() for v in _QUOTED_RE.findall(m.group("list"))}


def _nearest_effort(
    requested: str,
    rejected: set[str],
    supported: set[str] | None,
) -> str | None:
    """Closest acceptable effort to `requested`, or None to omit the field.

    `supported` is the server-advertised vocabulary when known; None means
    unknown, in which case every scale value is fair game until rejected.
    """
    if requested not in _EFFORT_SCALE:
        # Vocabulary we don't model (a provider-specific token). Try it once;
        # once rejected there's no scale position to walk from, so drop it.
        return None if requested in rejected else requested
    i = _EFFORT_SCALE.index(requested)
    for j in sorted(range(len(_EFFORT_SCALE)), key=lambda j: (abs(j - i), j)):
        cand = _EFFORT_SCALE[j]
        if cand in rejected:
            continue
        if supported is not None and cand not in supported:
            continue
        return cand
    return None


def _resolve_effort(key: tuple[str, str], requested: str | None) -> str | None:
    """Apply what we already learned about this endpoint+model."""
    if requested is None:
        return None
    with _EFFORT_LOCK:
        rejected = set(_EFFORT_REJECTED.get(key, ()))
        supported = _EFFORT_SUPPORTED.get(key)
        supported = set(supported) if supported else None
    if not rejected and supported is None:
        return requested
    if supported is None:
        # Rejected before, with no vocabulary offered — this endpoint doesn't
        # take the field at all. Don't re-litigate it on every later call.
        return None
    return _nearest_effort(requested, rejected, supported)


def _record_effort_rejection(
    key: tuple[str, str], value: str, body_text: str
) -> str | None:
    """Record a rejection and return the next value to try (None = omit).

    Two distinguishable rejections, and the difference matters:

    - The server **advertised a vocabulary** ("Supported values are: …"). It
      wants the field, just not that value — step to the nearest one it named.
    - The server **named no alternatives**. Then it is rejecting the *field*,
      not the value, and trying five more values is thrash that ends in the
      same place. Drop the field, which every server accepts.

    So a working request is always at most two round-trips away.
    """
    with _EFFORT_LOCK:
        _EFFORT_REJECTED.setdefault(key, set()).add(value)
        advertised = _parse_supported_efforts(body_text)
        if advertised:
            _EFFORT_SUPPORTED[key] = advertised
        rejected = set(_EFFORT_REJECTED[key])
        supported = _EFFORT_SUPPORTED.get(key)
        supported = set(supported) if supported else None
    if supported is None:
        return None
    return _nearest_effort(value, rejected, supported)


def call_openai_compatible(
    *,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    system: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    """POST to an OpenAI-compatible `/v1/chat/completions` endpoint.

    Covers LM Studio (default localhost:1234), vLLM, llama.cpp's
    `--server` mode, ollama-in-OpenAI-mode, and real OpenAI. Uses stdlib
    urllib to avoid pulling in the openai SDK as a runtime dep.

    Resolution order for `base_url`:
      1. explicit kwarg
      2. RW_LLM_BASE_URL environment variable
      3. _DEFAULT_LOCAL_BASE_URL (LM Studio default)

    `OPENAI_API_KEY` env var is forwarded as a Bearer token if set —
    real OpenAI requires it, LM Studio ignores it. Defaulted to
    `lm-studio` when unset so the request shape is always valid.

    `reasoning_effort` (when set) is forwarded as the OpenAI-spec field of
    the same name. Honored by Gemini 2.5 (Flash/Pro) via Google's OpenAI
    shim and by OpenAI o-series; ignored by servers that don't recognize
    it. Use it to bound thinking-token cost on flash/pro so the token
    budget reaches actual output. Local servers (LM Studio, vLLM) usually
    ignore it cleanly, but a strict server may 400 — omit it there.

    Prompt caching (Anthropic-only) is not supported; the spec doesn't
    expose KV-cache reuse hooks, so the call_prompt arg is silently
    ignored for this provider.
    """
    if base_url is None:
        env_base_url = os.environ.get("RW_LLM_BASE_URL")
        base_url = model_config.validate_env_base_url(
            env_base_url or _DEFAULT_LOCAL_BASE_URL,
            variable=(
                "RW_LLM_BASE_URL" if env_base_url
                else "OpenAI-compatible base_url"
            ),
        )
    else:
        base_url = model_config.validate_env_base_url(
            base_url, variable="OpenAI-compatible base_url",
        )

    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
    }
    # GPT-5+/o-series also reject any non-default `temperature` (only 1 is
    # allowed), so omit the field for them and let the server default apply;
    # every other model on this path honors the configured value.
    if _needs_max_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    api_key = os.environ.get("OPENAI_API_KEY", "lm-studio")
    url = f"{base_url.rstrip('/')}/chat/completions"

    # Start from whatever this endpoint+model has already accepted, so only the
    # first call of a run pays for discovery.
    effort_key = (base_url.rstrip("/"), model)
    effort = _resolve_effort(effort_key, reasoning_effort)

    def _encode_body() -> bytes:
        if effort is not None:
            payload["reasoning_effort"] = effort
        else:
            payload.pop("reasoning_effort", None)
        return json.dumps(payload).encode("utf-8")

    body = _encode_body()
    # Retry transient failures (401 / 429 / 5xx) with exponential backoff +
    # jitter. A fresh Request is built each attempt because urllib consumes the
    # body stream once. Non-retryable HTTP errors (other 4xx) and the final
    # exhausted attempt re-raise as RuntimeError, preserving prior behavior.
    data = None
    attempt = 1
    renegotiations = 0
    while True:
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:500]
            # A direct 400, or a LiteLLM-wrapped 500, naming reasoning effort
            # is the server telling us its vocabulary. Renegotiate and resend
            # without consuming the transient-retry budget: the next request
            # differs and the failure is deterministic rather than transient.
            if (
                e.code in (400, 500)
                and effort is not None
                and renegotiations < _EFFORT_MAX_RENEGOTIATIONS
                and _mentions_reasoning_effort(body_text)
            ):
                nxt = _record_effort_rejection(effort_key, effort, body_text)
                print(
                    f"[llm] {model} rejected reasoning_effort={effort!r}; "
                    + (
                        f"retrying with {nxt!r}" if nxt
                        else "retrying without the field"
                    ),
                    file=sys.stderr,
                )
                effort = nxt
                body = _encode_body()
                renegotiations += 1
                continue
            if e.code in _RETRY_STATUS and attempt < _RETRY_MAX_ATTEMPTS:
                delay = min(
                    _RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    _RETRY_MAX_DELAY,
                ) + random.uniform(0, 1)
                print(
                    f"[llm] HTTP {e.code} from {url} "
                    f"(attempt {attempt}/{_RETRY_MAX_ATTEMPTS}); "
                    f"retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise RuntimeError(
                f"OpenAI-compatible server returned HTTP {e.code} at {url}: "
                f"{body_text}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI-compatible server unreachable at {url}: {e}. "
                f"Is LM Studio (or your local server) running? "
                f"Override RW_LLM_BASE_URL to point elsewhere."
            ) from e

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"OpenAI-compatible server returned unexpected shape: {data!r}"
        ) from e

    # Some models (e.g. Gemma via this endpoint) don't support `reasoning_effort`
    # and instead inline their chain-of-thought as a literal <thought>...</thought>
    # prefix in `content` rather than a separate reasoning field. Strip it so
    # downstream phases (prose, JSON parsing) see only the actual answer.
    text = re.sub(r"^\s*<thought>.*?</thought>\s*", "", text, count=1, flags=re.DOTALL)

    usage = data.get("usage") or {}
    return LLMResponse(
        text=text,
        model=model,
        temperature=temperature,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
    )


# Chat-relay implementation lives in agents/relay.py — split out at phase 2
# because the schema-validation + retry logic outgrew this file. We re-export
# `call_chat_relay` here so the dispatch branch in `call()` reads naturally.
from .relay import call_chat_relay


def _anthropic_user_content(
    prompt: str, cache_prefix: str | None, cache_prompt: bool
):
    """Build the `messages[0].content` for an Anthropic call.

    - `cache_prefix` set → two text blocks: a cached prefix block (stable,
      reusable across calls that differ only in the suffix) followed by the
      variable `prompt`. This is the granular breakpoint the memory-evolution
      judge uses so the neighbor page (constant across the source papers judged
      against it) reads from cache at ~0.1× on repeat calls within the TTL.
    - `cache_prompt` set (and no prefix) → single cached block over the whole
      prompt (legacy author-phase behavior: two drafts share the full prefix).
    - neither → the bare string.
    """
    if cache_prefix is not None:
        return [
            {"type": "text", "text": cache_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    if cache_prompt:
        return [{"type": "text", "text": prompt,
                 "cache_control": {"type": "ephemeral"}}]
    return prompt


def call_anthropic(
    *,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    system: str | None = None,
    cache_prompt: bool = False,
    cache_prefix: str | None = None,
    disable_thinking: bool = False,
) -> LLMResponse:
    """Call the Anthropic API. Raises if ANTHROPIC_API_KEY missing or SDK absent.

    Some newer Anthropic models (Opus 4.7+) reject the `temperature` parameter
    outright. We attempt the call with temperature first; on a deprecation
    error, retry without it. Tournament draft diversity then comes from the
    model's natural stochasticity rather than explicit temperature variation.

    `cache_prompt`: when True, mark the user prompt with cache_control so a
    second call with the same `system` + `prompt` reads from cache at ~0.1×
    input cost. Used by the author phase, where two drafts share the same
    prefix and only differ in temperature.

    `cache_prefix`: when set, split the user turn into a cached prefix block
    (this string) + the variable `prompt`. Lets callers cache a large stable
    span (e.g. a neighbor page) across calls whose only difference is the
    trailing task/source content. Takes precedence over `cache_prompt`.
    """
    _validated_anthropic_base_url()
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed. `pip install anthropic`."
        ) from e
    if not is_real_mode_available():
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic()

    user_content = _anthropic_user_content(prompt, cache_prefix, cache_prompt)

    def _do_call(include_temperature: bool) -> "anthropic.types.Message":
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_content}],
        )
        if include_temperature:
            kwargs["temperature"] = temperature
        if disable_thinking:
            # Turn off thinking for this call. Adaptive-thinking-by-default
            # models (Sonnet 5, Opus 4.6+, Fable 5) otherwise spend the whole
            # max_tokens budget thinking and return an empty answer at tight
            # budgets — accepted by both Haiku 4.5 and the adaptive models.
            kwargs["thinking"] = {"type": "disabled"}
        if system is not None:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    try:
        resp = _do_call(include_temperature=True)
        used_temp = temperature
    except anthropic.BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg and ("deprecated" in msg or "not supported" in msg):
            resp = _do_call(include_temperature=False)
            used_temp = float("nan")  # signal: model controls temperature internally
        else:
            raise

    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return LLMResponse(
        text=text,
        model=model,
        temperature=used_temp,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def call_stub(
    *,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    system: str | None = None,
) -> LLMResponse:
    """Return a deterministic stub response keyed off (prompt, temperature).

    The text is recognizable as a stub (begins with 'STUB DRAFT') so a careless
    pipeline can't mistake it for a real generation. Token counts are zero so
    cost dashboards correctly attribute zero spend.
    """
    h = hashlib.sha1(f"{prompt}|{temperature}|{model}".encode()).hexdigest()[:10]
    text = (
        f"STUB DRAFT [{h} t={temperature:.2f}]: framework-test placeholder "
        f"in lieu of real LLM generation. Prompt prefix: "
        f"{prompt[:140]!r}..."
    )
    return LLMResponse(
        text=text,
        model=f"stub:{model}",
        temperature=temperature,
        input_tokens=0,
        output_tokens=0,
    )


def call(
    *,
    prompt: str,
    phase: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system: str | None = None,
    use_stub: bool = False,
    cache_prompt: bool = False,
    cache_prefix: str | None = None,
    disable_thinking: bool = False,
    schema: dict | None = None,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    """Dispatch to the right provider, resolving config from phase or kwargs.

    If `phase` is given, looks up the phase's ModelConfig in model_config
    (sourced from `config/models.yaml`). Explicit `model` / `temperature` /
    `max_tokens` / `provider` kwargs override the config — useful for ad-hoc
    testing or for the author phase, which varies temperature per parallel
    draft.

    Either `phase` or `model` must be specified; passing only `phase` is the
    preferred new style. Passing only `model` is the legacy style and still
    works (defaults to anthropic provider for backward compat).

    Provider dispatch:
      - "anthropic" → call_anthropic (Anthropic SDK; supports prompt caching)
      - "openai-compatible" / "lmstudio" / "openai" → call_openai_compatible
        (POSTs to /v1/chat/completions; covers LM Studio + vLLM + ollama
        OpenAI-mode + real OpenAI; cache_prompt is silently ignored)
      - "chat-relay" → call_chat_relay (filesystem protocol for delegating
        to a chat-platform agent in the same terminal; cache_prompt is
        silently ignored — caching happens via stable op_id reuse)
      - anything else → ValueError

    `schema` (optional JSON Schema dict): only honored by chat-relay today.
    The relay validates the chat's `structured` response against it and
    retries on mismatch (3 attempts max). Other providers ignore `schema`
    and return free text — callers that need validation against those
    providers parse and check at the call site, as they did before.

    `reasoning_effort` (optional, "minimal"|"low"|"medium"|"high"): hint to
    cap thinking-token cost on reasoning models. Forwarded only to the
    openai-compatible path (recognized by Gemini 2.5 Flash/Pro via Google's
    OpenAI shim, and by OpenAI o-series). Anthropic / chat-relay / stub
    silently ignore. Resolution: explicit kwarg → cfg.reasoning_effort
    (when phase=) → None.
    """
    rpm: int | None = None
    if phase is not None:
        cfg = model_config.for_phase(phase)
        if model is None:
            model = cfg.model
        if provider is None:
            provider = cfg.provider
        if temperature is None:
            temperature = cfg.temperature
        if max_tokens is None:
            max_tokens = cfg.max_tokens
        rpm = cfg.rpm
        # reasoning_effort: explicit kwarg wins (incl. caller passing None to
        # suppress); only fall back to cfg when the caller didn't set it.
        # We can't distinguish "unset" from "explicitly None" with kwargs,
        # so adopt the convention: kwarg None → use cfg value.
        if reasoning_effort is None:
            reasoning_effort = cfg.reasoning_effort

    # Cross-provider "no thinking". Anthropic honors `disable_thinking` directly
    # (thinking:{type:disabled}). The OpenAI-compatible path has no on/off
    # switch, so map it to the floor `reasoning_effort` when the caller hasn't
    # set one — this covers the reasoning models on that path (Gemini 2.5 via
    # Google's OpenAI shim, OpenAI o-series), which would otherwise hit the same
    # budget-consumed-by-reasoning empty-output failure on the fallback provider.
    # "minimal" is the request, not the final word: models disagree about the
    # vocabulary (some take "none" and reject "minimal"), so the transport
    # negotiates down to whatever the server accepts and omits the field if it
    # accepts none of them — see `_nearest_effort`. Local non-reasoning servers
    # ignore the field; chat-relay/stub don't use it.
    if disable_thinking and reasoning_effort is None:
        reasoning_effort = "minimal"

    if model is None:
        raise ValueError(
            "llm.call: either `phase` or `model` must be specified"
        )
    if provider is None:
        provider = "anthropic"  # legacy default
    if temperature is None:
        temperature = 0.7
    if max_tokens is None:
        max_tokens = 2000

    # Non-anthropic providers (incl. stub) have no cache_control primitive;
    # fold the cache prefix into the prompt so no content is lost — they just
    # forgo the cache discount. Only call_anthropic receives cache_prefix
    # separately (as its own cached block).
    _np_prompt = f"{cache_prefix}\n\n{prompt}" if cache_prefix else prompt

    if use_stub:
        return call_stub(
            model=model,
            prompt=_np_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

    p = model_config.canonical_provider_id(provider)
    supported = {"anthropic", "chat-relay"} | _OPENAI_COMPAT_PROVIDERS
    if p not in supported:
        raise ValueError(
            f"llm.call: unknown provider {provider!r}. Supported: "
            "'anthropic', 'openai-compatible' / 'lmstudio' / 'openai', "
            "'chat-relay'."
        )
    openai_endpoint = (
        resolve_openai_endpoint() if p in _OPENAI_COMPAT_PROVIDERS else None
    )
    anthropic_endpoint = (
        _validated_anthropic_base_url() if p == "anthropic" else None
    )
    effective_endpoint = (
        openai_endpoint.url if openai_endpoint is not None else anthropic_endpoint
    )
    call_is_free = bool(
        effective_endpoint and _is_local_endpoint(effective_endpoint)
    )

    # Optional per-ingest budget. The reservation happens before throttling or
    # network I/O, and is shared by parallel author threads, so two drafts
    # cannot both observe the same remaining allowance and oversubscribe it.
    from .budget import current_tracker
    tracker = current_tracker()
    reservation = None
    if tracker is not None:
        prompt_chars = len(_np_prompt) + len(system or "")
        reservation = tracker.reserve_call(
            model=model,
            provider=provider,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            free=call_is_free,
        )

    try:
        # The reservation already exists, so even an exceptional throttle
        # implementation must settle it through the same failure path.
        _throttle(model, rpm)
        if p == "anthropic":
            response = call_anthropic(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                cache_prompt=cache_prompt,
                cache_prefix=cache_prefix,
                disable_thinking=disable_thinking,
            )
        elif p in _OPENAI_COMPAT_PROVIDERS:
            if cache_prompt:
                # OpenAI-compatible transports expose no Anthropic-style
                # cache-control primitive; the caller accepts that no-op.
                pass
            # The same resolver drives preflight and status; no path may
            # report one endpoint and then send the prompt to another.
            assert openai_endpoint is not None
            base_url = openai_endpoint.url
            response = call_openai_compatible(
                model=model,
                prompt=_np_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                base_url=base_url,
                reasoning_effort=reasoning_effort,
            )
        elif p == "chat-relay":
            response = call_chat_relay(
                model=model,
                prompt=_np_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                phase=phase,
                fresh=bool(os.environ.get("RW_RELAY_FRESH")),
                schema=schema,
            )
        else:
            raise ValueError(
                f"llm.call: unknown provider {provider!r}. Supported: "
                "'anthropic', 'openai-compatible' / 'lmstudio' / 'openai', "
                "'chat-relay'."
            )
    except Exception:
        if tracker is not None and reservation is not None:
            tracker.fail(reservation)
        raise
    if tracker is not None and reservation is not None:
        tracker.finish(
            reservation,
            model=response.model or model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    return response
