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
import urllib.request
from dataclasses import dataclass

from . import model_config


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
    OpenAI), OR a local OpenAI-compatible endpoint is plausibly reachable
    (we don't probe the network — we infer from RW_LLM_BASE_URL being set
    or non-default, which is the user's signal that they have a server).

    Chat-relay does NOT count as synchronous. Each call blocks polling for
    a chat agent's response, with a default 10-minute timeout. Callers
    that need millisecond-to-second turnaround (interactive REPL
    completions, hot-loop validators) check this signal; callers that
    just need *some* LLM reachable use `has_any_llm()` instead.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if os.environ.get("OPENAI_API_KEY"):
        return True
    # Local LM-server signal: either explicitly configured or implied.
    if os.environ.get("RW_LLM_BASE_URL"):
        return True
    return False


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

_OPENAI_COMPAT_PROVIDERS = frozenset({"openai-compatible", "lmstudio", "openai"})

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
        base_url = os.environ.get("RW_LLM_BASE_URL", _DEFAULT_LOCAL_BASE_URL)

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
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode("utf-8")

    api_key = os.environ.get("OPENAI_API_KEY", "lm-studio")
    url = f"{base_url.rstrip('/')}/chat/completions"
    # Retry transient failures (401 / 429 / 5xx) with exponential backoff +
    # jitter. A fresh Request is built each attempt because urllib consumes the
    # body stream once. Non-retryable HTTP errors (other 4xx) and the final
    # exhausted attempt re-raise as RuntimeError, preserving prior behavior.
    data = None
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
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
    # Local non-reasoning servers ignore reasoning_effort (and the call retries
    # without it if the server rejects the field); chat-relay/stub don't use it.
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

    # Client-side RPM cap (config-driven) — block before any real network
    # call so free-tier models stay under their ceiling. Keyed by model.
    _throttle(model, rpm)

    p = provider.lower().strip()
    if p == "anthropic":
        return call_anthropic(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            cache_prompt=cache_prompt,
            cache_prefix=cache_prefix,
            disable_thinking=disable_thinking,
        )
    if p in _OPENAI_COMPAT_PROVIDERS:
        if cache_prompt:
            # Silently no-op: OpenAI-compatible spec doesn't expose KV-cache
            # reuse the way Anthropic's cache_control header does. Caller
            # accepts the loss of cache discount.
            pass
        # base_url precedence: RW_LLM_BASE_URL env (ad-hoc override) → the
        # config's top-level `base_url:` → None (call_openai_compatible then
        # falls back to the LM Studio localhost default). Folding the endpoint
        # into the config lets a backend switch ride on RW_MODELS_CONFIG alone.
        base_url = os.environ.get("RW_LLM_BASE_URL") or model_config.base_url()
        return call_openai_compatible(
            model=model,
            prompt=_np_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    if p == "chat-relay":
        # cache_prompt is silently ignored — relay caching is op_id-based,
        # not KV-cache-based. Setting RW_RELAY_FRESH=1 in the environment
        # bypasses op_id reuse and forces a fresh prompt every call.
        # `schema` (if provided) drives validation + retry-with-feedback
        # inside call_chat_relay; see agents/relay.py for the protocol.
        return call_chat_relay(
            model=model,
            prompt=_np_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            phase=phase,
            fresh=bool(os.environ.get("RW_RELAY_FRESH")),
            schema=schema,
        )
    raise ValueError(
        f"llm.call: unknown provider {provider!r}. "
        f"Supported: 'anthropic', 'openai-compatible' / 'lmstudio' / 'openai', "
        f"'chat-relay'."
    )
