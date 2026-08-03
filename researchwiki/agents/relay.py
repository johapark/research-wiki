"""Chat-relay LLM provider — file-based protocol for delegating to a chat agent.

Subscription users with no API key and no local LM can route every phase
through a chat-platform agent (Claude Code, Codex, Cursor, …) running in
the same terminal session. The CLI writes a prompt JSON to
`.llm-relay/pending/{op_id}.prompt.json`, blocks polling for a response
JSON at `.llm-relay/completed/{op_id}.response.json`, then returns. The
chat agent reads the pending file, drafts a response, writes the
completed file. No API key required.

Phase 1 (committed): blocking mode, no schema validation, op_id-based
cache reuse, atomic writes.
Phase 2 (this file): schema validation + retry-with-feedback. When the
caller passes a JSON Schema and the chat returns a structured response
that fails validation, the relay rewrites the pending file with
`retry_of` and `retry_feedback` populated, derives a new (deterministic)
op_id, and polls again. Hard cap at 3 attempts; after the third failure
the relay raises with the full op-id chain.
Phase 3 (deferred): RelayPending exception + checkpoint state machine.
Phase 4 (deferred): RW_LLM_PROVIDER env override + AGENTS.md.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

from ..paths import wiki_root

# `LLMResponse` is imported lazily inside call_chat_relay to dodge the
# llm.py ↔ relay.py cycle: llm.py imports `call_chat_relay` from us, so
# importing it back at module load time fails when relay.py loads first.


_RELAY_SCHEMA_VERSION = 1
_RELAY_DEFAULT_TIMEOUT = 600.0   # seconds; generous for slow / inattentive agents
_RELAY_POLL_INTERVAL = 0.5        # seconds between existence checks
_RELAY_MAX_RETRIES = 2            # retries on schema failure; 1 original + 2 retries = 3 attempts

# Counter for fresh-mode op_ids within a single process. Combined with PID
# this guarantees per-call uniqueness when the user opts out of cache reuse.
_fresh_counter = itertools.count(1)


# ───────────────────────── path helpers ─────────────────────────

def _relay_dir() -> Path:
    """Lazy resolver — wiki_root() may not be valid at module import time."""
    return wiki_root() / ".llm-relay"


def _paths_for(op_id: str) -> tuple[Path, Path]:
    """Return (prompt_path, response_path) for a given op_id."""
    base = _relay_dir()
    return (base / "pending" / f"{op_id}.prompt.json",
            base / "completed" / f"{op_id}.response.json")


def _stable_op_id(
    phase: str | None,
    prompt: str,
    *,
    fresh: bool = False,
    retry_of: str | None = None,
) -> str:
    """Derive a deterministic op_id from (phase, prompt[, retry_of]).

    Stable so a re-invocation after a crash re-derives the same id and reuses
    the prior response file if one exists. The `retry_of` discriminator means
    each retry attempt derives a different (still-deterministic) op_id, so a
    retry chain can survive a CLI crash and resume on rerun.

    Pass fresh=True (or set RW_RELAY_FRESH=1) to bypass determinism — appends
    pid+counter for forced uniqueness.
    """
    seed = f"{phase or '_'}|{prompt}"
    if retry_of:
        seed += f"|retry_of={retry_of}"
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    if fresh:
        h = f"{h}-{os.getpid()}-{next(_fresh_counter)}"
    return h


def _write_atomic_json(path: Path, data: dict) -> None:
    """Write JSON via tmp + rename — eliminates the partial-read race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _poll_until_exists(path: Path, timeout: float) -> None:
    """Block until `path` exists, or raise after `timeout` seconds.

    Uses time.monotonic so a wallclock jump (NTP correction, suspend/resume)
    can't shorten or extend the window.
    """
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"chat-relay: no response in {timeout:.0f}s for {path.name}. "
                f"Pending file remains; write the response file and retry, "
                f"or `rm` the pending file to abandon."
            )
        time.sleep(_RELAY_POLL_INTERVAL)


def _emit_handoff_message(prompt_path: Path, response_path: Path,
                          phase: str | None, timeout: float,
                          retry_of: str | None = None) -> None:
    """Print a single human-readable line to stderr so the user (and any
    chat agent watching the terminal) knows a prompt is waiting."""
    tag = f"{phase or 'ad-hoc'}"
    if retry_of:
        tag += f", retry of {retry_of}"
    print(
        f"📨 LLM relay pending [{tag}] → {prompt_path}\n"
        f"   Awaiting response at {response_path} (timeout {timeout:.0f}s)",
        file=sys.stderr,
        flush=True,
    )


# ───────────────────────── schema validation ─────────────────────────

class SchemaError(ValueError):
    """Validation error from the relay's JSON-Schema check.

    The message is propagated into the retry prompt's `retry_feedback` field
    so the chat agent sees a concrete description of what was wrong.
    """


def _validate_against_schema(data, schema: dict) -> None:
    """Validate `data` against a JSON Schema. Tries the `jsonschema` package
    if installed; falls back to a minimal type+required+items check that
    handles every shape this codebase actually uses today.

    Supported subset (fallback path):
      - type: object | array | string | number | integer | boolean | null
      - properties: {key: subschema}
      - required: [key, ...]
      - items: subschema (for arrays)
      - enum: [val, ...]

    Anything beyond this (allOf, anyOf, pattern, additionalProperties, etc.)
    silently passes in the fallback. Install `jsonschema` for stricter checks.
    """
    try:
        import jsonschema           # type: ignore[import-untyped]
    except ImportError:
        _lightweight_validate(data, schema, path="$")
        return
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:                       # type: ignore[attr-defined]
        # Build a path expression jsonschema.absolute_path doesn't quite give
        # us — `$.foo[0].bar` style — so the chat agent's retry prompt names
        # the failing field directly.
        path_parts = []
        for part in e.absolute_path:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            else:
                path_parts.append(f".{part}")
        path = "$" + "".join(path_parts)
        raise SchemaError(f"{path}: {e.message}") from e


_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "number": (int, float), "integer": int,
    "boolean": bool, "null": type(None),
}


def _lightweight_validate(data, schema: dict, *, path: str) -> None:
    """Minimal JSON Schema fallback. See _validate_against_schema for scope."""
    expected = schema.get("type")

    # Nullable-union form: type: ["X", "null"] — early-return for null,
    # otherwise pick the non-null type and fall through to the single-type
    # check below.
    if isinstance(expected, list):
        if data is None and "null" in expected:
            return
        non_null = [t for t in expected if t != "null"]
        # If it's still a multi-type union, just check membership and stop.
        if len(non_null) != 1:
            py_types = []
            for t in non_null:
                m = _TYPE_MAP.get(t)
                if m is None:
                    continue
                py_types.extend(m if isinstance(m, tuple) else (m,))
            if py_types and not isinstance(data, tuple(py_types)):
                raise SchemaError(
                    f"{path}: expected one of {expected}, got "
                    f"{type(data).__name__}"
                )
            return
        expected = non_null[0]

    if expected is not None:
        py_type = _TYPE_MAP.get(expected)
        if py_type is not None:
            # Tighten bool/int/number/integer — Python's bool subclasses int,
            # so plain isinstance checks would silently pass `True` as integer.
            if expected in ("integer", "number"):
                if isinstance(data, bool) or not isinstance(data, py_type):
                    raise SchemaError(
                        f"{path}: expected {expected}, got "
                        f"{type(data).__name__}"
                    )
            elif not isinstance(data, py_type):
                raise SchemaError(
                    f"{path}: expected {expected}, got {type(data).__name__}"
                )

    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(f"{path}: value {data!r} not in enum {schema['enum']!r}")

    if expected == "object" and isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                raise SchemaError(f"{path}: missing required field {req!r}")
        for key, subschema in (schema.get("properties") or {}).items():
            if key in data:
                _lightweight_validate(data[key], subschema, path=f"{path}.{key}")
    elif expected == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                _lightweight_validate(item, items_schema, path=f"{path}[{i}]")


# ───────────────────────── main entrypoint ─────────────────────────

def call_chat_relay(
    *,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    system: str | None = None,
    phase: str | None = None,
    timeout: float = _RELAY_DEFAULT_TIMEOUT,
    fresh: bool = False,
    schema: dict | None = None,
) -> LLMResponse:
    """Delegate the prompt to a chat-platform agent via the filesystem.

    Writes `.llm-relay/pending/{op_id}.prompt.json`, blocks polling for
    `.llm-relay/completed/{op_id}.response.json`, parses, validates, returns.

    Cache reuse: op_id is sha1(phase|prompt[|retry_of])[:12] by default. If a
    response file already exists for this op_id (e.g. from a prior run that
    crashed after the chat responded but before the CLI consumed), it is
    reused without re-prompting. Pass fresh=True (or set RW_RELAY_FRESH=1)
    to force a unique op_id and re-prompt unconditionally.

    Schema mode: when `schema` is non-None (a JSON Schema dict), the chat
    is expected to return a `structured` field that validates. On
    mismatch, the relay derives a fresh deterministic op_id (with the
    failed op as `retry_of`), writes a new pending file with the
    validator's error in `retry_feedback`, and polls again. Hard cap at
    3 attempts (1 original + 2 retries); after that, raises RuntimeError
    with the full op-id chain. Without `schema`, the response field is
    accepted as-is — same behavior as phase 1.

    Response file shape (one of `response` or `structured` is required):

        {
          "schema_version": 1,
          "op_id": "<echoed>",
          "via": "<platform/model identifier, optional>",
          "response":   "<free-text>",                  // OR
          "structured": { ... }                          // arbitrary JSON
        }

    `structured` is JSON-serialized into the returned text for callers
    that still parse JSON out of free text.
    """
    from .llm import LLMResponse           # lazy: see top-of-file note on cycle

    op_id = _stable_op_id(phase, prompt, fresh=fresh)
    chain: list[str] = [op_id]
    retry_of: str | None = None
    retry_feedback: str | None = None

    last_err: str | None = None
    for attempt in range(_RELAY_MAX_RETRIES + 1):           # 0, 1, 2 → 3 total
        prompt_path, response_path = _paths_for(op_id)

        # Skip rewriting if a prior response exists (cache hit) or a pending
        # prompt is still in flight (recovery). Otherwise emit fresh.
        if not response_path.exists() and not prompt_path.exists():
            _write_atomic_json(prompt_path, {
                "schema_version": _RELAY_SCHEMA_VERSION,
                "op_id": op_id,
                "phase": phase,
                "model_hint": model,
                "system": system,
                "prompt": prompt,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retry_of": retry_of,
                "retry_feedback": retry_feedback,
            })
            response_path.parent.mkdir(parents=True, exist_ok=True)
            _emit_handoff_message(prompt_path, response_path, phase, timeout,
                                  retry_of=retry_of)

        _poll_until_exists(response_path, timeout)

        try:
            data = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            # File-level corruption is the user's problem to fix manually —
            # retry-via-protocol won't help because the chat already wrote it.
            raise RuntimeError(
                f"chat-relay: response file {response_path} is not valid "
                f"JSON: {e}. Fix or delete the file and retry."
            ) from e

        # Validate basic shape and (optionally) schema.
        try:
            _check_response_shape(data, response_path, schema)
            if schema is not None:
                _validate_against_schema(data["structured"], schema)
            # Pass — exit the retry loop with `data` and the current paths.
            break
        except SchemaError as e:
            last_err = str(e)
            if attempt >= _RELAY_MAX_RETRIES:
                # Final failure: clean up the failing pair, then raise with
                # the full chain so the user can audit.
                prompt_path.unlink(missing_ok=True)
                response_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"chat-relay: schema validation failed "
                    f"{_RELAY_MAX_RETRIES + 1}× across op chain "
                    f"{' → '.join(chain)}. Last error: {last_err}"
                ) from e
            # Retry: clean up this attempt, derive next op_id deterministically
            # off the current one, and loop.
            prompt_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
            retry_of = op_id
            retry_feedback = last_err
            op_id = _stable_op_id(phase, prompt, retry_of=retry_of)
            chain.append(op_id)
            continue

    # `data` and the surviving (prompt_path, response_path) pair are from the
    # accepted attempt. Build the LLMResponse, then clean up.
    if isinstance(data.get("response"), str):
        text = data["response"]
    else:
        text = json.dumps(data["structured"], ensure_ascii=False)
    via = data.get("via", "unknown")

    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)

    return LLMResponse(
        text=text,
        model=f"chat-relay:{via}",
        temperature=temperature,
        input_tokens=0,
        output_tokens=0,
    )


def _check_response_shape(data, response_path: Path, schema: dict | None) -> None:
    """Enforce the basic response-file contract before schema validation runs.

    With schema mode, `structured` is required (a `response`-only reply means
    the chat ignored structured-output mode — that's a retryable failure).
    Without schema, either field is fine.
    """
    if not isinstance(data, dict):
        raise SchemaError(
            f"response file {response_path.name} must be a JSON object; "
            f"got {type(data).__name__}"
        )
    has_text = isinstance(data.get("response"), str)
    has_structured = isinstance(data.get("structured"), (dict, list))
    if schema is not None:
        if not has_structured:
            raise SchemaError(
                "schema mode requires a `structured` object/array, got "
                f"{'`response` text only' if has_text else 'neither field'}"
            )
        return
    if not (has_text or has_structured):
        raise SchemaError(
            "response must contain either a `response` string or a "
            f"`structured` object/array. Got keys: {sorted(data)}"
        )
