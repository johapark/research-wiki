"""Single-request OpenAI-compatible image dispatch; ordinary text calls bypass it."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import os
import time
import urllib.error
import urllib.request

from .llm import ProviderUnavailable
from ..operations.events import ModelCallObserver
from ..operations.observation import current_recorder
from . import model_config
from .budget import current_tracker


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    width: int
    height: int
    label: str


def validate_image_provider(provider: str) -> None:
    if model_config.canonical_provider_id(provider) not in model_config.OPENAI_COMPAT_PROVIDERS:
        raise ProviderUnavailable('Visual inspection requires an OpenAI-compatible image provider; '
                                  f'{provider!r} has no image transport.')


def preflight_visual_provider() -> None:
    from . import llm
    validate_image_provider(model_config.for_phase('visual_inspect').provider)
    problems = llm.missing_provider_credentials(phases=('visual_inspect',))
    if problems:
        raise ProviderUnavailable('; '.join(problems))


def image_token_bound(model: str, configured: int | None) -> int | None:
    # Full pages are capped at 3M pixels and a 4:1 aspect ratio. Known
    # GPT-4o/4.1 image accounting can be conservatively bounded at that size.
    # Other models/proxy aliases must supply an explicit bound under strict limits.
    if configured is not None:
        return configured
    if model.lower().startswith('gpt-4o-mini'):
        return 100000
    if model.lower().startswith(('gpt-4o', 'gpt-4.1')):
        return 32768
    return None


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('visual inspection deadline expired')
    return remaining


def _throttle_bounded(model: str, rpm: int | None, deadline: float) -> None:
    from . import llm
    if not rpm or rpm <= 0:
        return
    if not llm._throttle_lock.acquire(timeout=_remaining(deadline)):
        raise TimeoutError('visual inspection deadline expired waiting for throttle lock')
    try:
        last = llm._last_call_at.get(model)
        wait = max(0, 60.0 / rpm - (time.monotonic() - last)) if last is not None else 0
        if wait >= _remaining(deadline):
            raise TimeoutError('visual inspection throttle exceeds remaining deadline')
        if wait:
            time.sleep(wait)
        llm._last_call_at[model] = time.monotonic()
    finally:
        llm._throttle_lock.release()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # even HTTP redirects must not multiply the request cap


def _open_once(request, timeout):
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def _post_once(payload: dict, endpoint: str, deadline: float, request_state: dict) -> dict:
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(endpoint.rstrip('/') + '/chat/completions', data=body,
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', 'lm-studio')},
        method='POST')
    try:
        timeout = _remaining(deadline)
        request_state['outbound_requests'] = 1
        with _open_once(request, timeout=timeout) as response:
            chunks, size = [], 0
            while True:
                # Prevent trickle responses extending a socket timeout indefinitely.
                remaining = _remaining(deadline)
                try:
                    response.fp.raw._sock.settimeout(remaining)
                except AttributeError:
                    pass  # mocked response / alternative urllib handler
                chunk = response.read1(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 1_000_000:
                    raise ValueError('visual response exceeds 1MB')
                chunks.append(chunk)
        result = json.loads(b''.join(chunks))
        _remaining(deadline)
        return result
    except urllib.error.HTTPError as exc:
        # No response bodies: they can contain submitted context or credentials.
        raise ProviderUnavailable(f'Visual inspection HTTP {exc.code}; no retry was sent.') from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailable('Visual inspection endpoint unavailable; no retry was sent.') from exc


def dispatch_images(*, model: str, provider: str, prompt: str, system: str | None,
                    temperature: float, max_tokens: int, reasoning_effort: str | None,
                    rpm: int | None, images: list[ImageInput], phase: str | None,
                    use_stub: bool, event_sink=None, deadline_seconds: float = 30,
                    configured_bound: int | None = None, request_state: dict | None = None):
    from . import llm
    request_state = request_state if request_state is not None else {}
    request_state['outbound_requests'] = 0
    if (not 1 <= len(images) <= 2 or any(image.width <= 0 or image.height <= 0 for image in images)
            or any(max(image.width, image.height) / min(image.width, image.height) > 4 for image in images)
            or sum(image.width * image.height for image in images) > 3_000_000):
        raise ValueError('visual image/page/pixel cap exceeded')
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise TimeoutError('visual inspection has no remaining deadline')
    if use_stub:
        return llm.LLMResponse('{"observations":[]}', '(stub)', 0, 0, 0)
    validate_image_provider(provider)
    tracker = current_tracker()
    deadline = time.monotonic() + min(30, deadline_seconds)
    if tracker:
        tracker.check_wall()
        if tracker.budget.max_wall_seconds is not None:
            deadline = min(deadline, tracker.started + tracker.budget.max_wall_seconds)
    endpoint = llm.resolve_openai_endpoint().url
    bound = image_token_bound(model, configured_bound)
    if (tracker and bound is None and
            (tracker.budget.max_tokens is not None or tracker.budget.max_cost_usd is not None)):
        raise ValueError('No conservative image-token bound for model; configure image_token_bound.')
    reservation = tracker.reserve_call(model=model, provider=provider,
        prompt_chars=len(prompt) + len(system or ''), max_tokens=max_tokens,
        image_tokens=(bound or 32768) * len(images), free=llm._is_local_endpoint(endpoint)) if tracker else None
    if tracker and not llm._is_local_endpoint(endpoint) and model_config.rate_for(model) is None:
        tracker.mark_cost_unavailable()
    observer = ModelCallObserver(event_sink or current_recorder(), phase=phase,
                                 provider=provider, model=model)
    observer.started()
    try:
        _throttle_bounded(model, rpm, deadline)
        content = [{'type': 'text', 'text': prompt}]
        for image in images:
            content.extend([{'type': 'text', 'text': image.label},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                    base64.b64encode(image.data).decode('ascii'), 'detail': 'high'}}])
        messages = ([{'role': 'system', 'content': system}] if system is not None else [])
        messages.append({'role': 'user', 'content': content})
        payload = {'model': model, 'messages': messages,
                   'max_completion_tokens' if llm._needs_max_completion_tokens(model) else 'max_tokens':
                       min(max_tokens, 1200)}
        if not llm._needs_max_completion_tokens(model):
            payload['temperature'] = temperature
        if reasoning_effort is not None:
            payload['reasoning_effort'] = llm._resolve_effort((endpoint.rstrip('/'), model), reasoning_effort)
        data = _post_once(payload, endpoint, deadline, request_state)
        raw = data['choices'][0]['message']['content']
        if not isinstance(raw, str):
            raise ValueError('visual response has no text content')
        usage = data.get('usage') or {}
        known = all(isinstance(usage.get(k), int) and usage[k] >= 0
                    for k in ('prompt_tokens', 'completion_tokens'))
        response = llm.LLMResponse(raw, model, temperature,
            usage.get('prompt_tokens', 0) if known else 0,
            usage.get('completion_tokens', 0) if known else 0, usage_known=known)
        _remaining(deadline)
    except BaseException as exc:
        if tracker and reservation:
            tracker.fail(reservation)
        observer.failed(exc)
        raise
    if tracker and reservation:
        if response.usage_known:
            tracker.finish(reservation, model=model, input_tokens=response.input_tokens,
                           output_tokens=response.output_tokens)
        else:
            tracker.fail(reservation)  # unknown usage retains conservative charge
        tracker.check_wall()
    observer.finished(response)
    return response
