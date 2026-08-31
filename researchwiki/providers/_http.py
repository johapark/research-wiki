"""Shared curl transport for structured-metadata providers.

The narrow PubMed, ORCID, and bioRxiv clients all need the same distinction:
HTTP 404 is a valid "not found" result, while exhausted retries or a missing
curl executable are environment failures. Returning ``None`` for both made the
CLI report an outage as a successful empty lookup.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable

from .. import __version__
from ..errors import EnvironmentFailure
from ..log import log

USER_AGENT = (
    f"researchwiki/{__version__} "
    "(+https://github.com/johapark/research-wiki)"
)


class StructuredProviderUnavailable(EnvironmentFailure):
    """A whitelisted metadata API could not be reached after retries."""


def curl_json(
    url: str,
    *,
    provider: str,
    retries: int = 3,
    headers: Iterable[str] = (),
) -> dict:
    """GET JSON via curl, preserving not-found vs unavailable semantics."""
    last_problem = "unknown transport failure"
    for attempt in range(retries):
        if attempt > 0:
            delay = 2 ** attempt
            log(f"  retry {attempt} after {delay}s", tag=provider)
            time.sleep(delay)
        log(f"  fetch {url}", tag=provider)
        cmd = ["curl", "-sS", "-w", "\n%{http_code}", "-A", USER_AGENT]
        for header in headers:
            cmd.extend(["-H", header])
        cmd.append(url)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError as exc:
            raise StructuredProviderUnavailable(
                f"{provider} metadata lookup needs `curl`, but it is not installed"
            ) from exc
        except subprocess.TimeoutExpired:
            last_problem = "request timed out"
            log("  timeout", tag=provider)
            continue
        if proc.returncode != 0:
            last_problem = proc.stderr.strip() or f"curl exited {proc.returncode}"
            log(f"  curl error: {last_problem}", tag=provider)
            continue
        body, separator, status = proc.stdout.rpartition("\n")
        if not separator:
            last_problem = "curl response did not include an HTTP status"
            log(f"  {last_problem}", tag=provider)
            continue
        status = status.strip()
        if status == "404":
            return {}
        if status != "200":
            last_problem = f"HTTP {status or 'unknown'}"
            log(f"  {last_problem}", tag=provider)
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            last_problem = f"JSON parse error: {exc}"
            log(f"  {last_problem}", tag=provider)
            continue
        if not isinstance(payload, dict):
            last_problem = f"expected a JSON object, got {type(payload).__name__}"
            log(f"  {last_problem}", tag=provider)
            continue
        return payload
    raise StructuredProviderUnavailable(
        f"{provider} API unavailable after {retries} attempts ({last_problem})"
    )
