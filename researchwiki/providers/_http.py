"""Shared curl transport for structured-metadata providers.

The narrow PubMed, ORCID, and bioRxiv clients all need the same distinction:
HTTP 404 is a valid "not found" result, while exhausted retries or a missing
curl executable are environment failures. Returning ``None`` for both made the
CLI report an outage as a successful empty lookup.

Two more distinctions arrived with keyword search (`scout search`):

- **HTTP 400 is the caller's fault, not the network's.** ClinicalTrials.gov and
  Europe PMC answer a malformed query with 400. Retrying it with backoff and
  then reporting "API unavailable" would send the user off to check their
  connection over a typo, so with `reject_400=True` a 400 raises
  `ProviderRequestRejected` at once.
- **Secrets never reach argv, logs, or cache keys.** An NCBI API key passed as
  a query parameter would otherwise be visible to `ps`, echoed by the `fetch`
  log line, and baked into a readable cache filename. `secret_query` parameters
  travel in a curl config read from stdin; everything else sees the bare URL.

`curl_download` is the one binary path: open-access PDFs into `inbox/`. It
follows redirects by hand so every hop's host is checked against an allowlist
before anything is fetched from it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

from .. import __version__
from ..errors import EnvironmentFailure
from ..log import log

USER_AGENT = (
    f"researchwiki/{__version__} "
    "(+https://github.com/johapark/research-wiki)"
)

#: Statuses worth a longer wait than the 2**n backoff: the server is shedding
#: load, and arXiv in particular answers bursts with 503.
_THROTTLED = frozenset({"429", "503"})
_THROTTLE_SLEEP = 5
#: Query parameters that must never be logged, even if a caller forgets to
#: route them through `secret_query`.
_SECRET_PARAM_RE = re.compile(r"(?i)([?&](?:api_key|apikey|key|token|email)=)[^&#]*")
MAX_REDIRECTS = 5

T = TypeVar("T")


class StructuredProviderUnavailable(EnvironmentFailure):
    """A whitelisted metadata API could not be reached after retries."""


class ProviderRequestRejected(ValueError):
    """The API refused the request itself (HTTP 400) — a bad query, exit 1.

    Deliberately not an `EnvironmentFailure`: nothing about the machine is
    wrong, and retrying cannot help.
    """


class DownloadRefused(RuntimeError):
    """A PDF download that cannot complete automatically.

    The URL is still a usable lead — the caller lists it for a manual
    download — so this is neither an outage (exit 2) nor a bug. `reason` is a
    short machine-readable label (`http-403`, `not-pdf`, `off-allowlist`, …).
    """

    def __init__(self, reason: str, url: str, detail: str = ""):
        super().__init__(f"{reason}: {detail or url}")
        self.reason = reason
        self.url = url


def redact(url: str) -> str:
    """`url` with secret-looking query values blanked, for logs and errors."""
    return _SECRET_PARAM_RE.sub(r"\1***", url)


def _with_query(url: str, params: Mapping[str, str]) -> str:
    if not params:
        return url
    sep = "&" if urllib.parse.urlsplit(url).query else "?"
    return url + sep + urllib.parse.urlencode(dict(params))


def _curl_cmd(url: str, headers: Iterable[str], secret_query: Mapping[str, str] | None,
              extra: Iterable[str] = ()) -> tuple[list[str], str | None]:
    """argv plus optional stdin. A secret-bearing URL goes via `-K -` only."""
    cmd = ["curl", "-sS", "-A", USER_AGENT, *extra]
    for header in headers:
        cmd.extend(["-H", header])
    if not secret_query:
        cmd.append(url)
        return cmd, None
    full = _with_query(url, secret_query)
    # curl config syntax: a quoted value takes backslash escapes. urlencode
    # leaves neither `"` nor `\` in the URL, but escape anyway.
    quoted = full.replace("\\", "\\\\").replace('"', '\\"')
    cmd.extend(["-K", "-"])
    return cmd, f'url = "{quoted}"\n'


def _request(
    url: str,
    *,
    provider: str,
    retries: int,
    headers: Iterable[str],
    secret_query: Mapping[str, str] | None,
    parse: Callable[[str], T],
    reject_400: bool = False,
) -> T | None:
    """GET `url`; `parse(body)` the 200 response. None on 404.

    `parse` raising `ValueError` counts as a transient failure and retries,
    which is how `curl_json` has always treated a truncated JSON body.

    `reject_400` is opt-in: keyword-search callers catch
    `ProviderRequestRejected` and report the query as rejected, while the
    older DOI/ORCID lookups have no handler for it and keep the
    retry-then-unavailable behaviour they always had.
    """
    shown = redact(url)
    last_problem = "unknown transport failure"
    for attempt in range(retries):
        if attempt > 0:
            delay = 2 ** attempt
            if last_problem in {f"HTTP {s}" for s in _THROTTLED}:
                delay = max(delay, _THROTTLE_SLEEP * attempt)
            log(f"  retry {attempt} after {delay}s", tag=provider)
            time.sleep(delay)
        log(f"  fetch {shown}", tag=provider)
        cmd, stdin = _curl_cmd(url, headers, secret_query, ["-w", "\n%{http_code}"])
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                **({"input": stdin} if stdin is not None else {}),
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
            return None
        if status == "400" and reject_400:
            snippet = " ".join(body.split())[:200]
            raise ProviderRequestRejected(
                f"{provider} rejected the request (HTTP 400){': ' + snippet if snippet else ''}"
            )
        if status != "200":
            last_problem = f"HTTP {status or 'unknown'}"
            log(f"  {last_problem}", tag=provider)
            continue
        try:
            return parse(body)
        except ValueError as exc:
            last_problem = str(exc)
            log(f"  {last_problem}", tag=provider)
            continue
    raise StructuredProviderUnavailable(
        f"{provider} API unavailable after {retries} attempts ({last_problem})"
    )


def _json_object(body: str) -> dict:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def curl_json(
    url: str,
    *,
    provider: str,
    retries: int = 3,
    headers: Iterable[str] = (),
    secret_query: Mapping[str, str] | None = None,
    reject_400: bool = False,
) -> dict:
    """GET JSON via curl, preserving not-found vs unavailable semantics."""
    data = _request(url, provider=provider, retries=retries, headers=headers,
                    secret_query=secret_query, parse=_json_object,
                    reject_400=reject_400)
    return {} if data is None else data


def curl_body(
    url: str,
    *,
    provider: str,
    retries: int = 3,
    headers: Iterable[str] = (),
    secret_query: Mapping[str, str] | None = None,
    reject_400: bool = False,
) -> str | None:
    """GET a text body (Atom, XML) via curl. None on 404."""
    return _request(url, provider=provider, retries=retries, headers=headers,
                    secret_query=secret_query, parse=lambda body: body,
                    reject_400=reject_400)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def curl_download(
    url: str,
    dest: Path,
    *,
    provider: str,
    allowed_hosts: Iterable[str],
    max_bytes: int,
    retries: int = 3,
) -> Path:
    """Download a PDF to `dest`, which must not exist yet. Returns `dest`.

    - Every hop's host, including redirects, must be in `allowed_hosts`;
      redirects are followed here rather than by `curl -L` so an off-list
      host is refused before anything is requested from it.
    - The body lands in a dot-prefixed `.part` sibling, so a `*.pdf` glob over
      `inbox/` (status, batch ingest) never sees a partial file, and the final
      rename stays on one filesystem even when `inbox/` is a symlink.
    - The result must start with `%PDF-` and fit in `max_bytes`; a landing
      page or a bot challenge served with HTTP 200 is refused, not saved.

    Raises `DownloadRefused` for outcomes a person can still act on (403,
    persistent 429, off-allowlist redirect, not a PDF, 404), and
    `StructuredProviderUnavailable` when the network itself is failing.
    """
    allowed = {h.lower() for h in allowed_hosts}
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    part = dest.with_name(f".{dest.name}.part")
    current = url
    hops = 0
    last_problem = "unknown transport failure"
    attempt = 0
    try:
        while True:
            if not current.lower().startswith("https://"):
                raise DownloadRefused("not-https", current)
            if _host(current) not in allowed:
                raise DownloadRefused("off-allowlist", current, f"host {_host(current)!r}")
            if attempt >= retries:
                raise StructuredProviderUnavailable(
                    f"{provider} download failed after {retries} attempts ({last_problem})"
                )
            if attempt > 0:
                delay = max(2 ** attempt, _THROTTLE_SLEEP * attempt if last_problem == "HTTP 429" else 0)
                log(f"  retry {attempt} after {delay}s", tag=provider)
                time.sleep(delay)
            attempt += 1
            log(f"  download {redact(current)}", tag=provider)
            cmd, _ = _curl_cmd(current, (), None, [
                "--proto", "=https", "--max-filesize", str(max_bytes),
                "-o", str(part), "-w", "%{http_code}\n%{redirect_url}",
            ])
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            except FileNotFoundError as exc:
                raise StructuredProviderUnavailable(
                    f"{provider} download needs `curl`, but it is not installed"
                ) from exc
            except subprocess.TimeoutExpired:
                last_problem = "download timed out"
                continue
            if proc.returncode == 63:
                raise DownloadRefused("too-large", current, f"over {max_bytes} bytes")
            if proc.returncode != 0:
                last_problem = proc.stderr.strip() or f"curl exited {proc.returncode}"
                log(f"  curl error: {last_problem}", tag=provider)
                continue
            status, _, redirect = proc.stdout.strip().partition("\n")
            status, redirect = status.strip(), redirect.strip()
            if status in {"301", "302", "303", "307", "308"} and redirect:
                hops += 1
                if hops > MAX_REDIRECTS:
                    raise DownloadRefused("too-many-redirects", current)
                current = urllib.parse.urljoin(current, redirect)
                attempt = 0
                continue
            if status == "200":
                break
            if status in {"403", "404", "410", "451"}:
                raise DownloadRefused(f"http-{status}", current)
            last_problem = f"HTTP {status or 'unknown'}"
            if status == "429" and attempt >= retries:
                raise DownloadRefused("http-429", current, "rate limited")
            log(f"  {last_problem}", tag=provider)
        size = part.stat().st_size if part.exists() else 0
        if size > max_bytes:
            raise DownloadRefused("too-large", current, f"{size} bytes")
        with open(part, "rb") as fh:
            magic = fh.read(5)
        if magic != b"%PDF-":
            raise DownloadRefused("not-pdf", current, "response is not a PDF")
        if dest.exists():
            raise FileExistsError(dest)
        os.replace(part, dest)
        return dest
    finally:
        if part.exists():
            part.unlink()
