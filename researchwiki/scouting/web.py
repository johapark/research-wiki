"""Quarantined provenance receipts for agent-native web search.

The active chat agent owns search and conversational prose. This module does no
network access and stores no research prose: it creates a bounded request,
validates a minimal source receipt, and exposes a resumable local lifecycle.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

from ..errors import EnvironmentFailure
from ..fsatomic import exclusive_lock, file_sha256, write_json_atomic
from ..paths import scout_cache_dir


SCHEMA_VERSION = 3
DEFAULT_MAX_RESULTS = 20
DEFAULT_MAX_FETCHES = 10
HARD_MAX_RESULTS = 100
HARD_MAX_FETCHES = 50
MAX_QUERY_CHARS = 500
MAX_URL_CHARS = 4_096
MAX_TITLE_CHARS = 500
MAX_HARNESS_CHARS = 100
MAX_DOMAINS = 20
RUN_STATES = ("requested", "recorded", "invalid")

# How the recorded URLs were found. This is the host agent's self-attestation
# about its own capability, not something the CLI can verify — but stating it
# keeps a user-supplied exact-URL fetch from being mistaken for broad search.
# Model-prior URLs are intentionally not a mode: Rule 1 authorizes native web
# search or an exact URL supplied by the user, not arbitrary agent-recalled URLs.
DISCOVERY_METHODS = ("search", "user-provided-url")
_SEARCHLESS_METHODS = frozenset({"user-provided-url"})

_REQUEST_FIELDS = {
    "schema_version", "run_id", "mode", "query", "created_at",
    "evidence_class", "constraints", "receipt_contract",
}
_CONSTRAINT_FIELDS = {
    "max_results", "max_fetches", "domains", "since",
    "follow_transitive_links",
}
_MANIFEST_FIELDS = {
    "schema_version", "run_id", "status", "evidence_class", "recorded_at",
    "harness", "discovery_method", "source_count", "fetched_count",
    "duplicates_dropped", "request_sha256", "receipt_sha256",
}
_RECEIPT_FIELDS = {
    "schema_version", "run_id", "harness", "discovery_method", "sources",
}
_RECORDED_RECEIPT_FIELDS = {
    "schema_version", "run_id", "evidence_class", "recorded_at", "harness",
    "discovery_method", "sources",
}

_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_LEGACY_IPV4_LABEL_RE = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)", re.IGNORECASE)


def _receipt_contract() -> dict:
    return {
        "required_top_level": [
            "schema_version", "run_id", "harness", "discovery_method", "sources",
        ],
        "required_per_source": ["url", "fetched"],
        "optional_per_source": ["title", "published_at"],
        "discovery_methods": list(DISCOVERY_METHODS),
        "notes": [
            "Return conversational research through the host agent, not this receipt.",
            "Record only provenance here; research prose is not accepted or stored.",
            "fetched=true is the host harness's assertion that it opened the page.",
            "discovery_method states how the URLs were found, not what they say.",
            (
                "Under user-provided-url at least one source is required and "
                "all must be fetched."
            ),
            "A source remains discovery-only until its underlying PDF is ingested.",
        ],
    }


def _validate_discovery_method(value, *, sources: list[dict]) -> str:
    """Check the declared discovery mode and that the sources can support it."""
    if value not in DISCOVERY_METHODS:
        raise ScoutInputError(
            "discovery_method must be one of: " + ", ".join(DISCOVERY_METHODS)
        )
    if value in _SEARCHLESS_METHODS:
        if not sources:
            raise ScoutInputError(
                f"discovery_method {value!r} requires at least one opened source"
            )
        unopened = [source["url"] for source in sources if not source["fetched"]]
        if unopened:
            raise ScoutInputError(
                f"discovery_method {value!r} performed no search, so it cannot "
                "report search-only sources; these were never opened: "
                + ", ".join(unopened)
            )
    return value


class ScoutInputError(ValueError):
    """Malformed request/receipt input supplied by the operator or agent."""


class ScoutStorageUnavailable(EnvironmentFailure):
    """Scout artifact storage could not be read or written."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _clean_text(value, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ScoutInputError(f"{field} must be a string")
    text = " ".join(value.split()).strip()
    if not text:
        raise ScoutInputError(f"{field} must not be blank")
    if _CONTROL_RE.search(text):
        raise ScoutInputError(f"{field} contains control characters")
    if len(text) > max_chars:
        raise ScoutInputError(f"{field} exceeds {max_chars} characters")
    return text


def _bounded_int(value, *, field: str, minimum: int, maximum: int) -> int:
    """Validate an integer bound without accepting bool or coercing strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoutInputError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ScoutInputError(f"{field} must be between {minimum} and {maximum}")
    return value


def _validate_iso(value, *, field: str, date_only: bool = False) -> str:
    if not isinstance(value, str):
        raise ScoutInputError(f"{field} must be a string")
    raw = value.strip()
    if not raw:
        raise ScoutInputError(f"{field} must not be blank")
    try:
        if date_only:
            return date.fromisoformat(raw).isoformat()
        else:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone is required")
    except ValueError as exc:
        kind = "YYYY-MM-DD" if date_only else "ISO 8601 with timezone"
        raise ScoutInputError(f"{field} must be {kind}: {raw!r}") from exc
    return raw


def _normalize_domain(raw: str) -> str:
    if not isinstance(raw, str):
        raise ScoutInputError("domain must be a string")
    value = raw.strip().lower().rstrip(".")
    if not value or len(value) > 253:
        raise ScoutInputError(f"invalid domain: {raw!r}")
    if "://" in value or any(ch in value for ch in "/?#@"):
        raise ScoutInputError(f"domain must be a hostname, not a URL: {raw!r}")
    try:
        value = value.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ScoutInputError(f"invalid domain: {raw!r}") from exc
    if value == "localhost" or value.endswith((".localhost", ".local")):
        raise ScoutInputError(f"private/local domain is not allowed: {raw!r}")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ScoutInputError(f"non-public IP address is not allowed: {raw!r}")
    if address is None:
        labels = value.split(".")
        # URL clients still accept legacy shortened/octal/hex IPv4 spellings
        # that `ipaddress.ip_address()` deliberately rejects. Treating those as
        # DNS names lets values such as `127.1` or `0x7f.0.0.1` pass this check
        # even though browsers/curl resolve them to loopback.
        if labels and all(_LEGACY_IPV4_LABEL_RE.fullmatch(label) for label in labels):
            raise ScoutInputError(f"non-canonical IP address is not allowed: {raw!r}")
        if len(labels) < 2 or any(
            not label
            or len(label) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ScoutInputError(f"invalid public hostname: {raw!r}")
    return value


def _normalize_url(raw: str) -> tuple[str, str]:
    if not isinstance(raw, str):
        raise ScoutInputError("source url must be a string")
    value = raw.strip()
    if (
        not value
        or len(value) > MAX_URL_CHARS
        or _CONTROL_RE.search(value)
        or any(ch.isspace() for ch in value)
    ):
        raise ScoutInputError("source url is blank, too long, or contains whitespace")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ScoutInputError(f"invalid source url: {value!r}") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ScoutInputError(f"source url must be absolute HTTP(S): {value!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ScoutInputError("source url must not contain credentials")
    host = _normalize_domain(parsed.hostname)
    if port is not None and not (0 < port <= 65535):
        raise ScoutInputError(f"invalid source url port: {port}")
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    host_for_url = f"[{host}]" if ":" in host else host
    netloc = host_for_url if port is None or default_port else f"{host_for_url}:{port}"
    normalized = SplitResult(
        parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""
    )
    return urlunsplit(normalized), host


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise ScoutInputError("run id must be a string")
    value = run_id.strip()
    if not _RUN_ID_RE.fullmatch(value):
        raise ScoutInputError(f"invalid scout run id: {run_id!r}")
    return value


def _run_dir(run_id: str) -> Path:
    return scout_cache_dir() / "web" / "runs" / _validate_run_id(run_id)


def _new_run_id(query: str, now: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:36] or "query"
    stamp = re.sub(r"[^0-9]", "", now)[:14]
    return f"{stamp}-{slug}-{uuid.uuid4().hex[:8]}"


def create_request(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_fetches: int = DEFAULT_MAX_FETCHES,
    domains: list[str] | None = None,
    since: str | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> tuple[dict, Path]:
    """Create and persist a write-once agent-handoff request."""
    clean_query = _clean_text(query, field="query", max_chars=MAX_QUERY_CHARS)
    max_results = _bounded_int(
        max_results, field="max_results", minimum=1, maximum=HARD_MAX_RESULTS
    )
    max_fetches = _bounded_int(
        max_fetches, field="max_fetches", minimum=0,
        maximum=min(max_results, HARD_MAX_FETCHES),
    )
    if domains is not None and not isinstance(domains, list):
        raise ScoutInputError("domains must be a list")
    normalized_domains: list[str] = []
    for raw in domains or []:
        domain = _normalize_domain(raw)
        if domain not in normalized_domains:
            normalized_domains.append(domain)
    if len(normalized_domains) > MAX_DOMAINS:
        raise ScoutInputError(f"at most {MAX_DOMAINS} domains may be supplied")
    clean_since = _validate_iso(since, field="since", date_only=True) if since else None
    now = created_at or _utc_now()
    normalized_now = _validate_iso(now, field="created_at")
    if normalized_now != now:
        raise ScoutInputError("created_at is not normalized")
    rid = _validate_run_id(run_id) if run_id else _new_run_id(clean_query, now)

    request = {
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "mode": "web-agent-handoff",
        "query": clean_query,
        "created_at": now,
        "evidence_class": "discovery-only",
        "constraints": {
            "max_results": max_results,
            "max_fetches": max_fetches,
            "domains": normalized_domains,
            "since": clean_since,
            "follow_transitive_links": False,
        },
        "receipt_contract": _receipt_contract(),
    }
    directory = _run_dir(rid)
    try:
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "request.json"
        write_json_atomic(path, request)
    except FileExistsError as exc:
        raise ScoutInputError(f"scout run already exists: {rid}") from exc
    except OSError as exc:
        raise ScoutStorageUnavailable(f"cannot create web-scout request: {exc}") from exc
    return request, path


def _read_required_json(path: Path, *, label: str) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScoutInputError(f"{label} not found: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ScoutStorageUnavailable(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScoutInputError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ScoutInputError(f"{label} must be a JSON object")
    return raw


def _artifact_sha256(path: Path, *, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise ScoutStorageUnavailable(f"cannot hash {label} {path}: {exc}") from exc


def _validate_request_document(request: dict, *, run_id: str) -> None:
    version = request.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ScoutInputError(
            f"unsupported scout request schema_version {version!r} "
            f"(expected {SCHEMA_VERSION}); create a new request rather than "
            "editing this one"
        )
    unexpected = sorted(set(request) - _REQUEST_FIELDS)
    missing = sorted(_REQUEST_FIELDS - set(request))
    if unexpected or missing:
        details = []
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise ScoutInputError("invalid scout request schema (" + "; ".join(details) + ")")
    if request.get("run_id") != run_id:
        raise ScoutInputError("scout request run_id does not match its directory")
    if request.get("mode") != "web-agent-handoff":
        raise ScoutInputError("scout request has an unsupported mode")
    if request.get("evidence_class") != "discovery-only":
        raise ScoutInputError("scout request must remain discovery-only")
    query = _clean_text(request.get("query"), field="query", max_chars=MAX_QUERY_CHARS)
    if query != request.get("query"):
        raise ScoutInputError("scout request query is not normalized")
    created_at = request.get("created_at")
    if _validate_iso(created_at, field="created_at") != created_at:
        raise ScoutInputError("scout request created_at is not normalized")

    constraints = request.get("constraints")
    if not isinstance(constraints, dict):
        raise ScoutInputError("scout request constraints must be an object")
    if set(constraints) != _CONSTRAINT_FIELDS:
        raise ScoutInputError("scout request has an invalid constraints schema")
    max_results = constraints.get("max_results")
    max_fetches = constraints.get("max_fetches")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= HARD_MAX_RESULTS
    ):
        raise ScoutInputError("scout request has invalid max_results")
    if (
        isinstance(max_fetches, bool)
        or not isinstance(max_fetches, int)
        or not 0 <= max_fetches <= min(max_results, HARD_MAX_FETCHES)
    ):
        raise ScoutInputError("scout request has invalid max_fetches")
    domains = constraints.get("domains")
    if (
        not isinstance(domains, list)
        or len(domains) > MAX_DOMAINS
        or any(not isinstance(domain, str) for domain in domains)
    ):
        raise ScoutInputError("scout request has invalid domains")
    normalized_domains = [_normalize_domain(domain) for domain in domains]
    if normalized_domains != domains or len(set(domains)) != len(domains):
        raise ScoutInputError("scout request domains are not normalized")
    since = constraints.get("since")
    if since is not None:
        if _validate_iso(since, field="since", date_only=True) != since:
            raise ScoutInputError("scout request since is not normalized")
    if constraints.get("follow_transitive_links") is not False:
        raise ScoutInputError("scout request must forbid transitive links")

    contract = request.get("receipt_contract")
    if not isinstance(contract, dict):
        raise ScoutInputError("scout request has no receipt contract")
    if contract != _receipt_contract():
        raise ScoutInputError("scout request has an invalid receipt contract")


def load_request(run_id: str) -> tuple[dict, Path]:
    directory = _run_dir(run_id)
    if directory.is_symlink():
        raise ScoutInputError("scout run directory must not be a symlink")
    path = directory / "request.json"
    if path.is_symlink():
        raise ScoutInputError("scout request artifact must not be a symlink")
    request = _read_required_json(path, label="scout request")
    _validate_request_document(request, run_id=run_id)
    return request, path


def _domain_allowed(host: str, allowed: list[str]) -> bool:
    return not allowed or any(
        host == domain or host.endswith(f".{domain}") for domain in allowed
    )


def _normalize_sources(
    raw_sources: object, *, request: dict
) -> tuple[list[dict], int]:
    if not isinstance(raw_sources, list):
        raise ScoutInputError("receipt sources must be an array")
    max_results = int(request["constraints"]["max_results"])
    if len(raw_sources) > max_results:
        raise ScoutInputError(
            f"source count {len(raw_sources)} exceeds request limit {max_results}"
        )
    allowed = request["constraints"].get("domains") or []
    since = request["constraints"].get("since")
    by_url: dict[str, dict] = {}
    duplicates = 0
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ScoutInputError(f"sources[{index}] must be an object")
        unexpected = sorted(
            set(raw) - {"url", "fetched", "title", "published_at"}
        )
        if unexpected:
            raise ScoutInputError(
                f"sources[{index}] has unsupported fields: {', '.join(unexpected)}"
            )
        if "url" not in raw or "fetched" not in raw:
            raise ScoutInputError(f"sources[{index}] requires url and fetched")
        url, host = _normalize_url(raw["url"])
        if not _domain_allowed(host, allowed):
            raise ScoutInputError(
                f"sources[{index}] domain {host!r} is outside request bounds"
            )
        fetched = raw["fetched"]
        if not isinstance(fetched, bool):
            raise ScoutInputError(f"sources[{index}].fetched must be boolean")
        title = raw.get("title")
        title = (
            _clean_text(title, field=f"sources[{index}].title", max_chars=MAX_TITLE_CHARS)
            if title not in (None, "")
            else None
        )
        published = raw.get("published_at")
        published = (
            _validate_iso(
                published, field=f"sources[{index}].published_at", date_only=True
            )
            if published not in (None, "")
            else None
        )
        if since and published and published < since:
            raise ScoutInputError(
                f"sources[{index}].published_at predates the request's --since bound"
            )
        candidate = {
            "url": url,
            "fetched": fetched,
            "title": title,
            "published_at": published,
        }
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = candidate
            continue
        duplicates += 1
        if existing["published_at"] and published and existing["published_at"] != published:
            raise ScoutInputError(f"duplicate source {url} has conflicting published_at")
        existing["fetched"] = existing["fetched"] or fetched
        # Pick duplicate metadata deterministically so equivalent receipts
        # have one identity regardless of input ordering.
        titles = [value for value in (existing["title"], title) if value]
        existing["title"] = min(titles) if titles else None
        dates = [value for value in (existing["published_at"], published) if value]
        existing["published_at"] = min(dates) if dates else None

    normalized = list(by_url.values())
    fetched_count = sum(1 for source in normalized if source["fetched"])
    max_fetches = int(request["constraints"]["max_fetches"])
    if fetched_count > max_fetches:
        raise ScoutInputError(
            f"fetched source count {fetched_count} exceeds request limit {max_fetches}"
        )
    normalized.sort(key=lambda source: source["url"])
    return normalized, duplicates


def accept_submission(
    run_id: str,
    incoming: dict,
    *,
    recorded_at: str | None = None,
) -> tuple[dict, Path]:
    """Validate one minimal provenance receipt and persist a write-once copy."""
    request, request_path = load_request(run_id)
    if not isinstance(incoming, dict):
        raise ScoutInputError("scout receipt must be a JSON object")
    incoming_version = incoming.get("schema_version")
    if type(incoming_version) is not int or incoming_version != SCHEMA_VERSION:
        raise ScoutInputError(f"scout receipt schema_version must be {SCHEMA_VERSION}")
    if incoming.get("run_id") != run_id:
        raise ScoutInputError("receipt run_id does not match the request")
    prose_fields = {"findings", "coverage_gaps", "brief", "report", "excerpt"}
    present_prose = sorted(prose_fields & set(incoming))
    if present_prose:
        raise ScoutInputError(
            "research prose is not part of a source receipt; remove: "
            + ", ".join(present_prose)
        )
    unexpected = sorted(set(incoming) - _RECEIPT_FIELDS)
    if unexpected:
        raise ScoutInputError(
            "unsupported source-receipt fields: " + ", ".join(unexpected)
        )
    harness = _clean_text(
        incoming.get("harness"), field="harness", max_chars=MAX_HARNESS_CHARS
    )
    sources, duplicates = _normalize_sources(
        incoming.get("sources"), request=request
    )
    discovery_method = _validate_discovery_method(
        incoming.get("discovery_method"), sources=sources
    )
    fetched_count = sum(1 for source in sources if source["fetched"])

    directory = _run_dir(run_id)
    receipt_out = directory / "receipt.json"
    manifest_out = directory / "manifest.json"
    try:
        # Receipt + manifest are one logical recording operation. The lock
        # prevents two host agents from interleaving the two atomic writes.
        with exclusive_lock(directory / ".receipt"):
            if receipt_out.is_symlink() or manifest_out.is_symlink():
                raise ScoutInputError("recorded scout artifacts must not be symlinks")
            existing_receipt = (
                _read_required_json(receipt_out, label="recorded scout receipt")
                if receipt_out.exists()
                else None
            )
            now = recorded_at or (
                str(existing_receipt.get("recorded_at"))
                if existing_receipt is not None
                else _utc_now()
            )
            normalized_now = _validate_iso(now, field="recorded_at")
            if normalized_now != now:
                raise ScoutInputError("recorded_at is not normalized")
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "evidence_class": "discovery-only",
                "recorded_at": now,
                "harness": harness,
                "discovery_method": discovery_method,
                "sources": sources,
            }
            if existing_receipt is not None:
                if existing_receipt != receipt:
                    raise ScoutInputError(
                        f"scout run {run_id} already has a different receipt; "
                        "refusing to overwrite provenance"
                    )
                # A canonical receipt is the identity of the run. Replays may
                # have supplied different duplicate ordering/counts; preserve
                # the first manifest rather than turning those into conflicts.
                if manifest_out.exists():
                    existing_manifest = _read_required_json(
                        manifest_out, label="scout manifest"
                    )
                    _validate_recorded_documents(
                        request, existing_receipt, existing_manifest
                    )
                    if existing_manifest.get("request_sha256") != _artifact_sha256(
                        request_path, label="scout request"
                    ) or existing_manifest.get("receipt_sha256") != _artifact_sha256(
                        receipt_out, label="recorded scout receipt"
                    ):
                        raise ScoutInputError("recorded scout artifacts fail their hashes")
                    return existing_manifest, manifest_out
            elif manifest_out.exists():
                raise ScoutInputError("recorded manifest exists without its receipt")
            encoded = json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "status": "recorded",
                "evidence_class": "discovery-only",
                "recorded_at": now,
                "harness": harness,
                "discovery_method": discovery_method,
                "source_count": len(sources),
                "fetched_count": fetched_count,
                "duplicates_dropped": duplicates,
                "request_sha256": _artifact_sha256(request_path, label="scout request"),
                "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            if existing_receipt is None:
                write_json_atomic(receipt_out, receipt)
            write_json_atomic(manifest_out, manifest)
    except OSError as exc:
        raise ScoutStorageUnavailable(f"cannot persist scout receipt: {exc}") from exc
    return manifest, manifest_out


def accept_receipt(
    run_id: str,
    results_path: Path | str,
    *,
    recorded_at: str | None = None,
) -> tuple[dict, Path]:
    """Read and accept one source-receipt JSON file."""
    incoming = _read_required_json(Path(results_path), label="scout receipt")
    return accept_submission(run_id, incoming, recorded_at=recorded_at)


def record_sources(
    run_id: str,
    *,
    harness: str,
    discovery_method: str,
    fetched_urls: list[str] | None = None,
    snippet_urls: list[str] | None = None,
    published_at: dict[str, str] | None = None,
    recorded_at: str | None = None,
) -> tuple[dict, Path]:
    """Record provenance directly from CLI flags, with no JSON authoring."""
    for field, values in (("fetched_urls", fetched_urls), ("snippet_urls", snippet_urls)):
        if values is not None and not isinstance(values, list):
            raise ScoutInputError(f"{field} must be a list")
    if published_at is not None and not isinstance(published_at, dict):
        raise ScoutInputError("published_at must be a URL-to-date mapping")
    sources = [
        *({"url": url, "fetched": True} for url in fetched_urls or []),
        *({"url": url, "fetched": False} for url in snippet_urls or []),
    ]
    if published_at:
        normalized_dates: dict[str, str] = {}
        for raw_url, raw_date in published_at.items():
            url, _ = _normalize_url(raw_url)
            if not isinstance(raw_date, str):
                raise ScoutInputError("published_at dates must be strings")
            if url in normalized_dates and normalized_dates[url] != raw_date:
                raise ScoutInputError(f"published_at has conflicting dates for {url}")
            normalized_dates[url] = raw_date
        matched_dates: set[str] = set()
        for source in sources:
            normalized_url, _ = _normalize_url(source["url"])
            if normalized_url in normalized_dates:
                source["published_at"] = normalized_dates[normalized_url]
                matched_dates.add(normalized_url)
        unknown_dates = sorted(set(normalized_dates) - matched_dates)
        if unknown_dates:
            raise ScoutInputError(
                "published_at URL is not present in fetched/snippet sources: "
                + ", ".join(unknown_dates)
            )
    return accept_submission(
        run_id,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "harness": harness,
            "discovery_method": discovery_method,
            "sources": sources,
        },
        recorded_at=recorded_at,
    )


def _validate_recorded_documents(request: dict, receipt: dict, manifest: dict) -> None:
    receipt_version = receipt.get("schema_version")
    if type(receipt_version) is not int or receipt_version != SCHEMA_VERSION:
        raise ScoutInputError(
            f"unsupported recorded-receipt schema_version {receipt_version!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    if receipt.get("evidence_class") != "discovery-only":
        raise ScoutInputError("recorded scout receipt must remain discovery-only")
    unexpected = sorted(set(receipt) - _RECORDED_RECEIPT_FIELDS)
    if unexpected:
        raise ScoutInputError(
            "recorded scout receipt has unsupported fields: " + ", ".join(unexpected)
        )
    recorded_at = _validate_iso(receipt.get("recorded_at"), field="recorded_at")
    harness = _clean_text(
        receipt.get("harness"), field="harness", max_chars=MAX_HARNESS_CHARS
    )
    sources, duplicates = _normalize_sources(
        receipt.get("sources"), request=request
    )
    if duplicates or receipt.get("sources") != sources:
        raise ScoutInputError("recorded scout sources are not normalized")
    discovery_method = _validate_discovery_method(
        receipt.get("discovery_method"), sources=sources
    )
    fetched_count = sum(1 for source in sources if source["fetched"])
    expected = {
        "status": "recorded",
        "evidence_class": "discovery-only",
        "recorded_at": recorded_at,
        "harness": harness,
        "discovery_method": discovery_method,
        "source_count": len(sources),
        "fetched_count": fetched_count,
    }
    unexpected_manifest = sorted(set(manifest) - _MANIFEST_FIELDS)
    missing_manifest = sorted(_MANIFEST_FIELDS - set(manifest))
    if unexpected_manifest or missing_manifest:
        details = []
        if unexpected_manifest:
            details.append("unsupported fields: " + ", ".join(unexpected_manifest))
        if missing_manifest:
            details.append("missing fields: " + ", ".join(missing_manifest))
        raise ScoutInputError("invalid scout manifest schema (" + "; ".join(details) + ")")
    manifest_version = manifest.get("schema_version")
    if type(manifest_version) is not int or manifest_version != SCHEMA_VERSION:
        raise ScoutInputError("unsupported scout manifest schema_version")
    for field, value in expected.items():
        actual = manifest.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ScoutInputError(f"scout manifest disagrees on {field}")
    dropped = manifest.get("duplicates_dropped")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
        raise ScoutInputError("scout manifest has invalid duplicates_dropped")


def _load_recorded(run_id: str) -> tuple[dict, dict, dict]:
    request, request_path = load_request(run_id)
    directory = _run_dir(run_id)
    receipt_path = directory / "receipt.json"
    manifest_path = directory / "manifest.json"
    if receipt_path.is_symlink() or manifest_path.is_symlink():
        raise ScoutInputError("recorded scout artifacts must not be symlinks")
    receipt = _read_required_json(receipt_path, label="recorded scout receipt")
    manifest = _read_required_json(manifest_path, label="scout manifest")
    if receipt.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise ScoutInputError("recorded scout artifacts disagree on run_id")
    _validate_recorded_documents(request, receipt, manifest)
    if manifest.get("request_sha256") != _artifact_sha256(
        request_path, label="scout request"
    ):
        raise ScoutInputError("scout request changed after its receipt was recorded")
    if manifest.get("receipt_sha256") != _artifact_sha256(
        receipt_path, label="recorded scout receipt"
    ):
        raise ScoutInputError("recorded scout receipt fails its manifest hash")
    return request, receipt, manifest


def load_run(run_id: str) -> dict:
    """Load one request and its cached discovery result, when recorded.

    The request remains top-level so ``show --json`` is still a directly usable
    handoff for a resumed agent.  Once the run is recorded, ``cached_result``
    exposes the exact receipt and manifest already stored under
    ``.scout-cache/``; no second rendered artifact is created.
    """
    request, request_path = load_request(run_id)
    directory = _run_dir(run_id)
    receipt_path = directory / "receipt.json"
    manifest_path = directory / "manifest.json"
    receipt_exists = receipt_path.exists()
    manifest_exists = manifest_path.exists()
    if receipt_exists != manifest_exists:
        raise ScoutInputError("recorded receipt and manifest are incomplete")

    out = {
        **request,
        "state": "requested",
        "request_path": str(request_path),
        "cached_result": None,
    }
    if not receipt_exists:
        return out

    _recorded_request, receipt, manifest = _load_recorded(run_id)
    out["state"] = "recorded"
    out["cached_result"] = {
        "receipt": receipt,
        "manifest": manifest,
        "receipt_path": str(receipt_path),
        "manifest_path": str(manifest_path),
    }
    return out


def inspect_run(run_id: str) -> dict:
    """Return one local run's resumable state without writing."""
    rid = str(run_id)
    row = {
        "run_id": rid,
        "state": "invalid",
        "query": None,
        "created_at": None,
        "source_count": None,
        "discovery_method": None,
        "next_command": None,
        "error": None,
    }
    try:
        run = load_run(rid)
        row["query"] = run["query"]
        row["created_at"] = run["created_at"]
        if run["state"] == "requested":
            row["state"] = "requested"
            row["next_command"] = f"researchwiki scout web show {rid} --json"
            return row

        cached_result = run["cached_result"]
        row["source_count"] = cached_result["manifest"]["source_count"]
        row["discovery_method"] = cached_result["manifest"]["discovery_method"]
        row["state"] = "recorded"
        return row
    except (ScoutInputError, ScoutStorageUnavailable) as exc:
        # Storage failures degrade to an `invalid` row rather than propagating.
        # `status` inspects every run directory, so a single unreadable or
        # undecodable artifact — a permission error, a truncated write — would
        # otherwise abort the whole dashboard with an environment exit over a
        # subsystem the corpus may never have used. Catching only
        # ScoutInputError left exactly that gap: malformed JSON downgraded
        # cleanly, an unreadable file did not. `list_runs` still raises if the
        # runs directory itself cannot be enumerated, which is a real
        # environment failure rather than one bad run.
        row["error"] = str(exc)
        return row


def list_runs(*, state: str | None = None) -> list[dict]:
    """Enumerate local web-scout runs, newest first."""
    if state is not None and state not in RUN_STATES:
        raise ScoutInputError(f"state must be one of: {', '.join(RUN_STATES)}")
    base = scout_cache_dir() / "web" / "runs"
    if not base.exists():
        return []
    try:
        directories = [path for path in base.iterdir() if path.is_dir()]
    except OSError as exc:
        raise ScoutStorageUnavailable(f"cannot list web-scout runs: {exc}") from exc
    rows = [inspect_run(path.name) for path in directories]
    if state is not None:
        rows = [row for row in rows if row["state"] == state]
    rows.sort(key=lambda row: (row.get("created_at") or "", row["run_id"]), reverse=True)
    return rows
