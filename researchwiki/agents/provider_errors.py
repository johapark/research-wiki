"""Secret-safe diagnostics for terminal model-provider failures.

Classification is deliberately narrow. Body markers distinguish account quota
from ordinary throttling and model access from an unrelated 404; status-only
fallbacks are used only where the meaning is stable after retries are exhausted.
Unknown client errors return ``None`` so their original debugging path remains.
"""

from __future__ import annotations


_NO_ACCESS = (
    "model_not_found",
    "does not exist or you do not have access",
    "does not have access to model",
    "permission_error",
    "permission denied",
)
_NO_QUOTA = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "billing hard limit",
)


def friendly_provider_error(
    provider: str, model: str, *, status: int | None, body: str,
) -> str | None:
    """Return an actionable terminal-failure message, or ``None``.

    The response body is inspected but never included in the returned text.
    Callers may therefore show a recognized diagnostic without reflecting a
    provider payload that could contain request or account details.
    """
    text = (body or "").lower()
    label = f"{provider} model {model}"
    if any(marker in text for marker in _NO_QUOTA):
        return (
            f"{label} is out of quota — add credits or raise the provider's "
            "billing limit, or select a different model."
        )
    if any(marker in text for marker in _NO_ACCESS):
        return (
            f"the configured account cannot access {label} — select an "
            "available model or check access in the provider console."
        )
    # Anthropic's not-found body may carry only its type plus the exact model.
    bare_model = model.split(":")[-1].lower()
    if "not_found_error" in text and f"model: {bare_model}" in text:
        return (
            f"the configured account cannot access {label} — select an "
            "available model or check access in the provider console."
        )
    if status == 401:
        return (
            f"{provider} authentication failed after retries — check the active "
            "API key and model profile."
        )
    if status == 429:
        return (
            f"{label} remained rate-limited after retries — lower ingest "
            "concurrency or configure the model's rpm limit."
        )
    if status is not None and status >= 500:
        return (
            f"{provider} remained unavailable after retries (HTTP {status}) — "
            "retry later or select another provider."
        )
    return None
