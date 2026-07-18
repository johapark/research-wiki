"""Content-addressed claim slug — foundation for edges + page anchors.

`claim_slug = f"{SECTION_ABBR[section]}-{blake2s(normalize(text), digest_size=4).hex()}"`

Properties:
  - deterministic — same text ⇒ same slug ⇒ survives `db rebuild`
  - content-addressed — position-drift-immune; text edit ⇒ new slug ⇒
    correctly orphans stale edges
  - invariant-legal — a computed column, not LLM content, so it belongs
    in state.db even under the "no LLM in DB" invariant

The normalize() spec + SECTION_ABBR map are frozen at v1. A bump to
`SLUG_SCHEME_VERSION` is a corpus-wide re-slug + re-judge event, not an
in-place migration — the reconcile pass marks every mismatching edge stale.
"""

from __future__ import annotations

import hashlib
import re


# Version of the slug scheme (normalize() + SECTION_ABBR). Bump on any
# semantics-affecting change. `schema_meta.slug_scheme_version` mirrors this;
# `.claim-graph/edges.db` rows carry it too so reconcile() can flag mismatched
# edges as stale.
SLUG_SCHEME_VERSION = 1


# Section names emitted by researchwiki.grade.parser.parse_claims (see
# `section_keys` in that module) mapped to short deterministic prefixes.
# Unknown sections fall back to the first three letters of the section name;
# introducing a new section type is a v-bump event.
SECTION_ABBR: dict[str, str] = {
    "key_contributions": "kc",
    "results": "res",
    "limitations": "lim",
    "methodology": "met",
}


# Precompiled patterns for normalize_claim_text. Order matters — wikilinks
# and footnote refs must be stripped before whitespace collapse so their
# trailing whitespace doesn't accumulate.
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_FOOTNOTE_REF_RE = re.compile(r"\[\^[^\]]+\]:?")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_MARKERS_RE = re.compile(r"[*_`]+")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:]+$")
_SURROUNDING_QUOTE_RE = re.compile(r'^["\']|["\']$')


def normalize_claim_text(text: str) -> str:
    """Deterministic normalization of a claim before hashing.

    Frozen v1 spec:
      lowercase → strip wikilinks + footnote refs + markdown emphasis / code
      markers + HTML tags → collapse whitespace → drop trailing `.,;:` and
      surrounding quotes.

    No stemming, no synonym folding — those are judge-layer concerns. Same
    normalized string ⇒ same claim, definitionally, so any change here must
    bump SLUG_SCHEME_VERSION.
    """
    s = text.lower()
    s = _WIKILINK_RE.sub("", s)
    s = _FOOTNOTE_REF_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _MD_MARKERS_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    # Repeat quote-strip so `"foo."` → `foo`, not `foo.`
    while True:
        prev = s
        s = _SURROUNDING_QUOTE_RE.sub("", s).strip()
        s = _TRAILING_PUNCT_RE.sub("", s).strip()
        if s == prev:
            break
    return s


def _section_prefix(section: str) -> str:
    """SECTION_ABBR lookup with a deterministic fallback for unmapped sections.

    Fallback: first three lowercase-alphanumeric chars of the section name.
    Kept deterministic so a rebuild never surprises the cache — but a new
    section type SHOULD be added to SECTION_ABBR explicitly (and the version
    bumped) rather than relying on the fallback long-term.
    """
    if section in SECTION_ABBR:
        return SECTION_ABBR[section]
    clean = "".join(c for c in section.lower() if c.isalnum())
    return clean[:3] or "sec"


def compute_claim_slug(section: str, text: str) -> str:
    """Return `{section-prefix}-{8-hex-hash}` for a claim.

    Deterministic given (section, text); collisions (§3.1) are resolved by
    the caller via `disambiguate_slug` when the DB's UNIQUE constraint fires.
    """
    prefix = _section_prefix(section)
    digest = hashlib.blake2s(
        normalize_claim_text(text).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"{prefix}-{digest}"


def disambiguate_slug(base_slug: str, position: int) -> str:
    """Append the position suffix used when two claims in one paper normalize
    identically — a degenerate case.

    Note: the suffix is NOT durable across reorder. Callers are expected to
    treat position-suffixed edges as best-effort; a reorder that changes the
    position simply orphans the affected edges, which reconcile() marks stale.
    """
    return f"{base_slug}-{position}"
