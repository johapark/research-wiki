"""Permanent suppression list for discovery pairs judged not-a-relation.

`claim-overlap --discover` is stateless: it re-derives its ranked queue from
claim embeddings and term statistics on every call. A pair you looked at and
rejected — two papers whose claims share distinctive vocabulary without either
engaging the other — therefore resurfaces at the same rank on the next run, and
a queue that keeps re-proposing settled questions stops being read.

This is the pair-level analogue of `concepts.declines` and follows its shape
deliberately: a manual, explicit denylist that is permanent until a human
removes the entry, not a decay stamp. A decay stamp re-nags after a window
because the underlying condition can change; two papers do not become related
because time passed.

**Fingerprinted on the evidence, like `claim_overlap_runs`.** A dismissal is a
judgment about the claims that were on the table, so it should not outlive them.
Each entry records a hash of both papers' contribution-claim *slugs*, and
`dismissed_pairs` drops any entry whose fingerprint no longer matches — the pair
returns to the queue to be judged against its new evidence.

Claim slugs are the right fingerprint because they are already content-addressed
(`claim_graph.slug`: `blake2s(normalize(text))` prefixed by section). That gives
exactly the sensitivity wanted and no more: stable across `db rebuild`, since
unchanged text yields the same slug, but changed on a real edit, an added claim,
or a removed one. Hashing the claim *texts* directly would behave the same but
duplicate work the slug already did; hashing whole pages would churn on any prose
edit anywhere.

An entry written before fingerprints existed, or one whose papers have no claims
(the DB is unavailable, or a page was removed), is treated as **still valid** —
suppression is the conservative failure mode for a list of human decisions.

Keyed by the sorted stem pair, so dismissing (a, b) also suppresses (b, a).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..fsatomic import write_json_atomic

DISMISSALS_FILENAME = ".pair-dismissals.json"

# Sections whose claims constitute the evidence a dismissal was judged against —
# the same set concept attachment and the discovery ranker use.
_CONTRIBUTION_SECTIONS = ("key_contributions", "results", "methodology")


def _dismissals_path() -> Path:
    from ..paths import wiki_root
    return wiki_root() / DISMISSALS_FILENAME


def pair_key(stem_a: str, stem_b: str) -> str:
    """Order-independent key for a paper pair."""
    return "::".join(sorted((stem_a.strip(), stem_b.strip())))


def load_dismissals() -> dict[str, dict]:
    """key -> {stems, reason, dismissed_at, source}. Empty dict when absent or
    unreadable — never raises."""
    p = _dismissals_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def claims_fingerprint(stem_a: str, stem_b: str) -> str | None:
    """Hash of both papers' contribution-claim slugs, or None when unavailable.

    Slugs are content-addressed, so this changes exactly when the evidence does:
    a rewritten claim gets a new slug, an added or removed claim changes the set,
    and a `db rebuild` that changes nothing yields the same hash. None means
    "could not tell" (no DB, no claims) and callers must treat that as *valid*
    rather than stale — see the module docstring.
    """
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        rows = conn.execute(
            "SELECT claim_slug FROM claims "
            " WHERE paper_stem IN (?, ?) AND section IN (?, ?, ?) "
            "   AND claim_slug IS NOT NULL AND is_cross_ref = 0",
            (stem_a, stem_b, *_CONTRIBUTION_SECTIONS),
        ).fetchall()
    except Exception:
        return None
    finally:
        conn.close()
    slugs = sorted(r[0] for r in rows)
    if not slugs:
        return None
    return hashlib.blake2s("\n".join(slugs).encode("utf-8"),
                           digest_size=8).hexdigest()


def _slugs_for_stems(stems: set[str]) -> dict[str, list[str]]:
    """stem -> its contribution-claim slugs, in ONE query.

    The batched half of `claims_fingerprint`. `dismissed_pairs` is called from
    `discover_pairs`, which `status` runs every invocation — computing each
    entry's fingerprint through its own connection made that O(dismissals)
    SQLite connections per status run.
    """
    if not stems:
        return {}
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return {}
    try:
        placeholders = ",".join("?" * len(stems))
        rows = conn.execute(
            f"SELECT paper_stem, claim_slug FROM claims "
            f" WHERE paper_stem IN ({placeholders}) "
            f"   AND section IN (?, ?, ?) "
            f"   AND claim_slug IS NOT NULL AND is_cross_ref = 0",
            (*stems, *_CONTRIBUTION_SECTIONS),
        ).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()
    out: dict[str, list[str]] = {}
    for stem, slug in rows:
        out.setdefault(stem, []).append(slug)
    return out


def _hash_slugs(slugs: list[str]) -> str | None:
    """Fingerprint from an already-fetched slug list. None when empty."""
    if not slugs:
        return None
    return hashlib.blake2s("\n".join(sorted(slugs)).encode("utf-8"),
                           digest_size=8).hexdigest()


def is_stale(entry: dict, *, slugs_by_stem: dict[str, list[str]] | None = None) -> bool:
    """True when the claims a dismissal was judged against have changed.

    False for entries with no recorded fingerprint (written before the field
    existed) and whenever the current fingerprint can't be computed.

    `slugs_by_stem` lets a caller checking many entries pre-fetch the slug table
    once (see `_slugs_for_stems`); omitted, it falls back to a per-pair query,
    which is fine for the one-shot CLI listing.
    """
    recorded = entry.get("claims_fingerprint")
    if not recorded:
        return False
    stems = entry.get("stems")
    if not (isinstance(stems, list) and len(stems) == 2):
        return False
    if slugs_by_stem is None:
        current = claims_fingerprint(stems[0], stems[1])
    else:
        current = _hash_slugs(slugs_by_stem.get(stems[0], [])
                              + slugs_by_stem.get(stems[1], []))
    if current is None:
        return False
    return current != recorded


def dismissed_pairs(*, honor_fingerprints: bool = True) -> set[tuple[str, str]]:
    """Sorted stem tuples, in the shape `discover_pairs` filters against.

    Entries whose evidence has changed are omitted, so the pair returns to the
    queue for a fresh judgment. `honor_fingerprints=False` returns every recorded
    pair regardless — for listing and management, not for filtering.
    """
    entries = [e for e in load_dismissals().values()
               if isinstance(e.get("stems"), list) and len(e["stems"]) == 2]
    if not entries:
        return set()

    # One query for every stem involved, rather than one per entry: this runs
    # inside `discover_pairs`, which `status` calls on every invocation.
    slugs_by_stem = {}
    if honor_fingerprints and any(e.get("claims_fingerprint") for e in entries):
        slugs_by_stem = _slugs_for_stems(
            {s for e in entries for s in e["stems"]})

    out: set[tuple[str, str]] = set()
    for entry in entries:
        if honor_fingerprints and is_stale(entry, slugs_by_stem=slugs_by_stem):
            continue
        out.add(tuple(sorted(entry["stems"])))  # type: ignore[arg-type]
    return out


def _write(dismissals: dict[str, dict]) -> None:
    """Persist atomically.

    Same reasoning as `concepts.declines._write_declines`: this file is human
    judgment, and `load_dismissals` reads a corrupt file as an empty one — so a
    truncated write would silently un-suppress every past dismissal, with
    re-proposed noise as the only symptom.
    """
    write_json_atomic(_dismissals_path(), dismissals)


def add_dismissals(pairs: list[tuple[str, str, str]], *,
                   source: str = "manual") -> list[str]:
    """Record `(stem_a, stem_b, reason)` dismissals in ONE read-modify-write.

    Re-dismissing an existing pair updates its reason and timestamp rather than
    erroring. Returns the keys in input order.
    """
    if not pairs:
        return []
    dismissals = load_dismissals()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    keys: list[str] = []
    for stem_a, stem_b, reason in pairs:
        key = pair_key(stem_a, stem_b)
        a, b = stem_a.strip(), stem_b.strip()
        entry = {
            "stems": sorted((a, b)),
            "reason": reason,
            "dismissed_at": stamp,
            "source": source,
        }
        fp = claims_fingerprint(a, b)
        if fp:
            entry["claims_fingerprint"] = fp
        dismissals[key] = entry
        keys.append(key)
    _write(dismissals)
    return keys


def add_dismissal(stem_a: str, stem_b: str, reason: str, *,
                  source: str = "manual") -> str:
    """Record one pair as dismissed. Returns its key."""
    return add_dismissals([(stem_a, stem_b, reason)], source=source)[0]


def remove_dismissal(stem_a: str, stem_b: str) -> bool:
    """Un-dismiss a pair. True if an entry was removed."""
    dismissals = load_dismissals()
    key = pair_key(stem_a, stem_b)
    if key not in dismissals:
        return False
    del dismissals[key]
    _write(dismissals)
    return True
