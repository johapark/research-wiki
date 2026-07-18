"""Claim-graph primitives — content-addressed claim identity + edge cache.

  slug   — deterministic content-addressed identity for each claim
  edges  — SQLite-backed edge cache under .claim-graph/edges.db

The DB invariant (see researchwiki/db/schema.sql:1-18) forbids LLM-authored
content in state.db, so edges live in a **separate, derived cache** that
`researchwiki claim-graph reconcile` re-syncs after every `db rebuild`. The
only thing that lands in state.db is the deterministic `claim_slug` column
and the `slug_scheme_version` counter — both computed by hash, invariant-legal.
"""

from .slug import (
    SECTION_ABBR,
    SLUG_SCHEME_VERSION,
    compute_claim_slug,
    disambiguate_slug,
    normalize_claim_text,
)
from .edges import (
    Edge,
    RELATIONS,
    STATUSES,
    ReconcileStats,
    counts_by_relation,
    delete_edge,
    edges_db_path,
    open_edges_db,
    query,
    reconcile,
    set_status,
    upsert_edge,
)

__all__ = [
    "SECTION_ABBR",
    "SLUG_SCHEME_VERSION",
    "compute_claim_slug",
    "disambiguate_slug",
    "normalize_claim_text",
    "Edge",
    "RELATIONS",
    "STATUSES",
    "ReconcileStats",
    "counts_by_relation",
    "delete_edge",
    "edges_db_path",
    "open_edges_db",
    "query",
    "reconcile",
    "set_status",
    "upsert_edge",
]
