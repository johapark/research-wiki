"""Weighted paper-graph primitives: edges and modularity clustering.

Sits under `index/` because it operates on the semantic-embedding matrix
from `pages_semantic`; extracted from `tasks/_synthesis_candidates.py` (behind `researchwiki candidates
synthesis`) so the modularity algorithm is independently importable and
unit-tested (see `tests/test_synthesis_graph.py`).
Two representation-agnostic pieces live here:

  - `Edge`: a weighted paper-paper edge carrying per-signal weights.
  - `louvain`: phase-1 Louvain modularity community detection.

The Page→graph adapter (`_build_edges`, which reads wikilinks/keywords off wiki
pages to construct these edges) stays in the task module — it's coupled to the
`Page` representation and to wiki-parsing helpers shared with the coverage code.
This module knows nothing about pages; it only sees Edge weights and node keys.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------- edge-weight semantics + clustering cutoff ----------
# Multiple signals on one pair add; a pair's `Edge.total` is compared against
# these thresholds by the adapter (per-signal) and by `louvain` (aggregate).

WIKILINK_WEIGHT = 1.0
SEMANTIC_THRESHOLD = 0.65    # min cosine for a semantic edge to register
KEYWORD_THRESHOLD = 0.2      # min Jaccard for a keyword edge to register
EDGE_THRESHOLD = 1.0         # cluster cutoff after weight aggregation


@dataclass
class Edge:
    """One edge between two papers, with per-signal weights for transparency."""
    a: str                          # category/stem
    b: str
    wikilink: float = 0.0
    semantic: float = 0.0
    keyword: float = 0.0

    @property
    def total(self) -> float:
        return self.wikilink + self.semantic + self.keyword

    @property
    def signals(self) -> list[str]:
        out: list[str] = []
        if self.wikilink:
            out.append("wikilink")
        if self.semantic:
            out.append(f"semantic({self.semantic:.2f})")
        if self.keyword:
            out.append(f"keyword({self.keyword:.2f})")
        return out


# ---------- Louvain community detection ----------
#
# Why not connected components: on a dense paper-paper graph (many edges,
# transitive reachability), connected components collapses everything into
# one giant blob — first v0 surfaced 71/90 papers in a single "cluster".
# Louvain optimizes modularity, which rewards intra-cluster density and
# penalizes inter-cluster edges, so a hub paper linking two communities
# doesn't merge them.
#
# This is "phase 1 only" Louvain (Blondel et al. 2008): single greedy local
# optimization without the multi-level coarsening pass. For ~100-node
# graphs the single phase recovers comparable communities to the full
# algorithm, with a fraction of the code.

def louvain(
    nodes: list[str],
    edges: list[Edge],
    threshold: float,
    *,
    max_passes: int = 50,
) -> list[list[str]]:
    """Phase-1 Louvain modularity optimization.

    Modularity gain for moving node n from community C to C' (Blondel §2):

        ΔQ = (k_{n,C'} - k_{n,C}) / m
             + k_n * (Σ_tot[C] - k_n - Σ_tot[C']) / (2m²)

    where:
      m         = total edge weight (sum over unique pairs)
      k_n       = weighted degree of n
      k_{n,X}   = sum of weights of edges from n to community X members
      Σ_tot[X]  = sum of weighted degrees of all members of community X

    Convergence: iterate until no node move improves modularity, capped by
    `max_passes`. For 90-node graphs convergence is typically <10 passes.

    Returns a list of communities (each a sorted list of node keys),
    including singleton communities for isolated nodes — the caller
    filters by `min_cluster` size.
    """
    # Build weighted adjacency for edges meeting the threshold.
    adj: dict[str, dict[str, float]] = {n: {} for n in nodes}
    for e in edges:
        if e.total < threshold:
            continue
        if e.a not in adj or e.b not in adj:
            continue
        adj[e.a][e.b] = adj[e.a].get(e.b, 0.0) + e.total
        adj[e.b][e.a] = adj[e.b].get(e.a, 0.0) + e.total

    degrees: dict[str, float] = {n: sum(w for w in adj[n].values()) for n in nodes}
    m = sum(degrees.values()) / 2.0
    if m == 0:
        return [[n] for n in nodes]
    two_m_sq = (2.0 * m) ** 2

    # Initialize: each node in its own community (id = stable index).
    community: dict[str, int] = {n: i for i, n in enumerate(nodes)}
    sigma_tot: dict[int, float] = {i: degrees[nodes[i]] for i in range(len(nodes))}

    for _ in range(max_passes):
        improved = False
        # Fixed sorted order for determinism. Random shuffles help escape
        # local optima but make output non-reproducible — bad for a tool
        # whose output drives human review.
        for n in sorted(nodes):
            if degrees[n] == 0:
                continue
            c_curr = community[n]
            k_n = degrees[n]

            # Sum edge weight from n to each neighbor community.
            k_n_to: dict[int, float] = {}
            for neigh, w in adj[n].items():
                c = community[neigh]
                k_n_to[c] = k_n_to.get(c, 0.0) + w
            k_n_to_curr = k_n_to.get(c_curr, 0.0)

            # Find the move with the largest positive ΔQ.
            best_c = c_curr
            best_gain = 0.0
            for c_target, k_to_target in k_n_to.items():
                if c_target == c_curr:
                    continue
                gain = (k_to_target - k_n_to_curr) / m \
                    + k_n * (sigma_tot[c_curr] - k_n - sigma_tot[c_target]) / two_m_sq
                if gain > best_gain:
                    best_gain = gain
                    best_c = c_target

            if best_c != c_curr:
                sigma_tot[c_curr] -= k_n
                sigma_tot[best_c] = sigma_tot.get(best_c, 0.0) + k_n
                community[n] = best_c
                improved = True

        if not improved:
            break

    # Group nodes by community id, drop empty communities.
    groups: dict[int, list[str]] = {}
    for n, c in community.items():
        groups.setdefault(c, []).append(n)
    return [sorted(members) for members in groups.values()]
