"""Cluster detection: graph build, coverage checks, and the top-level pipeline.

Handles everything up to (and including) the point at which the LLM judge
takes over. The judge is imported lazily inside `find_candidates` to keep
this module free of a hard dependency on the LLM stack — a `--no-judge` run
never loads `agents.llm`.

Data flow:
  read_pages → _build_edges → louvain → _check_synthesis_coverage
    → Candidate list → (optional) _judge_candidate → stats
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..index import pages_semantic
from ..index.graph import (
    EDGE_THRESHOLD,
    KEYWORD_THRESHOLD,
    SEMANTIC_THRESHOLD,
    WIKILINK_WEIGHT,
    Edge,
    louvain,
)
from ..wiki import Page, read_pages, strip_non_prose


# ---------- thresholds ----------
# Edge-weight + cluster-cutoff constants (WIKILINK_WEIGHT, SEMANTIC_THRESHOLD,
# KEYWORD_THRESHOLD, EDGE_THRESHOLD) and the `Edge` type / `louvain` clusterer
# live in `researchwiki.index.graph`; imported above. The thresholds below are
# detection *policy* (cluster size + coverage bands), local to this module.

DEFAULT_MIN_CLUSTER = 4

# Tri-state coverage thresholds. The existing-synthesis overlap fraction
# determines what verdict each surfaced cluster gets:
#
#     overlap ≥ DEFAULT_COVERED:  "covered" → skip silently
#     DEFAULT_EXTEND ≤ overlap < DEFAULT_COVERED:  "extend" → recommend
#         augmenting the existing synthesis with the missing members
#     overlap < DEFAULT_EXTEND:    "new" → recommend creating a fresh
#         synthesis page
#
# The middle band is the load-bearing addition: a cluster at 0.67 overlap
# (just below 0.70) usually means "the existing synthesis is *almost*
# covering this — fill the gap" rather than "create a new synthesis."
DEFAULT_COVERED = 0.70
DEFAULT_EXTEND = 0.40

WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]*)?(?:\|[^\]]+)?\]\]")


# ---------- types ----------


@dataclass
class MemberVerdict:
    """One LLM judgment on whether a proposed member fits the synthesis's scope.

    Verdicts:
      - in_scope:     belongs in the synthesis's main structure
      - tangential:   related but better placed in tensions / open questions
      - out_of_scope: different axis or topic; should not be added
    """
    key: str
    verdict: str
    rationale: str


@dataclass
class Candidate:
    """One proposed synthesis cluster awaiting human review.

    `verdict` is the tri-state recommendation:
      - "new":    no existing synthesis substantially overlaps; create one
      - "extend": existing synthesis at `nearest_synthesis` partially
                  covers the cluster; add the missing members to it
                  (members listed in `members_missing_from_nearest`)
    Clusters with overlap ≥ DEFAULT_COVERED are dropped before becoming
    Candidates (they're "already covered").

    `member_verdicts` is the Phase-B4.5 LLM editorial-scoping output —
    per-member fit verdicts with rationale. Empty when judging skipped.
    `judge_topic` (for "new" candidates only) is the LLM's proposed topic
    title; `judge_*_tokens` are the LLM call's billing.
    """
    members: list[str]              # sorted list of category/stem
    titles: dict[str, str]          # key → title (for the proposal markdown)
    density: float                  # mean edge weight within the cluster
    edges: list[Edge]
    edge_signal_counts: dict        # {wikilink: int, semantic: int, keyword: int}
    common_keywords: list[str]      # keywords in ≥2 cluster members, ranked
    nearest_synthesis: str | None   # closest existing synthesis (with overlap)
    nearest_synthesis_overlap: float
    verdict: str                    # "new" | "extend"
    members_missing_from_nearest: list[str]   # for "extend": members not yet in nearest's refs
    # Editorial scoping via LLM judge:
    member_verdicts: list[MemberVerdict] = field(default_factory=list)
    judge_topic: str = ""           # for "new": LLM-proposed topic title
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_batches: int = 0           # attempted LLM calls (including an incomplete run)
    judged: bool = False

    @property
    def slug(self) -> str:
        """Filename slug derived from common keywords, falling back to first stem.

        Prefixed with the verdict so a directory listing groups by edit kind:
          extend__crispr-off-target-prediction-guide-seq.md
          new__protein-structure-prediction-alphafold-3.md
        """
        body = self.common_keywords[:3] if self.common_keywords else [self.members[0].split("/", 1)[-1]]
        cleaned = "-".join(re.sub(r"[^a-z0-9]+", "-", kw.lower()).strip("-")
                           for kw in body if kw)[:80]
        return f"{self.verdict}__{cleaned}" if cleaned else f"{self.verdict}__cluster"


# ---------- helpers ----------


def _paper_keywords(p: Page) -> set[str]:
    """The page's `keywords:` as a lowercased set (via the type-tolerant accessor)."""
    return {k.lower() for k in p.list_field("keywords")}


def _parse_referenced_papers(body: str) -> set[str]:
    """Extract paper wikilinks from a synthesis page body.

    Synthesis pages cite through inline links and ``## References`` footnotes,
    not a ``referenced_papers:`` frontmatter registry. The historical helper
    name remains for callers, but the body is the authoritative source.
    """
    keys: set[str] = set()
    for m in WIKILINK_RE.finditer(body):
        keys.add(m.group(1).strip())
    return keys


def _fm_referenced_papers(raw) -> set[str]:
    """Bare wikilink targets from a parsed `referenced_papers:` frontmatter value.

    PyYAML reads an unquoted `- [[cat/stem]]` entry as a nested list
    (`[[['cat/stem']]]`), while a quoted `- "[[cat/stem]]"` stays a string.
    Walk both shapes and collect the `cat/stem` targets.
    """
    out: set[str] = set()

    def walk(v):
        if isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            for m in WIKILINK_RE.finditer(v):
                out.add(m.group(1).strip())
            s = v.strip().strip("[]").strip()
            if "/" in s:
                out.add(s)

    walk(raw)
    return out


def _outlinks(body: str, known_keys: set[str]) -> set[str]:
    """Page→page outlinks parsed from page body, restricted to real wiki keys."""
    prose = strip_non_prose(body)
    out: set[str] = set()
    for m in WIKILINK_RE.finditer(prose):
        target = m.group(1).strip()
        if "/" in target and target in known_keys:
            out.add(target)
        elif "/" not in target:
            # Bare stem — match any category
            for k in known_keys:
                if k.split("/", 1)[1] == target:
                    out.add(k)
                    break
    return out


# ---------- graph build ----------


def _build_edges(
    paper_pages: list[Page],
    embeddings: np.ndarray,
    embed_keys: list[str],
) -> list[Edge]:
    """Compute all edges above per-signal thresholds.

    Returns a flat list of `Edge` objects — one per pair (a < b) with at
    least one signal active.
    """
    paper_keys = {p.key for p in paper_pages}
    by_key = {p.key: p for p in paper_pages}

    # Wikilink adjacency (asymmetric outlinks → undirected pair edges).
    out_links: dict[str, set[str]] = {}
    for p in paper_pages:
        out_links[p.key] = _outlinks(p.body, paper_keys)

    # Keyword sets per paper.
    keywords: dict[str, set[str]] = {p.key: _paper_keywords(p) for p in paper_pages}

    # Map embedding rows back to paper keys for indexed lookup.
    key_to_row = {k: i for i, k in enumerate(embed_keys)}

    # Pairwise cosine for the subset of paper rows. Embeddings are L2-normalized,
    # so cos(i,j) = arr[i] @ arr[j].
    paper_idx = [key_to_row[p.key] for p in paper_pages if p.key in key_to_row]
    paper_idx_to_key = {key_to_row[p.key]: p.key for p in paper_pages if p.key in key_to_row}
    if paper_idx:
        sub_arr = embeddings[paper_idx]
        sims = sub_arr @ sub_arr.T          # (N, N)
    else:
        sims = np.zeros((0, 0), dtype=np.float32)

    edges: list[Edge] = []
    keys_sorted = sorted(by_key.keys())
    sub_idx_to_key = {i: paper_idx_to_key[paper_idx[i]] for i in range(len(paper_idx))}
    sub_key_to_subidx = {v: k for k, v in sub_idx_to_key.items()}

    for i in range(len(keys_sorted)):
        a = keys_sorted[i]
        for j in range(i + 1, len(keys_sorted)):
            b = keys_sorted[j]
            edge = Edge(a=a, b=b)

            # Wikilink: undirected — add weight if either direction links.
            if b in out_links.get(a, set()) or a in out_links.get(b, set()):
                edge.wikilink = WIKILINK_WEIGHT

            # Semantic: cosine similarity from the sub-matrix.
            if a in sub_key_to_subidx and b in sub_key_to_subidx:
                cos = float(sims[sub_key_to_subidx[a], sub_key_to_subidx[b]])
                if cos >= SEMANTIC_THRESHOLD:
                    edge.semantic = max(0.0, min(1.0, cos))

            # Keyword Jaccard.
            ka, kb = keywords.get(a, set()), keywords.get(b, set())
            if ka and kb:
                inter = len(ka & kb)
                union = len(ka | kb)
                if union:
                    j_score = inter / union
                    if j_score >= KEYWORD_THRESHOLD:
                        edge.keyword = j_score

            if edge.total > 0:
                edges.append(edge)
    return edges


# ---------- coverage ----------


def _coverage(cluster: set[str], synthesis_refs: set[str]) -> float:
    """Fraction of cluster members cited by a synthesis page."""
    if not cluster:
        return 0.0
    return len(cluster & synthesis_refs) / len(cluster)


def _check_synthesis_coverage(
    cluster: list[str],
    syntheses: list[Page],
) -> tuple[str | None, float, set[str]]:
    """Find the existing synthesis with highest coverage of `cluster`.

    Returns (synthesis_key, coverage, refs_set) where refs_set is the
    referenced-papers set of the best-matching synthesis (used downstream
    to compute which cluster members are *missing* from it for the
    "extend" verdict). Returns (None, 0.0, set()) if no synthesis exists.
    """
    cluster_set = set(cluster)
    # Claim citations deliberately use the durable bare-stem form
    # ``[[stem#claim_slug]]``, while detector clusters use ``category/stem``
    # page keys. Canonicalize references against this cluster before taking
    # the intersection; otherwise a synthesis made entirely of claim-level
    # citations appears to cover none of its papers.
    cluster_keys_by_stem: dict[str, set[str]] = {}
    for key in cluster_set:
        cluster_keys_by_stem.setdefault(key.rsplit("/", 1)[-1], set()).add(key)

    best_key, best_cov, best_refs = None, 0.0, set()
    for s in syntheses:
        refs = _parse_referenced_papers(s.body)
        # The body is authoritative for synthesis citations. Retain the
        # frontmatter read only for older pages created before that field was
        # removed, so migration does not suddenly surface duplicate proposals.
        refs |= _fm_referenced_papers(s.fm.get("referenced_papers"))
        canonical_refs: set[str] = set()
        for ref in refs:
            if "/" in ref:
                canonical_refs.add(ref)
            else:
                canonical_refs.update(cluster_keys_by_stem.get(ref, set()))
        cov = _coverage(cluster_set, canonical_refs)
        if cov > best_cov:
            best_key, best_cov, best_refs = s.key, cov, canonical_refs
    return best_key, best_cov, best_refs


# ---------- common keywords (cluster naming hint, no LLM) ----------


def _common_keywords(cluster: list[str], paper_pages: list[Page]) -> list[str]:
    """Keywords appearing in ≥2 cluster members, ranked by frequency.

    Used in v0 instead of an LLM-named topic — the user picks a real name.
    """
    by_key = {p.key: p for p in paper_pages}
    counts: Counter = Counter()
    for k in cluster:
        p = by_key.get(k)
        if p is None:
            continue
        for kw in _paper_keywords(p):
            counts[kw] += 1
    # Keep only keywords present in ≥2 members; rank by frequency desc, then
    # alphabetically. The explicit (-count, keyword) sort is deterministic —
    # Counter.most_common() breaks count ties by insertion order, which here
    # comes from hash-randomized set iteration, so the derived slug varied
    # across runs (PYTHONHASHSEED).
    return [kw for kw, n in sorted(counts.items(), key=lambda kn: (-kn[1], kn[0])) if n >= 2]


# ---------- main detection ----------


def find_candidates(
    *,
    min_cluster: int = DEFAULT_MIN_CLUSTER,
    edge_threshold: float = EDGE_THRESHOLD,
    covered_threshold: float = DEFAULT_COVERED,
    extend_threshold: float = DEFAULT_EXTEND,
    judge: bool = False,
) -> tuple[list[Candidate], dict]:
    """Top-level entry point.

    Returns (candidates, stats). Each candidate carries a `verdict` of
    "new" or "extend" depending on how much existing synthesis coverage
    overlaps the cluster. Clusters with overlap ≥ `covered_threshold`
    are dropped before becoming Candidates (counted in stats but not
    surfaced).
    """
    pages = read_pages()
    paper_pages = [
        p for p in pages
        if p.fm.get("type", "paper") == "paper"
        and p.path.parent.name not in ("synthesis", "references", "concepts")
    ]
    syntheses = [p for p in pages if p.fm.get("type", "") == "synthesis"]

    loaded = pages_semantic.load_index()
    if loaded is None:
        return [], {"error": "semantic index not built — run `researchwiki reindex`"}
    embeddings, rows = loaded
    embed_keys = [r["key"] for r in rows]

    edges = _build_edges(paper_pages, embeddings, embed_keys)
    clusters = louvain(
        [p.key for p in paper_pages], edges, edge_threshold,
    )

    candidates: list[Candidate] = []
    n_covered = 0
    n_extend = 0
    n_new = 0
    for cluster in clusters:
        if len(cluster) < min_cluster:
            continue
        nearest, cov, nearest_refs = _check_synthesis_coverage(cluster, syntheses)
        if cov >= covered_threshold:
            n_covered += 1
            continue

        verdict = "extend" if cov >= extend_threshold else "new"
        if verdict == "extend":
            n_extend += 1
        else:
            n_new += 1

        # For "extend", which cluster members are NOT yet cited by the nearest
        # synthesis? That's the diff the user needs.
        missing = sorted(set(cluster) - nearest_refs) if verdict == "extend" else []

        cluster_edges = [e for e in edges
                         if e.a in cluster and e.b in cluster and e.total >= edge_threshold]
        density = (sum(e.total for e in cluster_edges) / len(cluster_edges)
                   if cluster_edges else 0.0)
        signal_counts: Counter = Counter()
        for e in cluster_edges:
            for s in e.signals:
                tag = s.split("(", 1)[0]
                signal_counts[tag] += 1
        titles = {p.key: p.fm.get("title", "") for p in paper_pages if p.key in cluster}
        candidates.append(Candidate(
            members=sorted(cluster),
            titles=titles,
            density=density,
            edges=cluster_edges,
            edge_signal_counts=dict(signal_counts),
            common_keywords=_common_keywords(cluster, paper_pages),
            nearest_synthesis=nearest,
            nearest_synthesis_overlap=cov,
            verdict=verdict,
            members_missing_from_nearest=missing,
        ))

    if judge:
        # Lazy import — keeps a --no-judge run free of the agents.llm import graph.
        from .judge import judge_candidate

        for c in candidates:
            judge_candidate(c, syntheses, paper_pages)

    stats = {
        "n_papers": len(paper_pages),
        "n_syntheses": len(syntheses),
        "n_edges_above_threshold": sum(1 for e in edges if e.total >= edge_threshold),
        "n_clusters_found": sum(1 for c in clusters if len(c) >= min_cluster),
        "n_already_covered": n_covered,
        "n_extend": n_extend,
        "n_new": n_new,
        "n_candidates": len(candidates),
        "n_judged": sum(1 for c in candidates if c.judged),
        "n_judge_batches": sum(c.judge_batches for c in candidates),
        "judge_input_tokens": sum(c.judge_input_tokens for c in candidates),
        "judge_output_tokens": sum(c.judge_output_tokens for c in candidates),
    }
    return candidates, stats
