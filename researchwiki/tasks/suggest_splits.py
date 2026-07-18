"""Cluster wiki/other/ and propose new categories or reassignments.

Usage:
  researchwiki suggest-splits
  researchwiki suggest-splits --threshold 0.75

Walks `wiki/other/`, clusters the papers using the existing semantic-cache
embeddings (no fresh compute), and asks an LLM per-cluster whether the
cluster should become a new category, be reassigned to an existing one,
or stay in `other`.

Designed as the back end of the "abstain to `other`" loop:
  1. The per-paper LLM classifier abstains to `other` when uncertain.
  2. `researchwiki status` flags `other` when it crosses a threshold.
  3. `suggest-splits` runs over the bucket, clusters the papers, and
     surfaces split/reassign/stay decisions per cluster.
  4. The user reviews and applies via the printed migration steps.

Cost: one Sonnet call per cluster (~$0.003 each). Typical run: 1-3
clusters, so total ~$0.01.

Known limitation (single-target verdict): the per-cluster LLM judge can
only pick one category per cluster. When a cluster has 8 papers that
fit 3 different existing categories (a real failure mode at low
thresholds and large N), the judge picks the modal fit and the
rationale flags the others, but the migration steps mis-route those
papers. Mitigation: tune `--threshold` upward to break heterogeneous
clusters apart. Future work: a per-paper reassignment pass after the
cluster judge for clusters flagged as internally heterogeneous.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from ..categories import is_valid, write_stamp
from ..index import pages_semantic
from ..log import log
from ..paths import wiki_root

# Cosine threshold for connected-component clustering. A pair of papers
# joins the same cluster when their embedding cosine ≥ this value.
#
# Empirically tuned to 0.70 on a 10-paper diverse wiki/other/ test:
# at 0.62 (the original default, matching B2 evolution proposals), union-
# find collapsed all 10 papers into a single blob — bge-small-en cosines
# between any pair of "scientific paper summaries" are usually >0.6, so
# transitive bridges merge unrelated topics. 0.70 produced the right two
# clusters (biology-heavy + pure-AI/KG). At larger N this may need to go
# higher; expose `--threshold` for tuning per-run.
CLUSTER_COSINE_THRESHOLD = 0.70
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SAMPLES = 8         # papers shown to the LLM per cluster


def _load_other_papers_with_embeddings() -> tuple[np.ndarray, list[dict]] | None:
    """Filter the semantic page index to rows where category == 'other'.

    Returns (embeddings, rows) — rows are the meta dicts from pages_meta.json
    keyed `key`, `stem`, `category`, `page_type`, `title`, `content_hash`.
    Returns None if the index isn't built or has no `other` papers.
    """
    loaded = pages_semantic.load_index()
    if loaded is None:
        print("ERROR: semantic page index not built — run `researchwiki reindex` first.",
              file=sys.stderr)
        return None
    embs, rows = loaded
    keep_idx = [i for i, r in enumerate(rows) if r.get("category") == "other"]
    if not keep_idx:
        return None
    return embs[keep_idx], [rows[i] for i in keep_idx]


def _connected_components(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster papers via connected components on cosine ≥ threshold.

    Embeddings are pre-normalized in pages_semantic, so dot product == cosine.
    Union-find for the components — handles the small-N case (≤50 papers)
    cleanly without needing sklearn.
    """
    n = len(embeddings)
    sim = embeddings @ embeddings.T
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= MIN_CLUSTER_SIZE]


def _sample_cluster(cluster_idx: list[int], embeddings: np.ndarray) -> list[int]:
    """Pick representative papers from a cluster: centroid + closest to it."""
    if len(cluster_idx) <= MAX_CLUSTER_SAMPLES:
        return cluster_idx
    cluster_embs = embeddings[cluster_idx]
    centroid = cluster_embs.mean(axis=0)
    # cosine to centroid (re-normalize centroid)
    centroid_n = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = cluster_embs @ centroid_n
    order = np.argsort(-sims)[:MAX_CLUSTER_SAMPLES]
    return [cluster_idx[i] for i in order]


def _read_paper_excerpt(row: dict) -> str:
    """Pull a short excerpt from the wiki page (title + summary section)."""
    page_path = wiki_root() / "wiki" / row["category"] / f"{row['stem']}.md"
    try:
        text = page_path.read_text()
    except OSError:
        return row.get("title") or ""
    # Best-effort: extract the Summary section
    m = re.search(r"##\s*Summary\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
    summary = m.group(1).strip() if m else ""
    if len(summary) > 800:
        summary = summary[:800].rstrip() + "..."
    title = row.get("title") or ""
    return f"{title}\n{summary}".strip()


def _build_cluster_prompt(rows: list[dict]) -> str:
    parts = ["Cluster papers ({} total):".format(len(rows)), ""]
    for i, r in enumerate(rows, 1):
        excerpt = _read_paper_excerpt(r)
        parts.append(f"{i}. [{r['stem']}] {excerpt}")
        parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("Existing categories (slug — scope):")
    # Render the live category list (derived from the wiki/ tree) via a light import
    from ..search import build_category_rules
    parts.append(build_category_rules().strip())
    return "\n".join(parts)


# JSON Schema for the suggest-splits judge verdict. Honored by chat-relay;
# ignored by other providers. `verdict` is enum-constrained because
# downstream code matches against specific lowercase strings;
# `slug` and `scope` are required only for new_category/reassign verdicts,
# but the schema keeps them optional and the call site validates the
# dependency in code (see _print_migration).
_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict":   {"type": "string",
                      "enum": ["stay", "new_category", "reassign"]},
        "slug":      {"type": ["string", "null"]},
        "scope":     {"type": ["string", "null"]},
        "rationale": {"type": ["string", "null"]},
    },
}


def _call_judge(rows: list[dict]) -> dict | None:
    from ..agents import llm

    system = (wiki_root() / "prompts" / "suggest-splits-system.md").read_text()
    user = _build_cluster_prompt(rows)
    try:
        # Same max_tokens override as bootstrap-categories — the classifier
        # role's default 200 tokens is sized for per-paper classification,
        # not structured verdicts with multi-sentence rationale.
        resp = llm.call(phase="classifier", prompt=user, system=system,
                        max_tokens=1000, schema=_JUDGE_SCHEMA)
    except Exception as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
        return None
    text = resp.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    print(f"  unparsable LLM response:", file=sys.stderr)
    print(f"  {resp.text[:300]}", file=sys.stderr)
    return None


def _print_migration(verdict: dict, rows: list[dict]) -> None:
    """Print actionable migration steps for a single cluster's verdict."""
    v = verdict.get("verdict")
    slug = (verdict.get("slug") or "").strip().lower()
    rationale = verdict.get("rationale") or ""
    stems = [r["stem"] for r in rows]

    if v == "stay":
        print(f"  → STAY in `other`. {rationale}")
        return

    if v == "new_category":
        scope = verdict.get("scope") or "(scope not provided)"
        print(f"  → NEW CATEGORY: `{slug}`")
        print(f"     scope: {scope}")
        print(f"     rationale: {rationale}")
        print(f"     migration:")
        print(f"       1. Create the category: mkdir -p wiki/{slug}  # its existence makes it valid")
        print(f"       2. Move {len(stems)} page(s):")
        for s in stems:
            print(f"            mv wiki/other/{s}.md wiki/{slug}/{s}.md")
        print(f"            # also update YAML `category: [other]` → `category: [{slug}]`")
        print(f"       3. researchwiki lint --fix  # auto-rebuild back-links")
        print(f"       4. researchwiki db rebuild && researchwiki reindex")
        return

    if v == "reassign":
        if not is_valid(slug):
            print(f"  → REASSIGN proposed `{slug}` but no wiki/{slug}/ exists — treating as STAY.")
            print(f"     rationale: {rationale}")
            return
        print(f"  → REASSIGN to existing `{slug}`")
        print(f"     rationale: {rationale}")
        print(f"     migration:")
        print(f"       1. Move {len(stems)} page(s):")
        for s in stems:
            print(f"            mv wiki/other/{s}.md wiki/{slug}/{s}.md")
        print(f"            # also update YAML `category: [other]` → `category: [{slug}]`")
        print(f"       2. researchwiki lint --fix")
        print(f"       3. researchwiki db rebuild && researchwiki reindex")
        return

    print(f"  → unknown verdict `{v}` — skipping. Raw: {verdict}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki suggest-splits",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--threshold", type=float, default=CLUSTER_COSINE_THRESHOLD,
                        help=f"Cosine threshold for cluster membership (default: {CLUSTER_COSINE_THRESHOLD}).")
    args = parser.parse_args(argv)

    loaded = _load_other_papers_with_embeddings()
    if loaded is None:
        print("No papers in `wiki/other/` (or index not built). Nothing to suggest.")
        write_stamp()  # dismiss the status warning anyway
        return 0
    embs, rows = loaded
    print(f"Found {len(rows)} paper(s) in `wiki/other/`.", file=sys.stderr)

    clusters = _connected_components(embs, threshold=args.threshold)
    if not clusters:
        print("No clusters above threshold — papers in `other` are too dispersed "
              "to suggest splits. Try `--threshold 0.55` for a looser grouping.")
        write_stamp()
        return 0

    print(f"Found {len(clusters)} candidate cluster(s) at cosine ≥ {args.threshold}.", file=sys.stderr)
    print()

    n_actionable = 0
    for ci, cluster_idx in enumerate(clusters, 1):
        sample_idx = _sample_cluster(cluster_idx, embs)
        sample_rows = [rows[i] for i in sample_idx]
        full_rows = [rows[i] for i in cluster_idx]

        print(f"=== Cluster {ci} ({len(cluster_idx)} papers; showing {len(sample_idx)}) ===")
        for r in sample_rows[:5]:
            title = (r.get("title") or "")[:80]
            print(f"  - [{r['stem']}] {title}")
        if len(sample_idx) > 5:
            print(f"  ... and {len(sample_idx) - 5} more")
        print()

        verdict = _call_judge(sample_rows)
        if verdict is None:
            print(f"  (judge call failed — skipping)")
            print()
            continue

        if verdict.get("verdict") in ("new_category", "reassign"):
            n_actionable += 1
        _print_migration(verdict, full_rows)
        print()

    write_stamp()
    log(f"suggest-splits | {len(clusters)} cluster(s), {n_actionable} actionable",
        tag="suggest-splits")
    if n_actionable == 0:
        print("No actionable splits this run. `wiki/other/` will be re-flagged "
              "by `status` when it grows further.")
    return 0
