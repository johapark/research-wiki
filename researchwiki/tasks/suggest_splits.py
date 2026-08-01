"""Cluster wiki/other/ — or a populated category — and propose taxonomy changes.

Usage:
  researchwiki suggest-splits                    # cluster wiki/other/ (default)
  researchwiki suggest-splits --category cgt     # find sub-clusters that should split OUT of `cgt`
  researchwiki suggest-splits --all              # scan every populated category for divergence
  researchwiki suggest-splits --threshold 0.75

Two modes share one clustering + LLM-judge engine:

* **other-bucket mode (default)** — the back end of the "abstain to `other`"
  loop:
    1. The per-paper LLM classifier abstains to `other` when uncertain.
    2. `researchwiki status` flags `other` when it crosses a threshold.
    3. This clusters the bucket and asks, per cluster, new_category / reassign
       / stay.
    4. The user reviews and applies via the printed migration steps.

* **within-category mode (`--category X` / `--all`)** — the *opposite* signal:
  a paper can be confidently classified into category X yet belong to a
  distinct sub-cluster that has grown large enough to deserve its own sibling
  category. This clusters the papers inside a populated category, isolates any
  sub-cluster that separates from the category's core, and asks — per
  sub-cluster — split_out / stay. `researchwiki status` surfaces a nudge
  (cluster-verified, decay-stamped) when such a sub-cluster exists. Splitting a
  populated category breaks back-links, so the judge is biased hard toward
  `stay`.

Cost: one Sonnet call per cluster (~$0.003 each). The status warning that
triggers within-category mode is structural-only (no LLM), so it's free.

Known limitation (single-target verdict): the per-cluster LLM judge can only
pick one category per cluster. When a cluster has 8 papers that fit 3 different
existing categories (a real failure mode at low thresholds and large N), the
judge picks the modal fit and the rationale flags the others, but the migration
steps mis-route those papers. Mitigation: tune `--threshold` upward to break
heterogeneous clusters apart. Within-category detection has a parallel caveat:
if a category fragments into many small components with no dominant core, every
non-largest component is offered as a candidate and the (stay-biased) judge is
the backstop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from ..categories import content_categories, is_valid, write_divergence_stamp, write_stamp
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

# Within-category divergence tunables.
CATEGORY_MIN_PAPERS = 12        # don't probe a category smaller than this in --all / the status warning
MIN_SPLIT_CLUSTER_SIZE = 3      # a split-out candidate needs ≥3 papers (matches the judge prompt's bar)

# Higher default than the `other`-bucket threshold: within a category the papers
# already share a topic, so their pairwise cosines run higher across the board.
# At 0.70 an already-cohesive category collapses into one blob and nothing ever
# separates; 0.80 is where genuine sub-structure starts to break out on the
# reference corpus. `--threshold` overrides per-run.
CATEGORY_DIVERGENCE_COSINE_THRESHOLD = 0.80


def _load_papers_with_embeddings(
    category: str, *, papers_only: bool = False
) -> tuple[np.ndarray, list[dict]] | None:
    """Filter the semantic page index to rows where category == `category`.

    Returns (embeddings, rows) — rows are the meta dicts from pages_meta.json
    keyed `key`, `stem`, `category`, `page_type`, `title`, `content_hash`.
    With `papers_only`, also drops non-paper page types (synthesis/idea/etc.
    that a content category should never hold, but guard anyway).
    Returns None if the index isn't built or has no matching rows.
    """
    loaded = pages_semantic.load_index()
    if loaded is None:
        print("ERROR: semantic page index not built — run `researchwiki reindex` first.",
              file=sys.stderr)
        return None
    embs, rows = loaded
    keep_idx = [
        i for i, r in enumerate(rows)
        if r.get("category") == category
        and (not papers_only or r.get("page_type") == "paper")
    ]
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


def _build_cluster_prompt(rows: list[dict], source_category: str | None = None) -> str:
    parts = ["Cluster papers ({} total):".format(len(rows)), ""]
    for i, r in enumerate(rows, 1):
        excerpt = _read_paper_excerpt(r)
        parts.append(f"{i}. [{r['stem']}] {excerpt}")
        parts.append("")
    parts.append("---")
    parts.append("")
    if source_category:
        parts.append(
            f"These papers all currently live in the `{source_category}` category. "
            f"Decide whether this sub-cluster should split OUT of `{source_category}` "
            f"into a new sibling category, or stay."
        )
        parts.append("")
    parts.append("Existing categories (slug — scope):")
    # Render the live category list (derived from the wiki/ tree) via a light import
    from ..search import build_category_rules
    parts.append(build_category_rules().strip())
    return "\n".join(parts)


# JSON Schema for the suggest-splits judge verdict. Honored by chat-relay;
# ignored by other providers. `verdict` is enum-constrained because
# downstream code matches against specific lowercase strings;
# `slug` and `scope` are required only for actionable verdicts,
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

# Within-category variant: split a cohesive sub-cluster OUT of its parent
# category into a new sibling, or leave it.
_CATEGORY_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict":   {"type": "string",
                      "enum": ["stay", "split_out"]},
        "slug":      {"type": ["string", "null"]},
        "scope":     {"type": ["string", "null"]},
        "rationale": {"type": ["string", "null"]},
    },
}


def _call_judge(
    rows: list[dict],
    *,
    system_filename: str = "suggest-splits-system.md",
    schema: dict = _JUDGE_SCHEMA,
    source_category: str | None = None,
    judge_fn=None,
) -> dict | None:
    """Ask the LLM to judge a cluster. `judge_fn`, when provided, replaces the
    LLM call entirely (dependency-injection seam for tests) and receives the
    sampled rows."""
    if judge_fn is not None:
        return judge_fn(rows)

    from ..agents import llm

    system = (wiki_root() / "prompts" / system_filename).read_text()
    user = _build_cluster_prompt(rows, source_category=source_category)
    try:
        # Same max_tokens override as bootstrap-categories — the classifier
        # role's default 200 tokens is sized for per-paper classification,
        # not structured verdicts with multi-sentence rationale.
        resp = llm.call(phase="classifier", prompt=user, system=system,
                        max_tokens=1000, schema=schema)
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


# Verdicts that represent an actionable taxonomy change (vs `stay`).
_ACTIONABLE = ("new_category", "reassign", "split_out")


def _print_migration(verdict: dict, rows: list[dict], *, source_category: str = "other") -> None:
    """Print actionable migration steps for a single cluster's verdict.

    `source_category` is the category the papers currently live in — `other`
    in bucket mode, the parent category in within-category mode. It's templated
    into the `mv` source path and the `category:` YAML rewrite.
    """
    v = verdict.get("verdict")
    slug = (verdict.get("slug") or "").strip().lower()
    rationale = verdict.get("rationale") or ""
    stems = [r["stem"] for r in rows]

    if v == "stay":
        print(f"  → STAY in `{source_category}`. {rationale}")
        return

    if v in ("new_category", "split_out"):
        if not slug:
            print(f"  → {v.upper()} proposed but no slug given — treating as STAY. {rationale}")
            return
        if slug == source_category or is_valid(slug):
            print(f"  → {v.upper()} proposed `{slug}` but that category already exists — treating as STAY.")
            print(f"     rationale: {rationale}")
            return
        scope = verdict.get("scope") or "(scope not provided)"
        label = (f"SPLIT OUT of `{source_category}` → NEW category `{slug}`"
                 if v == "split_out" else f"NEW CATEGORY: `{slug}`")
        print(f"  → {label}")
        print(f"     scope: {scope}")
        print(f"     rationale: {rationale}")
        print(f"     migration:")
        print(f"       1. Create the category: mkdir -p wiki/{slug}  # its existence makes it valid")
        print(f"       2. Move {len(stems)} page(s):")
        for s in stems:
            print(f"            mv wiki/{source_category}/{s}.md wiki/{slug}/{s}.md")
        print(f"            # also update YAML `category: [{source_category}]` → `category: [{slug}]`")
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
            print(f"            mv wiki/{source_category}/{s}.md wiki/{slug}/{s}.md")
        print(f"            # also update YAML `category: [{source_category}]` → `category: [{slug}]`")
        print(f"       2. researchwiki lint --fix")
        print(f"       3. researchwiki db rebuild && researchwiki reindex")
        return

    print(f"  → unknown verdict `{v}` — skipping. Raw: {verdict}")


def _judge_clusters(
    embs: np.ndarray,
    rows: list[dict],
    clusters: list[list[int]],
    *,
    source_category: str,
    system_filename: str,
    schema: dict,
    judge_fn=None,
) -> int:
    """Run the LLM judge over each cluster and print its migration. Shared by
    both modes. Returns the count of actionable (non-`stay`) verdicts."""
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

        verdict = _call_judge(
            sample_rows,
            system_filename=system_filename,
            schema=schema,
            source_category=None if source_category == "other" else source_category,
            judge_fn=judge_fn,
        )
        if verdict is None:
            print(f"  (judge call failed — skipping)")
            print()
            continue

        if verdict.get("verdict") in _ACTIONABLE:
            n_actionable += 1
        _print_migration(verdict, full_rows, source_category=source_category)
        print()

    return n_actionable


def _split_candidates(embs: np.ndarray, threshold: float) -> tuple[list[int], list[list[int]]]:
    """Given a category's embeddings, return (core, candidate_subclusters).

    Connected-component clustering partitions the papers. The largest component
    is treated as the category's retained *core*; every other component of at
    least MIN_SPLIT_CLUSTER_SIZE is a split-out candidate. Returns ([], []) when
    the category is one cohesive blob (nothing separates) — that's the healthy,
    no-divergence case.
    """
    comps = _connected_components(embs, threshold)
    if len(comps) < 2:
        return [], []
    comps_sorted = sorted(comps, key=len, reverse=True)
    core = comps_sorted[0]
    cands = [c for c in comps_sorted[1:] if len(c) >= MIN_SPLIT_CLUSTER_SIZE]
    return core, cands


def detect_divergence_candidates(
    threshold: float = CATEGORY_DIVERGENCE_COSINE_THRESHOLD,
    *,
    min_papers: int = CATEGORY_MIN_PAPERS,
) -> list[dict]:
    """Structural (no-LLM) scan: which populated categories contain a
    sub-cluster that separates from their core? Returns a list of
    {category, n_total, subclusters: [[stem, ...], ...]}. Used by both the
    status warning (cheap, no LLM) and `--all`."""
    loaded = pages_semantic.load_index()
    if loaded is None:
        return []
    embs, rows = loaded
    out: list[dict] = []
    for cat in sorted(c for c in content_categories() if c != "other"):
        idx = [i for i, r in enumerate(rows)
               if r.get("category") == cat and r.get("page_type") == "paper"]
        if len(idx) < min_papers:
            continue
        sub_embs = embs[idx]
        _core, cands = _split_candidates(sub_embs, threshold)
        if cands:
            out.append({
                "category": cat,
                "n_total": len(idx),
                "subclusters": [[rows[idx[i]]["stem"] for i in c] for c in cands],
            })
    return out


def divergence_warning(*, touch: bool = True,
                       threshold: float = CATEGORY_DIVERGENCE_COSINE_THRESHOLD) -> str | None:
    """Return a status warning if any populated category has a diverging
    sub-cluster and the divergence stamp is absent/stale, else None. Structural
    only — never calls the LLM. Mirrors `categories.other_saturation_warning`."""
    from ..categories import (CATEGORY_DIVERGENCE_DECAY_DAYS,
                              divergence_stamp_age_days)
    age = divergence_stamp_age_days()
    if age is not None and age < CATEGORY_DIVERGENCE_DECAY_DAYS:
        return None
    cands = detect_divergence_candidates(threshold)
    if not cands:
        return None
    if touch:
        write_divergence_stamp()
    lines = [f"⚠ {len(cands)} categor{'y' if len(cands) == 1 else 'ies'} "
             f"may contain a diverging sub-cluster:"]
    for c in cands:
        sizes = ", ".join(str(len(s)) for s in c["subclusters"])
        lines.append(f"    {c['category']} ({c['n_total']} papers; candidate sub-cluster size(s): {sizes})")
        lines.append(f"      → researchwiki suggest-splits --category {c['category']}")
    return "\n".join(lines)


def _run_other_mode(args, judge_fn=None) -> int:
    threshold = args.threshold if args.threshold is not None else CLUSTER_COSINE_THRESHOLD
    loaded = _load_papers_with_embeddings("other")
    if loaded is None:
        print("No papers in `wiki/other/` (or index not built). Nothing to suggest.")
        write_stamp()  # dismiss the status warning anyway
        return 0
    embs, rows = loaded
    print(f"Found {len(rows)} paper(s) in `wiki/other/`.", file=sys.stderr)

    clusters = _connected_components(embs, threshold=threshold)
    if not clusters:
        print("No clusters above threshold — papers in `other` are too dispersed "
              "to suggest splits. Try `--threshold 0.55` for a looser grouping.")
        write_stamp()
        return 0

    print(f"Found {len(clusters)} candidate cluster(s) at cosine ≥ {threshold}.", file=sys.stderr)
    print()

    n_actionable = _judge_clusters(
        embs, rows, clusters,
        source_category="other",
        system_filename="suggest-splits-system.md",
        schema=_JUDGE_SCHEMA,
        judge_fn=judge_fn,
    )

    write_stamp()
    log(f"suggest-splits | {len(clusters)} cluster(s), {n_actionable} actionable",
        tag="suggest-splits")
    if n_actionable == 0:
        print("No actionable splits this run. `wiki/other/` will be re-flagged "
              "by `status` when it grows further.")
    return 0


def _run_category_mode(args, judge_fn=None) -> int:
    threshold = (args.threshold if args.threshold is not None
                 else CATEGORY_DIVERGENCE_COSINE_THRESHOLD)
    if args.category:
        cat = args.category.strip().lower()
        if cat == "other":
            print("`--category other` — use the default mode (no flag) for the `other` bucket.",
                  file=sys.stderr)
            return 1
        if not is_valid(cat):
            print(f"ERROR: `{cat}` is not an existing content category "
                  f"(wiki/{cat}/ does not exist).", file=sys.stderr)
            return 1
        cats = [cat]
    else:  # --all
        cats = sorted(c for c in content_categories() if c != "other")
        if not cats:
            print("No content categories yet. Nothing to scan.")
            write_divergence_stamp()
            return 0

    total_actionable = 0
    analyzed = 0
    for cat in cats:
        loaded = _load_papers_with_embeddings(cat, papers_only=True)
        if loaded is None:
            if args.category:
                print(f"No papers in `{cat}` (or index not built — run `researchwiki reindex`).")
            continue
        embs, rows = loaded
        # In --all, skip categories too small to bother probing. An explicit
        # --category still analyzes (the user asked for it), but says so.
        if not args.category and len(rows) < CATEGORY_MIN_PAPERS:
            continue

        core, cand_clusters = _split_candidates(embs, threshold)
        if not cand_clusters:
            if args.category:
                if not core:
                    print(f"`{cat}`: {len(rows)} papers form one cohesive cluster at "
                          f"cosine ≥ {threshold} — no divergence. Try a higher "
                          f"`--threshold` to probe finer structure.")
                else:
                    print(f"`{cat}`: no sub-cluster ≥{MIN_SPLIT_CLUSTER_SIZE} papers "
                          f"separates from the core ({len(core)} papers) — no split candidate.")
            continue

        analyzed += 1
        print(f"=== Category `{cat}` ({len(rows)} papers; core {len(core)}, "
              f"{len(cand_clusters)} candidate sub-cluster(s)) ===")
        total_actionable += _judge_clusters(
            embs, rows, cand_clusters,
            source_category=cat,
            system_filename="suggest-category-splits-system.md",
            schema=_CATEGORY_JUDGE_SCHEMA,
            judge_fn=judge_fn,
        )
        print()

    write_divergence_stamp()  # dismiss the status nudge for the decay window
    log(f"suggest-splits --{'category ' + args.category if args.category else 'all'} | "
        f"analyzed {analyzed}, {total_actionable} actionable", tag="suggest-splits")
    if total_actionable == 0:
        print("No actionable split-outs this run. Categories look cohesive.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki suggest-splits",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Cosine threshold for cluster membership. Default depends on mode: "
                             f"{CLUSTER_COSINE_THRESHOLD} for the `other` bucket, "
                             f"{CATEGORY_DIVERGENCE_COSINE_THRESHOLD} for within-category "
                             f"(--category/--all), since same-category papers cluster tighter.")
    parser.add_argument("--category", default=None, metavar="CAT",
                        help="Within-category mode: find sub-clusters that should split OUT of "
                             "the existing category CAT (split_out/stay) instead of clustering wiki/other/.")
    parser.add_argument("--all", action="store_true", dest="all_categories",
                        help="Within-category mode across every populated content category "
                             f"(≥{CATEGORY_MIN_PAPERS} papers).")
    args = parser.parse_args(argv)

    if args.category and args.all_categories:
        print("ERROR: pass either --category or --all, not both.", file=sys.stderr)
        return 1

    if args.category or args.all_categories:
        return _run_category_mode(args)
    return _run_other_mode(args)
