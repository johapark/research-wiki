"""Build the wiki's link graph and render it as a self-contained HTML page.

Two edge kinds, and the second is the reason this exists rather than being a
generic wikilink viewer:

  - **wikilink** — a `[[target]]` in a page body. Structure the author wrote.
  - **claim** — a typed edge from `.claim-graph/edges.db` (`builds_on`,
    `refines`, `corroborates`, `measures_same`, `contradicts`, `instantiates`).
    Judged relations between *individual claims*, which no amount of reading the
    markdown would surface.

`contradicts` is the payload. `claim-graph --tensions` already lists unresolved
contradictions, but a list cannot show you that four of them cluster on one
paper, or that a synthesis page cites both sides of one. That is a picture.

Rendering is a string substitution into `templates/graph.html` — the template is
self-contained (no CDN, no build step) so the output is one file that opens from
`file://` and survives being emailed. Layout runs client-side; this module only
decides what is in the graph.

Read-only: no LLM call, no network, and the claim DB is opened only if it already
exists (`open_edges_db` would otherwise create an empty one as a side effect of
a *view* command).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .paths import claim_graph_dir, wiki_dir
from .wiki import Page, read_pages

# Statuses worth drawing: a retracted or rejected relation must not be drawn
# with the same weight as a live one, and `stale` dominates by volume (one corpus
# holds 13,535 stale `instantiates` against ~50 live edges).
#
# Volume alone is NOT the reason, though — the page-level collapse below absorbs
# it, and on that same corpus widening to `stale` adds only ~20 visible edges.
# The reason is semantic: `stale` means the claim it was judged against changed,
# so the edge is an assertion nobody currently stands behind.
LIVE_CLAIM_STATUSES: tuple[str, ...] = ("candidate", "confirmed", "promoted")

# `instantiates` edges are written by the concepts detector with this literal in
# `tgt_stem` and the concept slug in `tgt_slug`, so the target page is
# `concepts/<tgt_slug>` rather than a paper stem. Resolving it as a stem finds
# nothing and silently drops every concept spoke.
_CONCEPT_STEM_SENTINEL = "concepts"


@dataclass
class Graph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "meta": self.meta}


def _label_for(page: Page) -> str:
    """Short display label. `short_name` is the field written for exactly this
    job; fall back to the stem's author-year prefix rather than the full stem,
    which runs to 60+ characters and cannot be drawn beside a 6px dot."""
    short = page.str_field("short_name").strip()
    if short:
        return short
    parts = page.stem.split("-")
    if len(parts) >= 2 and parts[1][:4].isdigit():
        return f"{parts[0].title()} {parts[1]}"
    return page.stem.replace("-", " ")


def _is_content_page(page: Page) -> bool:
    """Exclude the bookkeeping pages at `wiki/` root.

    `index.md` links to every page in the corpus, so including it makes one node
    adjacent to all others — the layout collapses into a star and every real
    cluster disappears. Both meta files also lack frontmatter today, so
    `read_pages` already drops them; this is the guard for the day one gains a
    YAML block.
    """
    if page.path.parent.name == wiki_dir().name:
        return False
    return page.stem not in {"index", "log", "suggested-additions"}


def _collect_nodes(pages: list[Page]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for p in pages:
        nodes[p.key] = {
            "id": p.key,
            "label": _label_for(p),
            "type": p.page_type,
            "category": p.category,
            "title": p.str_field("title") or p.stem,
            "hook": p.str_field("hook"),
            "year": p.year_int(),
            "venue": p.str_field("venue"),
            "stem": p.stem,
            "deg": 0,
        }
    return nodes


def _wikilink_edges(
    pages: list[Page], nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One edge per distinct (source, target) wikilink pair.

    Link resolution is delegated to the lint walker so this graph and the
    `broken_wikilinks` check can never disagree about what a link points at —
    including the bare-stem form (`[[smith-2024-...]]` with no category), which
    Obsidian resolves and a naive parser drops.
    """
    from .tasks.lint.walk import extract_links

    known = set(nodes)
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for p in pages:
        for tgt in sorted(extract_links(p.body, known)):
            if tgt == p.key or (p.key, tgt) in seen:
                continue
            seen.add((p.key, tgt))
            edges.append({"source": p.key, "target": tgt, "kind": "wikilink"})
    return edges


def _resolve_claim_endpoint(
    stem: str, slug: str, by_stem: dict[str, str], known: set[str]
) -> str | None:
    """Map a claim-edge endpoint onto a page key, or None if it has no page."""
    if stem == _CONCEPT_STEM_SENTINEL:
        key = f"concepts/{slug}"
        return key if key in known else None
    return by_stem.get(stem)


def _claim_edges(
    nodes: dict[str, dict[str, Any]], statuses: tuple[str, ...]
) -> tuple[list[dict[str, Any]], int]:
    """Typed claim edges, collapsed to page level.

    Returns `(edges, dropped)` where `dropped` counts edges whose endpoints have
    no page — a removed paper leaves its edges behind until the next
    `claim-graph reconcile`, and silently omitting them would misreport the
    graph as complete.

    Several claim pairs commonly join the same two *pages*; they collapse into
    one edge per (source, target, relation) carrying `n` and the strongest
    confidence, because 30 parallel strands between two dots is noise.
    """
    db = claim_graph_dir() / "edges.db"
    if not db.exists():
        return [], 0

    from .claim_graph.edges import open_edges_db, query as query_edges

    by_stem = {n["stem"]: n["id"] for n in nodes.values()}
    known = set(nodes)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    dropped = 0

    conn = open_edges_db(db)
    try:
        for status in statuses:
            for e in query_edges(conn, status=status):
                src = _resolve_claim_endpoint(e.src_stem, e.src_slug, by_stem, known)
                tgt = _resolve_claim_endpoint(e.tgt_stem, e.tgt_slug, by_stem, known)
                if src is None or tgt is None or src == tgt:
                    dropped += 1
                    continue
                key = (src, tgt, e.relation)
                cur = grouped.get(key)
                conf = e.confidence if e.confidence is not None else 0.0
                if cur is None:
                    grouped[key] = {
                        "source": src,
                        "target": tgt,
                        "kind": "claim",
                        "relation": e.relation,
                        "status": e.status,
                        "directed": bool(e.directed),
                        "confidence": round(conf, 3) if conf else None,
                        "rationale": (e.rationale or "")[:240],
                        "n": 1,
                    }
                else:
                    cur["n"] += 1
                    if conf and (cur["confidence"] is None or conf > cur["confidence"]):
                        cur["confidence"] = round(conf, 3)
                        cur["rationale"] = (e.rationale or "")[:240]
                    # A confirmed edge outranks a candidate for display.
                    if e.status == "confirmed" or cur["status"] == "confirmed":
                        cur["status"] = "confirmed"
    finally:
        conn.close()

    return list(grouped.values()), dropped


def build_graph(
    *,
    claim_statuses: tuple[str, ...] = LIVE_CLAIM_STATUSES,
    include_claims: bool = True,
) -> Graph:
    """Assemble the graph from the current `wiki/` state. No network, no LLM."""
    pages = [p for p in read_pages() if _is_content_page(p)]
    nodes = _collect_nodes(pages)

    edges = _wikilink_edges(pages, nodes)
    dropped = 0
    if include_claims:
        claim, dropped = _claim_edges(nodes, claim_statuses)
        edges.extend(claim)

    for e in edges:
        nodes[e["source"]]["deg"] += 1
        nodes[e["target"]]["deg"] += 1

    type_counts: dict[str, int] = {}
    for n in nodes.values():
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    relation_counts: dict[str, int] = {}
    for e in edges:
        if e["kind"] == "claim":
            relation_counts[e["relation"]] = relation_counts.get(e["relation"], 0) + 1

    return Graph(
        nodes=sorted(nodes.values(), key=lambda n: n["id"]),
        edges=edges,
        meta={
            "n_nodes": len(nodes),
            "n_wikilinks": sum(1 for e in edges if e["kind"] == "wikilink"),
            "n_claim_edges": sum(1 for e in edges if e["kind"] == "claim"),
            "claim_edges_dropped": dropped,
            "claim_statuses": list(claim_statuses) if include_claims else [],
            "type_counts": dict(sorted(type_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "categories": sorted({n["category"] for n in nodes.values()}),
        },
    )


def render_html(graph: Graph) -> str:
    """Inject the graph as JSON into the self-contained HTML template.

    `</` is escaped because the payload lands inside a `<script>` block, where a
    `</script>` occurring inside a string literal — a page title or a judge
    rationale could contain one — would terminate the block and blank the page.
    `ensure_ascii=False` keeps diacritics in author names readable in the source.
    """
    template = (
        resources.files("researchwiki")
        .joinpath("templates/graph.html")
        .read_text(encoding="utf-8")
    )
    payload = json.dumps(graph.as_json(), ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__GRAPH_DATA__", payload)


def write_html(graph: Graph, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(graph), encoding="utf-8")
    return out
