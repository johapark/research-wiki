"""Smoke tests for the paper-graph clustering layer.

Written before extracting `Edge` / `louvain` out of
`tasks/synthesis_candidates.py` into `researchwiki/graph.py`, to lock the
observable behavior across the move: the modularity algorithm's community
splits, `Edge` weight aggregation, and the Page→graph adapter (`_build_edges`)
that constructs the edges the clusterer consumes.

The import block below is the ONLY thing that changes across the refactor —
`Edge` and `louvain` move to `researchwiki.graph`; `_build_edges` stays in the
task module as the Page-coupled adapter. The assertions are the behavior
contract and stay identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from researchwiki.index.graph import Edge, louvain
from researchwiki.synthesis_candidates.detect import _build_edges
from researchwiki.wiki import Page


def _page(stem: str, *, category: str = "cat", keywords=None, body: str = "") -> Page:
    return Page(
        path=Path(f"{category}/{stem}.md"),
        stem=stem,
        category=category,
        fm={"keywords": list(keywords or []), "title": stem.upper()},
        body=body,
    )


# ---------- Edge ----------

def test_edge_total_sums_signals():
    e = Edge(a="cat/a", b="cat/b", wikilink=1.0, semantic=0.7, keyword=0.3)
    assert e.total == 2.0


def test_edge_signals_labels():
    e = Edge(a="cat/a", b="cat/b", wikilink=1.0, semantic=0.72)
    assert "wikilink" in e.signals
    assert "semantic(0.72)" in e.signals
    assert not any(s.startswith("keyword") for s in e.signals)


# ---------- louvain ----------

def test_louvain_splits_two_triangles_joined_by_weak_bridge():
    """The property Louvain buys over connected-components: a single weak hub
    edge between two dense triangles must NOT merge them into one blob."""
    nodes = ["A", "B", "C", "D", "E", "F"]
    strong = 3.0
    edges = [
        Edge(a="A", b="B", keyword=strong),
        Edge(a="B", b="C", keyword=strong),
        Edge(a="A", b="C", keyword=strong),
        Edge(a="D", b="E", keyword=strong),
        Edge(a="E", b="F", keyword=strong),
        Edge(a="D", b="F", keyword=strong),
        Edge(a="C", b="D", keyword=1.0),  # weak bridge, exactly at threshold
    ]
    communities = louvain(nodes, edges, threshold=1.0)
    as_sets = sorted((sorted(c) for c in communities), key=lambda c: c[0])
    assert as_sets == [["A", "B", "C"], ["D", "E", "F"]]


def test_louvain_single_blob_when_fully_connected():
    nodes = ["A", "B", "C", "D"]
    edges = [
        Edge(a=x, b=y, keyword=2.0)
        for i, x in enumerate(nodes)
        for y in nodes[i + 1:]
    ]
    communities = louvain(nodes, edges, threshold=1.0)
    assert len(communities) == 1
    assert sorted(communities[0]) == nodes


def test_louvain_isolated_nodes_are_singletons():
    communities = louvain(["A", "B", "C"], [], threshold=1.0)
    assert sorted(sorted(c) for c in communities) == [["A"], ["B"], ["C"]]


def test_louvain_ignores_subthreshold_edges():
    edges = [Edge(a="A", b="B", keyword=0.5)]  # total 0.5 < threshold 1.0
    communities = louvain(["A", "B"], edges, threshold=1.0)
    assert sorted(sorted(c) for c in communities) == [["A"], ["B"]]


# ---------- _build_edges adapter + louvain end-to-end ----------

def test_build_edges_wikilink_plus_keyword_then_cluster():
    """Two papers linked by a wikilink AND sharing keywords form one edge whose
    weights add; an unrelated paper stays isolated. Semantic signal is disabled
    by passing an empty embedding table."""
    pages = [
        _page("a", keywords=["k1", "k2"], body="See [[cat/b]] for the method."),
        _page("b", keywords=["k1", "k2"]),
        _page("c", keywords=["unrelated"]),
    ]
    empty = np.zeros((0, 0), dtype=np.float32)
    edges = _build_edges(pages, empty, embed_keys=[])

    ab = [e for e in edges if {e.a, e.b} == {"cat/a", "cat/b"}]
    assert len(ab) == 1
    e = ab[0]
    assert e.wikilink == 1.0          # a → b link
    assert e.keyword == 1.0           # identical keyword sets → Jaccard 1.0
    assert e.semantic == 0.0          # disabled
    assert e.total == 2.0
    # cat/c shares nothing → no edge touches it
    assert not any("cat/c" in (e.a, e.b) for e in edges)

    communities = louvain([p.key for p in pages], edges, threshold=1.0)
    as_sets = sorted((sorted(c) for c in communities), key=lambda c: c[0])
    assert as_sets == [["cat/a", "cat/b"], ["cat/c"]]
