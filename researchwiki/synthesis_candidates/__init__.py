"""Detect emergent paper clusters not covered by an existing synthesis page.

The public entry point is `find_candidates` — it builds a weighted paper–paper
graph (wikilinks + semantic cosine + keyword Jaccard), runs Louvain clustering,
checks each cluster against existing syntheses' `referenced_papers:`, and
optionally invokes an LLM judge to score per-member scope-fit. Callers get
back a list of `Candidate` proposals + a stats dict.

The CLI wrapper lives at `researchwiki.tasks._synthesis_candidates` (hidden
from auto-discovery; invoked via `researchwiki candidates synthesis`); it
just parses flags, calls `find_candidates`, then hands each candidate to
`render_proposal` for the on-disk markdown.

Splitting rationale — the module was one 1090-LOC file that mixed
graph-clustering, LLM-judge, and markdown-rendering concerns. Each has a
different testing story and different reasons to change; separating them
makes the boundaries visible.
"""

from .detect import (
    Candidate,
    MemberVerdict,
    DEFAULT_MIN_CLUSTER,
    DEFAULT_COVERED,
    DEFAULT_EXTEND,
    find_candidates,
)
from .render import render_proposal

__all__ = [
    "Candidate",
    "MemberVerdict",
    "DEFAULT_MIN_CLUSTER",
    "DEFAULT_COVERED",
    "DEFAULT_EXTEND",
    "find_candidates",
    "render_proposal",
]
