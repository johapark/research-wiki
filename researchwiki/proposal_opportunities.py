"""Find proposal opportunities already latent in the corpus. No model calls.

Proposal generation only runs when someone asks for it, so a user who has
never heard of `proposals generate` never gets a proposal. The corpus, though,
already records the shapes a proposal wants — a judged contradiction between
two papers, a cluster of papers no synthesis page covers, a claim pair bridging
two categories. This module ranks those signals and turns each into the exact
`researchwiki proposals generate …` command that would explore it, so `status`,
ingest and the chat agent can offer one without spending anything.

Detection is deliberately conservative. Every source is an existing local
scan (the claim-graph edge cache, `synthesis_candidates.find_candidates`,
`claim_discover.discover_pairs`); nothing here embeds, judges or writes a wiki
page. A topic an existing proposal already explores is dropped, so an accepted
or rejected proposal stops the same opportunity resurfacing.

Ranking, strongest first:

1. **Tension** — a live `contradicts` claim-graph edge. An LLM judge already
   found the disagreement, which is the rarest and most proposal-shaped signal.
2. **Build-on** — a live `builds_on` / `refines` edge between two papers.
3. **Uncovered cluster** — a paper cluster with no synthesis page nearby,
   bounded in size: a 70-paper cluster is a field, not a question, and would
   overflow the eight-paper evidence packet.
4. **Bridge** — a cross-category claim pair sharing rare terms.

Advisory throughout: any failure yields fewer opportunities, never an error,
because this feeds `status` and ingest, which must not fail on account of it.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Kinds in rank order. `rank` in the JSON is the index here.
KINDS = ("tension", "build-on", "uncovered-cluster", "bridge")

#: Cluster sizes worth a proposal. Below the floor there is no cross-paper
#: question; above the ceiling the topic is a field survey, and the evidence
#: packet (eight papers) could not represent it.
CLUSTER_MIN = 3
CLUSTER_MAX = 40

#: At most this many papers are pinned with `--papers`; retrieval fills the rest.
MAX_PINNED_PAPERS = 4

#: Live claim-graph statuses. `stale` means the judged claim has since changed.
_LIVE_EDGE_STATUSES = ("candidate", "confirmed", "promoted")

#: Relations that make a proposal worth offering, mapped to their kind.
_EDGE_KINDS = {"contradicts": "tension", "builds_on": "build-on", "refines": "build-on"}

#: Shared terms too generic to justify a bridge on their own.
_GENERIC_TERMS = frozenset("""
analysis approach approaches based benchmark benchmarks data dataset datasets
method methods model models performance results study studies using used
incorporating lifestyle review reviews framework frameworks tool tools
""".split())

NUDGE_DECAY_DAYS = 14
_STAMP_FILENAME = ".proposal-opportunity-stamp"


@dataclass
class Opportunity:
    kind: str
    topic: str
    why: str
    papers: list[str] = field(default_factory=list)
    target_category: str = ""
    cross_category: bool = False
    #: Papers the opportunity involves but the command cannot pin (the source
    #: side of a bridge). Used for de-duplication and the post-ingest hint.
    related: list[str] = field(default_factory=list)

    @property
    def involved(self) -> list[str]:
        return self.papers + [s for s in self.related if s not in self.papers]

    @property
    def rank(self) -> int:
        return KINDS.index(self.kind)

    def command(self) -> str:
        """The exact command that explores this opportunity (preview only)."""
        parts = ["researchwiki proposals generate", _shell_quote(self.topic)]
        if self.papers:
            parts.append("--papers " + " ".join(self.papers[:MAX_PINNED_PAPERS]))
        if self.cross_category and self.target_category:
            parts.append(f"--target-category {self.target_category} --cross-category")
        return " ".join(parts)

    def as_json(self) -> dict:
        out = asdict(self)
        out["papers"] = self.papers[:MAX_PINNED_PAPERS]
        out["rank"] = self.rank
        out["command"] = self.command()
        return out


def _shell_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _stem(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _readable(stem: str) -> str:
    """`smith-2024-a-paper-about-things` → `Smith 2024`."""
    m = re.match(r"^([a-z0-9-]+?)-(\d{4})[a-z]?-", stem)
    if not m:
        return stem
    return f"{m.group(1).replace('-', ' ').title()} {m.group(2)}"


# ---------- sources ----------


def _live_edges() -> list:
    try:
        from .claim_graph.edges import edges_db_path, open_edges_db, query
    except Exception:
        return []
    if not edges_db_path().exists():
        return []
    try:
        conn = open_edges_db()
    except Exception:
        return []
    try:
        return [edge for relation in _EDGE_KINDS for status in _LIVE_EDGE_STATUSES
                for edge in query(conn, relation=relation, status=status)
                if edge.src_stem != edge.tgt_stem]
    except Exception:
        return []
    finally:
        conn.close()


def _edge_opportunities() -> list[Opportunity]:
    """One opportunity per connected group of judged edges, not per edge.

    Edges chain: five pangenome papers refine one another, and one structural
    paper is built on by two engineering papers. Offering each edge separately
    fragments a single question into several near-duplicates and lets one
    signal flood the list, so edges are grouped by shared papers first.
    A group containing a contradiction is a tension; otherwise build-on.
    """
    edges = _live_edges()
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src_stem, set()).add(edge.tgt_stem)
        adjacency.setdefault(edge.tgt_stem, set()).add(edge.src_stem)
    groups: list[set[str]] = []
    seen: set[str] = set()
    for start in adjacency:
        if start in seen:
            continue
        group, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node not in group:
                group.add(node)
                stack.extend(adjacency[node] - group)
        seen |= group
        groups.append(group)

    out: list[Opportunity] = []
    for group in groups:
        members = [e for e in edges if e.src_stem in group]
        contradictions = [e for e in members if e.relation == "contradicts"]
        # Most-connected papers first, so pinned `--papers` are the hubs.
        degree = {stem: len(adjacency[stem]) for stem in group}
        papers = sorted(group, key=lambda s: (-degree[s], s))
        names = [_readable(s) for s in papers]
        if contradictions:
            a, b = _readable(contradictions[0].src_stem), _readable(contradictions[0].tgt_stem)
            topic = (f"{a} and {b} report conflicting results; is it a real "
                     "disagreement or a difference in scope, and what would settle it?")
            why = f"judged contradiction between {a} and {b}"
            kind = "tension"
        elif len(group) == 2:
            edge = members[0]
            newer, older = _readable(edge.src_stem), _readable(edge.tgt_stem)
            verb = "builds on" if edge.relation == "builds_on" else "refines"
            topic = (f"How does {newer} extend {older}, and what does the "
                     "extension still leave unsolved?")
            why = f"{newer} {verb} {older}"
            kind = "build-on"
        else:
            listed = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
            topic = (f"What line of work runs through {listed}, and where "
                     "does it stop?")
            why = f"{len(group)} papers linked by judged build-on/refine relations ({listed})"
            kind = "build-on"
        out.append(Opportunity(kind=kind, topic=topic, why=why, papers=papers))
    # Larger groups first within each kind: more evidence, broader question.
    out.sort(key=lambda o: (o.rank, -len(o.papers)))
    return out


def _cluster_opportunities() -> list[Opportunity]:
    try:
        from .synthesis_candidates.detect import find_candidates
        candidates, _stats = find_candidates()
    except Exception:
        return []
    out: list[Opportunity] = []
    for cand in candidates:
        if cand.verdict != "new" or not CLUSTER_MIN <= len(cand.members) <= CLUSTER_MAX:
            continue
        terms = [t for t in cand.common_keywords if t.lower() not in _GENERIC_TERMS][:3]
        if not terms:
            continue
        categories = sorted({m.split("/", 1)[0] for m in cand.members})
        topic = f"What connects the {len(cand.members)} papers on {', '.join(terms)}?"
        why = (f"{len(cand.members)} papers on {terms[0]} with no synthesis page"
               + (f", across {len(categories)} categories" if len(categories) > 1 else ""))
        out.append(Opportunity(kind="uncovered-cluster", topic=topic, why=why,
                               papers=[_stem(m) for m in cand.members]))
    out.sort(key=lambda o: -len(o.papers))
    return out


#: A shared term that appears in this share of all bridge pairs is corpus-wide
#: jargon (`fine-tuning`, `zero-shot`), not what makes one pair a bridge.
_UBIQUITOUS_SHARE = 0.05


def _is_named_method(term: str) -> bool:
    """A compound or numbered token: `needleman-wunsch`, `atac-seq`, `k-mer`.

    Rare shared vocabulary alone admitted pairs that merely share register
    (`lifestyle`, `translating`, `guidance`); requiring one term shaped like a
    method, assay or statistic keeps the pairs a transfer could start from.
    """
    return "-" in term.strip("-") or any(ch.isdigit() for ch in term)


def _bridge_opportunities(limit: int = 30) -> list[Opportunity]:
    try:
        from .tasks.claim_discover import discover_pairs
        pairs = discover_pairs(limit=max(limit * 4, 100), cross_category_only=True)
    except Exception:
        return []
    frequency: dict[str, int] = {}
    for pair in pairs:
        for term in set(pair.shared_terms):
            frequency[term] = frequency.get(term, 0) + 1
    ubiquitous = {t for t, n in frequency.items()
                  if n > max(2, _UBIQUITOUS_SHARE * len(pairs))}
    out: list[Opportunity] = []
    for pair in pairs:
        methods = [t for t in pair.shared_terms
                   if _is_named_method(t) and t not in ubiquitous
                   and t.lower() not in _GENERIC_TERMS]
        if not methods:
            continue
        a, b = _readable(pair.stem_a), _readable(pair.stem_b)
        topic = (f"Could {methods[0]}, as used in {a} ({pair.category_a}), "
                 f"address a problem in {b} ({pair.category_b})?")
        why = (f"{pair.category_a} and {pair.category_b} papers both use "
               f"{', '.join(methods[:3])}, with no link between them")
        # Cross-category mode pins only target-category papers (the CLI rejects
        # any other); the planner's search is what finds the source side, so
        # the source paper is named in the topic and left for retrieval.
        out.append(Opportunity(kind="bridge", topic=topic, why=why,
                               papers=[pair.stem_b],
                               target_category=pair.category_b, cross_category=True,
                               related=[pair.stem_a]))
        if len(out) >= limit:
            break
    return out


def _explored_paper_sets() -> list[frozenset[str]]:
    """Paper sets existing proposals already cover, from their evidence links."""
    try:
        from .proposals import _proposals_on_disk
        records = _proposals_on_disk()     # wiki/proposals/ only, not a full walk
    except Exception:
        return []
    link = re.compile(r"\[\[(?:[^\]#|]*/)?([^\]#|/]+)#")
    out = []
    for record in records:
        try:
            text = record.path.read_text(encoding="utf-8")
        except OSError:
            continue
        stems = frozenset(link.findall(text))
        if stems:
            out.append(stems)
    return out


def _already_explored(opp: Opportunity, explored: list[frozenset[str]]) -> bool:
    """True when an existing proposal cites at least two of this opportunity's
    pinned papers — the same cross-paper connection, already on the ledger."""
    pinned = set(opp.papers[:MAX_PINNED_PAPERS]) | set(opp.related)
    return any(len(pinned & stems) >= 2 for stems in explored)


#: Most opportunities of one kind in a single list, so a plentiful signal
#: (build-on edges accumulate with every claim-overlap run) cannot push the
#: rarer kinds out of view.
PER_KIND_CAP = 5


def find_opportunities(
    *, limit: int = 10, include_clusters: bool = True, include_bridges: bool = True,
    per_kind: int = PER_KIND_CAP,
) -> list[Opportunity]:
    """Ranked opportunities, strongest kind first. Never raises.

    Cost by source, on a ~500-paper corpus: judged edges ~10 ms (one SQLite
    read), clusters ~0.7 s (graph build and Louvain over every paper), bridges
    ~2 s (a blocked matmul over every claim pair). Surfaces with a latency
    budget — `status`, ingest — use the edges alone; the dedicated
    `proposals opportunities` command pays for all three.
    """
    found = _edge_opportunities()
    if include_clusters:
        found += _cluster_opportunities()
    if include_bridges:
        found += _bridge_opportunities()
    explored = _explored_paper_sets()
    found = [opp for opp in found if not _already_explored(opp, explored)]
    found.sort(key=lambda o: o.rank)     # stable: keeps each source's own order
    kept: list[Opportunity] = []
    taken = dict.fromkeys(KINDS, 0)
    for opp in found:
        if taken[opp.kind] < per_kind:
            kept.append(opp)
            taken[opp.kind] += 1
    return kept[:limit]


def opportunities_for_paper(stem: str, *, limit: int = 3) -> list[Opportunity]:
    """Opportunities involving one paper — for the post-ingest hint.

    Judged edges only. They are the signal a fresh paper actually changes (its
    claim-overlap run writes them), and they cost one SQLite read; the cluster
    and pair scans would add seconds to every ingest.
    """
    return [opp for opp in find_opportunities(limit=50, include_clusters=False,
                                              include_bridges=False, per_kind=50)
            if stem in opp.involved][:limit]


# ---------- the `status` nudge ----------


def _stamp_path() -> Path:
    from .paths import wiki_root
    return wiki_root() / _STAMP_FILENAME


def _stamp_age_days() -> float | None:
    path = _stamp_path()
    if not path.exists():
        return None
    try:
        return (time.time() - int(path.read_text(encoding="utf-8").strip())) / 86400.0
    except (OSError, ValueError):
        return None


def _write_stamp() -> None:
    try:
        _stamp_path().write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def opportunity_warning(*, touch: bool = True) -> str | None:
    """The `status` line, or None when nothing is worth interrupting for.

    Decay-stamped like the other `status` nudges: once shown, it stays quiet
    for `NUDGE_DECAY_DAYS`. The stamp is checked first because the scan behind
    it is the slowest thing `status` would otherwise run on every invocation.
    `touch=False` peeks without starting the quiet period.
    """
    age = _stamp_age_days()
    if age is not None and age < NUDGE_DECAY_DAYS:
        return None
    # Judged edges only (~10 ms). The cluster and pair scans took `status`
    # from ~2.5 s to ~6 s; they stay in `proposals opportunities`, which the
    # line points to. Edges are also the strongest signal, so the top entry
    # is the same one the full ranking would lead with.
    opps = find_opportunities(limit=200, per_kind=200,
                              include_clusters=False, include_bridges=False)
    if not opps:
        return None
    if touch:
        _write_stamp()
    counts = {kind: sum(o.kind == kind for o in opps) for kind in KINDS}
    summary = ", ".join(f"{n} {kind}" for kind, n in counts.items() if n)
    top = opps[0]
    return (
        f"Proposal opportunities: {summary} (from judged claim relations)\n"
        f"  top: {top.why}\n"
        f"  → {top.command()}\n"
        f"  → researchwiki proposals opportunities   (adds clusters and cross-category bridges)"
    )
