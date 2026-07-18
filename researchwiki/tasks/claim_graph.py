"""Query and review the claim-edge graph.

The claim-graph cache lives at `.claim-graph/edges.db` — LLM-judged edges
between individual claims that survive `db rebuild` because they key on
content-addressed slugs. This CLI is the read + review surface over that
cache; the *write* side is driven by the judges themselves (cross-paper
contradiction lint side-writes `contradicts` edges; the concepts detector
side-writes `instantiates` edges).

Modes:
  claim-graph                                 (default: summary counts by relation)
  claim-graph --contradicting "<text|slug>"   find edges touching a claim
  claim-graph --neighbors <stem#slug>         all edges touching a specific claim
  claim-graph --tensions                      unresolved contradictions, grouped by hub
  claim-graph --stem <stem>                   all edges for a paper stem
  claim-graph --relation contradicts          filter by relation
  claim-graph --stale                         list edges reconcile() marked stale
  claim-graph --json                          machine output
  claim-graph reconcile                       re-sync against state.db
  claim-graph review                          walk `candidate` edges y/n/s
  claim-graph promote                         draft Tensions/Evidence bullets for `confirmed` edges
  claim-graph promote --apply                 walk drafts and apply the surviving ones

Exit codes: 0 = success (zero results still 0 for read-only modes). 1 = user
error. 2 = environment error (edges.db missing or state.db unreachable when
reconcile requires it).
"""

from __future__ import annotations

import argparse
import json
import sys

from ..claim_graph import (
    counts_by_relation, open_edges_db, query, reconcile, set_status,
)
from ..claim_graph.promote import apply_promotions, propose_promotions


def _print_edge(edge, *, verbose: bool = False) -> None:
    """One-line human-readable edge summary."""
    tag = f"[{edge.status}]"
    rel = edge.relation
    directed_arrow = "→" if edge.directed else "↔"
    src = f"[[{edge.src_stem}#{edge.src_slug}]]"
    tgt = f"[[{edge.tgt_stem}#{edge.tgt_slug}]]"
    print(f"  {tag:<12} {rel:<14} {src} {directed_arrow} {tgt}")
    if verbose and edge.rationale:
        print(f"               → {edge.rationale}")


def _resolve_slug_from_text_fragment(fragment: str) -> tuple[str, str] | None:
    """Best-effort resolver: `<text>` → (stem, slug) by substring match on claim text.

    Returns the first hit; None if nothing matches or state.db is unreachable.
    """
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT paper_stem, claim_slug FROM claims "
            " WHERE claim_slug IS NOT NULL AND text LIKE ? LIMIT 1",
            (f"%{fragment}%",),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row["claim_slug"]:
        return None
    return row["paper_stem"], row["claim_slug"]


def _run_reconcile(as_json: bool) -> int:
    """Post-`db rebuild` re-sync. Marks edges stale when endpoints or slug-
    scheme version drift."""
    try:
        from ..db.connection import get_connection
    except Exception as e:
        print(f"claim-graph reconcile: state.db unreachable — {e}", file=sys.stderr)
        return 2
    edges = open_edges_db()
    state = get_connection()
    try:
        stats = reconcile(edges, state)
    finally:
        edges.close()
        state.close()

    if as_json:
        print(json.dumps({
            "scanned": stats.scanned,
            "marked_stale_missing": stats.marked_stale_missing,
            "marked_stale_version": stats.marked_stale_version,
            "active": stats.active,
            "by_relation": stats.by_relation,
        }, indent=2))
    else:
        print(f"claim-graph reconcile:")
        print(f"  scanned:              {stats.scanned}")
        print(f"  marked stale (miss):  {stats.marked_stale_missing}")
        print(f"  marked stale (ver):   {stats.marked_stale_version}")
        print(f"  active (candidate+):  {stats.active}")
        for rel, n in sorted(stats.by_relation.items(), key=lambda kv: -kv[1]):
            print(f"    {rel:<16} {n}")
    return 0


def _run_review(argv: list[str]) -> int:
    """Interactive walker over `candidate` edges. y = confirm, n = reject,
    s = skip (leave candidate), q = quit."""
    parser = argparse.ArgumentParser(prog="researchwiki claim-graph review")
    parser.add_argument("--relation", default=None,
                        help="Restrict review to one relation (contradicts / corroborates / …).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of candidates walked in this session.")
    args = parser.parse_args(argv)

    edges = open_edges_db()
    try:
        rows = query(
            edges, relation=args.relation, status="candidate", limit=args.limit,
        )
        if not rows:
            print("no candidate edges to review.")
            return 0

        # Preload claim text for prompts — reads state.db in one shot.
        try:
            from ..db.connection import get_connection
            state = get_connection()
        except Exception:
            state = None

        confirmed = rejected = skipped = 0
        for i, edge in enumerate(rows, 1):
            print()
            print(f"--- review {i}/{len(rows)}  {edge.relation}  "
                  f"[[{edge.src_stem}#{edge.src_slug}]] ↔ "
                  f"[[{edge.tgt_stem}#{edge.tgt_slug}]] ---")
            if state is not None:
                for label, stem, slug in (("A", edge.src_stem, edge.src_slug),
                                          ("B", edge.tgt_stem, edge.tgt_slug)):
                    r = state.execute(
                        "SELECT text FROM claims "
                        " WHERE paper_stem=? AND claim_slug=?",
                        (stem, slug),
                    ).fetchone()
                    if r:
                        print(f"  {label}: {r['text']}")
            if edge.rationale:
                print(f"  judge: {edge.rationale}")
            print(f"  [y] confirm  [n] reject  [s] skip  [q] quit")
            try:
                choice = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if choice == "y":
                set_status(edges, edge.id, "confirmed")
                confirmed += 1
            elif choice == "n":
                set_status(edges, edge.id, "rejected")
                rejected += 1
            elif choice == "q":
                break
            else:
                skipped += 1
        edges.commit()
        if state is not None:
            state.close()
        print()
        print(f"session: {confirmed} confirmed, {rejected} rejected, {skipped} skipped")
        return 0
    finally:
        edges.close()


def _run_tensions(*, as_json: bool, verbose: bool = False) -> int:
    """List every unresolved contradiction, grouped by the concept hubs both
    endpoints instantiate.

    "Unresolved" means status in (candidate, confirmed). `promoted` edges are
    already reflected in canonical markdown and shouldn't re-surface here;
    `rejected` and `stale` shouldn't either. Grouping: for each contradicts
    edge, find the set of concept slugs that BOTH endpoints have an
    `instantiates` edge pointing to. That intersection is the hub set; edges
    outside any hub are grouped under `_no_hub_`.
    """
    edges = open_edges_db()
    try:
        contradictions = [
            e for e in query(edges, relation="contradicts")
            if e.status in ("candidate", "confirmed")
        ]
        # Build (stem, slug) → set(concept-slug) index from instantiates edges.
        instantiates = query(edges, relation="instantiates")
        inst_by_src: dict[tuple[str, str], set[str]] = {}
        for e in instantiates:
            if e.tgt_stem != "concepts":
                continue
            inst_by_src.setdefault((e.src_stem, e.src_slug), set()).add(e.tgt_slug)

        grouped: dict[str, list] = {}
        for e in contradictions:
            src_hubs = inst_by_src.get((e.src_stem, e.src_slug), set())
            tgt_hubs = inst_by_src.get((e.tgt_stem, e.tgt_slug), set())
            shared = src_hubs & tgt_hubs
            if not shared:
                grouped.setdefault("_no_hub_", []).append(e)
                continue
            for hub in shared:
                grouped.setdefault(hub, []).append(e)
    finally:
        edges.close()

    if as_json:
        print(json.dumps({
            hub: [e.as_json() for e in edge_list]
            for hub, edge_list in grouped.items()
        }, indent=2))
        return 0

    if not grouped:
        print("no unresolved contradictions.")
        return 0
    total = sum(len(v) for v in grouped.values())
    print(f"{total} unresolved contradiction(s), grouped by concept hub:")
    print()
    for hub, edge_list in sorted(grouped.items(),
                                  key=lambda kv: (kv[0] == "_no_hub_", -len(kv[1]))):
        header = f"— hub: {hub}" if hub != "_no_hub_" else "— (no concept hub in common)"
        print(f"{header}   [{len(edge_list)} edge(s)]")
        for e in edge_list:
            _print_edge(e, verbose=verbose)
        print()
    return 0


def _run_promote(argv: list[str]) -> int:
    """Draft or apply Tensions/Evidence bullets for confirmed edges."""
    parser = argparse.ArgumentParser(prog="researchwiki claim-graph promote")
    parser.add_argument("--apply", action="store_true",
                        help="Walk .ingest/*-claim-edges/ and apply every surviving "
                             "proposal to its target synthesis. Removed proposal files "
                             "are treated as rejected. Edges applied → status=promoted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Draft proposals but don't write them. Ignored with --apply.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    if args.apply:
        stats = apply_promotions(verbose=args.verbose)
        if args.as_json:
            print(json.dumps({
                "proposals_seen": stats.proposals_seen,
                "applied": stats.applied,
                "skipped_stale": stats.skipped_stale,
                "skipped_missing_edge": stats.skipped_missing_edge,
                "errors": stats.errors,
            }, indent=2))
            return 0 if not stats.errors else 1
        print(f"claim-graph promote --apply:")
        print(f"  proposals scanned:      {stats.proposals_seen}")
        print(f"  applied:                {stats.applied}")
        print(f"  skipped (stale edge):   {stats.skipped_stale}")
        print(f"  skipped (missing edge): {stats.skipped_missing_edge}")
        for err in stats.errors:
            print(f"  ! {err}")
        return 0 if not stats.errors else 1

    stats = propose_promotions(dry_run=args.dry_run)
    if args.as_json:
        print(json.dumps({
            "edges_scanned": stats.edges_scanned,
            "proposals_written": stats.proposals_written,
            "edges_no_target": stats.edges_no_target,
            "edges_promoted_already": stats.edges_promoted_already,
            "per_synthesis": stats.per_synthesis,
        }, indent=2))
        return 0
    verb = "would draft" if args.dry_run else "drafted"
    print(f"claim-graph promote:")
    print(f"  confirmed edges scanned: {stats.edges_scanned}")
    print(f"  {verb} proposals:        {stats.proposals_written}")
    if stats.edges_no_target:
        print(f"  skipped (no synthesis references both endpoints): "
              f"{stats.edges_no_target}")
    if stats.per_synthesis:
        print()
        for stem, n in sorted(stats.per_synthesis.items(), key=lambda kv: -kv[1]):
            print(f"    {stem:<50} {n} proposal(s)")
        print()
        print("Review each proposal under .ingest/*-claim-edges/, delete to reject, "
              "then `researchwiki claim-graph promote --apply` to write approved "
              "bullets into the synthesis pages.")
    return 0


def main(argv: list[str]) -> int:
    # Subcommand dispatch: `reconcile` / `review` / `promote` peel off first.
    if argv and argv[0] == "reconcile":
        parser = argparse.ArgumentParser(prog="researchwiki claim-graph reconcile")
        parser.add_argument("--json", dest="as_json", action="store_true")
        args = parser.parse_args(argv[1:])
        return _run_reconcile(args.as_json)
    if argv and argv[0] == "review":
        return _run_review(argv[1:])
    if argv and argv[0] == "promote":
        return _run_promote(argv[1:])

    parser = argparse.ArgumentParser(
        prog="researchwiki claim-graph",
        description="Query and review the claim-edge graph (LLM-judged, cache-resident).",
    )
    parser.add_argument("--contradicting", metavar="TEXT|SLUG",
                        help="List edges touching a claim (matched by claim_slug exactly, "
                             "or by claim-text substring — first hit wins).")
    parser.add_argument("--neighbors", metavar="STEM#SLUG",
                        help="Every edge (any relation) touching a specific claim. "
                             "Use `stem#slug` — the same form a citation uses.")
    parser.add_argument("--tensions", action="store_true",
                        help="All unresolved contradictions, grouped by concept hub "
                             "(via `instantiates` edges pointing at hub slugs).")
    parser.add_argument("--stem", metavar="STEM",
                        help="List edges touching any claim in this paper.")
    parser.add_argument("--relation", default=None,
                        help="Filter by relation (contradicts, corroborates, refines, …).")
    parser.add_argument("--status", default=None,
                        help="Filter by lifecycle status (candidate, confirmed, promoted, rejected, stale).")
    parser.add_argument("--stale", action="store_true",
                        help="Shorthand for --status stale.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also print each edge's judge rationale.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a JSON array instead of the prose listing.")
    args = parser.parse_args(argv)

    if args.stale and args.status:
        print("claim-graph: --stale and --status are mutually exclusive", file=sys.stderr)
        return 1
    status_filter = "stale" if args.stale else args.status

    # --tensions is a specialized report; runs first + returns before generic
    # query dispatch. Contradiction edges grouped by the concept hubs that both
    # endpoints instantiate.
    if args.tensions:
        return _run_tensions(as_json=args.as_json, verbose=args.verbose)

    # --neighbors: all edges (any relation) touching a specific claim.
    if args.neighbors:
        raw = args.neighbors.strip()
        if "#" not in raw:
            print(f"claim-graph --neighbors: expected 'stem#slug', got {raw!r}",
                  file=sys.stderr)
            return 1
        stem, _, slug = raw.partition("#")
        stem, slug = stem.strip().rsplit("/", 1)[-1], slug.strip()
        edges = open_edges_db()
        try:
            rows_a = query(edges, src_stem=stem, src_slug=slug,
                           status=status_filter, limit=args.limit)
            rows_b = query(edges, tgt_stem=stem, tgt_slug=slug,
                           status=status_filter, limit=args.limit)
            seen: set[int] = set()
            rows: list = []
            for e in rows_a + rows_b:
                if e.id in seen:
                    continue
                seen.add(e.id)
                rows.append(e)
        finally:
            edges.close()
        if args.as_json:
            print(json.dumps([e.as_json() for e in rows], indent=2))
            return 0
        if not rows:
            print(f"no edges touching [[{stem}#{slug}]].")
            return 0
        print(f"{len(rows)} edge(s) touching [[{stem}#{slug}]]:")
        for e in rows:
            _print_edge(e, verbose=args.verbose)
        return 0

    src_stem: str | None = None
    src_slug: str | None = None
    stem_filter: str | None = args.stem
    if args.contradicting:
        # Accept a slug directly, or resolve a text fragment via state.db.
        tok = args.contradicting.strip()
        if "-" in tok and not tok.startswith(" ") and " " not in tok:
            # Looks slug-shaped ("kc-9f3a2b1c"); try slug lookup first.
            pair = None
            try:
                from ..db.connection import get_connection
                state = get_connection()
                try:
                    r = state.execute(
                        "SELECT paper_stem, claim_slug FROM claims "
                        " WHERE claim_slug = ? LIMIT 1", (tok,),
                    ).fetchone()
                finally:
                    state.close()
                if r and r["claim_slug"]:
                    pair = (r["paper_stem"], r["claim_slug"])
            except Exception:
                pass
            if pair is None:
                pair = _resolve_slug_from_text_fragment(tok)
        else:
            pair = _resolve_slug_from_text_fragment(tok)
        if pair is None:
            print(f"claim-graph: could not resolve '{args.contradicting}' to a claim slug",
                  file=sys.stderr)
            return 1
        src_stem, src_slug = pair

    edges = open_edges_db()
    try:
        if not any([args.contradicting, args.stem, args.relation, status_filter]):
            # Default: summary counts by relation.
            if args.as_json:
                print(json.dumps(counts_by_relation(edges), indent=2))
                return 0
            counts = counts_by_relation(edges)
            if not counts:
                print("claim-graph cache is empty. Run `researchwiki lint --cross-paper` "
                      "to seed contradiction edges, or `researchwiki candidates concepts "
                      "--persist-edges` to seed instantiates edges.")
                return 0
            print("claim-graph — edges by relation:")
            for rel, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {rel:<16} {n}")
            print()
            print("`researchwiki claim-graph --relation contradicts` for details; "
                  "`review` walks candidates.")
            return 0

        # Contradicting: find edges where the resolved (stem, slug) matches either endpoint.
        if src_stem and src_slug:
            # An edge with the target claim at either src or tgt.
            rows_a = query(
                edges, src_stem=src_stem, src_slug=src_slug,
                relation=args.relation, status=status_filter, limit=args.limit,
            )
            rows_b = query(
                edges, tgt_stem=src_stem, tgt_slug=src_slug,
                relation=args.relation, status=status_filter, limit=args.limit,
            )
            # De-dup by edge id.
            seen: set[int] = set()
            rows = []
            for e in rows_a + rows_b:
                if e.id in seen:
                    continue
                seen.add(e.id)
                rows.append(e)
        else:
            rows = query(
                edges, stem=stem_filter,
                relation=args.relation, status=status_filter, limit=args.limit,
            )

        if args.as_json:
            print(json.dumps([e.as_json() for e in rows], indent=2))
            return 0

        if not rows:
            print("_no edges match._")
            return 0
        print(f"{len(rows)} edge(s):")
        for e in rows:
            _print_edge(e, verbose=args.verbose)
        return 0
    finally:
        edges.close()
