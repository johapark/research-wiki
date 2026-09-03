"""Cross-paper contradiction lint — find claim pairs across papers that disagree.

Pattern from Starling (jones-2026-self-driving-datasets-from-20): when a curated
biomedical database is suspect, find molecules where ≥3 independent extractions
disagree with the database label, then audit those — and they consistently
discover the curated label is wrong (5.5–19.7% real error rates on Bioavailability_Ma,
BBB_Martins, LD50_Zhu, B3DB).

Adapted to a wiki at this scale: pair every claim with every other claim that
sits at high embedding cosine across papers, then ask an LLM judge whether the
pair *agrees*, *disagrees on a number*, *disagrees on direction* (negation), or is
*about a different topic* (false positive). Surface only the disagreement verdicts.

Why opt-in. The LLM judge is the most expensive lint surface by far, so
`--cross-paper` keeps it out of the default lint path.

What the pool actually costs, measured at 12.4k graded claims (the earlier
"typically under 100 pairs" note was written against a ~6,900-claim corpus and no
longer holds): **1,106 cross-paper claim pairs at 0.85**, spanning 643 paper
pairs — ~3,500 pairs if the floor drops to 0.83. Per-paper the distribution is
median 4 / p90 17 / max 44.

That distribution is worth reading before lowering `max_pairs`'s implicit ceiling
into a sweep. Only 19 of 305 papers exceed the ingest alert's `max_pairs=20`, so
at most ~171 of the 1,106 pairs ever escaped that cap: the great majority of this
pool has already been judged across the corpus's ingests, for a single
disagreement. Treat 0.85 as a largely mined seam, not an untapped one.

Note also what this judge is *for*. It keeps only `disagree_numeric` and
`disagree_direction`, and routes anything with a different cohort/dataset/run to
`different_topic`. That finds **errors** — one of two papers must be wrong. It
does not find **arguments**: two papers taking incompatible methodological
positions score `different_topic` here, correctly by this judge's own definition.
Anything wanting the latter needs its own verdict vocabulary, and at a lower
cosine band (methodological disagreements sit well below 0.85).

Function contract:
  find_cross_paper_contradictions(
      *,
      sim_threshold: float = 0.85,
      max_pairs: int = 50,
      judge_fn: callable | None = None,
      db_conn=None,
      only_stem: str | None = None,
      stats: dict | None = None,
      rejudge: bool = False,
  ) -> list[dict]

  Returns [{verdict, rationale, similarity, pair: [{claim_slug, paper_stem,
  section, position, text}, ...]}, ...] with only the disagree_numeric /
  disagree_direction verdicts kept.

  Every verdict — including the clears — is recorded in
  `cross_paper_judgements`, so a second run judges only what the first never
  reached. `rejudge=True` overrides. `stats`, when passed, is populated in place
  with pool/judged/skipped/disagreement counts.

  `judge_fn` (optional) lets tests stub the LLM call. Default uses
  researchwiki.agents.llm.call() with phase='cross_paper_judge'. When the LLM
  cannot be reached (no API key configured / stub mode), returns []
  with a single-line stderr note rather than failing the whole lint run.
"""

from __future__ import annotations

from typing import Callable

from ...errors import EnvironmentFailure
from ...log import log

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    np = None
    _NUMPY_AVAILABLE = False


# Rows per similarity block. Bounds peak memory at _BLOCK x N instead of N x N.
# Measured over this corpus's 12.4k claims: the full-matrix formulation peaks at
# 611 MB, the blocked corpus sweep at 62 MB, and the `only_stem` path — the one
# that runs on every ingest via `alert_after_ingest` — at 13 MB. The residual 62 MB
# is the float32 block plus its boolean temporaries, so it scales with this
# constant if a tighter ceiling is ever wanted. Kept at 512 to match
# `tasks/claim_discover._BLOCK`, which fixed the identical cliff.
_BLOCK = 512


def _pair_slugs(a: dict, b: dict) -> tuple[str, str] | None:
    """Both claims' slugs, or None when either is missing.

    Read straight off the claim dicts — the corpus query already selects
    `claim_slug`, so the old re-query by (stem, section, position) was resolving
    a value it had been handed. A null slug means the claim predates slug
    assignment; such a pair is judged but not recorded, since the slug is the
    only durable key either store has.
    """
    sa, sb = a.get("claim_slug"), b.get("claim_slug")
    if not sa or not sb:
        return None
    return sa, sb


def _canonical_endpoints(a: dict, b: dict, slug_a: str, slug_b: str):
    """Order a pair deterministically: ((stem, slug), (stem, slug)) sorted.

    The `contradicts` edge and the judgement row both key on this, so they can
    never disagree about which endpoint is `src` — and a symmetric relation can
    not be stored twice under opposite orientations.
    """
    return tuple(sorted([(a["paper_stem"], slug_a), (b["paper_stem"], slug_b)]))


def _judged_pairs(conn) -> set[tuple[str, str]]:
    """Canonical (src_slug, tgt_slug) keys the judge has already ruled on.

    Threshold-independent on purpose: a verdict is a statement about the pair,
    not about the floor that surfaced it, so a pair judged under a looser floor
    needs no re-judging under a tighter one. `sim_threshold` is recorded for
    analysis, not for invalidation.
    """
    try:
        return {
            (r["src_slug"], r["tgt_slug"])
            for r in conn.execute(
                "SELECT src_slug, tgt_slug FROM cross_paper_judgements"
            )
        }
    except Exception:
        return set()   # table absent on an un-migrated DB -> judge everything


def _record_judgement(
    conn, a: dict, b: dict, verdict: str, similarity: float, sim_threshold: float,
) -> None:
    """Record that this pair was judged, whatever the verdict.

    The clears are the point: `agree` and `different_topic` are the overwhelming
    majority and previously left no trace, so every run re-paid for every pair it
    had already dismissed. Best-effort — a bookkeeping failure must not lose the
    report the caller is building.
    """
    try:
        slugs = _pair_slugs(a, b)
        if slugs is None:
            return
        (src_stem, src_slug), (tgt_stem, tgt_slug) = _canonical_endpoints(a, b, *slugs)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO cross_paper_judgements "
                " (src_stem, src_slug, tgt_stem, tgt_slug, verdict, similarity, "
                "  sim_threshold, judged_at, judge_phase) "
                " VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%s','now'), ?)",
                (src_stem, src_slug, tgt_stem, tgt_slug, verdict,
                 float(similarity), float(sim_threshold), "cross_paper_judge"),
            )
    except Exception as e:
        log(f"judgement-record skipped: {type(e).__name__}: {e}", tag="cross_paper")


def _persist_contradicts_edge(
    conn, a: dict, b: dict, verdict: str, rationale: str, similarity: float,
) -> None:
    """Side-write a `contradicts` edge into the claim-graph cache.

    Skipped silently on any error — this is a side-effect of the lint report,
    not the report itself. The lint output is unchanged if persistence fails.

    A pair already recorded as `rejected` (human dismissed it) is not
    re-surfaced: `upsert_edge` has `skip_if_rejected=True` by default.
    """
    try:
        from ...claim_graph import Edge, SLUG_SCHEME_VERSION, open_edges_db, upsert_edge

        slugs = _pair_slugs(a, b)
        if slugs is None:
            return
        (src_stem, src_slug), (tgt_stem, tgt_slug) = _canonical_endpoints(a, b, *slugs)

        edge = Edge(
            src_stem=src_stem, src_slug=src_slug,
            tgt_stem=tgt_stem, tgt_slug=tgt_slug,
            relation="contradicts",
            directed=False,
            confidence=float(similarity),
            rationale=rationale[:400] if rationale else "",
            judge_phase="cross_paper_judge",
            judge_model="",
            slug_scheme_version=SLUG_SCHEME_VERSION,
            status="candidate",
        )
        edges = open_edges_db()
        try:
            upsert_edge(edges, edge)
            edges.commit()
        finally:
            edges.close()
    except Exception as e:
        log(f"contradicts-edge persistence skipped: {type(e).__name__}: {e}",
            tag="claim_graph")


_JUDGE_SYSTEM = """\
You are auditing a research wiki for cross-paper contradictions.

Two claims, A and B, were extracted from DIFFERENT wiki papers and embedded as
near-paraphrases. Decide whether they actually contradict — i.e., whether one
of them must be wrong — or whether the embedding similarity is incidental.

Verdicts:
  agree              - both claims describe the same fact and are mutually consistent
  disagree_numeric   - SAME fact, SAME experiment/cohort/model/run, different number
                       (e.g., both papers claim the headline accuracy of method X on
                       benchmark Y, but report different values — at least one is wrong)
  disagree_direction - SAME fact, opposite framing (e.g., A says X reduces Y, B says
                       X increases Y, in contexts where this MUST be the same result)
  different_topic    - the embeddings clustered them but the claims describe different
                       things — including different trials/cohorts/models/runs that
                       could BOTH be true simultaneously

CRITICAL: "different trial / cohort / dataset / model run" → `different_topic`,
not `disagree_numeric`. Two clinical trials of the same therapy with different
median engraftment times are NOT contradicting — they are independent measurements,
both can be correct. Only flag `disagree_numeric` when the SAME experiment is
being described and the numbers cannot both be right.

Output strict JSON: {"verdict": "...", "rationale": "one short sentence"}.
"""


_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "rationale"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["agree", "disagree_numeric", "disagree_direction", "different_topic"],
        },
        "rationale": {"type": "string"},
    },
}


def _format_pair_prompt(a: dict, b: dict) -> str:
    """Render the pair for the LLM judge, including supporting_text when present."""
    parts = []
    for label, c in (("A", a), ("B", b)):
        parts.append(f"Claim {label} — paper [[{c['paper_stem']}]] ({c['section']}#{c['position']}):")
        parts.append(f"  text: {c['text']}")
        if c.get("supporting_text"):
            parts.append(f"  supporting passage: {c['supporting_text']}")
        parts.append("")
    parts.append("Output JSON only.")
    return "\n".join(parts)


def _candidate_pairs(
    claims: list[dict], embs, sim_threshold: float, *, only_stem: str | None = None,
) -> list[tuple[int, int, float]]:
    """Cross-paper claim pairs above `sim_threshold`, similarity-descending.

    Blocked upper-triangle scan. The full N x N product is the obvious
    formulation and does not scale: at this corpus's 12.4k claims it is a 611 MB
    float32 allocation plus a 78-million-iteration Python loop, and
    `alert_after_ingest` pays both on every ingest. Per block the peak is bounded
    by _BLOCK x N and the comparison is vectorized. Same fix as
    `tasks/claim_discover.discover_pairs`.

    `only_stem` is pushed *into* the scan rather than filtered afterwards. Scoped
    to one paper, the pairs that survive are (that paper's claims) x (everything
    else) — roughly 25 x 12,361 comparisons instead of 78 million, for identical
    output. The upper-triangle constraint is dropped in that branch and is not
    needed: every row is the target paper and the different-stem mask excludes
    its own columns, so each cross-paper pair is emitted exactly once.

    Sorted by similarity descending because the caller slices `max_pairs` off the
    front — losing this order silently changes which pairs get judged.
    """
    n = len(claims)
    if n < 2 or not _NUMPY_AVAILABLE:
        return []
    stems = np.array([c["paper_stem"] for c in claims])
    # Compare stems as integer codes, not strings. The same-paper mask is a
    # (block x N) comparison — 6.3 M element pairs per block here — and numpy
    # string comparison at that width is measurably slower (0.54 s -> 0.41 s over
    # 12.4k claims). It does *not* move peak memory: that is dominated by the
    # float32 block itself plus the boolean temporaries, not by this mask.
    codes = np.unique(stems, return_inverse=True)[1].astype(np.int32, copy=False)

    if only_stem is None:
        row_idx = np.arange(n)
        upper_only = True
    else:
        row_idx = np.nonzero(stems == only_stem)[0]
        upper_only = False
        if not len(row_idx):
            return []

    cols = np.arange(n)
    ii_parts: list = []
    jj_parts: list = []
    sim_parts: list = []
    for start in range(0, len(row_idx), _BLOCK):
        rows_here = row_idx[start:start + _BLOCK]
        block = embs[rows_here] @ embs.T                      # (b, N)
        keep = block >= sim_threshold
        keep &= codes[rows_here][:, None] != codes[None, :]
        if upper_only:
            keep &= cols[None, :] > rows_here[:, None]
        bi, bj = np.where(keep)
        if len(bi):
            ii_parts.append(rows_here[bi])
            jj_parts.append(bj)
            sim_parts.append(block[bi, bj])

    if not ii_parts:
        return []
    ii = np.concatenate(ii_parts)
    jj = np.concatenate(jj_parts)
    sims = np.concatenate(sim_parts)
    pairs = [(int(i), int(j), float(s)) for i, j, s in zip(ii, jj, sims)]
    pairs.sort(key=lambda t: -t[2])
    return pairs


def _default_judge(prompt: str) -> dict | None:
    """Run the production LLM judge; return parsed JSON dict or None on failure."""
    from ...agents.judge import run_llm_judge
    return run_llm_judge(
        phase="cross_paper_judge", system=_JUDGE_SYSTEM, prompt=prompt, schema=_JUDGE_SCHEMA,
    )


def find_cross_paper_contradictions(
    *,
    sim_threshold: float = 0.85,
    max_pairs: int = 50,
    judge_fn: Callable[[str], dict | None] | None = None,
    db_conn=None,
    only_stem: str | None = None,
    stats: dict | None = None,
    rejudge: bool = False,
) -> list[dict]:
    """Surface contradictory claims across paper pairs.

    Pulls all gradable claims with non-null grader output, embeds them via the
    append-only claim cache, finds cross-paper pairs above sim_threshold, then
    asks the judge to classify each pair. Only `disagree_numeric` and
    `disagree_direction` verdicts are returned.

    `only_stem`: restrict to pairs where at least one endpoint is in that paper.
    Pushed into the scan, so the cost now scales with that paper's claims rather
    than the corpus.

    `stats`: optional dict, populated in place with `pool` / `judged` /
    `skipped_already_judged` / `disagreements` / `sim_threshold`. Filled before
    the `max_pairs` truncation and before every early return, so the pool size is
    readable even when nothing is judged — which makes `max_pairs=0` a zero-cost
    way to size the pool before paying for a sweep.

    `rejudge`: re-judge pairs already recorded in `cross_paper_judgements`.
    Off by default, so a second run costs only the pairs the first one never
    reached.

    Returns [] when:
      - numpy or the embedding model isn't available
      - the LLM judge isn't reachable
      - no candidate pair clears the similarity threshold
      - no judged pair came back as a disagreement
    """
    if stats is not None:
        stats.update({"pool": 0, "judged": 0, "skipped_already_judged": 0,
                      "disagreements": 0, "sim_threshold": sim_threshold,
                      "stopped_early": None})

    if not _NUMPY_AVAILABLE:
        log("numpy unavailable — skipping.", tag="cross_paper")
        return []

    # Pull eligible claims. We want graded, non-cross-ref claims. supporting_text
    # is included so the judge sees experimental context; claim_slug is what both
    # the edge and the judgement row key on. Keep the conn open across the judge
    # loop so persistence can reuse it.
    owns_conn = db_conn is None
    if owns_conn:
        from ...db.connection import get_connection
        conn = get_connection()
    else:
        conn = db_conn
    try:
        rows = conn.execute(
            """
            SELECT claim_slug, paper_stem, section, position, text, supporting_text
              FROM claims
             WHERE is_cross_ref = 0
               AND last_graded_at IS NOT NULL
            """
        ).fetchall()

        claims = [
            {
                "claim_slug": r["claim_slug"],
                "paper_stem": r["paper_stem"],
                "section": r["section"],
                "position": r["position"],
                "text": r["text"],
                "supporting_text": r["supporting_text"],
            }
            for r in rows
        ]
        if len(claims) < 2:
            return []

        # Append-only warm path, not `embed_texts`: that has no cache of its own,
        # so this function was re-encoding the whole corpus (~29 s at 12.4k
        # claims) on every call including every ingest. Not the cache-only reader
        # either — a newly promoted paper's claims are never cached, and this is
        # the one caller that exists to judge exactly those (E7).
        from ...index.claim_embeddings import warm_claim_embeddings
        embs = warm_claim_embeddings([
            {k: c[k] for k in ("paper_stem", "section", "position", "text")}
            for c in claims
        ])
        if embs is None or embs.size == 0:
            log("bi-encoder unavailable — skipping.", tag="cross_paper")
            return []

        pairs = _candidate_pairs(claims, embs, sim_threshold, only_stem=only_stem)
        if stats is not None:
            stats["pool"] = len(pairs)
        if not pairs:
            return []

        if not rejudge:
            already = _judged_pairs(conn)
            if already:
                kept = []
                for i, j, s in pairs:
                    slugs = _pair_slugs(claims[i], claims[j])
                    if slugs is not None:
                        (_, sa), (_, ta) = _canonical_endpoints(
                            claims[i], claims[j], *slugs)
                        if (sa, ta) in already:
                            continue
                    kept.append((i, j, s))
                if stats is not None:
                    stats["skipped_already_judged"] = len(pairs) - len(kept)
                pairs = kept
                if not pairs:
                    return []

        pairs = pairs[:max_pairs]
        if not pairs:
            return []

        judge = judge_fn or _default_judge
        out: list[dict] = []
        for i, j, sim in pairs:
            a, b = claims[i], claims[j]
            try:
                verdict_obj = judge(_format_pair_prompt(a, b))
            except EnvironmentFailure as exc:
                # House rule 3 (errors.py): stop the sweep, keep what it found.
                # Letting this unwind discarded the ~30 free local checks the
                # caller had already computed; tolerating it per pair would turn
                # one unanswered chat-relay prompt into one per remaining pair,
                # each costing the full RW_RELAY_TIMEOUT. Every verdict so far is
                # already committed by `_record_judgement`, so a re-run resumes
                # from here rather than re-paying.
                reason = f"{type(exc).__name__}: {exc}"
                if stats is not None:
                    stats["stopped_early"] = reason
                log(f"judging stopped after {stats['judged'] if stats else '?'} "
                    f"of {len(pairs)} pair(s) — {reason}", tag="cross_paper")
                break
            if not verdict_obj:
                continue
            verdict = verdict_obj.get("verdict")
            if stats is not None:
                stats["judged"] += 1
            # Record every verdict, including the clears — that is what makes a
            # second run cheap and the coverage question answerable at all.
            _record_judgement(conn, a, b, verdict, sim, sim_threshold)
            if verdict not in {"disagree_numeric", "disagree_direction"}:
                continue
            rationale = verdict_obj.get("rationale", "")
            # Persist as a `contradicts` edge in the claim-graph cache.
            # Side-write; the lint report itself is unchanged.
            _persist_contradicts_edge(conn, a, b, verdict, rationale, sim)
            out.append({
                "verdict": verdict,
                "rationale": rationale,
                "similarity": round(sim, 3),
                "pair": [
                    {k: a[k] for k in ("claim_slug", "paper_stem", "section", "position", "text")},
                    {k: b[k] for k in ("claim_slug", "paper_stem", "section", "position", "text")},
                ],
            })
        if stats is not None:
            stats["disagreements"] = len(out)
        return out
    finally:
        if owns_conn:
            conn.close()
