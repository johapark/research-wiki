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

Why opt-in. The LLM judge is the most expensive lint surface by far. At
sim_threshold=0.85 across ~6,900 claims, the candidate pool is small (typically
under 100 pairs); but each judged pair costs one LLM call and we can't yet
estimate signal/noise without running it. `--cross-paper` keeps the cost out of
the default lint path until calibration shows the rule is worth running on every
invocation.

Function contract:
  find_cross_paper_contradictions(
      *,
      sim_threshold: float = 0.85,
      max_pairs: int = 50,
      judge_fn: callable | None = None,
  ) -> list[dict]

  Returns [{verdict, rationale, pair: [{claim_slug, paper_stem, text, supporting_text}, ...]}, ...]
  with only the disagree_numeric / disagree_direction verdicts kept.

  `judge_fn` (optional) lets tests stub the LLM call. Default uses
  researchwiki.agents.llm.call() with phase='cross_paper_judge'. When the LLM
  cannot be reached (no API key configured / stub mode), returns []
  with a single-line stderr note rather than failing the whole lint run.
"""

from __future__ import annotations

from typing import Callable
from ...log import log

try:
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


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

        # Resolve slugs from state.db — we have (paper_stem, section, position)
        # in the pair dicts and need `claim_slug` for the edge key.
        row_a = conn.execute(
            "SELECT claim_slug FROM claims "
            " WHERE paper_stem=? AND section=? AND position=?",
            (a["paper_stem"], a["section"], a["position"]),
        ).fetchone()
        row_b = conn.execute(
            "SELECT claim_slug FROM claims "
            " WHERE paper_stem=? AND section=? AND position=?",
            (b["paper_stem"], b["section"], b["position"]),
        ).fetchone()
        if row_a is None or row_b is None:
            return
        if not row_a["claim_slug"] or not row_b["claim_slug"]:
            return

        # Canonical ordering: (stem, slug) sorted so a symmetric edge doesn't
        # get inserted twice under different (src, tgt) orientations.
        endpoints = sorted([
            (a["paper_stem"], row_a["claim_slug"]),
            (b["paper_stem"], row_b["claim_slug"]),
        ])
        (src_stem, src_slug), (tgt_stem, tgt_slug) = endpoints

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


def _candidate_pairs(claims: list[dict], embs, sim_threshold: float) -> list[tuple[int, int, float]]:
    """Pairwise cosine across claims from different papers, sorted by similarity desc.

    `embs` is shape (N, dim) L2-normalized — cosine is dot product. We compute
    the upper triangle once and filter on (i) different paper_stem, (ii) cosine
    above threshold. Returns [(i, j, sim), ...] in similarity-descending order.
    """
    n = len(claims)
    if n < 2:
        return []
    sims = embs @ embs.T  # (N, N)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if claims[i]["paper_stem"] == claims[j]["paper_stem"]:
                continue
            s = float(sims[i, j])
            if s >= sim_threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda p: -p[2])
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
) -> list[dict]:
    """Surface contradictory claims across paper pairs.

    Pulls all gradable claims with non-null grader output, embeds them, finds
    cross-paper pairs above sim_threshold, then asks the judge to classify each
    pair. Only `disagree_numeric` and `disagree_direction` verdicts are returned.

    `only_stem`: restrict output to pairs where at least one endpoint is in
    that paper. The corpus-wide embedding is still computed (we need it to
    find neighbors of the new-paper claims), but the pair filter is cheap and
    the judge cost scales with the returned pairs, not with the corpus.

    Returns [] when:
      - numpy or the embedding model isn't available
      - the LLM judge isn't reachable
      - no candidate pair clears the similarity threshold
      - no judged pair came back as a disagreement
    """
    if not _NUMPY_AVAILABLE:
        log("numpy unavailable — skipping.", tag="cross_paper")
        return []

    from ...index.embeddings import embed_texts, is_available as _embed_available
    if not _embed_available():
        log("bi-encoder unavailable — skipping.", tag="cross_paper")
        return []

    # Pull eligible claims. We want graded, non-cross-ref claims. supporting_text
    # is included so the judge sees experimental context. Keep the conn open
    # across the judge loop so the edge-persistence side-write can resolve
    # slugs without reopening.
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

        embs = embed_texts([c["text"] for c in claims])
        if embs is None or embs.size == 0:
            return []

        pairs = _candidate_pairs(claims, embs, sim_threshold)
        if only_stem is not None:
            pairs = [
                (i, j, s) for i, j, s in pairs
                if claims[i]["paper_stem"] == only_stem
                or claims[j]["paper_stem"] == only_stem
            ]
        pairs = pairs[:max_pairs]
        if not pairs:
            return []

        judge = judge_fn or _default_judge
        out: list[dict] = []
        for i, j, sim in pairs:
            a, b = claims[i], claims[j]
            verdict_obj = judge(_format_pair_prompt(a, b))
            if not verdict_obj:
                continue
            verdict = verdict_obj.get("verdict")
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
        return out
    finally:
        if owns_conn:
            conn.close()
