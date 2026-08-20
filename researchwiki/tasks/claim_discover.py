"""Zero-token discovery pass over claim pairs the auto-link tier never sees.

`claim-overlap` judges claim pairs above cosine 0.83 and writes reciprocal
bullets on confirmed matches. That threshold is correct for *writing*: a wrong
bullet is a visible defect, so precision wins. It is wrong for *discovery*,
where a missed connection is invisible and permanent.

The obvious repair — lower the threshold — does not work. Measured on a
117-paper corpus (6,786 possible paper pairs):

    cosine >= 0.83     99 paper pairs   (1.5% of all possible)
    cosine >= 0.75  2,487 paper pairs   (37%)
    cosine >= 0.70  5,415 paper pairs   (80%)

At any threshold low enough to be interesting, most of the corpus is "related
to" most of the corpus. And the relation that motivated this module — Parks
2018 vs van Iterson 2017 disagreeing about whether a mixture model can serve as
an empirical null, the sharpest finding on `wiki/concepts/mixture-model.md` —
peaks at cosine **0.743**. Reaching it by threshold costs ~2,400 pairs.

So this module does not lower the threshold. It uses a cosine *band* as a coarse
filter and ranks within it by **IDF-weighted shared-term mass**: the summed
inverse-document-frequency of the content words two claims share. That puts the
Parks/van Iterson pair at rank 210 of 54,792 — top 0.4% — with genuinely related
pairs above it (two superpixel papers sharing "bsds"/"undersegmentation"; an LNP
paper and a CRISPR trial sharing "d-dimer"/"elevations").

Why it works: a 384-dimension embedding compresses away the rare, distinctive
vocabulary that marks two claims as being about the same *specific thing*. Two
claims sit at 0.73 because both are methods prose; two claims sharing "empirical
null" are about one subject. Cosine measures register, IDF overlap measures
subject. This is the hybrid the framework already trusts in `search` (BM25 fused
with semantic), applied to claim pairs instead of queries.

**Nothing here judges, writes, or costs tokens.** The output is a ranked review
queue. A pair worth acting on is confirmed by running `claim-overlap <stem>` on
it, which routes through the existing judged path unchanged.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..log import log

# Cosine band. The floor keeps the pair list to claims in a comparable register;
# the ceiling excludes what `claim-overlap` already handles at 0.83, so this is
# strictly the layer below it rather than a competing view of the same pairs.
DEFAULT_COS_LO = 0.72
DEFAULT_COS_HI = 0.83

# Ranked pairs returned. The band holds ~55k pairs on a 117-paper corpus; the
# signal is concentrated in the head, and a review queue nobody finishes is the
# same as no queue.
DEFAULT_LIMIT = 40

# Rows per similarity block. Bounds peak memory at _BLOCK × N floats instead of
# N × N, which is what keeps `status` viable as the corpus grows.
_BLOCK = 512

# Same reasoning as `semantic_members._NO_CLAIMS`: an empty substrate has to say
# so, or a migrated corpus reads its own absence as "no candidates found".
# Public so the CLI can print it instead of guessing at the cause.
NO_CLAIMS = (
    "no claims in the corpus — nothing to rank. Run `researchwiki db rebuild`; "
    "if it stays empty, check `researchwiki lint --json` -> zero_claim_papers "
    "(a migrated wiki whose H2 headings don't match the extractor produces no "
    "claims, which makes every discovery surface silently empty)"
)

# Content words only: 4+ letters, so "the"/"and"/"with" never carry IDF mass.
_TOKEN_RE = re.compile(r"[a-z][a-z-]{3,}")

# Terms that are distinctive by IDF but say nothing about subject matter — they
# mark a claim as *methods prose*, which is exactly the false signal the cosine
# band already admits too much of.
_STOP_TERMS = frozenset("""
using used uses show shows shown showed demonstrate demonstrates introduced
introduce presents present proposed propose method methods approach approaches
results result performance improve improved improvement compared comparison
between across within based including such these those their there where when
also however therefore thus while both each other than then this that with
""".split())


@dataclass(frozen=True)
class DiscoveredPair:
    """One cross-paper claim pair proposed for review."""
    stem_a: str
    stem_b: str
    category_a: str
    category_b: str
    slug_a: str
    slug_b: str
    text_a: str
    text_b: str
    cosine: float
    idf_mass: float
    shared_terms: list[str] = field(default_factory=list)

    @property
    def cross_category(self) -> bool:
        return self.category_a != self.category_b

    def citation_a(self) -> str:
        return f"[[{self.stem_a}#{self.slug_a}]]"

    def citation_b(self) -> str:
        return f"[[{self.stem_b}#{self.slug_b}]]"


def _tokens(text: str) -> set[str]:
    """Content words, with hyphenated compounds also split into their parts.

    The corpus writes the same idea both ways — one paper says "empirical-null
    method", another "constructs the null distribution ... empirical FDR" — and
    matching only whole tokens makes those disjoint. Emitting the compound *and*
    its parts unifies them without losing the compound's own specificity, which
    is often the more distinctive term ("d-dimer", "one-hot", "gradient-based").
    """
    out: set[str] = set()
    for w in _TOKEN_RE.findall((text or "").lower()):
        if w not in _STOP_TERMS:
            out.add(w)
        if "-" in w:
            out.update(part for part in w.split("-")
                       if len(part) >= 4 and part not in _STOP_TERMS)
    return out


def _load_claims() -> list[dict]:
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT c.paper_stem, c.claim_slug, c.section, c.position, c.text, "
            "       p.category "
            "  FROM claims c JOIN papers p ON p.stem = c.paper_stem "
            " WHERE c.claim_slug IS NOT NULL AND c.is_cross_ref = 0 "
            "   AND p.page_type = 'paper'"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def discover_pairs(
    *,
    cos_lo: float = DEFAULT_COS_LO,
    cos_hi: float = DEFAULT_COS_HI,
    limit: int = DEFAULT_LIMIT,
    cross_category_only: bool = False,
    exclude_linked: bool = True,
    exclude_dismissed: bool = True,
) -> list[DiscoveredPair]:
    """Rank cross-paper claim pairs in the cosine band by shared-term mass.

    `cross_category_only` narrows to pairs bridging categories — the ones no
    other structure in the wiki connects, and the reason `concept_span >= 2`
    hubs are worth building. `exclude_linked` drops pairs whose papers already
    cite each other, since those need no discovering. `exclude_dismissed` drops
    pairs a human has judged not-a-relation, so a reviewed queue shrinks instead
    of re-proposing settled questions.

    Returns at most `limit` pairs, best first. Empty on any failure (no numpy,
    cold embedding cache, no claims) — this is an advisory surface and must
    never be the reason a command fails.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    rows = _load_claims()
    if not rows:
        log(NO_CLAIMS, tag="discover")
        return []

    from ..index.claim_embeddings import load_cached_claim_embeddings
    loaded = load_cached_claim_embeddings(rows)
    if loaded is None:
        log("claim discovery skipped: embedding cache cold "
            "(warm it with any `claim-overlap` run)", tag="discover")
        return []
    vecs, row_indices = loaded
    rows = [rows[i] for i in row_indices]

    docs = [_tokens(r["text"]) for r in rows]
    df: Counter = Counter()
    for d in docs:
        df.update(d)
    n_docs = len(docs)
    idf = {w: math.log(n_docs / (1 + c)) for w, c in df.items()}

    stems = np.array([r["paper_stem"] for r in rows])
    cats = np.array([r["category"] for r in rows])

    # Blocked upper-triangle scan. The full N×N similarity matrix is the
    # obvious formulation and does not scale: 3k claims is a 37 MB float32
    # array, but 13k claims (a ~500-paper corpus) is 676 MB, and `status` calls
    # this on every run. Per-block the peak is bounded by _BLOCK × N instead.
    ii_parts: list = []
    jj_parts: list = []
    sim_parts: list = []
    n = len(rows)
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        block = vecs[start:stop] @ vecs.T                    # (b, N)
        keep = (block >= cos_lo) & (block < cos_hi)
        keep &= stems[start:stop, None] != stems[None, :]
        if cross_category_only:
            keep &= cats[start:stop, None] != cats[None, :]
        # Upper triangle only: global column index must exceed global row index.
        cols = np.arange(n)[None, :]
        keep &= cols > (np.arange(start, stop)[:, None])
        bi, bj = np.where(keep)
        if len(bi):
            ii_parts.append(bi + start)
            jj_parts.append(bj)
            sim_parts.append(block[bi, bj])
    if not ii_parts:
        return []
    ii = np.concatenate(ii_parts)
    jj = np.concatenate(jj_parts)
    pair_sims = np.concatenate(sim_parts)

    suppressed = _linked_pairs() if exclude_linked else set()
    if exclude_dismissed:
        from .pair_dismissals import dismissed_pairs
        suppressed = suppressed | dismissed_pairs()

    scored: list[tuple[float, int, int, float, list[str]]] = []
    for a, b, cos in zip(ii.tolist(), jj.tolist(), pair_sims.tolist()):
        sa, sb = rows[a]["paper_stem"], rows[b]["paper_stem"]
        if suppressed and tuple(sorted((sa, sb))) in suppressed:
            continue
        shared = docs[a] & docs[b]
        if not shared:
            continue
        mass = sum(idf.get(w, 0.0) for w in shared)
        scored.append((mass, a, b, cos,
                       sorted(shared, key=lambda w: -idf.get(w, 0.0))))
    scored.sort(key=lambda t: -t[0])

    # One entry per paper pair — the best claim pair represents it. A hub-worthy
    # relation shows up as many claim pairs; listing them all buries the tail.
    out: list[DiscoveredPair] = []
    seen: set[tuple[str, str]] = set()
    for mass, a, b, cos, shared in scored:
        ra, rb = rows[a], rows[b]
        key = tuple(sorted((ra["paper_stem"], rb["paper_stem"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(DiscoveredPair(
            stem_a=ra["paper_stem"], stem_b=rb["paper_stem"],
            category_a=ra["category"], category_b=rb["category"],
            slug_a=ra["claim_slug"], slug_b=rb["claim_slug"],
            text_a=ra["text"], text_b=rb["text"],
            cosine=float(cos), idf_mass=float(mass),
            shared_terms=shared[:8],
        ))
        if len(out) >= limit:
            break
    return out


# Nudge thresholds. Higher bar than the claim-overlap backlog's 10: this is an
# opportunity signal, not a coverage gap — nothing is *wrong* when the queue has
# entries, so it should only speak when there is enough to be worth a sitting.
DISCOVERY_THRESHOLD = 15
DISCOVERY_DECAY_DAYS = 14

_STAMP_FILENAME = ".claim-discovery-stamp"


def _stamp_path() -> Path:
    from ..paths import wiki_root
    return wiki_root() / _STAMP_FILENAME


def write_discovery_stamp() -> None:
    try:
        _stamp_path().write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def discovery_stamp_age_days() -> float | None:
    """Days since the stamp was written, or None if absent/unreadable."""
    p = _stamp_path()
    if not p.exists():
        return None
    try:
        ts = int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return (time.time() - ts) / 86400.0


def discovery_warning(*, touch: bool = True) -> str | None:
    """Nudge string when the discovery queue is worth a look, else None.

    Counts only **cross-category** pairs. Same-category pairs are frequently
    just two papers from one subfield sharing its vocabulary; a cross-category
    pair sharing distinctive terms is the case nothing else in the wiki
    connects, so it is the only part of the queue worth interrupting for.

    `touch=False` peeks without advancing decay state.
    """
    try:
        pairs = discover_pairs(limit=DISCOVERY_THRESHOLD * 4,
                               cross_category_only=True)
    except Exception:
        return None      # advisory surface — never the reason a command fails
    n = len(pairs)
    if n < DISCOVERY_THRESHOLD:
        return None
    age = discovery_stamp_age_days()
    if age is not None and age < DISCOVERY_DECAY_DAYS:
        return None
    if touch:
        write_discovery_stamp()
    return (
        f"Claim-pair discovery: {n}+ unreviewed cross-category pair(s)\n"
        f"  → researchwiki claim-overlap --discover --cross-category"
    )


def _linked_pairs() -> set[tuple[str, str]]:
    """Paper pairs that already cite each other in either direction."""
    try:
        from ..wiki import read_pages
    except Exception:
        return set()
    pages = [p for p in read_pages() if p.page_type == "paper"]
    by_stem = {p.stem: p for p in pages}
    out: set[tuple[str, str]] = set()
    for p in pages:
        try:
            body = p.path.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in re.finditer(r"\[\[([^\]|#]+)", body):
            target = m.group(1).strip().split("/")[-1]
            if target in by_stem and target != p.stem:
                out.add(tuple(sorted((p.stem, target))))
    return out
