"""Concept-hub candidate detection + persistence.

Detects recurring vocabulary terms across paper pages using two parallel
sources, then merges + labels them:

  1. LLM-authored `keywords:` and `tags:` YAML fields aggregated across pages
     (find_candidates_from_keywords). Primary source — high precision.
  2. Regex extraction over graded-claim text (find_candidates_from_claims).
     Fallback recall net — catches vocabulary that never became a keyword.

Both feed into `collect_candidates`, which merges the two, filters against
stopword sets, ranks by (span, pages) descending, and optionally
side-writes `instantiates` edges into the claim-graph.

The `n_bridge_candidates()` helper is imported by `status` to surface
the top-line "N bridge term(s) with no hub yet" opportunity signal.

Nothing here mutates wiki pages — that's `scaffold.run()` (creates a hub)
and `attach.attach_after_ingest()` (attaches a paper to existing hubs).
"""

from __future__ import annotations

import re
from math import sqrt
from pathlib import Path

from ..log import log
from ..stems import slugify_phrase
from ..wiki import read_page, strip_non_prose


ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9\-]+[A-Z0-9])\b")
# TitleCase multi-word phrases. The optional leading `\d+[- ]` absorbs a
# numeric prefix so named entities like "1000 Genomes Project" are captured
# whole instead of truncated to the fragment "Genomes Project" (which then
# double-counts as its own candidate — see docs/concept-vs-glossary.md denoise).
PHRASE_RE = re.compile(r"\b((?:\d+[- ])?[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+)\b")

STOP_ACRONYMS = {
    # infrastructure / formats
    "DOI", "PMID", "YAML", "PDF", "PDFs", "URL", "HTTP", "HTTPS", "API",
    "JSON", "XML", "CSV", "ASCII", "UTF", "CLI", "LLM", "LLMs",
    "YYYY", "TODO", "ID", "IDs", "USA", "UK", "EU",
    # all-caps English words that leak from headings, emphasis, or
    # sentence-initial capitals — pure noise as concept candidates.
    "THIS", "THAT", "THESE", "THOSE", "AND", "FOR", "WITH", "NOT", "ARE",
    "WAS", "WERE", "BUT", "ALL", "ANY", "THE", "THEN", "THAN", "WHEN",
    "WHERE", "WHICH", "WHILE", "FROM", "INTO", "OUR", "ITS", "HAS", "HAD",
    "CAN", "MAY", "ONE", "TWO", "USE", "VIA", "PER", "SEE", "NEW", "KEY",
    "HOW", "WHY", "WHO", "YES", "ONLY", "ALSO", "SUCH", "MOST", "MORE",
    "EACH", "BOTH", "OUT", "OFF", "OWN", "SET", "GET", "RUN", "END",
    # ubiquitous domain primitives — a hub linking a third of the corpus
    # carries no discriminating signal. Drop from this set to resurface one.
    "RNA", "DNA", "PCR", "GPU", "CPU", "RAM",
}

# Multi-word phrase candidates containing any of these tokens are figure /
# table / section cross-references or document scaffolding, never a concept
# (e.g. "Extended Data Fig", "Supplementary Table").
STRUCTURAL_TOKENS = {
    "Fig", "Figure", "Table", "Panel", "Supplementary", "Appendix",
    "Equation", "Eq", "Extended",
}

# Journal / venue names that recur across citation prose but aren't concepts.
STOP_PHRASES = {
    "Nature Machine Intelligence", "Nature Methods", "Nature Biotechnology",
    "Nature Communications", "Nature Genetics", "Nature Medicine",
    "Nature Reviews", "Nature Cell Biology", "Nature Neuroscience",
    "Cell Systems", "Cell Reports", "Cell Genomics", "Genome Biology",
    "Genome Research", "Molecular Systems Biology", "Nucleic Acids Research",
    "Science Advances", "Proceedings",
}

# Head nouns that anchor a lowercase methodological phrase. Detection only
# fires when a match ends in one of these — a curated whitelist keeps the
# false-positive space bounded (vs. general noun-phrase chunking). Extend as
# domains grow; adding a noun is a one-line change.
HEAD_NOUNS: frozenset[str] = frozenset({
    # methods / mechanisms
    "editing", "repair", "recombination", "mutagenesis", "mechanism", "pathway",
    "screen", "screening", "assay", "profiling",
    # effects / outcomes
    "effect", "effects", "outcome", "outcomes", "response", "activity",
    # substrates / entities
    "vector", "vectors", "nanoparticle", "nanoparticles", "atlas",
    "contact", "contacts", "variant", "variants",
    # design surfaces
    "design", "prediction", "alignment", "architecture", "structure",
    # ML / statistical objects
    "model", "models", "embedding", "embeddings", "tokenization",
    "pretraining", "fine-tuning", "objective", "benchmark", "benchmarks",
    "dataset", "datasets",
    # delivery / systems
    "delivery", "system", "systems",
})

# Precompiled regex for lowercase phrases ending in a head noun. Matches:
#   - 2 or 3 lowercase-hyphen tokens
#   - the last token is a HEAD_NOUN
# Case-sensitive on purpose: TitleCase phrases are covered by PHRASE_RE, and
# ALL-CAPS is ACRONYM_RE — this layer targets specifically the mid-sentence
# lowercase form ("prime editing", "off-target effects").
PHRASE_RE_LC = re.compile(
    r"\b(?:[a-z][a-z\-]*(?:\s+|-)){1,2}"
    r"(?:" + "|".join(sorted(HEAD_NOUNS, key=len, reverse=True)) + r")\b"
)

# Claim-writing tics — dominate the corpus if left unfiltered ("this work",
# "the method", ...). Compared against the lowercased phrase AFTER leading-
# stopword stripping (see LEADING_STOPWORDS below).
CLAIM_STOP_PHRASES: frozenset[str] = frozenset({
    "this work", "these results", "method", "approach", "results",
    "model", "models", "authors", "paper", "study",
    "framework", "algorithm", "pipeline", "system",
    "analysis", "experiment", "experiments",
    "dataset", "datasets", "benchmark",
    # generic post-strip phrases that end in a head noun but carry no signal
    "design", "effect", "effects", "outcome",
    "pathway", "mechanism", "response",
    "activity",  # bare word matches many-noun contexts
    # short bigrams often built from noise
    "full model", "single model", "based model", "based models",
    "such model", "such models", "based effect", "based effects",
    "rare variants", "common variants",   # too generic without a modifier
    "seq dataset", "seq datasets",         # hyphenation split artefact ("-seq datasets")
})

# Leading stopwords that a lowercase phrase may accidentally consume — strip
# these from the *start* of a match, then re-check the residual against
# CLAIM_STOP_PHRASES. Prevents "the foundation model" and "for rare variants"
# from being reported as distinct terms from "foundation model" / "variants".
LEADING_STOPWORDS: frozenset[str] = frozenset({
    # determiners
    "the", "a", "an", "our", "their", "its", "this", "that", "these", "those",
    "any", "each", "every", "both", "no", "some",
    # prepositions / conjunctions
    "of", "for", "and", "or", "but", "with", "to", "on", "in", "at", "by",
    "as", "than", "into", "onto", "from", "over", "under",
    "across", "beyond", "when", "where", "while",
    # participial / verbal (creep in on comparative phrases)
    "using", "based", "given", "showing", "including", "excluding",
    "restricted", "limited", "outperforms", "outperforming",
    "improves", "improving", "exceeds", "exceeding",
    "requires", "requiring", "yields", "yielding",
    # quantifier / adverbial
    "most", "same", "only", "also", "here", "now", "still", "already",
    # possessive apostrophe-s artefact
    "s",
})

# Section weights for candidate ranking. Occurrences in load-bearing sections
# (key_contributions, results) count 2× vs. context / limitations. Preserves
# per-paper coverage (a term still needs to appear in ≥3 papers), but bumps
# it in the ranking when centrality is real.
SECTION_WEIGHTS: dict[str, float] = {
    "key_contributions": 2.0,
    "results": 2.0,
    "methodology": 1.0,
    "limitations": 0.5,
    "background": 0.5,
}



def _is_structural(phrase: str) -> bool:
    """True if any whitespace-delimited word is a figure/table/section token."""
    return any(w in STRUCTURAL_TOKENS for w in phrase.split())

# --- Glossary-suspect denoise (see docs/concept-vs-glossary.md) ------------
# The `--thesis` gate (concepts.scaffold) catches glossary terms at *creation*
# time. This catches them at *detection* time so they stop dominating the
# candidate list and inflating the `status` bridge count (the deferred
# "detector heuristic" that doc anticipated building "if those pressures ever
# materialise"). Two signals, both proxies for the doc's concept-vs-glossary
# discriminator (a concept is an idea the corpus *disagrees about*; glossary is
# vocabulary used *consistently*):
#
#   1. Bare acronym / code — a single token of only capitals, digits, and
#      hyphens (≤10 chars). You can write a *definition* for SNP/PAM/CNN/AUC/WGS
#      or a cell-line/measurement code like HEK293T/LDL-C, but not a concept-
#      *thesis*; every retracted hub (PAM/RNP/LNP/DSB) had this shape. Mixed-
#      case method names (PrediXcan, ProteinMPNN, NanoSeq) keep their casing and
#      are unaffected; all-caps method names (GPN-MSA) are demoted-not-dropped,
#      which is correct — a lone method name isn't a concept the corpus argues
#      about, and a reviewer can still scaffold it with a thesis if warranted.
#   2. Corpus ubiquity — a term appearing in more than CEILING of the paper
#      corpus is ambient vocabulary with no discriminating signal (the same
#      rationale the hard-listed RNA/DNA/PCR primitives encode, generalized
#      from a fixed denylist to a corpus-relative threshold).
#
# Demote, don't drop: glossary-suspects still appear in the plain candidate
# list (labelled, sorted last) so nothing is silently lost — a reviewer who
# judges a specific acronym genuinely conceptual can still scaffold it with a
# thesis. They are excluded only from the bridge tier and the `status` count.
GLOSSARY_UBIQUITY_CEILING = 0.05
_BARE_ACRONYM_RE = re.compile(r"[A-Z0-9][A-Z0-9\-]{1,9}")

_GLOSSARY_UBIQUITY_FLOOR = 10  # absolute page minimum before the ceiling applies

def _is_glossary_suspect(term: str, pages: int, corpus_size: int | None) -> bool:
    """True if a candidate reads as glossary/metric vocabulary, not a concept."""
    if _BARE_ACRONYM_RE.fullmatch(term):
        return True
    # Ubiquity backstop for non-acronym ambient phrases. Gated on an absolute
    # floor as well as the corpus-relative ceiling: the fraction is meaningless
    # on a tiny corpus (in a 3-paper corpus every shared term is "100%
    # ubiquitous"), so a term must appear in ≥ FLOOR papers *and* exceed the
    # ceiling to be demoted on ubiquity grounds.
    if (corpus_size and pages >= _GLOSSARY_UBIQUITY_FLOOR
            and pages / corpus_size > GLOSSARY_UBIQUITY_CEILING):
        return True
    return False

def _label_for(pages: int, categories: int, *,
               term: str | None = None, corpus_size: int | None = None) -> str:
    """Threshold-tier label for a candidate.

    glossary-suspect        — bare acronym or corpus-ubiquitous (demoted; only
                              applied when `term` + `corpus_size` are supplied)
    concept-ready (bridge)  — span ≥ 2, pages ≥ 3   → scaffold this
    concept-ready (deep)    — span == 1, pages ≥ 5  → consider (synthesis may fit)
    candidate               — everything else at pages ≥ 3 (advisory)
    """
    if term is not None and _is_glossary_suspect(term, pages, corpus_size):
        return "glossary-suspect"
    if categories >= 2 and pages >= 3:
        return "concept-ready (bridge)"
    if categories == 1 and pages >= 5:
        return "concept-ready (deep)"
    return "candidate"

def _term_slug(term: str) -> str:
    """Canonical slug for a detected term — same shape `tasks.synthesize.
    _slugify` produces for hub filenames, so a scaffolded page's slug == the
    edge target. Both now delegate to one shared helper rather than keeping
    two copies in sync by hand."""
    return slugify_phrase(term)


# Suffixes where a trailing "s" is part of the stem, not a plural marker —
# stripping it would corrupt the term (bias→bia, analysis→analysi, virus→viru).
_SINGULARIZE_SKIP_SUFFIXES = ("ss", "us", "is", "os")
# Whole tokens that end in "s" (often vowel+"s") but are already singular /
# invariant — the suffix guards above miss "-as" singulars. Small, stable set;
# the co-occurrence gate makes any miss harmless (forms just stay distinct).
_SINGULARIZE_SKIP_WORDS = frozenset({
    "species", "series", "bias", "atlas", "canvas", "alias", "lens",
})


def _singularize(word: str) -> str:
    """Best-effort singularization of ONE lowercased alnum token, used only
    to *detect* a collision between two surface forms that both appear as
    candidates (see `_canonical_key` / the co-occurrence-gated merge) — never
    to rewrite a lone term. Conservative by design: irregular plurals
    (analyses→analysis, indices→index) are deliberately NOT handled, because
    a missed merge is harmless (the forms stay distinct) while a wrong merge
    is not. First rule that matches wins.
    """
    if word in _SINGULARIZE_SKIP_WORDS or len(word) <= 3:
        return word
    if word.endswith(_SINGULARIZE_SKIP_SUFFIXES):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"                 # studies → study
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]                        # batches → batch, boxes → box
    if word.endswith("s"):                      # (not "ss" — guarded above)
        return word[:-1]                        # models → model
    return word


def _canonical_key(term: str) -> str:
    """A grouping key that folds morphological near-dupes together
    ("foundation models" / "foundation-model" → "foundation-model"). Built
    ON TOP of `_term_slug` (never mutating it — that stays the frozen filename
    / decline / edge key) by singularizing only the LAST hyphen segment, where
    English plurals live ("systems biology" keeps "systems"). Returns "" iff
    `_term_slug` does.
    """
    slug = _term_slug(term)
    if not slug:
        return ""
    parts = slug.split("-")
    parts[-1] = _singularize(parts[-1])
    return "-".join(p for p in parts if p)


def _normalize_lc_phrase(tok: str) -> str | None:
    """Post-process a PHRASE_RE_LC match: lowercase, whitespace-collapse, strip
    leading stopwords, filter against CLAIM_STOP_PHRASES. Returns None if the
    residual is under 2 tokens (bare head noun = not concept-worthy) or fully
    stopwords / stop-listed.
    """
    tok = re.sub(r"\s+", " ", tok.strip().lower())
    # Strip apostrophe-s from any token (e.g. "author's design" → "design").
    tok = re.sub(r"'s\b", "", tok)
    parts = tok.split()
    while parts and parts[0] in LEADING_STOPWORDS:
        parts.pop(0)
    if len(parts) < 2:
        return None
    residual = " ".join(parts)
    if residual in CLAIM_STOP_PHRASES:
        return None
    # Also filter if the residual after stripping leading stopwords is a
    # single-word HEAD_NOUN plus a stopword we didn't catch (defensive).
    if len(parts) == 2 and parts[0] in LEADING_STOPWORDS:
        return None
    return residual

def _extract_terms(text: str) -> set[str]:
    """Run all three detection layers on a single claim's text, return
    {token} set. Filtered against stop-lists; no per-paper counting yet.
    """
    found: set[str] = set()
    for m in ACRONYM_RE.finditer(text):
        tok = m.group(1)
        if tok in STOP_ACRONYMS:
            continue
        found.add(tok)
    for m in PHRASE_RE.finditer(text):
        tok = m.group(1)
        if tok in STOP_PHRASES or _is_structural(tok):
            continue
        found.add(tok)
    for m in PHRASE_RE_LC.finditer(text):
        residual = _normalize_lc_phrase(m.group(0))
        if residual is not None:
            found.add(residual)
    return found

def find_concept_candidates(
    pages_body: dict[Path, str], existing_slugs: set[str],
) -> list[tuple[str, int, int]]:
    """Legacy pages-body-based detector. Kept as a fallback when state.db is
    empty; the primary paths are `find_candidates_from_keywords` (LLM keywords)
    and `find_candidates_from_claims` (regex over claims text).

    Returns [(token, n_pages, n_categories)] sorted by descending page count.
    """
    occurrences: dict[str, set[Path]] = {}
    for md, body in pages_body.items():
        found: set[str] = set()
        for m in ACRONYM_RE.finditer(body):
            tok = m.group(1)
            if tok in STOP_ACRONYMS or tok.lower() in existing_slugs:
                continue
            found.add(tok)
        for m in PHRASE_RE.finditer(body):
            tok = m.group(1)
            if tok in STOP_PHRASES or _is_structural(tok):
                continue
            slug = tok.lower().replace(" ", "-")
            if slug in existing_slugs:
                continue
            found.add(tok)
        for tok in found:
            occurrences.setdefault(tok, set()).add(md)

    cands: list[tuple[str, int, int]] = []
    for tok, pgs in occurrences.items():
        if len(pgs) < 3:
            continue
        n_categories = len({p.parent.name for p in pgs})
        cands.append((tok, len(pgs), n_categories))
    cands.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return cands

def find_candidates_from_claims(
    claim_rows: list[dict],
    existing_slugs: set[str],
    *,
    persist_edges: bool = False,
    corpus_size: int | None = None,
    existing_canon: set[str] | None = None,
) -> list[dict]:
    """Claim-substrate concept-hub detector.

    `claim_rows`: [{paper_stem, section, claim_slug, text, category}], one row
    per graded claim in state.db. `category` is the parent directory (drives
    the span calculation).

    `existing_slugs`: set of concept-page slugs already on disk. Terms whose
    slug matches are filtered out — they already have a hub.

    `persist_edges`: when True, side-write `instantiates(claim_slug → concept-
    term-slug)` edges to `.claim-graph/edges.db`. Silent no-op on any failure
    (candidates output is independent of edge persistence).

    Returns [{
        "term":          str,     # verbatim as detected (preserved casing)
        "slug":          str,     # canonical concept-page slug
        "pages":         int,     # distinct paper stems containing the term
        "categories":    int,     # distinct wiki categories the term spans
        "weighted":      float,   # section-weighted paper count (ranking signal)
        "sections":      dict[str, int],  # {section: count} across the corpus
        "label":         str,     # concept-ready (bridge|deep) | candidate
    }] ranked by `pages × sqrt(categories)` primary, weighted score tiebreak.
    """
    # Aggregate: term → {paper_stem → {section: count, ...}}
    # Also track (paper_stem, claim_slug) pairs per term for edge writes.
    per_term: dict[str, dict[str, dict[str, int]]] = {}
    edge_pairs: dict[str, set[tuple[str, str]]] = {}
    category_by_stem: dict[str, str] = {}

    for row in claim_rows:
        stem = row["paper_stem"]
        section = row["section"]
        text = row["text"]
        slug = row.get("claim_slug")
        category_by_stem.setdefault(stem, row.get("category") or "")
        terms = _extract_terms(text)
        for tok in terms:
            term_slug = _term_slug(tok)
            if (not term_slug or term_slug in existing_slugs
                    or _canonical_key(tok) in (existing_canon or set())):
                continue
            per_term.setdefault(tok, {}).setdefault(stem, {})
            per_term[tok][stem][section] = per_term[tok][stem].get(section, 0) + 1
            if slug:
                edge_pairs.setdefault(tok, set()).add((stem, slug))

    out: list[dict] = []
    for term, by_stem in per_term.items():
        pages = len(by_stem)
        if pages < 3:
            continue
        categories = len({category_by_stem.get(s, "") for s in by_stem if category_by_stem.get(s)})
        # Weighted score: sum over (paper, section) of SECTION_WEIGHTS[section];
        # multiple claims from the same paper in the same section count once.
        weighted = 0.0
        sections_agg: dict[str, int] = {}
        for stem, sec_counts in by_stem.items():
            for section in sec_counts:
                weighted += SECTION_WEIGHTS.get(section, 1.0)
                sections_agg[section] = sections_agg.get(section, 0) + 1
        out.append({
            "term": term,
            "slug": _term_slug(term),
            "pages": pages,
            "categories": categories,
            "weighted": round(weighted, 2),
            "sections": sections_agg,
            "label": _label_for(pages, categories, term=term, corpus_size=corpus_size),
        })
    # Ranking (§3.3): glossary-suspects sink to the bottom, then
    # pages × sqrt(categories) primary, weighted score tiebreak, term name.
    out.sort(key=lambda r: (r["label"] == "glossary-suspect",
                            -(r["pages"] * sqrt(max(r["categories"], 1))),
                            -r["weighted"], r["term"]))

    if persist_edges and out:
        _persist_instantiates_edges(edge_pairs, out)

    return out

def _persist_instantiates_edges(
    edge_pairs: dict[str, set[tuple[str, str]]],
    candidates: list[dict],
) -> None:
    """Side-write `instantiates(claim_slug → concept-term-slug)` edges.

    Only writes for terms that made it into the candidate list (≥3 papers) —
    otherwise every one-off mention floods the cache. Silent no-op on any
    failure so `--candidates` output survives cache trouble.
    """
    try:
        from ..claim_graph import Edge, SLUG_SCHEME_VERSION, open_edges_db, upsert_edge

        candidate_terms = {c["term"] for c in candidates}
        conn = open_edges_db()
        try:
            n_written = 0
            for term, pairs in edge_pairs.items():
                if term not in candidate_terms:
                    continue
                term_slug = _term_slug(term)
                if not term_slug:
                    continue
                for (stem, claim_slug) in pairs:
                    upsert_edge(conn, Edge(
                        src_stem=stem, src_slug=claim_slug,
                        tgt_stem="concepts", tgt_slug=term_slug,
                        relation="instantiates",
                        directed=True,
                        slug_scheme_version=SLUG_SCHEME_VERSION,
                        status="candidate",
                        rationale="detected in claim",
                        judge_phase="concepts_detector",
                    ))
                    n_written += 1
            conn.commit()
            if n_written:
                log(f"instantiates edges persisted: {n_written}", tag="concepts")
        finally:
            conn.close()
    except Exception as e:
        log(f"instantiates persistence skipped: {type(e).__name__}: {e}",
            tag="concepts")

def _load_claim_rows() -> list[dict]:
    """Pull graded claims from state.db, joined with each paper's category.

    Returns [] when the DB is empty or unreachable. Only paper-type rows are
    considered — synthesis / ideas / concepts pages don't produce claims.
    """
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT c.paper_stem, c.section, c.claim_slug, c.text, p.category "
            "  FROM claims c "
            "  JOIN papers p ON c.paper_stem = p.stem "
            " WHERE c.is_cross_ref = 0 AND c.claim_slug IS NOT NULL "
            "   AND p.page_type = 'paper'"
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]

def _load_paper_metadata() -> list[dict]:
    """Pull LLM-authored per-paper metadata from state.db.

    Returns [{stem, category, keywords, tags}]. Keywords and tags are parsed
    from raw_frontmatter (JSON). Empty list on any DB failure. Only paper-type
    pages are considered — the concept-hub detector shouldn't count keywords
    from synthesis / concept / reference pages.
    """
    try:
        from ..db.connection import get_connection
        import json as _json
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT stem, category, raw_frontmatter "
            "  FROM papers WHERE page_type = 'paper'"
        ).fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            fm = _json.loads(r["raw_frontmatter"])
        except (TypeError, ValueError):
            continue
        kws = fm.get("keywords") or []
        tags = fm.get("tags") or []
        out.append({
            "stem": r["stem"],
            "category": r["category"],
            "keywords": [k for k in kws if isinstance(k, str) and k.strip()],
            "tags": [t for t in tags if isinstance(t, str) and t.strip()],
        })
    conn.close()
    return out


def _load_hub_aliases() -> list[str]:
    """Every existing concept hub's `topic_seed` + `topic_seed_aliases`, read
    from state.db — NOT the filesystem: this runs on `status`'s fast path via
    `n_bridge_candidates`, so it's one query, no file opens or YAML parses
    (concept pages already carry `raw_frontmatter` in `papers`). Used to
    canonically exclude candidates that are near-dupes of a hub that already
    exists (e.g. "protein language model" vs the `protein-language-models`
    page, or "FH" for familial hypercholesterolemia). Empty list on any DB
    failure — the filename-stem exclusion still applies without it.
    """
    try:
        from ..db.connection import get_connection
        import json as _json
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT raw_frontmatter FROM papers WHERE page_type = 'concept'"
        ).fetchall()
    except Exception:
        return []
    out: list[str] = []
    for r in rows:
        try:
            fm = _json.loads(r["raw_frontmatter"])
        except (TypeError, ValueError):
            continue
        seed = fm.get("topic_seed")
        if isinstance(seed, str) and seed.strip():
            out.append(seed)
        aliases = fm.get("topic_seed_aliases") or []
        if isinstance(aliases, list):
            out.extend(a for a in aliases if isinstance(a, str) and a.strip())
    conn.close()
    return out

# Framework/agent-artifact tags that leak into `tags:` YAML but aren't concepts.
_FRAMEWORK_TAG_STOPLIST: frozenset[str] = frozenset({
    "ingested-via-agent",
    "auto-added",
    "concept-link",
    "review",  # occasional legacy tag; keep an eye on it
})

def find_candidates_from_keywords(
    papers_meta: list[dict],
    existing_slugs: set[str],
    *,
    existing_canon: set[str] | None = None,
) -> list[dict]:
    """Primary detector: aggregate the LLM-authored `keywords` + `tags`.

    Each paper contributes its ingest-phase `keywords` (usually 7-11 terms
    per paper, LLM-normalized) plus any conceptual `tags`. Terms are grouped
    by canonical slug so hyphen/space variants merge ("off-target" ↔
    "off target"). Framework tags (`ingested-via-agent` etc.) are stripped.

    Returns the same {term, slug, pages, categories, weighted, sections,
    label, source} shape as find_candidates_from_claims so callers don't
    branch on source.
    """
    corpus_size = len(papers_meta) or None
    if existing_canon is None:   # canonical hub/alias exclusion is opt-in (real
        existing_canon = set()   # callers pass it; tests keep the exact-slug path)

    # {canonical_slug → {stem → category}}, plus a companion map keeping
    # the human-readable form for display (whichever variant is longest).
    by_slug_stems: dict[str, dict[str, str]] = {}
    display_form: dict[str, str] = {}

    for paper in papers_meta:
        stem = paper["stem"]
        category = paper.get("category") or ""
        raw_terms: list[str] = []
        for src in ("keywords", "tags"):
            v = paper.get(src) or []
            if isinstance(v, list):
                raw_terms.extend(v)
        for raw in raw_terms:
            if not isinstance(raw, str):
                continue
            term = raw.strip()
            if not term:
                continue
            if term.lower() in _FRAMEWORK_TAG_STOPLIST:
                continue
            slug = _term_slug(term)
            if not slug or slug in existing_slugs or _canonical_key(term) in existing_canon:
                continue
            by_slug_stems.setdefault(slug, {})[stem] = category
            # Prefer the longer form as display (usually the fuller phrase).
            prev = display_form.get(slug, "")
            if len(term) > len(prev):
                display_form[slug] = term

    # Fold morphological near-dupes ("foundation model" / "foundation models")
    # into one representative — but ONLY when ≥2 distinct surface slugs share a
    # canonical key (co-occurrence gate). A lone slug is never rewritten, so a
    # false plural (e.g. "biases" with no "bias") can't be corrupted. Page/
    # category counts come from the UNIONED stem map, not summed integers.
    canon_members: dict[str, list[str]] = {}
    for slug in by_slug_stems:
        ckey = _canonical_key(display_form.get(slug) or slug)
        canon_members.setdefault(ckey, []).append(slug)

    out: list[dict] = []
    for slugs in canon_members.values():
        if len(slugs) >= 2:
            stems: dict[str, str] = {}
            for s in slugs:
                stems.update(by_slug_stems[s])   # union stem→category by stem key
            # Representative: longest display form, tie → most pages → lexical.
            slug = sorted(slugs, key=lambda s: (-len(display_form.get(s, s)),
                                                -len(by_slug_stems[s]), s))[0]
        else:
            slug = slugs[0]
            stems = by_slug_stems[slug]
        pages = len(stems)
        if pages < 3:
            continue
        categories = len({c for c in stems.values() if c})
        term = display_form.get(slug) or slug
        out.append({
            "term": term,
            "slug": slug,
            "pages": pages,
            "categories": categories,
            "weighted": float(pages),  # keyword-derived; no section weighting
            "sections": {},
            "label": _label_for(pages, categories, term=term, corpus_size=corpus_size),
            "source": "keywords",
        })
    out.sort(key=lambda r: (r["label"] == "glossary-suspect",
                            -(r["pages"] * sqrt(max(r["categories"], 1))),
                            -r["weighted"], r["term"]))
    return out

def _merge_candidate_sources(
    primary: list[dict], secondary: list[dict],
) -> list[dict]:
    """Union two candidate lists keyed by slug. Primary wins on term display,
    label, pages, categories. Secondary adds only rows whose slug isn't
    already present. Preserves primary's ordering; appends new secondaries.
    """
    seen = {r["slug"] for r in primary}
    out = list(primary)
    for r in secondary:
        if r["slug"] in seen:
            continue
        seen.add(r["slug"])
        out.append(r)
    return out


_SOURCE_RANK = {"keywords": 2, "claims": 1, "page-body": 0}


def _dedup_by_canonical(rows: list[dict], *,
                        corpus_size: int | None = None) -> list[dict]:
    """Collapse morphological near-dupes that survive `_merge_candidate_sources`
    because they came from different detectors (keywords "foundation models" vs
    claims "foundation model") or from the claims path's raw-token keying — the
    exact-slug merge leaves those separate. Keep ONE representative per
    `_canonical_key`: keywords source wins, then more pages, then longer term.

    The representative supplies `term`/`slug`/`source` — that's a pick-best, so
    the keyword detector's display form and provenance win. But `pages` and
    `categories` are taken as the MAX across the group, because they describe
    the *concept*, not the detector that happened to find it: each detector
    sees a subset of the papers instantiating the term, so every row's count is
    ≤ the true union and `max` is a valid (strictly tighter) lower bound.
    Summing is what would double-count the overlap; `max` cannot.

    Pick-best on `categories` was a silent signal loss: a claims row spanning 3
    categories collapsed into a 1-category keywords row dropped the term out of
    the bridge tier and out of `status`'s bridge count — the exact signal
    concept hubs exist to surface. `label` is therefore recomputed from the
    merged counts (an earned `glossary-suspect` demotion is never undone —
    re-derived from the representative's own term, so a bare acronym stays
    demoted while a real phrase form in the group rightly promotes).

    `weighted` stays the representative's: it's a ranking tiebreak on
    incommensurable scales (keyword rows use page count, claim rows use
    SECTION_WEIGHTS), so a cross-source max would be meaningless.

    Single-member canonical groups are returned verbatim — a lone term is never
    rewritten, and gains no keys it didn't have (co-occurrence-safe). First-seen
    order is preserved so the keyword detector's ranking survives.
    """
    def rank(r: dict) -> tuple:
        return (_SOURCE_RANK.get(r.get("source", ""), 0), r.get("pages", 0),
                len(r.get("term", "")))

    best: dict[str, dict] = {}
    members: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        ck = _canonical_key(r["term"])
        if ck not in best:
            best[ck] = r
            members[ck] = [r]
            order.append(ck)
            continue
        members[ck].append(r)
        if rank(r) > rank(best[ck]):
            best[ck] = r

    out: list[dict] = []
    for ck in order:
        group = members[ck]
        rep = best[ck]
        if len(group) == 1:
            out.append(rep)          # verbatim — no merged keys added
            continue
        merged = dict(rep)
        merged["pages"] = max(r.get("pages", 0) for r in group)
        merged["categories"] = max(r.get("categories", 0) for r in group)
        merged["label"] = _label_for(
            merged["pages"], merged["categories"],
            term=merged.get("term"), corpus_size=corpus_size,
        )
        out.append(merged)
    return out

def _persist_keyword_instantiates(
    keyword_rows: list[dict], claim_rows: list[dict],
    papers_meta: list[dict],
) -> None:
    """Attribute `instantiates` edges from keyword-derived terms to specific
    claims. For each paper/keyword pair, find the paper's claims whose text
    contains the keyword (case-insensitive substring) and emit one edge per
    match. Skips a paper's contribution if no claim mentions the term
    (keyword-only membership is real but not claim-anchored).
    """
    try:
        from ..claim_graph import Edge, SLUG_SCHEME_VERSION, open_edges_db, upsert_edge

        # Group claims by paper for O(N) lookup.
        by_paper: dict[str, list[dict]] = {}
        for c in claim_rows:
            by_paper.setdefault(c["paper_stem"], []).append(c)
        # Which keywords each paper carries.
        paper_keywords: dict[str, set[str]] = {}
        for p in papers_meta:
            terms: set[str] = set()
            for src in ("keywords", "tags"):
                for raw in p.get(src, []) or []:
                    if isinstance(raw, str) and raw.strip():
                        terms.add(raw.strip().lower())
            paper_keywords[p["stem"]] = terms
        # Canonical key → representative slug. Keyed canonically so a paper
        # keyword that is a morphological variant of the merged representative
        # (e.g. "foundation model" when the row emitted "foundation models")
        # still attributes its edge to the representative, not dropped.
        candidate_by_canon = {_canonical_key(r["term"]): r["slug"] for r in keyword_rows}

        conn = open_edges_db()
        try:
            n_written = 0
            for stem, keywords in paper_keywords.items():
                stem_claims = by_paper.get(stem, [])
                if not stem_claims:
                    continue
                for kw in keywords:
                    slug = candidate_by_canon.get(_canonical_key(kw))
                    if slug is None:
                        continue
                    # Word-boundary match (not naive substring) so a keyword
                    # like "gene" doesn't attribute an edge to a claim about
                    # "generation" — matches _matching_claims' boundary logic.
                    kw_re = re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(kw)}(?![A-Za-z0-9])", re.IGNORECASE
                    )
                    for claim in stem_claims:
                        if kw_re.search(claim.get("text") or ""):
                            upsert_edge(conn, Edge(
                                src_stem=stem,
                                src_slug=claim["claim_slug"],
                                tgt_stem="concepts",
                                tgt_slug=slug,
                                relation="instantiates",
                                directed=True,
                                slug_scheme_version=SLUG_SCHEME_VERSION,
                                status="candidate",
                                rationale="keyword+claim match",
                                judge_phase="concepts_detector",
                            ))
                            n_written += 1
            conn.commit()
            if n_written:
                log(f"instantiates edges (keyword-anchored) persisted: {n_written}",
                    tag="concepts")
        finally:
            conn.close()
    except Exception as e:
        log(f"keyword instantiates persistence skipped: {type(e).__name__}: {e}",
            tag="concepts")

def collect_candidates(
    *, bridges_only: bool = False, persist_edges: bool = False,
) -> list[dict]:
    """Discover concept candidates over the current wiki state.

    **Primary source: LLM-authored `keywords` + `tags`** on each paper page.
    Already normalized, topical, and semantically picked during ingest —
    strictly higher signal than regex over free text.

    **Secondary source: claim-substrate regex** over `claims.text` with the
    layered acronym + TitleCase + head-noun detectors. Catches acronyms +
    phrases that never made it into a paper's `keywords`.

    **Fallback: page-body scan.** Only used when state.db is empty
    (fresh install / pre-rebuild state).

    Sources are unioned by slug — keywords wins on term display and label
    when both surface the same concept. Rows carry a `source` field
    (`keywords` | `claims` | `page-body`) so downstream can weight them.

    Terms recorded in `.concept-declines.json` (via
    `researchwiki candidates concepts --decline`) are always excluded —
    a manual, permanent override for terms that failed the
    concept-vs-glossary thesis test (docs/concept-vs-glossary.md) and
    would otherwise keep resurfacing every call, since detection here is
    stateless.

    Returns [{term, slug, pages, categories, weighted, sections, label, source}].
    `bridges_only` restricts to concept-ready (bridge) tier. `persist_edges`
    side-writes `instantiates` edges into `.claim-graph/edges.db` — attributed
    to specific claims whose text contains the keyword.
    """
    from ..categories import content_categories
    from ..tasks.lint.walk import all_pages
    # Deferred import: `.declines` imports `_term_slug` from this module, so
    # importing it at module level here would cycle. By call time both
    # modules are fully loaded.
    from .declines import declined_canon, declined_slugs

    declined = declined_slugs()
    declined_ckeys = declined_canon()
    pages = all_pages()
    known_stems = {p.stem.lower() for p in pages}
    # Canonical exclusion = filename stems + every existing hub's topic_seed
    # and aliases, singularized. Catches near-dupes of a scaffolded hub
    # ("protein language model" vs `protein-language-models`; "FH").
    existing_canon = {_canonical_key(s) for s in known_stems} | {
        _canonical_key(a) for a in _load_hub_aliases()
    }

    # Primary: LLM keywords + tags.
    papers_meta = _load_paper_metadata()
    keyword_rows: list[dict] = []
    if papers_meta:
        keyword_rows = find_candidates_from_keywords(
            papers_meta, known_stems, existing_canon=existing_canon)

    # Secondary: claim-substrate regex. Persist edges only from the
    # secondary path when it surfaces something keywords missed; for terms
    # that ARE in keywords, prefer the keyword-anchored persistence below
    # (attributes to matching claims, not to every claim where the term
    # appears in body text).
    claim_rows_data = _load_claim_rows()
    # Corpus size for the glossary ubiquity ceiling: prefer the paper count,
    # fall back to distinct stems seen in the claim rows.
    corpus_size = len(papers_meta) or len({r["paper_stem"] for r in claim_rows_data}) or None
    claim_regex_rows: list[dict] = []
    if claim_rows_data:
        claim_regex_rows = find_candidates_from_claims(
            claim_rows_data, known_stems, persist_edges=False, corpus_size=corpus_size,
            existing_canon=existing_canon,
        )
        # Tag them for provenance.
        for r in claim_regex_rows:
            r.setdefault("source", "claims")

    # Union: keywords primary, claim-regex fills gaps.
    if keyword_rows or claim_regex_rows:
        combined = _merge_candidate_sources(keyword_rows, claim_regex_rows)
        # Decline filter = exact-slug OR canonical-key (union). Exact honors
        # every legacy decline verbatim (even if a merged representative slug
        # shifted); canonical adds "decline one form → suppress its near-dupes".
        combined = [r for r in combined
                    if r["slug"] not in declined
                    and _canonical_key(r["term"]) not in declined_ckeys]
        # Cross-source near-dupe collapse (keywords "foundation models" vs
        # claims "foundation model"): the exact-slug merge above leaves these
        # separate; fold them to one representative per canonical key.
        combined = _dedup_by_canonical(combined, corpus_size=corpus_size)
        # Re-sort: the collapse can lift a row's pages/categories (and so its
        # label), which invalidates the per-detector ordering it arrived with.
        # Same key both detectors sort by, so the ranking contract holds.
        combined.sort(key=lambda r: (r["label"] == "glossary-suspect",
                                     -(r["pages"] * sqrt(max(r["categories"], 1))),
                                     -r["weighted"], r["term"]))
        if persist_edges:
            # Keyword-anchored edges for candidates surfaced by keywords.
            _persist_keyword_instantiates(keyword_rows, claim_rows_data, papers_meta)
            # Regex-anchored edges only for terms keywords missed.
            keyword_slugs = {r["slug"] for r in keyword_rows}
            regex_only = [r for r in claim_regex_rows if r["slug"] not in keyword_slugs]
            if regex_only:
                # Re-run claim-substrate with persistence for the missing set.
                # (Cheap — the regex has already run once; this just re-scans
                # with edge-writes enabled, filtered to regex-only slugs.)
                _persist_regex_only_edges(regex_only, claim_rows_data)
        if bridges_only:
            combined = [r for r in combined if r["label"] == "concept-ready (bridge)"]
        return combined

    # Full fallback: page-body scan (no DB, no claims).
    if not pages:
        return []
    content_cats = content_categories()
    paper_prose: dict[Path, str] = {}
    for md in pages:
        if md.parent.name not in content_cats:
            continue
        p = read_page(md)
        body = p.body if p else md.read_text(encoding="utf-8")
        paper_prose[md] = strip_non_prose(body)

    legacy_rows = find_concept_candidates(paper_prose, known_stems)
    out: list[dict] = []
    legacy_corpus_size = len(paper_prose) or None
    for tok, n, c in legacy_rows:
        label = _label_for(n, c, term=tok, corpus_size=legacy_corpus_size)
        if bridges_only and label != "concept-ready (bridge)":
            continue
        if _term_slug(tok) in declined or _canonical_key(tok) in declined_ckeys:
            continue
        out.append({
            "term": tok, "slug": _term_slug(tok),
            "pages": n, "categories": c,
            "weighted": float(n),
            "sections": {},
            "label": label,
            "source": "page-body",
        })
    return out

def _persist_regex_only_edges(regex_only: list[dict], claim_rows: list[dict]) -> None:
    """Emit `instantiates` edges for terms surfaced ONLY by the regex path
    (not in any paper's keywords). Uses the same claim-mention attribution
    as _persist_keyword_instantiates for consistency.
    """
    try:
        from ..claim_graph import Edge, SLUG_SCHEME_VERSION, open_edges_db, upsert_edge

        # slug → term (canonical form for matching)
        needed = {r["slug"]: r["term"] for r in regex_only}
        if not needed:
            return
        # Word-boundary matchers per needed term (not naive substring), so a
        # term like "ai" doesn't attribute an edge to "domain"/"training".
        # Matches _matching_claims' boundary logic; precompiled once.
        needed_res = {
            slug: re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
            for slug, term in needed.items()
        }
        conn = open_edges_db()
        try:
            n_written = 0
            for claim in claim_rows:
                text = claim.get("text") or ""
                if not text:
                    continue
                for slug, matcher in needed_res.items():
                    if matcher.search(text):
                        upsert_edge(conn, Edge(
                            src_stem=claim["paper_stem"],
                            src_slug=claim["claim_slug"],
                            tgt_stem="concepts",
                            tgt_slug=slug,
                            relation="instantiates",
                            directed=True,
                            slug_scheme_version=SLUG_SCHEME_VERSION,
                            status="candidate",
                            rationale="regex+claim match",
                            judge_phase="concepts_detector",
                        ))
                        n_written += 1
            conn.commit()
            if n_written:
                log(f"instantiates edges (regex-anchored) persisted: {n_written}",
                    tag="concepts")
        finally:
            conn.close()
    except Exception as e:
        log(f"regex instantiates persistence skipped: {type(e).__name__}: {e}",
            tag="concepts")

def n_bridge_candidates() -> int | None:
    """Fast count of bridge-tier candidates (span ≥ 2). Used by `status`.

    Never raises — `status` must not fail because a helper hiccupped. But it
    returns **None**, not 0, when the scan breaks: CLAUDE.md makes the count
    `status` prints the trigger for scaffolding a hub, and `status` prints
    nothing at 0, so reporting a crash as 0 made "the scan died" indistin-
    guishable from "nothing to do" — the trigger would silently stop firing.
    None lets the caller say so out loud. The reason still goes to the log.
    """
    try:
        return len(collect_candidates(bridges_only=True))
    except Exception as e:
        log(f"bridge-candidate scan failed: {type(e).__name__}: {e}",
            tag="concepts")
        return None
