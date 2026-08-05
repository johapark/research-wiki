"""Does this metadata record actually describe *this* paper?

One shared answer to a question three call sites ask, all of which are about to
write a DOI (or a venue) into a page's frontmatter:

  - `tasks.backfill doi` — a title search returned a candidate; adopt it?
  - `agents.phases.reconcile` — a DOI scavenged from the PDF resolved on
    Crossref; adopt it?
  - `tasks.backfill doi --verify` — a DOI already on a page: is it the right one?

The three used to disagree, and the middle one didn't ask at all. `reconcile`'s
URL-DOI hunt validated a candidate by checking only that it **resolves**, which
is a different question from whether it is this paper's DOI. On
`kim-2019-spcas9-activity-prediction-by-deepspcas9` the hunt found
`arxiv.org/abs/1412.6980` in the *reference list* — the Adam optimizer, cited as
the training optimizer — and offered it up as the paper's own DOI. It was saved
only by an accident of coverage: arXiv DOIs aren't in Crossref, so the lookup
404'd. Had the cited work been a journal article with a registered DOI, the
lookup would have succeeded and the page would have shipped with another
paper's DOI — silently, and `audit`'s citation graph, `retraction-check` and
`preprint-check` all key off that field.

So resolvability is not adoption-worthiness, and this module is the difference.
"""

from __future__ import annotations

import re

# Same stop list the stem rules use (CLAUDE.md → File Naming Convention), plus
# "is". Title comparison is over content words only: two records for one paper
# routinely disagree on articles and prepositions after a publisher's
# copy-editing pass, and on hyphen/colon handling.
STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "with", "and", "or", "in", "on", "at", "to",
    "from", "by", "as", "across", "over", "all", "that", "this", "these",
    "those", "is",
})

#: Minimum share of content words that must overlap between the two titles.
#: Measured against the smaller token set, not the union, so a candidate whose
#: title is a strict superset (subtitle retained upstream, dropped in the PDF's
#: running head) still matches.
TITLE_OVERLAP_MIN = 0.5

#: Years may disagree by this much. A preprint-to-journal transition shifts the
#: year by one, and December/January issues routinely straddle.
YEAR_TOLERANCE = 1

#: Tokens that look like a surname after naive parsing but identify nobody. A
#: page whose `authors:` reads `Chuai et al.` yields `al` from last-token
#: extraction, which then matches no candidate and reads as "wrong paper" — four
#: such pages surfaced as false MISMATCHes on the first `--verify` sweep. The
#: real defect is the page's `authors:` field, so callers should report these
#: separately rather than as a metadata mismatch.
UNUSABLE_SURNAMES = frozenset({"al", "et", "etal", "others", "unknown", "anonymous"})


def normalise_author(s: str) -> str:
    """Lowercase, ASCII-fold, strip non-alphanumerics. For surname matching.

    Folds via `stems.strip_diacritics`, not bare NFKD, because NFKD cannot
    decompose every Latin letter: `Szałata` has a `ł` (U+0142) with no canonical
    decomposition, so NFKD + `encode('ascii','ignore')` silently *deletes* it and
    yields `szaata`. Stem derivation already learned this (see `stems`'
    `_TRANSLITERATE`), and the mismatch between the two spellings surfaced as a
    false MISMATCH against `single-cell/szaata-2024-…` on the first
    `backfill doi --verify` sweep: the page's own `authors:` folded one way and
    the provider's the other, for the same person.
    """
    if not s:
        return ""
    from .stems import strip_diacritics
    return re.sub(r"[^a-z0-9]", "", strip_diacritics(s).lower())


def title_tokens(s: str) -> set[str]:
    """Lowercase content-word tokens of a title, for Jaccard-style comparison."""
    if not s:
        return set()
    from .stems import strip_diacritics
    toks = re.findall(r"[a-z0-9]+", strip_diacritics(s).lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def title_overlap(a: str, b: str) -> float | None:
    """Content-word overlap of two titles, or None when either has no tokens.

    None means "cannot judge" and is deliberately distinct from 0.0 ("judged,
    and they share nothing") — callers must not read an unjudgeable pair as a
    mismatch, or a PDF whose title didn't extract would veto every candidate.
    """
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return None
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def sanity_ok(
    cand_year: int | None,
    cand_authors: list[str],
    cand_title: str,
    first_author: str,
    wiki_year: int | None,
    wiki_title: str,
) -> bool:
    """True when a metadata record plausibly describes the same paper.

    Requires ALL of:
      - the paper's first-author surname appears (normalised) in some candidate
        author — the check that catches a *different paper entirely*;
      - the years agree within `YEAR_TOLERANCE`, when both are known;
      - the titles share at least `TITLE_OVERLAP_MIN` of their content words,
        when both are judgeable — this is what catches a common surname
        returning an unrelated paper by the same author.

    Unknown inputs weaken the test rather than failing it, with one exception:
    an empty `first_author` returns False. Author is the load-bearing signal;
    without it the remaining checks are too weak to justify writing a DOI.
    """
    return reject_reason(
        cand_year, cand_authors, cand_title, first_author, wiki_year, wiki_title
    ) is None


def reject_reason(
    cand_year: int | None,
    cand_authors: list[str],
    cand_title: str,
    first_author: str,
    wiki_year: int | None,
    wiki_title: str,
) -> str | None:
    """Why `sanity_ok` said no, as a short log-ready phrase. None when it said yes.

    Kept separate so `sanity_ok` stays a cheap predicate, and so the ingest log
    can say *which* check killed a candidate — the difference between "the hunt
    found nothing" and "the hunt found the Adam optimizer" is the whole reason
    this module exists, and a bare False hides it.
    """
    fa = normalise_author(first_author)
    if not fa:
        return "no first-author surname to check against"
    if fa in UNUSABLE_SURNAMES:
        return (f"page's authors field is malformed ({first_author!r}) — "
                f"nothing to match against")
    ov = title_overlap(wiki_title, cand_title)
    have_authors = any(normalise_author(a) for a in cand_authors)
    if have_authors:
        if not any(fa in normalise_author(a) for a in cand_authors):
            names = ", ".join(a for a in cand_authors[:3] if a) or "none"
            return f"first author {first_author!r} not among candidate authors ({names})"
    else:
        # The record carries no author list, so the author check is unjudgeable
        # rather than failed — Crossref returns exactly this for some ACL
        # Anthology deposits, and treating it as a failure reported
        # `ai/wadden-2022-multivers-…` as a wrong DOI when Crossref's own title
        # for it was the paper's. With no author to lean on, the title must
        # positively confirm instead; if neither is judgeable there is nothing
        # to go on and the record is refused.
        if ov is None:
            return "candidate record has neither authors nor a comparable title"
        if ov < TITLE_OVERLAP_MIN:
            return (f"no candidate authors, and title overlap {ov:.2f} < "
                    f"{TITLE_OVERLAP_MIN} (candidate: {cand_title[:60]!r})")
    if wiki_year is not None and cand_year is not None:
        if abs(int(cand_year) - int(wiki_year)) > YEAR_TOLERANCE:
            return f"year {cand_year} vs paper's {wiki_year}"
    if ov is not None and ov < TITLE_OVERLAP_MIN:
        return (f"title overlap {ov:.2f} < {TITLE_OVERLAP_MIN} "
                f"(candidate: {cand_title[:60]!r})")
    return None


# --- venue plausibility -----------------------------------------------------
# Strings that are typesetting furniture rather than a journal of record. Only
# consulted when no provider (S2/Crossref) supplied a venue, i.e. the PDF
# masthead is the sole source — the situation in which a venue hint picks up
# whatever text sits near the title.
#
# Deliberately restricted to names **no real journal has**, because the wider
# alternatives were measured and are worse:
#
#   - A subject-word deny-list (GENETICS, IMMUNOLOGY, …) is unsafe. `GENETICS`
#     was wrong on `kim-2019-spcas9-…` (it is *Science Advances*' subject-category
#     label) but *Genetics* is a real journal, and
#     `genomics/li-2003-modeling-linkage-disequilibrium-and-identifying`
#     legitimately carries it ("Copyright © 2003 by the Genetics Society of
#     America"). Such a list would have corrupted a correct value.
#   - Reading the venue from a repeated `JOURNAL | SECTION` running header was
#     tried and rejected on measurement: across a 70-PDF corpus sample only 4 had
#     such a header at all, and 2 of those 4 were wrong (`UAA` lifted from a
#     sequence; `RESEARCH` from Science's own header). 6% recall at 50% precision
#     is not worth a wrong venue.
#
# The subject-label class is left to a reviewer instead, via `lint`'s
# `venue_suspect` and `backfill doi --verify`, which can compare against provider
# metadata. Leaving `venue` unset is the better failure: absent reads as
# incomplete, wrong reads as authoritative.
VENUE_FURNITURE = (
    "journal of latex class files",
    "latex class files",
    "manuscript submitted to",
    "submitted to",
    "to appear in",
    "preprint",
    "unpublished",
    "in preparation",
)


def is_venue_furniture(venue: str | None) -> bool:
    """True when a venue string names typesetting furniture, not a journal."""
    v = (venue or "").strip().lower()
    if not v:
        return False
    return any(f in v for f in VENUE_FURNITURE)


# --- placeholder DOIs -------------------------------------------------------
#: A DOI whose suffix is template boilerplate rather than an identifier. ACM's
#: LaTeX class ships `10.1145/nnnnnnn.nnnnnnn` in its sample document, and
#: `ai/formal-2021-splade-v2-sparse-lexical-and-expansion` shipped with exactly
#: that value — the PDF really does print it, so extraction was faithful and the
#: result was still not a DOI. Crossref returns nothing for it, so it reads as
#: "unresolved" rather than as the defect it is unless named explicitly.
_PLACEHOLDER_DOI_RE = re.compile(
    r"""(?ix)
    ^10\.\d{4,9}/
    (?: [nx]{4,}(?:[.\-_][nx]{4,})*        # nnnnnnn.nnnnnnn / xxxxxxx
      | 0+(?:[.\-_]0+)*                    # all-zero suffix
      | (?:doi|todo|tbd|xxx|placeholder)
    )$
    """
)


def is_placeholder_doi(doi: str | None) -> bool:
    """True when a DOI-shaped string is template boilerplate, not an identifier."""
    d = (doi or "").strip().lower()
    if not d:
        return False
    return bool(_PLACEHOLDER_DOI_RE.match(d))
