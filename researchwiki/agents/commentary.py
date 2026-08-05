"""Commentary-shaped-PDF guard — refuse to auto-promote a Research Highlight
as if it were the paper it describes.

## The failure this exists for

`wiki/single-cell/li-2026-quantifying-gene-importance-with-signature.md` was a
one-page Nature Genetics **Research highlight** about a *different* paper
(`gold-2026-scoring-gene-importance-by-interpreting`, 32 pages, Nature
Biotechnology). It was auto-promoted as `type: paper`, and claim extraction
then produced 19 claims that credited Gold et al.'s contributions to the
highlight's author — "Introduces SIGnature, a framework…" and so on.

Both fidelity gates passed, and they were right to: the claims *are* faithfully
present in the highlight's PDF. `grade synthesis` and `check-grounding` check
**faithfulness**, not **entitlement**. Nothing downstream of promotion can
recover the distinction, because by then the page asserts the wrong authorship
in its own frontmatter.

## Why upstream metadata cannot fix it

Every authority agrees the highlight is an ordinary article:

| Source   | Field           | Value for the highlight | Value for the 32-page primary |
|----------|-----------------|-------------------------|-------------------------------|
| Crossref | `type`          | `journal-article`       | `journal-article`             |
| Crossref | `subtype`       | `None`                  | `None`                        |
| PubMed   | `pubtype`       | `['Journal Article']`   | `['Journal Article']`         |

So a type lookup is a dead end. What *does* separate them is local and
structural: the highlight is one page, deposits zero references, and prints
`Research highlights` as a section label in its masthead.

## The rule

High precision is the explicit design target — a false positive blocks a real
paper from the wiki, which is far worse than missing a highlight that a human
then notices. So no single weak signal fires, and **page count alone never
fires**: genuine one-page Correspondence, Matters Arising, and short Reports
exist and belong in the wiki as papers.

A PDF is commentary-shaped when ANY of:

1. **A strong section label** appears in the masthead zone (first
   `_MASTHEAD_CHARS` characters) of page 1 — `Research Highlight(s)`,
   `News & Views` / `News and Views`. These are journal *section* names, not
   phrases a research article's own masthead contains.
1b. **A publisher DOI namespace reserved for non-research content** —
   Springer Nature assigns `10.1038/d<5 digits>-…` to news, News & Views,
   Editorials and features, and `10.1038/s…` to research articles. The
   namespace is the publisher's own type declaration, so it is strong.

   This tier exists because tier 1 has a real blind spot, found on
   `koralov-2026-hard-to-detect-mutations-in-autoimmune-diseases`: a 2-page
   Nature News & Views whose page 1 opens directly on body prose ("Mutations
   that are acquired throughout life…"), printing no section label in the
   masthead zone at all. Two pages defeats the structural pair too, so every
   text- and count-based tier missed it while the DOI said `d41586` outright.
   Verified on the corpus: 4 DOIs match `10.1038/d`, all four non-research
   (the two mistyped News & Views plus two already-correct `whitepaper` news
   features); zero research articles match.
2. **The structural pair**: `page_count == 1` AND `reference_count == 0`. A
   real one-page paper still cites prior work; a highlight deposits nothing.
3. **A weak label plus corroboration**: `Comment(ary)` /
   `Correspondence` / `Books & Arts` standing alone on its own line, AND at
   least one of `page_count == 1`, `reference_count == 0`, or a single-page
   Crossref extent (`"1458-1458"`). The weak labels are line-anchored and
   never fire on their own, because `Comment` appears in
   legitimate contexts (a `Comment` heading inside a
   longer piece).

`reference_count is None` (Crossref never asked, or DOI unknown) means "no
signal" and is never read as zero — the guard must not convert a cache miss
into a block.

## Where this runs

`phases.reconcile_metadata` calls `detect_commentary` once, stashes the verdict
in the metadata dict (`commentary_signals`, `page_type`), and every downstream
consumer reads from there — so the decision is computed once per ingest, is
idempotent, and costs no LLM call. `promote.should_auto_promote` turns a
non-empty signal list into a gate failure naming each signal that fired;
`promote._build_frontmatter` writes `type: commentary` instead of `type: paper`
so that even a `--auto-promote` override lands a correctly-typed page (and
`db rebuild` skips claim extraction for any non-`paper` type, which is what
stops the 19 misattributed claims at the source).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Section labels sit in the masthead / running-head zone at the very top of
# page 1. Bounding the strong-label search to this window is what keeps a body
# sentence ("a recent News & Views argued…") from tripping the guard.
_MASTHEAD_CHARS = 1200

# The page type a commentary-shaped PDF should land as.
COMMENTARY_PAGE_TYPE = "commentary"

# Strong labels — sufficient on their own inside the masthead zone. Multi-word
# patterns use `\s+` so a label typeset across two lines still matches.
_STRONG_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("research-highlight", re.compile(r"research\s+highlights?\b", re.IGNORECASE)),
    ("news-and-views", re.compile(r"news\s*(?:&|and)\s*views\b", re.IGNORECASE)),
    # `Editorial` alone on a line, inside the masthead zone. Promoted from the
    # weak tier after `editorial-2026-circling-back-to-rna-vaccines` (a 2-page
    # Nature Biotechnology Editorial) proved undetectable otherwise: it deposits
    # **9 references** and spans pages 673–674, so no structural signal can ever
    # corroborate a weak label for it, and Crossref calls it `journal-article`.
    #
    # Safe here for the same reason the other two are: the masthead zone is
    # where journals print the section name. The weak tier searched the *whole*
    # page, where `editorial board` and a `Editorial` line inside a longer
    # piece live; scoping to the masthead is what removes those. Verified across
    # all 401 corpus PDFs: 2 matches, both genuine editorials, zero research
    # articles.
    ("editorial", re.compile(r"^[\s|]*editorials?[\s|]*$", re.IGNORECASE | re.MULTILINE)),
)

# Weak labels — only count when corroborated by a structural signal. Anchored
# to a whole line so `editorial board` and mid-sentence `comment` don't match.
#
# `Correspondence` was tried here and REMOVED. Cell Press typesets an author-
# contact block on page 1 whose label is the bare word `Correspondence` on its
# own line, so the pattern matched 8 genuine research articles in the corpus —
# including 17-, 21-, 27-, 33- and 39-page primaries. No line-anchoring tweak
# separates that block from a Correspondence *section* label, and a false
# positive here blocks a real paper, so the label is not worth its recall.
_WEAK_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("comment", re.compile(r"^[\s|]*comment(?:ary|s)?[\s|]*$", re.IGNORECASE | re.MULTILINE)),
    ("books-and-arts", re.compile(r"^[\s|]*books?\s*(?:&|and)\s*arts[\s|]*$", re.IGNORECASE | re.MULTILINE)),
    ("in-this-issue", re.compile(r"^[\s|]*in\s+this\s+issue[\s|]*$", re.IGNORECASE | re.MULTILINE)),
)

# A Crossref `page` extent covering exactly one page — an EXPLICIT same-page
# range like "1458-1458" or "e123-e123".
#
# A bare value ("1458") is deliberately NOT a signal even though it usually
# does mean one page: Cell Press and other article-number publishers put the
# article number in this field, so `page: "100762"` on a 17-page Cell Genomics
# paper would otherwise read as a one-page extent. Requiring the repeated
# endpoint costs a little recall on journals that abbreviate single pages and
# buys immunity to the whole article-number family.
_SINGLE_PAGE_EXTENT_RE = re.compile(r"^\s*([A-Za-z]?\d+)\s*[-–—]\s*\1\s*$")

# Publisher DOI namespaces reserved for non-research content. Springer Nature
# splits its own output by DOI prefix: research articles get `10.1038/s…`
# (`s41586` Nature, `s41588` Nature Genetics, …) while news, News & Views,
# Editorials and features get `10.1038/d…` (`d41586` for Nature). Matching the
# whole `d<digits>` family rather than hardcoding `d41586` covers the sibling
# journals' news namespaces without needing to enumerate them.
#
# Strong enough to fire alone because it is the publisher's own type
# declaration, not an inference from layout. Scoped to prefixes we have
# verified: do NOT generalize to "any DOI with a letter after the slash" —
# other registrants use letter-prefixed suffixes for ordinary articles.
_NEWS_DOI_RE = re.compile(r"^\s*(?:https?://(?:dx\.)?doi\.org/)?10\.1038/d\d{5,6}-", re.IGNORECASE)


def is_news_namespace_doi(doi: str | None) -> bool:
    """True when the DOI sits in a publisher namespace reserved for non-research
    content (see `_NEWS_DOI_RE`). Tolerates a full `doi.org/` URL form."""
    if not doi:
        return False
    return bool(_NEWS_DOI_RE.match(doi))


@dataclass
class CommentaryVerdict:
    """Outcome of the guard. `signals` is the observable record — every fired
    signal is named, in the order the rule evaluates them, so the log line and
    the gate-failure reason say *why* rather than just *no*."""

    is_commentary: bool = False
    signals: list[str] = field(default_factory=list)
    page_type: str | None = None        # COMMENTARY_PAGE_TYPE, or None
    considered: list[str] = field(default_factory=list)  # signals seen but not sufficient

    def reason(self) -> str:
        """One-line, actionable gate-failure text."""
        return gate_reason(self.signals)


def gate_reason(signals: list[str]) -> str:
    """The promotion-gate failure string for a commentary-shaped PDF.

    Module-level so `promote.should_auto_promote` can render it from a bare
    signal list (which is all the reconcile metadata carries) without
    reconstructing a verdict, while keeping one definition of the wording.
    """
    return (
        f"commentary-shaped PDF (signals: {', '.join(signals)}) — "
        f"looks like a Research Highlight / News & Views about another paper, "
        f"not the paper itself. Suggested type: {COMMENTARY_PAGE_TYPE}. "
        f"Re-run with --auto-promote to override, or ingest the primary paper "
        f"instead."
    )


def is_single_page_extent(page_field: str | None) -> bool:
    """True when a Crossref `page` value covers exactly one page."""
    if not page_field:
        return False
    return bool(_SINGLE_PAGE_EXTENT_RE.match(page_field))


def crossref_lookup_worthwhile(
    first_page_text: str | None, page_count: int | None
) -> bool:
    """Should the caller spend a Crossref request to complete the signal set?

    True only when a purely-local pre-trigger already fired: the document is a
    single page, or page 1 carries a strong/weak commentary label. An ordinary
    multi-page research article answers False, so the guard adds **zero**
    network calls on the common path — which is the "cheap" half of the design
    contract. On the suspicious minority it buys `reference-count` and `page`,
    the two fields that make the structural tiers decidable.
    """
    if page_count == 1:
        return True
    head = first_page_text or ""
    masthead = head[:_MASTHEAD_CHARS]
    if any(pat.search(masthead) for _, pat in _STRONG_LABELS):
        return True
    return any(pat.search(head) for _, pat in _WEAK_LABELS)


def detect_commentary(
    *,
    first_page_text: str | None,
    page_count: int | None = None,
    reference_count: int | None = None,
    crossref_page: str | None = None,
    doi: str | None = None,
) -> CommentaryVerdict:
    """Classify a PDF as commentary-shaped or not. See the module docstring for
    the rule and the reasoning behind each tier.

    All inputs are optional and independently degradable: missing page count,
    missing Crossref, or unreadable first page each remove signals rather than
    forcing a verdict. With no inputs at all the answer is "not commentary".
    """
    head = first_page_text or ""
    masthead = head[:_MASTHEAD_CHARS]

    strong = [f"label:{name}" for name, pat in _STRONG_LABELS if pat.search(masthead)]
    # Tier 1b — publisher's own namespace declaration. Sufficient alone, like a
    # strong label, but named as its own signal kind: calling it `label:` would
    # tell a reader of the gate reason to go looking for text on page 1 that
    # isn't there, which is exactly the case this tier exists to cover.
    if is_news_namespace_doi(doi):
        strong.append("doi-news-namespace")
    weak = [name for name, pat in _WEAK_LABELS if pat.search(head)]

    one_page = page_count == 1
    # `is None` guard is load-bearing: a Crossref cache miss must not read as 0.
    zero_refs = reference_count == 0
    single_extent = is_single_page_extent(crossref_page)

    structural: list[str] = []
    if one_page:
        structural.append("page-count-1")
    if zero_refs:
        structural.append("reference-count-0")
    if single_extent:
        structural.append(f"single-page-extent-{crossref_page}")

    fired: list[str] = []
    # Tier 1 — strong masthead label, sufficient alone.
    fired.extend(strong)  # already prefixed at detection (`label:…` / `doi-…`)
    # Tier 2 — the structural pair. Page count alone is deliberately not enough.
    structural_pair = one_page and zero_refs
    # Tier 3 — weak label, corroborated by any structural signal.
    corroborated_weak = bool(weak) and bool(structural)

    if structural_pair or corroborated_weak:
        if corroborated_weak:
            fired.extend(f"label:{name}" for name in weak)
        fired.extend(structural)
    elif strong:
        # Strong label carried the verdict; still record structure we saw, so
        # the log shows the full picture rather than one lonely signal.
        fired.extend(structural)

    if not fired:
        # Nothing sufficient. Report what we noticed anyway (`considered`) so a
        # near-miss is diagnosable without re-running with a debugger.
        return CommentaryVerdict(
            is_commentary=False,
            signals=[],
            page_type=None,
            considered=[f"label:{n}" for n in weak] + structural,
        )

    # Preserve first-seen order while dropping duplicates.
    seen: set[str] = set()
    ordered = [s for s in fired if not (s in seen or seen.add(s))]
    return CommentaryVerdict(
        is_commentary=True,
        signals=ordered,
        page_type=COMMENTARY_PAGE_TYPE,
        considered=[],
    )
