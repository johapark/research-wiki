"""Style-fitness metrics for wiki pages — compression and extractiveness.

Two page-level diagnostics from SEI's evaluation framework, surfaced as a
sixth informational pane in benchmark-fixture (alongside the five fixture-
anchored axes):

  - compression: page-tokens / paper-tokens. Catches "too short to carry
    the paper's content" and "padded with verbose abstract paraphrasing."
  - extractiveness: fraction of page sentences containing a verbatim
    ≥10-word span from the PDF. Catches "heavy paraphrase, drift risk"
    on the low end and "cargo-culted excerpts" on the high end.

These don't roll into `overall_weighted_recall` — they're orthogonal to
the fixture-coverage question. A page can hit 95% recall AND be too
extractive (recall + drift risk) or too compressed (recall + missing
context). Reported as their own pane via `--with-style`.

The thresholds are heuristic and empirical, calibrated against the
committed wiki pages in this repo:
  - compression: page-tokens / paper-tokens. <1% → too compressed
    (incomplete); >30% → too verbose (padded). Section word caps
    constrain the upper end naturally — wiki pages typically run
    1500–2000 tokens regardless of paper length, so for a long paper
    the ratio is small (~5%) and for a Brief Communication it can hit
    20–25%, both fine.
  - extractiveness: <5% → heavy paraphrase (drift risk); >40% → cargo-
    cult (low synthesis). Most committed pages sit at 10–25%.

Implementation detail: extractiveness uses a 10-gram set lookup over the
PDF text — O(n) build, O(1) per check — so a paper with 50K tokens and
a page with 30 sentences runs in ~50ms.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


# Match latin/greek alphabetic word tokens with internal hyphens / digits.
# Used by both the compression count and the extractiveness n-gram build.
_TOKEN_RE = re.compile(r"[A-Za-zα-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# n-gram width for the extractiveness verbatim check. 10 is the SEI
# default — short enough to catch genuine quotation, long enough to avoid
# spurious matches on common phrasing.
NGRAM_N = 10

# Compression thresholds (page tokens / paper tokens).
# Calibrated against committed wiki pages: typical compression ratio is
# 5–25%, so a tight band would flag normal pages as "verbose." The flags
# fire only when the page is clearly underwritten (<1%) or clearly padded
# (>30%) — the latter is rare given the section word caps.
COMPRESSION_LOW = 0.01    # <1%:  too compressed (incomplete)
COMPRESSION_HIGH = 0.30   # >30%: too verbose (padded)
# Independent absolute floor: a page below this many tokens is almost
# certainly incomplete regardless of paper length.
COMPRESSION_MIN_PAGE_TOKENS = 300

# Extractiveness thresholds (fraction of page sentences with verbatim span).
EXTRACTIVENESS_LOW = 0.05   # <5%:  heavy paraphrase, drift risk
EXTRACTIVENESS_HIGH = 0.40  # >40%: cargo-culted excerpts


@dataclass
class StyleReport:
    page_tokens: int
    paper_tokens: int
    compression_ratio: float
    compression_verdict: str        # "compressed" | "normal" | "verbose"
    n_page_sentences: int           # sentences ≥ NGRAM_N tokens long
    n_extractive_sentences: int     # of those, how many have a verbatim span
    extractiveness_fraction: float
    extractiveness_verdict: str     # "paraphrased" | "normal" | "extractive"

    def to_dict(self) -> dict:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _ngram_set(text: str, n: int = NGRAM_N) -> set[str]:
    """Build the set of lowercased n-grams from a body of text. Uses
    space-joined token tuples as the dictionary key. Repeated n-grams
    collapse — set semantics is correct for "did this span appear at
    all" lookups."""
    tokens = [t.lower() for t in _tokenize(text)]
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def compute_style(page_body: str, paper_text: str) -> StyleReport:
    """Compute page-level style metrics. Caller passes the page markdown
    body and the full PDF text (already extracted; same source the agent
    uses for L1's `pdf_full_text`).

    Returns a StyleReport with both compression and extractiveness, each
    classified into a verdict tier."""
    page_tokens_list = _tokenize(page_body)
    paper_tokens_list = _tokenize(paper_text)
    page_n = len(page_tokens_list)
    paper_n = len(paper_tokens_list)

    if paper_n == 0:
        # Defensive: empty PDF. Treat as missing data rather than divide-by-zero.
        return StyleReport(
            page_tokens=page_n, paper_tokens=0,
            compression_ratio=0.0, compression_verdict="unknown",
            n_page_sentences=0, n_extractive_sentences=0,
            extractiveness_fraction=0.0, extractiveness_verdict="unknown",
        )

    ratio = page_n / paper_n
    if ratio < COMPRESSION_LOW or page_n < COMPRESSION_MIN_PAGE_TOKENS:
        comp_verdict = "compressed"
    elif ratio > COMPRESSION_HIGH:
        comp_verdict = "verbose"
    else:
        comp_verdict = "normal"

    # Extractiveness — build the paper's n-gram set once, then for each
    # page sentence with ≥ NGRAM_N tokens, check whether ANY of its
    # candidate n-grams matches.
    paper_ngrams = _ngram_set(paper_text, NGRAM_N)
    page_lower = page_body.lower()
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(page_lower) if s.strip()]
    eligible_sentences = []
    extractive_count = 0
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if len(sent_tokens) < NGRAM_N:
            continue
        eligible_sentences.append(sent)
        sent_lower = [t.lower() for t in sent_tokens]
        # Check each n-gram in the sentence against the paper's set.
        for i in range(len(sent_lower) - NGRAM_N + 1):
            span = " ".join(sent_lower[i:i + NGRAM_N])
            if span in paper_ngrams:
                extractive_count += 1
                break

    n_eligible = len(eligible_sentences)
    fraction = (extractive_count / n_eligible) if n_eligible else 0.0
    if n_eligible == 0:
        ext_verdict = "unknown"
    elif fraction < EXTRACTIVENESS_LOW:
        ext_verdict = "paraphrased"
    elif fraction > EXTRACTIVENESS_HIGH:
        ext_verdict = "extractive"
    else:
        ext_verdict = "normal"

    return StyleReport(
        page_tokens=page_n,
        paper_tokens=paper_n,
        compression_ratio=ratio,
        compression_verdict=comp_verdict,
        n_page_sentences=n_eligible,
        n_extractive_sentences=extractive_count,
        extractiveness_fraction=fraction,
        extractiveness_verdict=ext_verdict,
    )
