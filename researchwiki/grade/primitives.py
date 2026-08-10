"""Stem-agnostic claim-grading primitives.

Shared by the paper-page fidelity grader (`fidelity/paper.py`, one claim →
the paper's own PDF) and the synthesis fidelity grader
(`fidelity/synthesis.py`, one claim → the PDF(s) of the paper(s) it cites).
Retrieval lives in the callers; these are pure functions over
(claim_text, evidence_text).

Two deterministic signals:

  - numeric integrity  — every numeric token in the claim must appear in the
    evidence (verbatim or numerically-equivalent after normalization), else
    flagged. The anti-fabrication / anti-misattribution guard.
  - negation parity    — the claim asserts a negation the evidence doesn't
    echo. Coarse (lexicon-based); a soft signal, never an auto-fail.

`fidelity/paper.py` re-exports these under their historical `_`-prefixed
names so its internal call sites and the existing unit tests keep working
unchanged.

The module name is `primitives.py` (was `scoring.py`) to disambiguate from
`grade/scorer.py` — the fixture-based scorer is distinct from these
deterministic primitives, even though the historical name suggested
otherwise.
"""

from __future__ import annotations

import re


# A numeric mantissa (with thousands separators / decimals) plus an optional
# magnitude suffix (K/M/B/G, one space allowed, not glued to a longer word).
# No trailing word-boundary/unit requirement: a unit stuck to the number
# ("3.22Å", "128nm") must not prevent extraction of the value — the old
# `…(?:×|x|%|…)?\b` form silently dropped numbers followed by an unlisted
# letter-unit, which the substring matcher used to paper over. The leading
# `(?<![\w.])` stops mid-number/glued-to-a-word matches ("562" in "K562",
# "8" in "128").
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s?[KkMmBbGg](?![A-Za-z]))?"
)

# Letter-prefixed *decimal* measurements, admitted to the evidence value-set only.
#
# The `(?<![\w.])` above is load-bearing and stays: without it "K562" contributes
# 562 and "GRCh38" contributes 38, and a fabricated claim could be satisfied by a
# cell line or a reference-build name. But it also hides values a paper only ever
# writes with a letter prefix. Phred/QV notation is the case that bit: Hansen 2026
# states "QV increased from Q63.1 for the v0.7 assembly to Q68.9", and 68.9 appears
# nowhere else in the paper unprefixed, so a correct page claiming "QV from 63.1 to
# 68.9" was reported as numeric drift and vetoed at promote time. 63.1 escaped only
# because that PDF happens to also say "The initial QV was 63.1" — i.e. whether the
# veto fired was luck. Observed 2026-08-10 on hansen-2026-a-complete-diploid-human-genome
# (38 Q-prefixed numerics in that PDF alone).
#
# The discriminator is the decimal point. Identifiers that must stay excluded are
# letter+integer (K562, HG002, chr2, GRCh38, rs45512696, T2T); measurement notation
# that should be admitted carries a fractional part (Q68.9, v1.1). Requiring
# `\.\d` keeps the whole identifier class out without enumerating prefixes.
#
# Evidence-side only, and additive: this can only grow the value-set, so it can
# only suppress false drift — it can never invalidate a match the plain tokenizer
# already made. The claim side is deliberately left alone. A page that writes
# "Q68.9" simply has that number unchecked (fails open, no false positive);
# admitting prefixed forms as *claim* tokens would instead create new ways to
# flag correct prose.
_PREFIXED_DECIMAL_RE = re.compile(
    r"(?<![\w.])[A-Za-z](\d[\d,]*\.\d+)(?![\w.])"
)

# Magnitude-suffix multipliers so prose "350 K" / "8.6 M" compares equal to a
# PDF's "350,000" / "8,600,000". A number's canonical value is added to its
# form-set *in addition to* its plain normalized form (never replacing it), so
# magnitude handling can only make matching more permissive — it can never drop
# a match that the plain form would have made.
_MAGNITUDE = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "g": 1_000_000_000}
_MAGNITUDE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?([KkMmBbGg])\b")

# Space-separated thousands (ISO 31-0 / SI style), which many European journals
# typeset instead of commas — often with a non-breaking or thin space.
# `NUMERIC_TOKEN_RE` allows `,` inside a number but not whitespace, so a PDF's
# "1 in 300 000" tokenized as two numbers (300, 000) and a page citing "300,000"
# was reported as numeric drift against a paper that states the figure plainly.
# Observed 2026-08-05 on an EAS pediatric-FH consensus PDF.
#
# Only groups of exactly three digits are joined, so this cannot glue unrelated
# numbers together: "2024" never starts a match (no 1-3 digit prefix is followed
# by space+3 digits), and "8 heads" is left alone. Applied as a haystack/claim
# pre-normalization rather than by loosening the tokenizer, which is shared with
# the paper-page grader and the fixture scorer.
# The trailing `(?!,\d)` guard is not optional. PDF extraction routinely glues a
# superscript citation marker to the preceding word, so a table row reads
# "Borzoi (2023)73 524,000" — and without the guard "73" is a valid 1-3 digit lead
# and "524" a valid 3-digit group, so the two joined into "73524,000" and the real
# figure 524,000 vanished from the evidence set, surfacing as a false
# MISATTRIBUTED on a page whose number was verifiably in the cited PDF. A
# following ",<digit>" proves the group was already comma-delimited, so it cannot
# also be space-delimited; a comma used as punctuation ("300 000, and") still
# joins, because that comma is not followed by a digit.
_SPACED_THOUSANDS_RE = re.compile(
    r"(?<![\d.])(\d{1,3})((?:[    ]\d{3})+)(?!\d)(?!,\d)"
)


def collapse_spaced_thousands(text: str) -> str:
    """Rewrite `300 000` / `1 158 017` to `300000` / `1158017`.

    Value-preserving and idempotent: it only removes separators inside a
    digit-group run, so it can make matching more permissive and never less.
    """
    if not text:
        return text
    return _SPACED_THOUSANDS_RE.sub(
        lambda m: m.group(1) + re.sub(r"[    ]", "", m.group(2)),
        text,
    )

# Negation lexicon. Catches the dominant contradiction failure mode: the page
# asserts a negation that the cited evidence doesn't echo. Deliberately coarse —
# recall over precision; callers treat it as a soft signal, not an auto-fail.
NEGATION_RE = re.compile(
    r"\b(not|no|never|without|cannot|can'?t|don'?t|doesn'?t|didn'?t|"
    r"won'?t|wouldn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|"
    r"fails?\s+to|failed\s+to|unable\s+to|lack\s+of|absence\s+of|absent)\b",
    re.IGNORECASE,
)


def normalize_numeric(s: str) -> str:
    """Strip thousands-separators and trailing zero-only decimal so tokens
    like '8081' / '8081.0' / '8,081' / '8,081.0' all compare equal. Also
    strips a single trailing punctuation/unit char so '8081.' matches '8081'."""
    bare = re.sub(r"[^\d.,]", "", s)
    bare = bare.replace(",", "")
    if "." in bare:
        # Strip trailing zeros after decimal; drop the decimal point if the
        # result has no fractional part. "8081.0" → "8081", "8081.50" →
        # "8081.5", "0.500" → "0.5". Skip when there's no integer part (".5"
        # stays ".5") to avoid losing information.
        intp, frac = bare.split(".", 1)
        frac = frac.rstrip("0")
        bare = intp if not frac else f"{intp}.{frac}"
    return bare


def numeric_forms(token: str) -> set[str]:
    """Canonical normalized form(s) of a numeric token, for value-based matching.

    Always includes the plain normalized form (thousands separators / trailing
    zeros stripped). When the token carries a magnitude suffix (K/M/B/G), also
    includes the expanded integer value, so "350 K" → {"350", "350000"} matches
    a PDF's "350,000" → {"350000"} while still not matching a bare "350".
    """
    forms: set[str] = set()
    base = normalize_numeric(token)
    if base:
        forms.add(base)
    m = _MAGNITUDE_RE.search(token)
    if m:
        try:
            val = float(m.group(1).replace(",", "")) * _MAGNITUDE[m.group(2).lower()]
        except ValueError:
            return forms
        forms.add(str(int(val)) if val == int(val) else normalize_numeric(repr(val)))
    return forms


def check_numerics(
    claim_text: str,
    retrieved_text: str,
    full_text: str,
) -> tuple[list[str], list[str]]:
    """Return (all_numeric_tokens, unmatched).

    A number is matched if it appears verbatim (with or without trailing unit)
    OR as a numerically-equivalent token (after normalizing thousands separators
    and trailing-zero decimals) in either `retrieved_text` OR `full_text`. The
    two haystacks let callers tune scope: both the paper-page grader and the
    synthesis grader pass the retrieved neighborhood as the first arg and the
    PDF's full text as the second (matched if near the retrieval *or* anywhere
    in the source). The synthesis grader runs this once per cited paper and
    treats a number as drift only when it is unmatched across *all* cited
    papers — catching a number ascribed to a paper that contains it nowhere.
    """
    # Normalize space-separated thousands on both sides before tokenizing, so a
    # page's "300,000" matches a PDF's "300 000". See collapse_spaced_thousands.
    claim_text = collapse_spaced_thousands(claim_text)
    retrieved_text = collapse_spaced_thousands(retrieved_text)
    full_text = collapse_spaced_thousands(full_text)
    tokens = NUMERIC_TOKEN_RE.findall(claim_text)
    if not tokens:
        return [], []
    # Evidence value-set: the canonical form(s) of every numeric token in the
    # retrieved chunks and the full text. Value-based (not substring): a claim
    # of "8 heads" won't match a PDF's "128 samples" (8 ≠ 128), and a rounded
    # "510,000" won't match "510,495" — but "350 K" matches "350,000".
    evidence_forms: set[str] = set()
    for t in NUMERIC_TOKEN_RE.findall(retrieved_text):
        evidence_forms |= numeric_forms(t)
    for t in NUMERIC_TOKEN_RE.findall(full_text):
        evidence_forms |= numeric_forms(t)
    # Values the source only ever writes letter-prefixed ("Q68.9"). See
    # _PREFIXED_DECIMAL_RE: additive, so this only ever suppresses false drift.
    for t in _PREFIXED_DECIMAL_RE.findall(retrieved_text):
        evidence_forms |= numeric_forms(t)
    for t in _PREFIXED_DECIMAL_RE.findall(full_text):
        evidence_forms |= numeric_forms(t)
    unmatched = []
    for t in tokens:
        if numeric_forms(t) & evidence_forms:
            continue
        unmatched.append(t)
    return tokens, unmatched


def negation_mismatch(claim_text: str, evidence_text: str) -> bool:
    """True iff the claim contains a negation token but the evidence contains
    none. One direction only (claim-negates, evidence-doesn't). False positives
    possible when the source negates via vocabulary outside the lexicon — a
    documented soft signal.
    """
    if not NEGATION_RE.search(claim_text):
        return False
    return not NEGATION_RE.search(evidence_text)
