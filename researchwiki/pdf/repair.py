"""Repair of pypdfium2 text-extraction noise: ligature dropouts and soft hyphens.

Split out of `pdf/text.py`, which is about getting bytes out of a PDF; this
module is about the two ways those bytes come out wrong. The split is not
cosmetic — the repair rules carry this code's real judgement, and they are what
a reader needs to be able to find and argue with.

`repair_text` is the whole public surface. Everything else is the reasoning
behind it: which fragment counts as damage, and which "fix" would be a
fabrication.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- PDF-extraction noise repair ---------------------------------------------
# Two failure modes show up in pypdfium2 output:
#
# 1. Ligature dropouts. Some PDFs ship Type 1 / Type 3 fonts that include
#    ligature glyphs (`fi`, `ff`, `ffi`, `ffl`, `fl`, `ft`, `tt`) but no
#    `ToUnicode` CMap, or a CMap that omits the ligature CIDs. PDFium falls
#    back to the raw CID, which usually lands in the C1 control range
#    (U+0080–U+009F). The result is words like `effective`, `efficient`,
#    `difference`, `off-target` coming out with a single non-printing byte
#    where the ligature used to be:
#
#        'e<\x82>ective'  →  effective
#        'di<\x85>erence' →  difference
#        'o<\x97>-target' →  off-target
#
#    The byte value isn't stable — different fonts use different CIDs for the
#    same ligature, sometimes within one paper. What's stable is the
#    *surrounding letters*: only one ligature substitution yields a valid
#    English word. We exploit that with a dictionary lookup over a bundled
#    30K-word frequency-sorted list.
#
#    Rarer cousin: the ligature is dropped entirely with no replacement byte,
#    producing words like `ne-grained` (was `fine-grained`). Same dictionary
#    trick works — try inserting each ligature at each gap, accept the unique
#    fix that resolves to a known word.
#
# 2. Soft-hyphen / line-break-hyphenation artifacts. PDFs hyphenate at line
#    ends; some encodings preserve the soft hyphen as a C0 control char
#    (most commonly U+0002 STX, also U+0012, U+001E, U+001F). The corpus
#    shows 7,500+ occurrences of these between letter clusters — by far the
#    largest class of noise. Stripped unconditionally between two letters.
#
# The dictionary is loaded lazily on first repair call and cached in a
# module-global, so paths that don't extract PDFs (`status`, `search`,
# `claims`, etc.) don't pay the ~1.7 MB resident cost.

_C1 = r"[\x80-\x9f]"
_C1_RE = re.compile(_C1)
# A "word" candidate for repair: ASCII letters + apostrophes + hyphens, with
# at least one C1 control byte embedded. We skip words without C1 since the
# dictionary check on a clean word would just say "yes" and do nothing.
_C1_WORD_RE = re.compile(rf"[A-Za-z]*{_C1}[A-Za-z'-]*")

# Mode-B candidates: clean words with no C1 byte that *might* be missing a
# ligature. We require the word boundary on both sides to be free of C1
# bytes — otherwise we'd "repair" fragments left behind when Mode A failed
# to fix a C1 word, e.g. `pre\x93lter` → fragment `lter` → "filter" but the
# `\x93` is still there. Capital first letter is allowed because legitimate
# damage shows up at sentence starts (`Dierent` was `Different`); the
# dictionary check rejects author surnames like `Gurevych` naturally.
# Bound work by only probing 3-15-letter words.
_BARE_WORD_RE = re.compile(r"(?<![\x80-\x9f])\b[A-Za-z][A-Za-z'-]{2,14}\b(?![\x80-\x9f])")

# Candidate ligatures, ordered by typical frequency in English. When two
# candidates both yield real words, the first one wins (mostly cosmetic;
# ambiguity is rare in practice).
_LIGATURES = ("fi", "ff", "ffi", "fl", "ffl", "ft", "tt")

# Soft-hyphen / line-break artifacts: any C0 control char (except \t\n\r)
# sitting between two letters is treated as an elidable soft hyphen.
_INTERIOR_CTRL_RE = re.compile(
    r"(?<=[A-Za-z])[\x00-\x08\x0b\x0c\x0e-\x1f](?=[A-Za-z])"
)

_DICTIONARY: frozenset[str] | None = None
_DICTIONARY_PATH = Path(__file__).resolve().parent / "data" / "english_words.txt"


def _load_dictionary() -> frozenset[str]:
    """Lazy-load the bundled wordlist. Returns empty set if missing (graceful
    degradation: ligature repair becomes a no-op rather than crashing on a
    fresh checkout that hasn't built the wordlist yet)."""
    global _DICTIONARY
    if _DICTIONARY is None:
        try:
            words = _DICTIONARY_PATH.read_text(encoding="utf-8").split()
            _DICTIONARY = frozenset(w.lower() for w in words if w)
        except FileNotFoundError:
            _DICTIONARY = frozenset()
    return _DICTIONARY


def _is_known_word(candidate: str, dictionary: frozenset[str]) -> bool:
    """Whole-word lookup with hyphen-split fallback.

    The bundled list is unigrams (no hyphenated entries), so for hyphenated
    candidates like `off-target` or `fine-grained` we accept the candidate
    iff every hyphen-separated part is itself a dictionary word. Apostrophes
    are stripped (English contractions: `don't` → `dont` not in dict, but
    `don` is — kept simple by lookup of the leading part).
    """
    cand = candidate.lower()
    if cand in dictionary:
        return True
    if "-" in cand:
        parts = [p for p in cand.split("-") if p]
        return bool(parts) and all(p in dictionary for p in parts)
    return False


def _repair_c1_word(word: str, dictionary: frozenset[str]) -> str:
    """Try each ligature in turn at each C1 byte; accept the first
    substitution that yields a dictionary word. If none does, return the
    original (possibly with multiple C1 bytes still in it)."""
    # Multiple C1 bytes in one word (e.g. `Co\x83\x83dence`): collapse runs
    # of consecutive C1 bytes into one slot. This handles the common case
    # where the same byte represents one ligature; rare cases where two
    # adjacent unmapped CIDs represent different things (e.g. `\x83` = `n`
    # and `\x83` = `fi` in `Confidence`) are out of scope.
    collapsed = re.sub(rf"{_C1}+", "\x00", word)
    if "\x00" not in collapsed:
        return word
    parts = collapsed.split("\x00")
    if len(parts) - 1 == 1:
        prefix, suffix = parts
        for lig in _LIGATURES:
            candidate = prefix + lig + suffix
            if _is_known_word(candidate, dictionary):
                return candidate
    elif len(parts) - 1 == 2:
        a, b, c = parts
        for lig1 in _LIGATURES:
            for lig2 in _LIGATURES:
                candidate = a + lig1 + b + lig2 + c
                if _is_known_word(candidate, dictionary):
                    return candidate
    # 3+ gaps: don't try; combinatorial blow-up and almost never seen.
    return word


def _repair_bare_word(word: str, dictionary: frozenset[str]) -> str:
    """For a word *without* C1 bytes that's missing from the dictionary, try
    inserting each ligature at each gap. Accept only if exactly one
    insertion produces a dictionary word — otherwise leave alone."""
    if _is_known_word(word, dictionary):
        return word
    matches: list[str] = []
    for i in range(len(word) + 1):
        for lig in _LIGATURES:
            candidate = word[:i] + lig + word[i:]
            if _is_known_word(candidate, dictionary):
                matches.append(candidate)
                if len(matches) > 1:
                    return word  # ambiguous, bail
    return matches[0] if len(matches) == 1 else word


def repair_text(text: str) -> str:
    """Repair PDF-extraction noise: soft-hyphen joins and ligature dropouts.

    Two passes: (1) elide soft-hyphen control bytes between letters, then
    (2) for each word containing a C1 byte, try the ligature substitutions
    against the bundled English wordlist and accept the unique fix.

    Mode-B repair (whole-word drops with no C1 byte) is bounded — we only
    probe lowercase words 3-11 letters long that fail dictionary lookup,
    which keeps the work proportional to actual damage. Ambiguous matches
    (two ligatures both yield valid words) are left alone rather than
    guessed at; the LLM downstream of extraction tolerates a few damaged
    words better than it tolerates a confidently-wrong repair.
    """
    text = _INTERIOR_CTRL_RE.sub("", text)

    dictionary = _load_dictionary()
    if not dictionary:
        return text  # graceful degradation: no wordlist, no repair

    # Mode A: words containing C1 bytes
    text = _C1_WORD_RE.sub(lambda m: _repair_c1_word(m.group(), dictionary), text)
    # Mode B: bare lowercase words that fail dictionary lookup
    text = _BARE_WORD_RE.sub(lambda m: _repair_bare_word(m.group(), dictionary), text)
    return text
