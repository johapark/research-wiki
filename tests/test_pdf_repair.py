"""Ligature repair and soft-hyphen elision in PDF extraction.

The repair pass corrects two failure modes in pypdfium2 output:
  1. Ligature glyphs without ToUnicode mappings emerge as C1 control bytes
     in the C1 range (U+0080–U+009F). The byte value isn't stable across
     fonts; we repair by trying each candidate ligature and accepting the
     unique substitution that yields a dictionary word.
  2. Soft-hyphen / line-break artifacts emerge as low C0 control bytes
     between letter clusters. Always elidable.

Tests use synthetic damage strings — no PDF I/O — so they're hermetic.
"""

from researchwiki.pdf.text import _repair_ligatures


# --- Mode A: C1 byte repairs --------------------------------------------------

def test_c1_ff_in_effective():
    assert _repair_ligatures("e\x82ective") == "effective"


def test_c1_ffi_via_short_suffix():
    # Suffix `cient` (no leading `i`) means the ligature ate three characters.
    assert _repair_ligatures("e\x81cient") == "efficient"


def test_c1_ff_via_long_suffix():
    # Suffix `icient` keeps the `i` — only `ff` was dropped.
    assert _repair_ligatures("e\x82icient") == "efficient"


def test_c1_ffi_in_official():
    assert _repair_ligatures("o\x81cial") == "official"


def test_c1_fi_in_modified():
    assert _repair_ligatures("Modi\x92ed") == "Modified"


def test_c1_in_hyphenated_off_target():
    # `off-target` is hyphenated and not a single dictionary entry; the
    # hyphen-split fallback in `_is_known_word` accepts it because both
    # `off` and `target` are dictionary words.
    assert _repair_ligatures("o\x97-target") == "off-target"


def test_c1_fi_word_start():
    # Word-start C1 byte representing `fi` ligature.
    assert _repair_ligatures("signi\x80cant") == "significant"


def test_c1_in_url():
    assert _repair_ligatures("h\x8aps://example.com") == "https://example.com"


def test_c1_word_end():
    # Cutoff: C1 at the end of the word; pattern still matches the word.
    assert "cutoff" in _repair_ligatures("cuto\x82.")


def test_c1_unrepairable_left_alone():
    # `Khattab` (author name) gets emitted as `Kha\x8aab` because the `tt`
    # ligature glyph is unmapped in some fonts. `khattab` isn't in the top-30K
    # English wordlist, so the dictionary check rightly leaves it alone
    # rather than guessing wrong. Damage stays — the LLM downstream handles
    # this better than a confidently-wrong repair would.
    assert "\x8a" in _repair_ligatures("Kha\x8aab")


# --- Mode B: whole-word ligature drops (no replacement byte) ------------------

def test_modeb_finally():
    assert _repair_ligatures("nally we") == "finally we"


def test_modeb_findings():
    assert _repair_ligatures("ndings") == "findings"


# --- Hyphenated Mode B has too much ambiguity in 30K dict so we accept the leakage. ---


# --- Soft-hyphen elision ------------------------------------------------------

def test_soft_hyphen_stx_between_letters():
    # `\x02` is a common PDF soft-hyphen marker. Strip when it sits between
    # letters; line-break-hyphenated words rejoin to their canonical form.
    assert _repair_ligatures("orthogo\x02nal") == "orthogonal"


def test_soft_hyphen_unit_separator():
    assert _repair_ligatures("inter\x1fnal") == "internal"


def test_soft_hyphen_does_not_join_across_whitespace():
    # The control char must be flanked by letters on both sides; otherwise
    # leave it alone (it might be intentional structure).
    out = _repair_ligatures("foo \x02 bar")
    assert "\x02" in out


# --- False-positive guards ----------------------------------------------------

def test_clean_text_unchanged():
    text = "The quick brown fox jumps over the lazy dog."
    assert _repair_ligatures(text) == text


def test_author_surnames_unchanged():
    # These are common scientific surnames that look superficially like
    # damage (start with consonant clusters that COULD be missing `fi`).
    # The dictionary rejects them, so Mode B leaves them alone.
    for name in ["Gurevych", "Diercks", "Khattab"]:
        assert _repair_ligatures(name) == name


def test_short_words_left_alone():
    # Words shorter than 3 chars don't get probed (false-positive risk
    # outweighs catch rate). `is`, `or`, `it` stay as-is.
    assert _repair_ligatures("is or it") == "is or it"
