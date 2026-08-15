"""Ligature repair and soft-hyphen elision in PDF extraction.

The repair pass corrects two failure modes in pypdfium2 output:
  1. Ligature glyphs without ToUnicode mappings emerge as C1 control bytes
     in the C1 range (U+0080–U+009F). The byte value isn't stable across
     fonts; we repair by trying each candidate ligature and accepting the
     unique substitution that yields a dictionary word.
  2. Soft-hyphen / line-break artifacts emerge as low C0 control bytes
     between letter clusters. These stand for two different hyphens — the
     invisible one that broke a word across lines, and a compound's own that
     happened to land there — and the wordlist decides which.

The guards matter as much as the repairs: Mode B infers damage from a failed
dictionary lookup alone, so without them it rewrites `p-values` into
`ftp-values`. Both directions are asserted here.

Tests use synthetic damage strings — no PDF I/O — so they're hermetic.
"""

import pytest

from researchwiki.pdf.repair import repair_text


# --- Mode A: C1 byte repairs --------------------------------------------------

def test_c1_ff_in_effective():
    assert repair_text("e\x82ective") == "effective"


def test_c1_ffi_via_short_suffix():
    # Suffix `cient` (no leading `i`) means the ligature ate three characters.
    assert repair_text("e\x81cient") == "efficient"


def test_c1_ff_via_long_suffix():
    # Suffix `icient` keeps the `i` — only `ff` was dropped.
    assert repair_text("e\x82icient") == "efficient"


def test_c1_ffi_in_official():
    assert repair_text("o\x81cial") == "official"


def test_c1_fi_in_modified():
    assert repair_text("Modi\x92ed") == "Modified"


def test_c1_in_hyphenated_off_target():
    # `off-target` is hyphenated and not a single dictionary entry; the
    # hyphen-split fallback in `_is_known_word` accepts it because both
    # `off` and `target` are dictionary words.
    assert repair_text("o\x97-target") == "off-target"


def test_c1_fi_word_start():
    # Word-start C1 byte representing `fi` ligature.
    assert repair_text("signi\x80cant") == "significant"


def test_c1_in_url():
    assert repair_text("h\x8aps://example.com") == "https://example.com"


def test_c1_word_end():
    # Cutoff: C1 at the end of the word; pattern still matches the word.
    assert "cutoff" in repair_text("cuto\x82.")


def test_c1_unrepairable_left_alone():
    # `Khattab` (author name) gets emitted as `Kha\x8aab` because the `tt`
    # ligature glyph is unmapped in some fonts. `khattab` isn't in the top-30K
    # English wordlist, so the dictionary check rightly leaves it alone
    # rather than guessing wrong. Damage stays — the LLM downstream handles
    # this better than a confidently-wrong repair would.
    assert "\x8a" in repair_text("Kha\x8aab")


# --- Mode B: whole-word ligature drops (no replacement byte) ------------------

def test_modeb_finally():
    assert repair_text("nally we") == "finally we"


def test_modeb_findings():
    assert repair_text("ndings") == "findings"


def test_modeb_repairs_only_the_damaged_hyphen_part():
    # Each fragment is judged on its own: `ne` is damage, `grained` is not.
    assert repair_text("ne-grained") == "fine-grained"


def test_modeb_repairs_a_fragment_the_wordlist_happens_to_contain():
    # `rst` IS in the 30K list (rank 19463), so membership alone says "leave
    # it"; `first` at rank 56 is what identifies it as damage.
    assert repair_text("rst") == "first"
    assert repair_text("nal") == "final"


# --- Soft-hyphen resolution ---------------------------------------------------

def test_soft_hyphen_stx_between_letters():
    # `\x02` is a common PDF soft-hyphen marker. Strip when it sits between
    # letters; line-break-hyphenated words rejoin to their canonical form.
    assert repair_text("orthogo\x02nal") == "orthogonal"


def test_soft_hyphen_unit_separator():
    assert repair_text("inter\x1fnal") == "internal"


def test_soft_hyphen_does_not_join_across_whitespace():
    # The control char must be flanked by letters on both sides; otherwise
    # leave it alone (it might be intentional structure).
    out = repair_text("foo \x02 bar")
    assert "\x02" in out


@pytest.mark.parametrize("damaged,expected", [
    ("off\x02target", "off-target"),
    ("high\x02throughput", "high-throughput"),
    ("multi\x02modal", "multi-modal"),
    ("hyper\x02parameters", "hyper-parameters"),
    ("next\x02generation", "next-generation"),
])
def test_a_real_hyphen_at_a_line_break_survives(damaged, expected):
    """The byte stands for a hyphen the compound genuinely has, so eliding it
    welds two words. `off-target` → `offtarget` fired nine times in a 12-paper
    sample, in a corpus where that term is central: the chunk index then holds
    a token no reader's query matches."""
    assert repair_text(damaged) == expected


@pytest.mark.parametrize("damaged,expected", [
    ("con\x02text", "context"),
    ("there\x02fore", "therefore"),
    ("perfor\x02mance", "performance"),
    ("revolution\x02ized", "revolutionized"),
])
def test_a_line_break_hyphen_still_vanishes(damaged, expected):
    """The other half of the same decision — and `there`/`fore` are both words,
    so this is what pins the rule order."""
    assert repair_text(damaged) == expected


def test_a_line_break_that_also_dropped_a_ligature_welds_then_repairs():
    """`dif` + `cult` are both words, so the halves-are-words rule would strand
    `dif-cult`; the fragments have to be joined for Mode B to see `difcult`."""
    assert repair_text("dif\x02cult") == "difficult"


def test_a_truncated_abbreviation_does_not_earn_a_hyphen():
    """`technol` is in the wordlist only as list junk (rank 26462), which is
    too weak to prove `Bio-technol.` is a compound rather than a line break."""
    assert repair_text("Bio\x02technol.") == "Biotechnol."


# --- False-positive guards ----------------------------------------------------

def test_clean_text_unchanged():
    text = "The quick brown fox jumps over the lazy dog."
    assert repair_text(text) == text


def test_author_surnames_unchanged():
    # These are common scientific surnames that look superficially like
    # damage (start with consonant clusters that COULD be missing `fi`).
    # The dictionary rejects them, so Mode B leaves them alone.
    for name in ["Gurevych", "Diercks", "Khattab"]:
        assert repair_text(name) == name


def test_short_words_left_alone():
    # Words shorter than 3 chars don't get probed (false-positive risk
    # outweighs catch rate). `is`, `or`, `it` stay as-is.
    assert repair_text("is or it") == "is or it"


# --- Mode-B structural guards -------------------------------------------------
#
# Every case below is a real corruption measured over a 20-paper corpus sample
# before the guards existed: 166 of 187 Mode-B insertions landed on an acronym
# or a hyphenated term. Frequency cannot separate these — `ftp` (rank 3871) and
# `floor` (1446) are common words — so the guards are structural, and these
# tests are what pin them. The suite previously asserted only that repair FIRES,
# which is why none of this was caught.

@pytest.mark.parametrize("text", [
    "We report p-values below 0.05",     # -> ftp-values, 21x in the sample
    "P-values were adjusted",            # -> ftP-values
    "the p-arms of each chromosome",     # -> ftp-arms
])
def test_a_one_letter_hyphen_part_is_never_repaired(text):
    """`p` is a variable name. `ftp` and `values` both being words is what let
    the old whole-token rule rewrite the statistics out of a results section."""
    assert repair_text(text) == text


@pytest.mark.parametrize("text", [
    "The UNG gene encodes uracil-DNA glycosylase",   # -> flUNG
    "Cells on the OOR panel were excluded",          # -> flOOR, 44x in the sample
    "OOD detection on held-out data",                # -> flOOD
    "de-enrichment of HLA-U alleles",                # -> HLA-flU
    "an OS-inspired design",                         # -> OffS-inspired
    "kNN overlap was computed",
])
def test_an_acronym_is_never_repaired(text):
    """A dropped ligature removes a glyph; it does not change the case of the
    letters around it. An interior capital therefore means acronym or proper
    noun, not damage."""
    assert repair_text(text) == text


@pytest.mark.parametrize("text", [
    "results were re-produced independently",   # -> fire-produced
    "de-enrichment was observed",               # -> fide-enrichment
    "the “Ex-boyfriend” example",               # -> flEx-boyfriend
])
def test_an_english_affix_is_never_repaired(text):
    """A short leading particle is ordinary hyphenation, never a ligature."""
    assert repair_text(text) == text


def test_a_token_whose_other_parts_are_not_words_is_left_alone():
    """`Tz-TCO` is a reagent. A repair only makes sense when the rest of the
    token is already English — otherwise `fiTz-TCO` is as good a guess as any,
    which is exactly the problem."""
    assert repair_text("Tz-TCO ligation") == "Tz-TCO ligation"


def test_a_rare_capitalised_word_is_treated_as_a_name():
    """`Neff` is a surname and sits in the list's tail; the rare-word repair
    path must not rewrite it into `Ne` + a commoner reading."""
    assert repair_text("Neff et al.") == "Neff et al."


def test_scientific_vocabulary_survives_a_full_sentence():
    text = ("Chen et al. reported that SpCas9 and AsCas12a differ in PAM "
            "preference; kmers from Nanopore HiFi contigs were aligned with "
            "Bowtie and GATK, and p-values are reported for each ATAC-seq run.")
    assert repair_text(text) == text

