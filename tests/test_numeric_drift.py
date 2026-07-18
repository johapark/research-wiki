"""Numeric-drift detection — the grader's hallucinated-number check.

`_check_numerics` is the anti-fabrication guard: a number in a claim that
appears nowhere in the source PDF is "drift" and blocks auto-promote. The
normalization (`_normalize_numeric`) exists to suppress false positives from
thousands-separators and trailing-zero rounding. Both are pure and worth
pinning precisely — a regression here silently lets fabricated numbers through
(false negative) or rejects faithful pages (false positive).
"""

from researchwiki.grade.fidelity.paper import _check_numerics, _normalize_numeric


# ---------- _normalize_numeric ----------

def test_normalize_thousands_separator():
    assert _normalize_numeric("8,081") == "8081"


def test_normalize_trailing_zero_decimal():
    assert _normalize_numeric("8081.0") == "8081"


def test_normalize_keeps_significant_decimal():
    assert _normalize_numeric("8081.50") == "8081.5"


def test_normalize_strips_unit_char():
    assert _normalize_numeric("12%") == "12"


def test_normalize_trailing_dot():
    assert _normalize_numeric("8081.") == "8081"


def test_normalize_combined():
    assert _normalize_numeric("8,081.0") == "8081"


# ---------- _check_numerics ----------

def test_verbatim_match_no_drift():
    tokens, unmatched = _check_numerics(
        "we found 8081 variants", "table shows 8081 variants", "full text 8081"
    )
    assert tokens == ["8081"]
    assert unmatched == []


def test_hallucinated_number_is_unmatched():
    _, unmatched = _check_numerics(
        "the model hit 9999 accuracy", "nothing relevant here", "still nothing"
    )
    assert unmatched == ["9999"]


def test_normalized_match_suppresses_false_positive():
    # Claim rounds "8,081" where the PDF prints "8081" — must NOT be drift.
    _, unmatched = _check_numerics("8,081 cells", "the assay used 8081 cells", "8081")
    assert unmatched == []


def test_full_pdf_fallback_when_retrieval_misses():
    # Retrieved chunk lacks the number but the full PDF has it → not drift.
    _, unmatched = _check_numerics(
        "the answer is 42", "an unrelated retrieved passage", "deep thought returned 42"
    )
    assert unmatched == []


def test_no_numbers_returns_empty():
    assert _check_numerics("a purely qualitative claim", "anything", "anything") == ([], [])


# ---------- substring false-match regression (review finding T1.3) ----------

def test_bare_number_not_matched_as_substring_of_larger_number():
    # Claim "8" must NOT be satisfied by a coincidental "128" in the evidence —
    # substring matching would silently pass a fabricated number through the
    # synthesis anti-fabrication gate.
    _, unmatched = _check_numerics(
        "the model used 8 attention heads", "trained on 128 documents", "trained on 128 documents"
    )
    assert unmatched == ["8"]


def test_decimal_not_matched_as_substring():
    _, unmatched = _check_numerics("accuracy 0.5", "reached 10.53 percent", "")
    assert unmatched == ["0.5"]


def test_genuine_exact_number_still_matches():
    _, unmatched = _check_numerics("improves by 8%", "an 8% improvement", "")
    assert unmatched == []


# ---------- magnitude-suffix matching (review T1.3 follow-up) ----------

def test_magnitude_suffix_matches_full_number():
    # "350 K" must match a PDF's "350,000" (mantissa token vs full number).
    _, um = _check_numerics("~350 K sequences", "supplemented with 350,000 sequences", "")
    assert um == []


def test_magnitude_million_matches():
    _, um = _check_numerics("8.6 M training pairs", "8,600,000 pairs from BindingDB", "")
    assert um == []


def test_magnitude_lowercase_k():
    _, um = _check_numerics("440 k peaks", "trained on 440,000 peaks", "")
    assert um == []


def test_magnitude_does_not_reintroduce_substring_false_negative():
    # Fabrication still caught: "8" must not match "128".
    _, um = _check_numerics("the model used 8 attention heads", "trained on 128 documents", "")
    assert um == ["8"]


def test_magnitude_does_not_mask_real_rounding():
    # ">510,000" is a rounding of 510,495; the exact value differs, so still drift.
    _, um = _check_numerics("SILVA SSU (>510,000 taxa)", "SSU alignment 510,495 sequences", "")
    assert "510,000" in um


def test_gb_unit_not_expanded_to_billion():
    # "23 GB" tokenizes to "23" (GB breaks the boundary) — plain form matches.
    _, um = _check_numerics("using 23 GB of GPU memory", "224 GB and 23 GB of memory", "")
    assert um == []


def test_number_glued_to_letter_unit_is_extracted():
    # PDF writes "3.22Å" (no space); the value must still match a claim's "3.22".
    # (Pre-T1.3 the substring matcher hid this; the tokenizer must extract it.)
    _, um = _check_numerics("multipolar interaction at d = 3.22", "distance (d = 3.22Å) resolved", "")
    assert um == []
    _, um = _check_numerics("128nm resolution", "imaged at 128nm depth", "")
    assert um == []
