"""`eval classifier` accounting — abstention is not a prediction.

The bug this pins: `other` is *both* a real content category and the abstention
bucket, and `suggest_category_llm` expresses "I don't know" by returning
`category="other"`. The eval had no way to tell that apart from a deliberate
`other`, so it counted every abstention as an ordinary prediction and reported
**0% abstention on a run that abstained ten times** — a number whose label did
not describe what it measured.

`Suggestion.abstained` carries the decision the classifier already made, and the
eval now reports two different things that were previously conflated:

  - **placement** — where the paper ends up on disk. An abstention still files
    it under `other`, so it belongs in the confusion matrix under `other`, not
    hidden in a `·` column. Hiding it would under-report how `other` fills up.
  - **commitment** — whether the classifier named a category or declined.

An abstention on a paper that genuinely lives in `other` is therefore *correct
placement* and *a declined commitment* at the same time, which is exactly the
distinction the old single counter could not express.

Hermetic: the classifier is stubbed, no index build, no LLM.
"""

from __future__ import annotations

import pytest

from researchwiki.search import Suggestion


def _suggestion(category, confidence=0.9, abstained=False):
    return Suggestion(category=category, confidence=confidence,
                      top_3=[(category, 3)], strength="weak" if abstained else "strong",
                      abstained=abstained)


# ---------- the flag itself ----------

def test_suggestion_defaults_to_not_abstained():
    assert _suggestion("compbio").abstained is False


def test_low_confidence_llm_call_abstains(monkeypatch):
    """Below `LLM_ABSTAIN_THRESHOLD` the classifier routes to `other` — and now
    says so, instead of being indistinguishable from choosing `other`."""
    from researchwiki import search
    from researchwiki.agents import llm as agents_llm

    monkeypatch.setattr(search, "_is_valid_category", lambda c: c in {"compbio", "other"})
    monkeypatch.setattr(agents_llm, "call", lambda **kw: type(
        "R", (), {"text": '{"category": "compbio", "confidence": 0.3}'})())

    class _Backend:
        def more_like_text(self, seed, limit, page_type=None):
            return [type("H", (), {"category": "compbio", "key": "compbio/x",
                                   "title": "A neighbour"})()]

    out = search.suggest_category_llm(_Backend(), "T", "A")
    assert out.category == "other"
    assert out.abstained is True


def test_invalid_category_abstains(monkeypatch):
    from researchwiki import search
    from researchwiki.agents import llm as agents_llm

    monkeypatch.setattr(search, "_is_valid_category", lambda c: c == "compbio")
    monkeypatch.setattr(agents_llm, "call", lambda **kw: type(
        "R", (), {"text": '{"category": "astrology", "confidence": 0.99}'})())

    class _Backend:
        def more_like_text(self, seed, limit, page_type=None):
            return [type("H", (), {"category": "compbio", "key": "compbio/x",
                                   "title": "A neighbour"})()]

    out = search.suggest_category_llm(_Backend(), "T", "A")
    assert out.category == "other"
    assert out.abstained is True


def test_a_confident_call_does_not_abstain(monkeypatch):
    from researchwiki import search
    from researchwiki.agents import llm as agents_llm

    monkeypatch.setattr(search, "_is_valid_category", lambda c: c in {"compbio", "other"})
    monkeypatch.setattr(agents_llm, "call", lambda **kw: type(
        "R", (), {"text": '{"category": "compbio", "confidence": 0.95}'})())

    class _Backend:
        def more_like_text(self, seed, limit, page_type=None):
            return [type("H", (), {"category": "compbio", "key": "compbio/x",
                                   "title": "A neighbour"})()]

    out = search.suggest_category_llm(_Backend(), "T", "A")
    assert out.category == "compbio"
    assert out.abstained is False


def test_a_deliberate_other_is_not_an_abstention(monkeypatch):
    """The case the whole flag exists for: the LLM confidently says a paper is
    genuinely cross-cutting. Same `category` as an abstention, different event."""
    from researchwiki import search
    from researchwiki.agents import llm as agents_llm

    monkeypatch.setattr(search, "_is_valid_category", lambda c: c in {"compbio", "other"})
    monkeypatch.setattr(agents_llm, "call", lambda **kw: type(
        "R", (), {"text": '{"category": "other", "confidence": 0.95}'})())

    class _Backend:
        def more_like_text(self, seed, limit, page_type=None):
            return [type("H", (), {"category": "other", "key": "other/x",
                                   "title": "A neighbour"})()]

    out = search.suggest_category_llm(_Backend(), "T", "A")
    assert out.category == "other"
    assert out.abstained is False, \
        "a confident `other` must be distinguishable from a shrug"


# ---------- the eval's accounting ----------

@pytest.fixture
def run_eval(monkeypatch, capsys):
    """Drive `evaluate()` over a fixed set of (actual, suggestion) pairs."""
    def _run(cases):
        from researchwiki.tasks import eval_classifier as ec

        docs = [
            type("D", (), {"stem": f"p{i}", "category": actual, "title": f"T{i}",
                           "summary": "s", "body": "b", "page_type": "paper"})()
            for i, (actual, _) in enumerate(cases)
        ]
        by_stem = {f"p{i}": sug for i, (_, sug) in enumerate(cases)}

        monkeypatch.setattr(ec, "build_documents_from_wiki", lambda: docs)
        monkeypatch.setattr(ec, "TantivySearchBackend",
                            lambda path=None: type("B", (), {"build": lambda s, d: None})())
        monkeypatch.setattr(
            ec, "suggest_category",
            lambda backend, title, seed: by_stem[f"p{docs.index(next(d for d in docs if d.title == title))}"])
        ec.evaluate()
        return capsys.readouterr().out
    return _run


def test_abstention_is_counted_as_an_abstention(run_eval):
    """The regression. Three abstentions must not read as three predictions."""
    out = run_eval([
        ("compbio", _suggestion("compbio")),
        ("single-cell", _suggestion("other", 0.3, abstained=True)),
        ("other", _suggestion("other", 0.3, abstained=True)),
        ("ai", _suggestion("other", 0.0, abstained=True)),
    ])
    assert "abstained:          3 (75.0%)" in out


def test_abstention_splits_by_whether_other_was_right(run_eval):
    out = run_eval([
        ("other", _suggestion("other", 0.3, abstained=True)),        # right
        ("single-cell", _suggestion("other", 0.3, abstained=True)),  # miss
        ("ai", _suggestion("other", 0.2, abstained=True)),           # miss
    ])
    assert "paper is 'other': 1" in out
    assert "real category:    2" in out


def test_a_confident_other_is_a_commitment_not_an_abstention(run_eval):
    out = run_eval([
        ("other", _suggestion("other", 0.95)),
        ("compbio", _suggestion("compbio")),
    ])
    assert "abstained:          0 (0.0%)" in out
    assert "committed:          2 (100.0%)" in out


def test_placement_and_commitment_are_reported_separately(run_eval):
    """An abstention onto an `other` paper is correct placement *and* a
    declined commitment. One counter could not say both."""
    out = run_eval([
        ("other", _suggestion("other", 0.3, abstained=True)),
        ("compbio", _suggestion("compbio")),
    ])
    assert "correct:            2 (100.0%)" in out, "both papers land correctly"
    assert "committed:          1 (50.0%)" in out, "only one was a positive call"


def test_accuracy_when_committed_excludes_abstentions(run_eval):
    out = run_eval([
        ("compbio", _suggestion("compbio")),
        ("ai", _suggestion("compbio")),
        ("single-cell", _suggestion("other", 0.1, abstained=True)),
    ])
    assert "Accuracy when committed: 1/2 = 50.0%" in out


def test_abstentions_land_under_other_in_the_matrix(run_eval):
    """Not in a `·` column: the paper really is filed under `other`, and
    hiding that would under-report how `other` fills up."""
    out = run_eval([("single-cell", _suggestion("other", 0.2, abstained=True))])
    matrix = out[out.index("## Confusion matrix"):]
    assert "single-cell" in matrix
    row = next(ln for ln in matrix.splitlines() if ln.strip().startswith("single-cell"))
    assert "1" in row


def test_other_precision_notes_how_much_came_from_abstention(run_eval):
    """The number that says whether `other` is a judgement or a shrug."""
    out = run_eval([
        ("other", _suggestion("other", 0.2, abstained=True)),
        ("other", _suggestion("other", 0.95)),
    ])
    assert "via abstention" in out


def test_confidence_is_labelled_as_self_reported(run_eval):
    out = run_eval([("compbio", _suggestion("compbio"))])
    assert "self-report" in out
