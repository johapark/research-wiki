"""Preconditions that must survive `python -O`.

`assert` is stripped entirely under `-O`/`PYTHONOPTIMIZE`. That's harmless for a
type-narrowing hint the surrounding code already guarantees, but not for a check
on data that arrives from outside the process — there the assert is the only
thing standing between bad input and a wrong answer computed silently.

Two sites, handled differently on purpose:

  - `grade/scorer.py` had two asserts narrowing `it.relation` away from `None`
    after a filter comprehension had already guaranteed it. Those were removed by
    restructuring — the relation is now carried alongside its item, so there is
    nothing left to narrow. Nothing to test; the behavior is unchanged and the
    existing scorer tests cover it.

  - `tasks/benchmark_fixture.py` dispatches on which dataclass `load_fixture`
    parsed out of a YAML file. A third fixture type is a real possibility (the
    file is user-authored, and `RetrievalFixture` was itself added later), and
    under `-O` it would have fallen through to the content-coverage scorer and
    produced a plausible-looking score for a fixture that path can't score. That
    one became a raise, and it's tested here.

Hermetic: `load_fixture` is stubbed, no fixture YAML or PDF is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from researchwiki.benchmark.fixture import ContentFixture, RetrievalFixture
from researchwiki.tasks import benchmark_fixture as bf


@dataclass
class _ThirdFixtureType:
    """Stands in for a fixture kind added later, as RetrievalFixture once was."""
    stem: str = "some-2026-paper"


def test_unknown_fixture_type_raises_typeerror(monkeypatch):
    monkeypatch.setattr(bf, "load_fixture", lambda stem: _ThirdFixtureType())
    with pytest.raises(TypeError, match="unsupported fixture type"):
        bf.main(["some-2026-paper"])


def test_the_check_is_a_raise_not_an_assert():
    """The point of the fix. A `-O` run must still reject the fixture, so the
    check cannot be an `assert` — inspect the compiled source rather than trying
    to re-exec the module under PYTHONOPTIMIZE."""
    import inspect
    src = inspect.getsource(bf.main)
    marker = "unsupported fixture type"
    assert marker in src
    line = next(ln for ln in src.splitlines() if marker in ln)
    # The raise statement is a couple of lines above the message; what matters is
    # that no `assert` introduces it.
    head = src[:src.index(line)].splitlines()[-3:]
    assert not any(ln.strip().startswith("assert ") for ln in head), \
        "the fixture-type precondition regressed to an assert; -O would strip it"
    assert "raise TypeError(" in src


def test_known_fixture_types_are_not_rejected(monkeypatch):
    """Guard against the precondition being too strict — a false positive here
    would break every real benchmark run. Both known types must get past it.

    Each is short-circuited immediately after the check so the test doesn't need
    a PDF, an index, or an LLM: the retrieval branch returns before it, and the
    content branch is stopped at the first thing it reaches.
    """
    monkeypatch.setattr(bf, "_run_retrieval_fixture", lambda fixture, args: 0)
    monkeypatch.setattr(bf, "load_fixture",
                        lambda stem: RetrievalFixture.__new__(RetrievalFixture))
    assert bf.main(["some-2026-paper"]) == 0

    # ContentFixture passes the isinstance check, then fails further down for an
    # unrelated reason (no page/PDF). Any outcome other than the TypeError proves
    # the precondition let it through.
    monkeypatch.setattr(bf, "load_fixture",
                        lambda stem: ContentFixture.__new__(ContentFixture))
    try:
        bf.main(["some-2026-paper"])
    except TypeError as e:
        assert "unsupported fixture type" not in str(e)
    except Exception:
        pass
