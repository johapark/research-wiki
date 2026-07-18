"""Keyword-proposal failures must be loud and retried, not silently empty.

A malformed `keywords` response used to return `[]` with no log and no retry,
and promote wrote the page anyway — which is how 38% of the corpus ended up
with no `keywords:` field despite `lint` requiring one. These tests pin the
three behaviours that close that hole:

  1. a non-JSON / malformed response logs rather than failing silently,
  2. an empty first parse triggers exactly one retry,
  3. a successful retry supplies the keywords (and sums both calls' tokens).

`llm.call` is monkeypatched throughout, so no network call happens.
"""
from __future__ import annotations

import importlib

# `researchwiki.agents.phases` re-exports a `commit` *function*, which shadows
# the submodule of the same name — import the module explicitly.
commit = importlib.import_module("researchwiki.agents.phases.commit")


def test_non_json_response_is_logged(monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(commit, "log", lambda msg, **kw: logged.append(msg))

    assert commit._parse_keywords_response("sorry, I cannot do that") == []
    assert any("no JSON object" in m for m in logged)


def test_malformed_json_is_logged(monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(commit, "log", lambda msg, **kw: logged.append(msg))

    # Braces present (so the object regex matches) but the body is not JSON —
    # this is the branch that reaches json.loads and raises.
    assert commit._parse_keywords_response('{"keywords": [oops}') == []
    assert any("not valid JSON" in m for m in logged)


def test_empty_parse_triggers_one_retry_that_succeeds(monkeypatch):
    monkeypatch.setattr(commit, "log", lambda *a, **kw: None)
    calls = {"n": 0}

    def fake_call(**kw):
        calls["n"] += 1
        text = (
            "no json here"
            if calls["n"] == 1
            else '{"keywords": ["matrix factorization", "parts-based representation"]}'
        )
        return commit.llm.LLMResponse(
            text=text, model="m", temperature=0.0, input_tokens=5, output_tokens=7
        )

    monkeypatch.setattr(commit.llm, "call", fake_call)

    out = commit.propose_keywords(
        metadata={"title": "T"}, draft_text="## Summary\nx"
    )
    assert calls["n"] == 2, "expected exactly one retry"
    assert "matrix factorization" in out.keywords
    # tokens from both attempts are accounted for, not just the retry
    assert out.input_tokens == 10 and out.output_tokens == 14


def test_still_empty_after_retry_warns_and_returns_empty(monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(commit, "log", lambda msg, **kw: logged.append(msg))
    calls = {"n": 0}

    def fake_call(**kw):
        calls["n"] += 1
        return commit.llm.LLMResponse(
            text="still not json", model="m", temperature=0.0,
            input_tokens=1, output_tokens=1,
        )

    monkeypatch.setattr(commit.llm, "call", fake_call)

    out = commit.propose_keywords(
        metadata={"title": "T"}, draft_text="## Summary\nx"
    )
    assert calls["n"] == 2  # one retry, then give up
    assert out.keywords == []
    assert any("backfill keywords" in m for m in logged), \
        "operator needs a pointer to the remediation command"
