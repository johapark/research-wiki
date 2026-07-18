"""`disable_thinking` routing in researchwiki.agents.llm.call.

Anthropic honors it directly (thinking:{type:disabled}); the OpenAI-compatible
path maps it to the floor `reasoning_effort` (its reasoning-model analog).
Both branches are monkeypatched so no network call happens.
"""
from __future__ import annotations

from researchwiki.agents import llm


def _fake_capture(store):
    def fake(**kw):
        store.update(kw)
        return llm.LLMResponse(text="{}", model=kw.get("model", ""),
                               temperature=0.0, input_tokens=1, output_tokens=1)
    return fake


def test_disable_thinking_forwarded_to_anthropic(monkeypatch):
    cap = {}
    monkeypatch.setattr(llm, "call_anthropic", _fake_capture(cap))
    llm.call(model="claude-sonnet-5", provider="anthropic", prompt="p",
             disable_thinking=True, temperature=0.2, max_tokens=100)
    assert cap["disable_thinking"] is True


def test_disable_thinking_maps_to_min_reasoning_effort_for_openai(monkeypatch):
    cap = {}
    monkeypatch.setattr(llm, "call_openai_compatible", _fake_capture(cap))
    llm.call(model="gemini-2.5-flash", provider="openai-compatible", prompt="p",
             disable_thinking=True, temperature=0.2, max_tokens=100)
    assert cap.get("reasoning_effort") == "minimal"


def test_explicit_reasoning_effort_wins_over_disable_thinking(monkeypatch):
    cap = {}
    monkeypatch.setattr(llm, "call_openai_compatible", _fake_capture(cap))
    llm.call(model="gemini-2.5-flash", provider="openai-compatible", prompt="p",
             disable_thinking=True, reasoning_effort="high", temperature=0.2, max_tokens=100)
    assert cap.get("reasoning_effort") == "high"


def test_no_disable_thinking_leaves_openai_reasoning_effort_unset(monkeypatch):
    cap = {}
    monkeypatch.setattr(llm, "call_openai_compatible", _fake_capture(cap))
    llm.call(model="gemini-2.5-flash", provider="openai-compatible", prompt="p",
             temperature=0.2, max_tokens=100)
    assert cap.get("reasoning_effort") is None
