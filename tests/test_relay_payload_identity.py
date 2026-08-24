"""Chat-relay prompt payloads must name the paper they belong to.

Nothing serializes relay calls -- each writes its own `{op_id}.prompt.json` and
polls its own response path -- so `agent ingest inbox/*.pdf -w 4` can leave four
pending prompts in flight. Before `stem`/`pdf` were in the payload, a responder
answering them concurrently (one subagent per ingest) had to guess ownership by
reading the prompt body, because the payload named the phase but never the paper.

The other half of the contract is that these fields must NOT reach `_stable_op_id`.
op_id is the cache key that lets a crashed run reuse a response already on disk;
folding identity into it would silently invalidate every cached response.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.agents import relay


@pytest.fixture(autouse=True)
def _reset_identity():
    """Identity is process-scoped module state; don't leak across tests."""
    relay._current_stem = None
    relay._current_pdf = None
    yield
    relay._current_stem = None
    relay._current_pdf = None


def _relay_once(tmp_path, monkeypatch, *, phase="author", prompt="hello"):
    """Run one relay call and return (result, the payload it wrote).

    Two things to work around. The real call blocks until the response file
    appears, so the poll is patched to answer immediately -- which keeps the
    prompt-writing path intact, where pre-seeding the response would make the
    code skip writing a prompt at all. And on success `call_chat_relay` unlinks
    both files, so the payload has to be captured at write time rather than read
    off disk afterwards.
    """
    monkeypatch.chdir(tmp_path)
    written: list[dict] = []
    real_write = relay._write_atomic_json

    def _capture(path, data):
        if path.name.endswith(".prompt.json"):
            written.append(data)
        return real_write(path, data)

    def _answer(path, timeout):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 1,
            "op_id": path.name.split(".")[0],
            "via": "test/model-1",
            "response": "ok",
        }), encoding="utf-8")

    monkeypatch.setattr(relay, "_write_atomic_json", _capture)
    monkeypatch.setattr(relay, "_poll_until_exists", _answer)
    monkeypatch.setattr(relay, "_emit_handoff_message", lambda *a, **k: None)
    out = relay.call_chat_relay(system="sys", prompt=prompt, model="m", phase=phase)
    assert len(written) == 1, written
    return out, written[0]


def test_payload_carries_stem_and_pdf(tmp_path, monkeypatch):
    relay.set_relay_identity(pdf="raw-drop.pdf")
    relay.set_relay_identity(stem="smith-2024-a-worked-example-of-something")
    _out, payload = _relay_once(tmp_path, monkeypatch)
    assert payload["stem"] == "smith-2024-a-worked-example-of-something"
    assert payload["pdf"] == "raw-drop.pdf"


def test_stem_is_null_before_reconcile_but_pdf_is_not(tmp_path, monkeypatch):
    # Reconcile is the phase that derives the stem, so it cannot name one. `pdf`
    # is set at runner entry precisely so that prompt is still attributable.
    relay.set_relay_identity(pdf="inbox-scan-3.pdf")
    _out, payload = _relay_once(tmp_path, monkeypatch, phase="reconcile")
    assert payload["stem"] is None
    assert payload["pdf"] == "inbox-scan-3.pdf"


def test_identity_does_not_change_op_id(tmp_path, monkeypatch):
    # op_id is the cache key. If identity leaked into it, every response already
    # on disk would stop being found and a crashed run would re-prompt for work
    # the responder had already done.
    bare = relay._stable_op_id("author", "the same prompt")
    relay.set_relay_identity(stem="smith-2024-something", pdf="x.pdf")
    assert relay._stable_op_id("author", "the same prompt") == bare


def test_set_relay_identity_updates_each_field_independently():
    # The runner sets pdf at entry and stem later; the second call must not wipe
    # the first field by passing only one of them.
    relay.set_relay_identity(pdf="a.pdf")
    relay.set_relay_identity(stem="jones-2025-title-words-here")
    assert relay._current_pdf == "a.pdf"
    assert relay._current_stem == "jones-2025-title-words-here"


def test_schema_version_unchanged_by_additive_fields(tmp_path, monkeypatch):
    # The fields are additive and nullable, so responders written against v1 keep
    # working and the response shape is untouched. Pinned so a future edit that
    # bumps the version has to be deliberate.
    relay.set_relay_identity(stem="a-2020-b", pdf="c.pdf")
    _out, payload = _relay_once(tmp_path, monkeypatch)
    assert payload["schema_version"] == 1


@pytest.mark.parametrize("via", ["codex/gpt-5.6", "claude-code/claude-4"])
def test_relay_rejects_an_ambiguous_model_family_alias(tmp_path, via):
    with pytest.raises(relay.SchemaError, match="specific model variant"):
        relay._check_response_shape(
            {"via": via, "response": "ok"},
            tmp_path / "response.json",
            schema=None,
        )


def test_relay_accepts_an_exact_gpt_variant(tmp_path):
    relay._check_response_shape(
        {"via": "codex/gpt-5.6-terra", "response": "ok"},
        tmp_path / "response.json",
        schema=None,
    )


def test_relay_accepts_an_exact_anthropic_variant(tmp_path):
    relay._check_response_shape(
        {"via": "claude-code/opus-4-7", "response": "ok"},
        tmp_path / "response.json",
        schema=None,
    )
