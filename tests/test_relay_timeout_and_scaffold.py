"""Two first-run/concurrency gaps found while running chat-relay end to end.

1. The relay's poll deadline was a hard-coded 600 s. The clock starts when the
   prompt is *written*, not when anyone notices it, so under concurrency each
   in-flight ingest holds its own 600 s budget and a serially-working responder
   can lose workers it has not reached. `RW_RELAY_TIMEOUT` makes it settable
   without raising the floor for everyone.

2. `ensure_scaffold` created every content directory but not `wiki/index.md`, and
   `promote._append_index_entry` returns False when that file is absent. On a
   fresh clone the *first* paper ingested therefore never got a catalog line while
   every later one did — the kind of off-by-one-run bug that looks like a fluke.
   Observed 2026-08-10 promoting into a scaffold-only root.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.agents import relay
from researchwiki.errors import EnvironmentFailure


# ---------- RW_RELAY_TIMEOUT ----------

def test_default_timeout_without_env(monkeypatch):
    monkeypatch.delenv("RW_RELAY_TIMEOUT", raising=False)
    assert relay._default_timeout() == relay._RELAY_DEFAULT_TIMEOUT


def test_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("RW_RELAY_TIMEOUT", "90")
    assert relay._default_timeout() == 90.0
    monkeypatch.setenv("RW_RELAY_TIMEOUT", "  1800  ")
    assert relay._default_timeout() == 1800.0


@pytest.mark.parametrize("bad", ["abc", "", "0", "-5", "nan-ish"])
def test_unusable_values_fall_back_rather_than_raise(monkeypatch, bad):
    # A typo'd env var must not turn every relay call into an instant failure.
    monkeypatch.setenv("RW_RELAY_TIMEOUT", bad)
    got = relay._default_timeout()
    assert got == relay._RELAY_DEFAULT_TIMEOUT or got > 0


def test_timeout_is_resolved_per_call_not_at_import(monkeypatch):
    # Pinned because the obvious implementation — `timeout=_default_timeout()` as a
    # default argument — evaluates once at import and freezes whatever the env held
    # then, which is invisible until someone sets the var and nothing changes.
    import inspect
    sig = inspect.signature(relay.call_chat_relay)
    assert sig.parameters["timeout"].default is None


def test_timeout_is_a_retryable_environment_failure(tmp_path):
    missing = tmp_path / "missing.response.json"
    with pytest.raises(relay.RelayTimeout, match="Pending file remains"):
        relay._poll_until_exists(missing, timeout=0)
    assert issubclass(relay.RelayTimeout, EnvironmentFailure)


def test_timed_out_prompt_can_be_answered_then_reused(tmp_path, monkeypatch):
    """The deterministic op id turns a timeout into a resumable handoff."""
    monkeypatch.chdir(tmp_path)
    phase = "author"
    prompt = "grounded test prompt"
    op_id = relay._stable_op_id(phase, prompt)
    prompt_path, response_path = relay._paths_for(op_id)

    with pytest.raises(relay.RelayTimeout):
        relay.call_chat_relay(
            model="gpt-5.6-terra",
            prompt=prompt,
            phase=phase,
            timeout=0,
        )
    assert prompt_path.exists()

    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "op_id": op_id,
                "via": "codex/gpt-5.6-terra",
                "response": "recovered",
            }
        ),
        encoding="utf-8",
    )
    result = relay.call_chat_relay(
        model="gpt-5.6-terra",
        prompt=prompt,
        phase=phase,
        timeout=0,
    )
    assert result.text == "recovered"
    assert not prompt_path.exists()
    assert not response_path.exists()


def test_existing_pending_prompt_is_announced_again(tmp_path, monkeypatch, capsys):
    """A resumed worker must expose the request it is about to wait on."""
    monkeypatch.chdir(tmp_path)
    relay.set_relay_identity(pdf="paper.pdf")
    phase = "author"
    prompt = "resumed prompt"
    op_id = relay._stable_op_id(phase, prompt)
    prompt_path, _ = relay._paths_for(op_id)
    relay._write_atomic_json(prompt_path, {"op_id": op_id})

    def answer(path, timeout):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 1,
            "op_id": op_id,
            "via": "codex/gpt-5.6-terra",
            "response": "ok",
        }), encoding="utf-8")

    monkeypatch.setattr(relay, "_poll_until_exists", answer)
    relay.call_chat_relay(model="gpt-5.6-terra", prompt=prompt, phase=phase)
    err = capsys.readouterr().err
    assert err.startswith(relay.HANDOFF_PREFIX)
    assert "paper.pdf" in err
    assert err.count("\n") == 1


def test_agent_cli_maps_relay_timeout_to_exit_2(tmp_path, monkeypatch):
    """`agent ingest` must not relabel a responder timeout as an internal bug."""
    from researchwiki.__main__ import main as cli_main
    from researchwiki.tasks import agent as agent_task

    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def _timeout(*args, **kwargs):
        raise relay.RelayTimeout("chat-relay responder timed out")

    monkeypatch.setattr(agent_task, "run_ingest", _timeout)
    assert cli_main(["agent", "ingest", "--stub", str(pdf)]) == 2


# ---------- wiki/index.md in the scaffold ----------

def test_ensure_scaffold_creates_index_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from researchwiki.paths import ensure_scaffold
    created = ensure_scaffold()
    idx = tmp_path / "wiki" / "index.md"
    assert idx.exists(), "first ingest would silently skip its catalog line"
    assert idx in created
    assert idx.read_text().startswith("# index.md")


def test_ensure_scaffold_is_idempotent_and_preserves_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from researchwiki.paths import ensure_scaffold
    ensure_scaffold()
    idx = tmp_path / "wiki" / "index.md"
    idx.write_text("# index.md\n\n## other\n\n- [[other/kept-2024-a-real-entry]] — keep me\n")
    created = ensure_scaffold()
    assert idx not in created
    assert "kept-2024-a-real-entry" in idx.read_text(), "must never clobber a live catalog"


def test_first_entry_splices_into_the_fresh_scaffold(tmp_path, monkeypatch):
    # The whole point: existence is the requirement, because a missing
    # `## <category>` section is created by the splice rather than refused.
    monkeypatch.chdir(tmp_path)
    from researchwiki.paths import ensure_scaffold
    from researchwiki.agents.promote import _append_index_entry
    ensure_scaffold()
    ok = _append_index_entry(
        stem="smith-2024-a-worked-example-of-something", category="other",
        short_name="Example", title="A worked example of something",
        year=2024, venue="Journal of Tests", hook="A one-line gloss.",
    )
    assert ok is True
    text = (tmp_path / "wiki" / "index.md").read_text()
    assert "## other" in text
    assert "- [[other/smith-2024-a-worked-example-of-something]]" in text
