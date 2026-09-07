"""Real process boundaries for relay mailbox ownership."""

import json
import multiprocessing
import queue
from pathlib import Path

import pytest


def _relay_process(root, label, prompt, started, notices, outcomes):
    from researchwiki.agents import relay

    relay.wiki_root = lambda: Path(root)
    relay._RELAY_POLL_INTERVAL = 0.01
    relay.set_relay_identity(pdf=f"{label}.pdf")
    relay._emit_handoff_message = lambda p, r, *a, **k: notices.put(
        (label, str(p), str(r))
    )
    started.set()
    try:
        result = relay.call_chat_relay(
            model="test", phase="author", prompt=prompt, timeout=5,
        )
        outcomes.put((label, result.text))
    except Exception as exc:
        outcomes.put((label, f"ERROR: {type(exc).__name__}: {exc}"))


def _answer(notice):
    from researchwiki.agents.relay import _write_atomic_json

    label, pending, response = notice
    payload = json.loads(Path(pending).read_text())
    assert payload["pdf"] == f"{label}.pdf"
    _write_atomic_json(Path(response), {
        "via": "test/model-1", "op_id": payload["op_id"], "response": label,
    })


@pytest.mark.parametrize("same_prompt", [True, False])
def test_mailbox_ownership_across_processes(tmp_path, same_prompt):
    ctx = multiprocessing.get_context("spawn")
    notices, outcomes = ctx.Queue(), ctx.Queue()
    started_a, started_b = ctx.Event(), ctx.Event()
    a = ctx.Process(target=_relay_process, args=(
        str(tmp_path), "A", "prompt A", started_a, notices, outcomes,
    ))
    b = ctx.Process(target=_relay_process, args=(
        str(tmp_path), "B", "prompt A" if same_prompt else "prompt B",
        started_b, notices, outcomes,
    ))
    processes = []
    try:
        a.start()
        processes.append(a)
        first = notices.get(timeout=5)
        b.start()
        processes.append(b)
        assert started_b.wait(5)
        if same_prompt:
            with pytest.raises(queue.Empty):
                notices.get(timeout=0.2)
            _answer(first)
            second = notices.get(timeout=5)
            assert first[1:] == second[1:]
        else:
            # Distinct prompts must both be serviceable before either is answered.
            second = notices.get(timeout=5)
            assert first[1:] != second[1:]
            _answer(first)
        _answer(second)
        assert {outcomes.get(timeout=5), outcomes.get(timeout=5)} == {
            ("A", "A"), ("B", "B"),
        }
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        notices.close()
        outcomes.close()


def test_exited_owner_releases_mailbox_and_refreshes_identity(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    notices, outcomes = ctx.Queue(), ctx.Queue()
    started_a, started_b = ctx.Event(), ctx.Event()
    a = ctx.Process(target=_relay_process, args=(
        str(tmp_path), "A", "same prompt", started_a, notices, outcomes,
    ))
    b = ctx.Process(target=_relay_process, args=(
        str(tmp_path), "B", "same prompt", started_b, notices, outcomes,
    ))
    processes = []
    try:
        a.start()
        processes.append(a)
        first = notices.get(timeout=5)
        a.terminate()
        a.join(5)
        b.start()
        processes.append(b)
        second = notices.get(timeout=5)
        assert first[1:] == second[1:]
        _answer(second)  # also verifies that the pending file now names B
        assert outcomes.get(timeout=5) == ("B", "B")
        b.join(5)
        assert b.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        notices.close()
        outcomes.close()
