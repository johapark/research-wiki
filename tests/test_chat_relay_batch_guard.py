"""Provider-aware worker defaults, warnings, and resume policy for batches.

The policy lives in `tasks._ingest_batch` rather than in one CLI because four
entry points need the same answer — `agent ingest`, `import apply`,
`researchwiki ingest`, and `resume_batch` reading a plan written by any of them.
These tests pin each entry point and the conservative all-phase detector used
to avoid silently missing a newly relay-backed ingest phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from researchwiki.tasks import _ingest_batch
from researchwiki.tasks import agent as agent_task


def _make_batch_args(tmp_path, **overrides) -> argparse.Namespace:
    """Argparse namespace for a two-PDF batch run (the shape that triggers it)."""
    a, b = tmp_path / "one.pdf", tmp_path / "two.pdf"
    a.write_bytes(b"%PDF-1.4\n")
    b.write_bytes(b"%PDF-1.4\n")
    ns = argparse.Namespace(
        pdfs=[str(a), str(b)],
        workers=None,
        resume=None,
        no_retry=False,
        # False by default: a `--stub` run never reaches a provider, so it is the
        # wrong shape for testing provider policy. `test_stub_batch_*` covers it.
        stub=False,
        no_semantic=False,
        verify_claim_entailment=False,
        n_drafts=1,
        max_evolve=1,
        auto_promote=False,
        force_sandbox=False,
        no_cross_link=False,
        claim_overlap=False,
        doi=None,
        title=None,
        authors=None,
        year=None,
        author_prompt_file=None,
        allow_rename=False,
        llm_reconcile=None,
        supplementary=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def _batch_args(tmp_path):
    return _make_batch_args(tmp_path)


@pytest.fixture(autouse=True)
def _dont_actually_spawn(monkeypatch):
    """Stub the batch runner so no subprocesses start."""
    monkeypatch.setattr(_ingest_batch, "new_batch", lambda *a, **k: 0)


def _capture_new_batch(monkeypatch) -> dict:
    seen: dict = {}
    monkeypatch.setattr(
        _ingest_batch, "new_batch", lambda *a, **k: seen.update(k) or 0
    )
    return seen


# ---------- agent ingest ----------

def test_chat_relay_batch_defaults_to_one_visible_worker(
    _batch_args, monkeypatch, capsys
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = _capture_new_batch(monkeypatch)
    rc = agent_task._cmd_ingest(_batch_args)
    err = capsys.readouterr().err
    assert rc == 0
    assert "defaults to 1 worker" in err
    assert "Pass -w N" in err
    assert "prompts/chat-relay.md" in err
    assert seen["workers"] == 1
    assert seen["workers_explicit"] is False


def test_stub_batch_is_never_serialized(tmp_path, monkeypatch, capsys):
    """`--stub` writes no relay prompt, so the 1-worker default cannot apply.

    Regression: the resolver was consulted with no `--stub` gate, so the offline
    framework test the module docstring advertises ran 4x slower on any
    chat-relay profile. This test previously asserted the bug, via a fixture
    that set `stub=True`.
    """
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = _capture_new_batch(monkeypatch)
    rc = agent_task._cmd_ingest(_make_batch_args(tmp_path, stub=True))
    err = capsys.readouterr().err
    assert rc == 0
    assert seen["workers"] == 4
    assert "chat-relay" not in err


def test_silent_for_an_api_provider(_batch_args, monkeypatch, capsys):
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    # Non-stub, so `_cmd_ingest` runs its provider preflight for real.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    seen = _capture_new_batch(monkeypatch)
    rc = agent_task._cmd_ingest(_batch_args)
    err = capsys.readouterr().err
    assert rc == 0
    assert "chat-relay" not in err
    assert seen["workers"] == 4
    assert seen["workers_explicit"] is False


def test_explicit_parallel_chat_relay_is_honored_and_warned(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = _capture_new_batch(monkeypatch)
    assert agent_task._cmd_ingest(_make_batch_args(tmp_path, workers=3)) == 0
    assert seen["workers"] == 3
    assert seen["workers_explicit"] is True
    err = capsys.readouterr().err
    assert "3 concurrent chat-relay workers explicitly requested" in err
    assert "not isolated chat responders" in err


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_worker_count_is_a_command_line_error(
    tmp_path, monkeypatch, capsys, bad
):
    """`-w 0` reached `ThreadPoolExecutor` and surfaced as exit 3 with a
    traceback — after the batch dir and plan.json had already been written."""
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    rc = agent_task._cmd_ingest(_make_batch_args(tmp_path, workers=bad))
    assert rc == 1, "bad command line is exit 1, not an internal bug"
    assert "greater than zero" in capsys.readouterr().err
    assert not list(tmp_path.glob(".ingest/batch-*")), "nothing should be created"


def test_non_positive_worker_count_rejected_before_resume(tmp_path, capsys):
    """The check sits ahead of `--resume`, which returns before argv validation."""
    rc = agent_task._cmd_ingest(
        _make_batch_args(tmp_path, workers=0, resume=str(tmp_path), pdfs=[])
    )
    assert rc == 1
    assert "greater than zero" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [0, -1])
def test_digest_resume_rejects_non_positive_worker_count(tmp_path, capsys, bad):
    """The digest CLI returns through ``resume_batch`` before its fresh-batch
    validation block, so the batch driver must enforce the shared invariant."""
    from researchwiki.tasks import ingest as ingest_cli

    rc = ingest_cli.main([
        "--resume", str(tmp_path / "batch"), "--workers", str(bad),
    ])
    assert rc == 1
    assert "greater than zero" in capsys.readouterr().err


def test_malformed_config_fails_before_chat_relay_batch_spawns(
    _batch_args, tmp_path, monkeypatch
):
    from researchwiki.agents import model_config

    bad = tmp_path / "models.yaml"
    bad.write_text("roles: [this is not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("RW_MODELS_CONFIG", str(bad))
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    spawned = []
    monkeypatch.setattr(
        _ingest_batch, "new_batch", lambda *a, **k: spawned.append((a, k))
    )
    model_config.clear_caches()
    try:
        with pytest.raises(
            model_config.ModelConfigUnavailable, match="cannot be parsed"
        ):
            agent_task._cmd_ingest(_batch_args)
        assert spawned == []
    finally:
        model_config.clear_caches()


# ---------- detector ----------

def test_detector_catches_env_override(monkeypatch):
    from researchwiki.agents import model_config

    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    assert model_config.uses_chat_relay() is True
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    assert model_config.uses_chat_relay() is False


def _route_one_phase_to_relay(monkeypatch, target: str):
    from researchwiki.agents import model_config

    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    real = model_config.for_phase

    def _one(name):
        cfg = real(name)
        if name == target:
            return type(cfg)(
                provider="chat-relay",
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                reasoning_effort=cfg.reasoning_effort,
                rpm=cfg.rpm,
            )
        return cfg

    monkeypatch.setattr(model_config, "for_phase", _one)


def test_detector_catches_a_single_phase_routed_by_config(monkeypatch):
    # `config/models.yaml` can route one phase to chat-relay while the rest stay
    # on an API provider. A guard that checked only the env var would miss it.
    from researchwiki.agents import model_config

    _route_one_phase_to_relay(monkeypatch, "author")
    assert model_config.uses_chat_relay() is True
    assert _ingest_batch.resolve_batch_workers(None) == 1


def test_worker_policy_conservatively_honors_any_relay_phase(monkeypatch):
    from researchwiki.agents import model_config

    _route_one_phase_to_relay(monkeypatch, "synthesis_judge")
    assert model_config.uses_chat_relay() is True
    assert _ingest_batch.resolve_batch_workers(None) == 1


def test_explicit_count_is_preserved_when_any_phase_uses_relay(monkeypatch):
    _route_one_phase_to_relay(monkeypatch, "classifier")
    assert _ingest_batch.resolve_batch_workers(2) == 2
    assert _ingest_batch.resolve_batch_workers(None) == 1


# ---------- resume ----------

def _write_plan(tmp_path: Path, **plan_overrides) -> tuple[Path, Path]:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    batch = tmp_path / "batch"
    batch.mkdir()
    plan = {
        "started_at": "2026-09-02T00:00:00",
        "subcommand": ["agent", "ingest"],
        "workers": 4,
        "inputs": [str(pdf.resolve())],
        "extra_args": [],
    }
    plan.update(plan_overrides)
    (batch / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return batch, pdf


def _resume_runtime(tmp_path, monkeypatch, *, provider, workers_override=None,
                    **plan_overrides) -> int:
    """Resume a synthetic pending batch and capture its runtime policy."""
    batch, _ = _write_plan(tmp_path, **plan_overrides)
    captured = {}

    def _capture(batch_dir, pending, effective_workers, subcommand, extra_args,
                 per_input_args=None):
        captured["workers"] = effective_workers
        return 0

    monkeypatch.setattr(_ingest_batch, "_run_batch", _capture)
    monkeypatch.setenv("RW_LLM_PROVIDER", provider)
    assert _ingest_batch.resume_batch(
        batch, no_retry=False, workers_override=workers_override) == 0
    return captured["workers"]


@pytest.mark.parametrize(
    ("plan_extra", "provider", "override", "expected"),
    [
        # An implicit default follows the profile in force at resume time.
        ({"workers": 4, "workers_explicit": False}, "chat-relay", None, 1),
        ({"workers": 1, "workers_explicit": False}, "anthropic", None, 4),
        # An explicit count is the user's, and survives a provider switch.
        ({"workers": 3, "workers_explicit": True}, "chat-relay", None, 3),
        ({"workers": 1, "workers_explicit": True}, "chat-relay", None, 1),
        # A resume-time override always wins.
        ({"workers": 4, "workers_explicit": False}, "chat-relay", 2, 2),
        # Legacy plans (no `workers_explicit`): a stored 4 is ambiguous with the
        # old unconditional default, so it is re-resolved …
        ({"workers": 4}, "chat-relay", None, 1),
        # … but any other stored count was typed by hand and must be honored.
        # Re-resolving these escalated a deliberate `-w 1` — which CLAUDE.md
        # prescribes for the Gemini free tier — back up to four on resume.
        ({"workers": 1}, "anthropic", None, 1),
        ({"workers": 2}, "anthropic", None, 2),
        ({"workers": 8}, "chat-relay", None, 8),
    ],
)
def test_resume_recomputes_provider_policy_and_preserves_explicit_intent(
    tmp_path, monkeypatch, plan_extra, provider, override, expected
):
    assert _resume_runtime(
        tmp_path, monkeypatch, provider=provider,
        workers_override=override, **plan_extra
    ) == expected


def test_resume_uses_one_policy_for_both_entry_points(tmp_path, monkeypatch):
    """`researchwiki ingest --resume` and `agent ingest --resume` on the same
    batch dir must agree; they previously disagreed because only one injected
    the resolver, so the digest CLI ran four chat-relay workers."""
    from researchwiki.tasks import ingest as ingest_cli

    seen: list[int] = []

    def _capture(batch_dir, pending, workers, subcommand, extra_args,
                 per_input_args=None):
        seen.append(workers)
        return 0

    monkeypatch.setattr(_ingest_batch, "_run_batch", _capture)
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    batch, _ = _write_plan(tmp_path, workers=4, workers_explicit=False)

    agent_task.main(["ingest", "--resume", str(batch)])
    ingest_cli.main(["--resume", str(batch)])
    assert seen[0] == seen[1] == 1


def test_resume_survives_an_unreadable_model_config(tmp_path, monkeypatch, capsys):
    """Reading the routing snapshot is new on this path. On its own it turned an
    unparseable RW_MODELS_CONFIG into "this half-finished batch cannot be
    resumed at all" — including for a --stub batch the config is irrelevant to.
    """
    from researchwiki.agents import model_config

    bad = tmp_path / "models.yaml"
    bad.write_text("roles: [not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("RW_MODELS_CONFIG", str(bad))
    model_config.clear_caches()
    try:
        workers = _resume_runtime(
            tmp_path, monkeypatch, provider="chat-relay",
            workers=2, workers_explicit=True)
    finally:
        model_config.clear_caches()
    assert workers == 2, "falls back to the plan's recorded intent"
    err = capsys.readouterr().err
    assert "could not re-check provider routing" in err
    assert "recorded 2 worker(s)" in err
