"""Provider-aware worker defaults, warnings, and resume policy for batches.

The policy lives in `tasks._ingest_batch` rather than in one CLI because four
entry points need the same answer — `agent ingest`, `import apply`,
`researchwiki ingest`, and `resume_batch` reading a plan written by any of them.
These tests pin each of those, and pin the phase set the policy reads against
the live registry, because `uses_chat_relay` skips names it cannot resolve and
would otherwise fail open to four unmirrored workers.
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


# ---------- the phase set the policy reads ----------

def test_ingest_phase_set_matches_the_live_registry():
    """Every name must resolve, or the policy silently under-reports.

    `model_config.uses_chat_relay` swallows `PhaseNotRegistered`, so a phase
    renamed without updating this set fails *open*: four workers and no prompt
    mirroring under chat-relay, which is the exact stall the mirror exists for.
    """
    from researchwiki.agents.model_config import _FALLBACK_PHASES

    unknown = sorted(_ingest_batch._INGEST_LLM_PHASES - set(_FALLBACK_PHASES))
    assert unknown == [], f"not registered in _FALLBACK_PHASES: {unknown}"
    unknown_digest = sorted(_ingest_batch._DIGEST_LLM_PHASES - set(_FALLBACK_PHASES))
    assert unknown_digest == []


def test_phases_are_selected_per_subcommand():
    """The digest path reaches only the classifier; agent ingest reaches the rest."""
    assert _ingest_batch._llm_phases_for(["ingest"]) == frozenset({"classifier"})
    assert _ingest_batch._llm_phases_for(["agent", "ingest"]) == (
        _ingest_batch._INGEST_LLM_PHASES
    )
    # `classifier` is common to both: the digest path calls it via
    # `search.suggest_category`, so a digest batch is not provider-free.
    assert "classifier" in _ingest_batch._INGEST_LLM_PHASES


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
    assert seen["relay_watch"] is True


def test_stub_batch_is_never_serialized_or_mirrored(tmp_path, monkeypatch, capsys):
    """`--stub` writes no relay prompt, so neither the 1-worker default nor the
    "requests will be surfaced here" promise can be true of it.

    Regression: the resolver was consulted with no `--stub` gate, so the offline
    framework test the module docstring advertises ran 4x slower on any
    chat-relay profile and polled an empty directory for its whole life. This
    test previously asserted the bug, via a fixture that set `stub=True`.
    """
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = _capture_new_batch(monkeypatch)
    rc = agent_task._cmd_ingest(_make_batch_args(tmp_path, stub=True))
    err = capsys.readouterr().err
    assert rc == 0
    assert seen["workers"] == 4
    assert seen["relay_watch"] is False
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
    assert seen["relay_watch"] is False


def test_explicit_parallel_chat_relay_is_honored_and_warned(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = _capture_new_batch(monkeypatch)
    assert agent_task._cmd_ingest(_make_batch_args(tmp_path, workers=3)) == 0
    assert seen["workers"] == 3
    assert seen["workers_explicit"] is True
    assert seen["relay_watch"] is True
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
    assert _ingest_batch.resolve_batch_workers(
        None, subcommand=["agent", "ingest"]) == (1, True)


def test_ingest_detector_ignores_an_unrelated_relay_phase(monkeypatch):
    """A mixed synthesis route must not serialize an API-backed ingest."""
    from researchwiki.agents import model_config

    _route_one_phase_to_relay(monkeypatch, "synthesis_judge")
    assert model_config.uses_chat_relay() is True
    assert _ingest_batch.resolve_batch_workers(
        None, subcommand=["agent", "ingest"]) == (4, False)


def test_digest_batch_is_mirrored_when_the_classifier_uses_relay(monkeypatch):
    """`researchwiki ingest -w N` is not provider-free: `suggest_category`
    resolves `phase="classifier"` per paper, and without mirroring each paper
    blocked for the full relay timeout with its notice in a worker log."""
    _route_one_phase_to_relay(monkeypatch, "classifier")
    assert _ingest_batch.resolve_batch_workers(
        2, subcommand=["ingest"]) == (2, True)
    assert _ingest_batch.resolve_batch_workers(
        None, subcommand=["ingest"]) == (1, True)


def test_digest_batch_ignores_a_phase_it_cannot_reach(monkeypatch):
    _route_one_phase_to_relay(monkeypatch, "author")
    assert _ingest_batch.resolve_batch_workers(
        None, subcommand=["ingest"]) == (4, False)


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
                    **plan_overrides) -> tuple[int, bool]:
    """Resume a synthetic pending batch and capture its runtime policy."""
    batch, _ = _write_plan(tmp_path, **plan_overrides)
    captured = {}

    def _capture(batch_dir, pending, effective_workers, subcommand, extra_args,
                 per_input_args=None, relay_watch=False):
        captured.update(workers=effective_workers, relay_watch=relay_watch)
        return 0

    monkeypatch.setattr(_ingest_batch, "_run_batch", _capture)
    monkeypatch.setenv("RW_LLM_PROVIDER", provider)
    assert _ingest_batch.resume_batch(
        batch, no_retry=False, workers_override=workers_override) == 0
    return captured["workers"], captured["relay_watch"]


@pytest.mark.parametrize(
    ("plan_extra", "provider", "override", "expected"),
    [
        # An implicit default follows the profile in force at resume time.
        ({"workers": 4, "workers_explicit": False}, "chat-relay", None, (1, True)),
        ({"workers": 1, "workers_explicit": False}, "anthropic", None, (4, False)),
        # An explicit count is the user's, and survives a provider switch.
        ({"workers": 3, "workers_explicit": True}, "chat-relay", None, (3, True)),
        ({"workers": 1, "workers_explicit": True}, "chat-relay", None, (1, True)),
        # A resume-time override always wins.
        ({"workers": 4, "workers_explicit": False}, "chat-relay", 2, (2, True)),
        # Legacy plans (no `workers_explicit`): a stored 4 is ambiguous with the
        # old unconditional default, so it is re-resolved …
        ({"workers": 4}, "chat-relay", None, (1, True)),
        # … but any other stored count was typed by hand and must be honored.
        # Re-resolving these escalated a deliberate `-w 1` — which CLAUDE.md
        # prescribes for the Gemini free tier — back up to four on resume.
        ({"workers": 1}, "anthropic", None, (1, False)),
        ({"workers": 2}, "anthropic", None, (2, False)),
        ({"workers": 8}, "chat-relay", None, (8, True)),
    ],
)
def test_resume_recomputes_provider_policy_and_preserves_explicit_intent(
    tmp_path, monkeypatch, plan_extra, provider, override, expected
):
    assert _resume_runtime(
        tmp_path, monkeypatch, provider=provider,
        workers_override=override, **plan_extra
    ) == expected


def test_resume_of_a_digest_batch_mirrors_relay_prompts(tmp_path, monkeypatch):
    """The digest CLI never injected the policy, so its batches resumed with
    `relay_watch=False` no matter what the profile said."""
    _route_one_phase_to_relay(monkeypatch, "classifier")
    workers, relay_watch = _resume_runtime(
        tmp_path, monkeypatch, provider="", subcommand=["ingest"],
        workers=2, workers_explicit=True)
    assert (workers, relay_watch) == (2, True)


def test_resume_uses_one_policy_for_both_entry_points(tmp_path, monkeypatch):
    """`researchwiki ingest --resume` and `agent ingest --resume` on the same
    batch dir must agree; they previously disagreed because only one injected
    the resolver, so the digest CLI ran four unmirrored chat-relay workers."""
    from researchwiki.tasks import ingest as ingest_cli

    seen: list[tuple[int, bool]] = []

    def _capture(batch_dir, pending, workers, subcommand, extra_args,
                 per_input_args=None, relay_watch=False):
        seen.append((workers, relay_watch))
        return 0

    monkeypatch.setattr(_ingest_batch, "_run_batch", _capture)
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    batch, _ = _write_plan(tmp_path, workers=4, workers_explicit=False)

    agent_task.main(["ingest", "--resume", str(batch)])
    ingest_cli.main(["--resume", str(batch)])
    assert seen[0] == seen[1] == (1, True)


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
        workers, relay_watch = _resume_runtime(
            tmp_path, monkeypatch, provider="chat-relay",
            workers=2, workers_explicit=True)
    finally:
        model_config.clear_caches()
    assert workers == 2, "falls back to the plan's recorded intent"
    assert relay_watch is False
    err = capsys.readouterr().err
    assert "could not re-check provider routing" in err
    assert "recorded 2 worker(s)" in err
