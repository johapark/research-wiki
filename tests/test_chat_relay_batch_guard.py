"""Provider-aware worker defaults and warnings for chat-relay batches."""

from __future__ import annotations

import argparse

import pytest

from researchwiki.tasks import agent as agent_task


@pytest.fixture
def _batch_args(tmp_path):
    """Argparse namespace for a two-PDF batch run (the shape that triggers it)."""
    a, b = tmp_path / "one.pdf", tmp_path / "two.pdf"
    a.write_bytes(b"%PDF-1.4\n")
    b.write_bytes(b"%PDF-1.4\n")
    return argparse.Namespace(
        pdfs=[str(a), str(b)],
        workers=None,
        resume=None,
        no_retry=False,
        stub=True,
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


@pytest.fixture(autouse=True)
def _dont_actually_spawn(monkeypatch):
    """Stub the batch runner so no subprocesses start."""
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(_ingest_batch, "new_batch", lambda *a, **k: 0)


def test_chat_relay_batch_defaults_to_one_visible_worker(
    _batch_args,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    seen = {}
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(
        _ingest_batch,
        "new_batch",
        lambda *a, **k: seen.update(k) or 0,
    )
    rc = agent_task._cmd_ingest(_batch_args)
    err = capsys.readouterr().err
    assert rc == 0
    assert "defaults to 1 worker" in err
    assert "Pass -w N" in err
    assert "prompts/chat-relay.md" in err
    assert seen["workers"] == 1
    assert seen["relay_watch"] is True


def test_silent_for_an_api_provider(_batch_args, monkeypatch, capsys):
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    seen = {}
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(
        _ingest_batch,
        "new_batch",
        lambda *a, **k: seen.update(k) or 0,
    )
    rc = agent_task._cmd_ingest(_batch_args)
    err = capsys.readouterr().err
    assert rc == 0
    assert "chat-relay" not in err
    assert seen["workers"] == 4
    assert seen["relay_watch"] is False


def test_explicit_parallel_chat_relay_is_honored_and_warned(
    _batch_args,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    _batch_args.workers = 3
    seen = {}
    from researchwiki.tasks import _ingest_batch

    monkeypatch.setattr(
        _ingest_batch,
        "new_batch",
        lambda *a, **k: seen.update(k) or 0,
    )
    assert agent_task._cmd_ingest(_batch_args) == 0
    assert seen["workers"] == 3
    assert seen["relay_watch"] is True
    err = capsys.readouterr().err
    assert "3 concurrent chat-relay workers explicitly requested" in err
    assert "not isolated chat responders" in err


def test_detector_catches_env_override(monkeypatch):
    from researchwiki.agents import model_config

    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    assert model_config.uses_chat_relay() is True
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    assert model_config.uses_chat_relay() is False


def test_malformed_config_fails_before_chat_relay_batch_spawns(
    _batch_args,
    tmp_path,
    monkeypatch,
):
    from researchwiki.agents import model_config
    from researchwiki.tasks import _ingest_batch

    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text("roles: [unterminated\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    spawned = []
    monkeypatch.setattr(
        _ingest_batch,
        "new_batch",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
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


def test_detector_catches_a_single_phase_routed_by_config(monkeypatch):
    # The env var is not the only way in: models.yaml can route individual phases
    # to chat-relay while the rest stay on an API provider. A guard that checked
    # only the env var would miss exactly that mixed config.
    from researchwiki.agents import model_config

    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    real = model_config.for_phase

    def _one_relay_phase(name):
        cfg = real(name)
        if name == "author":
            return type(cfg)(
                provider="chat-relay",
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                reasoning_effort=cfg.reasoning_effort,
                rpm=cfg.rpm,
            )
        return cfg

    monkeypatch.setattr(model_config, "for_phase", _one_relay_phase)
    assert model_config.uses_chat_relay() is True
