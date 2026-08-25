"""A/B model profiles switch model, endpoint, and ingest settings together."""

from __future__ import annotations

import os

from researchwiki.agents import llm, model_config as mc
from researchwiki.tasks import benchmark_fixture as bf


def _write_config(path, *, model: str, endpoint: str, drafts: int) -> None:
    path.write_text(
        f"base_url: {endpoint}\n"
        "ingest:\n"
        f"  n_drafts: {drafts}\n"
        "roles:\n"
        "  author:\n"
        "    provider: openai-compatible\n"
        f"    model: {model}\n",
        encoding="utf-8",
    )


def test_benchmark_profile_switch_clears_every_routing_cache(tmp_path, monkeypatch):
    baseline = tmp_path / "models.baseline.yaml"
    candidate = tmp_path / "models.candidate.yaml"
    _write_config(
        baseline, model="baseline-model", endpoint="https://baseline.invalid/v1", drafts=1,
    )
    _write_config(
        candidate, model="candidate-model", endpoint="https://candidate.invalid/v1", drafts=2,
    )
    monkeypatch.setenv("RW_MODELS_CONFIG", str(baseline))
    mc.clear_caches()
    seen: list[tuple[str, str, int | None]] = []

    def fake_replicate_score(*_args, **_kwargs):
        seen.append((
            mc.for_phase("author").model,
            llm.resolve_openai_endpoint().url,
            mc.default_n_drafts(),
        ))
        return object()

    monkeypatch.setattr(bf, "replicate_score", fake_replicate_score)
    try:
        bf._run_replicate_under_config(
            object(), config_path=str(baseline), n=2, use_llm=False, verbose=False,
        )
        bf._run_replicate_under_config(
            object(), config_path=str(candidate), n=2, use_llm=False, verbose=False,
        )
    finally:
        mc.clear_caches()

    assert seen == [
        ("baseline-model", "https://baseline.invalid/v1", 1),
        ("candidate-model", "https://candidate.invalid/v1", 2),
    ]
    assert os.environ["RW_MODELS_CONFIG"] == str(baseline)
