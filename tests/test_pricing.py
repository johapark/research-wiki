"""`researchwiki.pricing` — rate lookup, dated model IDs, time-boxed rates.

The bug this module was extracted to fix: the old table was a dict literal in
`tasks/status.py` keyed on bare family names (`claude-haiku-4-5`), but the API
echoes back dated build IDs (`claude-haiku-4-5-20251001`). `dict.get` missed, so
429 calls over 2.7M input tokens priced at $0.00 in every report — and two of the
three rates it did carry were a release out of date. Across all recorded history
that understated spend by 22%.

So the prefix matching and the "unpriced is visible" behavior are the load-bearing
parts, and they're what these tests pin.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from researchwiki import pricing


@pytest.fixture(autouse=True)
def _clean_cache():
    pricing.reset_cache()
    yield
    pricing.reset_cache()


# ---------- the shipped table ----------

def test_shipped_table_parses_and_is_dated():
    assert pricing.as_of(), "config/pricing.yaml must carry an as_of date"
    dt.date.fromisoformat(pricing.as_of())          # raises if malformed
    assert set(pricing.sources()) >= {"anthropic", "openai"}


def test_every_shipped_entry_is_well_formed():
    """A typo'd rate silently becomes an unpriced model, so validate the file."""
    with pricing.pricing_path().open(encoding="utf-8") as fh:
        table = (yaml.safe_load(fh) or {})["models"]
    for key, entry in table.items():
        variants = entry if isinstance(entry, list) else [entry]
        for v in variants:
            assert isinstance(v, dict), f"{key}: not a mapping"
            assert isinstance(v["in"], (int, float)) and v["in"] >= 0, f"{key}: bad in"
            assert isinstance(v["out"], (int, float)) and v["out"] >= 0, f"{key}: bad out"
            assert v.get("provider"), f"{key}: missing provider"
        assert pricing.resolve(key) is not None, f"{key} does not resolve"


def test_models_this_repo_actually_configures_are_priced():
    """Guards the original failure: a model the configs route to must not be
    silently free. Local backends are excluded — they genuinely cost nothing."""
    for model in ("claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-7",
                  "claude-haiku-4-5-20251001", "gpt-5.6-luna"):
        assert pricing.resolve(model) is not None, f"{model} is unpriced"


# ---------- prefix matching ----------

def test_dated_build_id_resolves_to_its_family():
    r = pricing.resolve("claude-haiku-4-5-20251001")
    assert r is not None and r.model_key == "claude-haiku-4-5"
    assert (r.input_per_mtok, r.output_per_mtok) == (1.0, 5.0)


def test_longest_prefix_wins_so_a_family_cannot_be_shadowed():
    """`claude-haiku-3-5` must not match a shorter `claude-haiku-*` key."""
    assert pricing.resolve("claude-haiku-3-5").model_key == "claude-haiku-3-5"
    assert pricing.resolve("claude-haiku-4-5").model_key == "claude-haiku-4-5"


def test_unknown_and_sentinel_models_are_unpriced():
    for m in ("gemma-4-31b-it", "qwen3.6-35b-a3b-mlx", "(local)", "(skipped)",
              "(no calls)", "stub", "", None):
        assert pricing.resolve(m) is None
        assert pricing.estimate_usd(m, 1_000_000, 1_000_000) == 0.0


def test_unpriced_models_separates_local_from_missing_cloud():
    got = pricing.unpriced_models(
        ["claude-sonnet-5", "gemma-4-31b-it", "(local)", "(skipped)"]
    )
    assert got == ["gemma-4-31b-it"], "sentinels must not be reported as unpriced"


# ---------- arithmetic ----------

def test_cost_is_per_million_tokens():
    r = pricing.resolve("claude-sonnet-4-6")        # $3 in / $15 out
    assert r.usd(1_000_000, 0) == pytest.approx(3.0)
    assert r.usd(0, 1_000_000) == pytest.approx(15.0)
    assert r.usd(500_000, 100_000) == pytest.approx(1.5 + 1.5)


def test_estimate_matches_a_hand_computed_figure():
    # 2.73M in / 302k out of Haiku 4.5 at $1/$5.
    assert pricing.estimate_usd("claude-haiku-4-5-20251001", 2_730_649, 302_406) \
        == pytest.approx(2.730649 + 1.51203, rel=1e-6)


def test_zero_tokens_is_zero():
    assert pricing.estimate_usd("claude-sonnet-5", 0, 0) == 0.0


# ---------- time-boxed rates ----------

@pytest.mark.parametrize("day,expected", [
    ("2026-08-03", (2.0, 10.0)),     # inside the introductory window
    ("2026-08-31", (2.0, 10.0)),     # `until` is inclusive
    ("2026-09-01", (3.0, 15.0)),     # lapsed
    ("2030-01-01", (3.0, 15.0)),     # far future falls through to standard
])
def test_sonnet_5_introductory_rate_lapses_on_the_stated_date(day, expected):
    r = pricing.resolve("claude-sonnet-5", today=dt.date.fromisoformat(day))
    assert (r.input_per_mtok, r.output_per_mtok) == expected


def test_a_single_rate_entry_ignores_the_date():
    a = pricing.resolve("claude-sonnet-4-6", today=dt.date(2026, 1, 1))
    b = pricing.resolve("claude-sonnet-4-6", today=dt.date(2030, 1, 1))
    assert a == b


# ---------- degradation ----------

def test_a_missing_table_degrades_instead_of_raising(tmp_path, monkeypatch):
    """`status` must still print a report when the table is gone."""
    monkeypatch.setenv("RW_PRICING_CONFIG", str(tmp_path / "nope.yaml"))
    pricing.reset_cache()
    assert pricing.as_of() == ""
    assert pricing.resolve("claude-sonnet-5") is None
    assert pricing.estimate_usd("claude-sonnet-5", 10**6, 10**6) == 0.0


def test_a_malformed_table_degrades_instead_of_raising(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [this is: not a mapping\n")
    monkeypatch.setenv("RW_PRICING_CONFIG", str(bad))
    pricing.reset_cache()
    assert pricing.resolve("claude-sonnet-5") is None


def test_an_override_table_is_honoured(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({
        "as_of": "2099-01-01",
        "models": {"my-model": {"in": 1.0, "out": 2.0, "provider": "x"}},
    }))
    monkeypatch.setenv("RW_PRICING_CONFIG", str(custom))
    pricing.reset_cache()
    assert pricing.as_of() == "2099-01-01"
    assert pricing.estimate_usd("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert pricing.resolve("claude-sonnet-5") is None


# ---------- callers ----------

def test_status_and_insights_share_one_table():
    """They used to, via `insights` importing a private constant out of
    `status`. Both must now read the module, so a rate fix lands in both."""
    import researchwiki.tasks.insights as insights
    import researchwiki.tasks.status as status

    assert not hasattr(status, "_PRICING")
    assert not hasattr(status, "_PRICING_AS_OF")
    assert insights._estimate_usd("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)
