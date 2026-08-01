"""Suite-wide isolation.

CONTRIBUTING requires the test suite to be hermetic — no network, no LLM calls,
no model downloads. Reading the developer's `.env` breaks that in a subtler way:
it makes results depend on an untracked file on one machine.

`researchwiki.__main__.main` calls `_load_dotenv()` before dispatching, which
`os.environ.setdefault`s every key in the repo's `.env` for the rest of the
process. This repo's `.env` sets `RW_MODELS_CONFIG`, and `model_config` reads
that env var — so a single test that drove the CLI end-to-end silently retargeted
the model-config loader for every test that ran after it, and
`tests/test_model_config.py` failed 18 assertions purely on ordering. Passing
alone, failing in the suite.

Neutralized here rather than per-file because the trap is invisible at the call
site: nothing about `cli.main([...])` suggests it mutates global state. A test
that genuinely wants `.env` semantics should set the variables it needs with
`monkeypatch.setenv`, which unwinds.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv_leak(monkeypatch):
    from researchwiki import __main__ as cli
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
