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


@pytest.fixture(autouse=True)
def _isolate_state_db(tmp_path, monkeypatch):
    """Point `state.db` at a per-test temp file.

    Same class of trap as `_no_dotenv_leak`, and invisible at the call site for
    the same reason. `db_path()` resolves from the *current working directory*
    (`_repo_key()` over `wiki_root()`), so any test that reaches a write path
    lands in the developer's real per-repo DB. `backlinks.append_related_paper`
    is the live example: it calls `commit_page` → `db.upsert_page` whenever it
    actually appends (`backlinks.py:81`), so `test_appends_when_absent` — the one
    backlinks test where the append succeeds — inserted a `papers` row keyed on a
    pytest tmp dir into the real DB on every suite run. That row then showed up
    as `db_drift.extra` in `researchwiki lint` and as a phantom zero-claim paper,
    i.e. the suite manufactured corpus defects.

    `RESEARCHWIKI_DB_PATH` is the documented override and wins over the per-repo
    path (`db/connection.py:91-95`). Nothing memoizes it, so a per-test value
    takes effect immediately. Tests that assert on path *resolution* already
    `monkeypatch.delenv` it (see `tests/test_db_path.py`), and monkeypatch
    unwinds in LIFO order, so this fixture doesn't fight them.
    """
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "_state" / "state.db"))
