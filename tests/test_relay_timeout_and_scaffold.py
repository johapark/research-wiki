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

import pytest

from researchwiki.agents import relay


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
