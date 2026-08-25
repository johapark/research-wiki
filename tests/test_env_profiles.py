"""Explicit dotenv profiles select one routing environment, fail closed.

The repository's default ``.env`` remains optional and auto-loaded.  Named
profiles are deliberately explicit so choosing one cannot be shadowed by a
different root profile.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from researchwiki import __main__ as cli
from researchwiki import env_profiles


_REAL_LOAD_DOTENV = cli._load_dotenv


@pytest.fixture
def fake_task(tmp_path, monkeypatch):
    (tmp_path / "wiki").mkdir()
    monkeypatch.chdir(tmp_path)
    seen: dict[str, str | None] = {}

    mod = types.ModuleType("researchwiki.tasks.faketask")
    mod.__doc__ = "Synthetic env-profile task."

    def run(argv):
        seen["config"] = os.environ.get("RW_MODELS_CONFIG")
        return 0

    mod.main = run
    monkeypatch.setitem(sys.modules, "researchwiki.tasks.faketask", mod)
    monkeypatch.setattr(cli, "_discover_tasks", lambda: {"faketask": "faketask"})
    # Undo tests/conftest.py's suite-wide dotenv isolation for this module.
    monkeypatch.setattr(cli, "_load_dotenv", _REAL_LOAD_DOTENV)
    # `_load_dotenv` mutates os.environ directly rather than through
    # monkeypatch, so explicitly restore this key after each test. Otherwise a
    # selected profile leaks into later tests and makes their implicit missing
    # config look like an explicit fail-closed override.
    sentinel = object()
    previous = {
        key: os.environ.pop(key, sentinel)
        for key in (
            "RW_MODELS_CONFIG",
            "LITELLM_API_KEY",
            "OPENAI_API_KEY",
            cli._ACTIVE_ENV_FILE_VAR,
            env_profiles.DOTENV_PROVENANCE_VAR,
        )
    }
    try:
        yield tmp_path, seen
    finally:
        for key, value in previous.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_named_profile_is_loaded_instead_of_root_env(fake_task):
    root, seen = fake_task
    (root / ".env").write_text("RW_MODELS_CONFIG=models.chatgpt.yaml\n")
    (root / ".env.litellm").write_text(
        "RW_MODELS_CONFIG=models.openai-compatible.yaml\n"
    )

    assert cli.main(["--env-file", ".env.litellm", "faketask"]) == 0
    assert seen["config"] == "models.openai-compatible.yaml"
    assert os.environ[cli._ACTIVE_ENV_FILE_VAR] == str(
        (root / ".env.litellm").resolve()
    )


def test_env_file_equals_form_and_export_syntax(fake_task):
    root, seen = fake_task
    (root / ".env.local").write_text(
        "export RW_MODELS_CONFIG=models.lmstudio.yaml\n"
    )

    assert cli.main(["--env-file=.env.local", "faketask"]) == 0
    assert seen["config"] == "models.lmstudio.yaml"


def test_missing_named_profile_is_environment_error(fake_task, capsys):
    assert cli.main(["--env-file", ".env.missing", "faketask"]) == 2
    assert "env profile does not exist" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics are POSIX-specific")
def test_broken_optional_root_profile_is_not_treated_as_absent(fake_task, capsys):
    root, _seen = fake_task
    (root / ".env").symlink_to("missing-profile")

    assert cli.main(["faketask"]) == 2
    err = capsys.readouterr().err
    assert "broken symbolic link" in err
    assert "Traceback" not in err


def test_non_regular_profile_is_environment_error(fake_task, capsys):
    root, _seen = fake_task
    (root / ".env").mkdir()

    assert cli.main(["faketask"]) == 2
    assert "not a regular file" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="FIFO semantics are POSIX-specific")
def test_fifo_profile_is_rejected_without_reading_from_it(tmp_path):
    profile = tmp_path / ".env.fifo"
    os.mkfifo(profile)

    with pytest.raises(env_profiles.EnvProfileFailure, match="not a regular file"):
        env_profiles.snapshot_profile(profile)


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics are POSIX-specific")
def test_symlink_to_regular_profile_remains_supported(tmp_path):
    target = tmp_path / "real-profile"
    target.write_text('OPENAI_API_KEY="secret"\n')
    target.chmod(0o600)
    profile = tmp_path / ".env.link"
    profile.symlink_to(target)

    snapshot = env_profiles.snapshot_profile(profile)

    assert snapshot.text == 'OPENAI_API_KEY="secret"\n'
    env_profiles.warn_permissive_credentials(snapshot)


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics are POSIX-specific")
def test_atomic_profile_write_preserves_symlink_and_updates_target(tmp_path):
    target = tmp_path / "real-profile"
    target.write_text('OPENAI_API_KEY="old"\n')
    target.chmod(0o600)
    profile = tmp_path / ".env.link"
    profile.symlink_to(target.name)

    env_profiles.write_profile_atomic(profile, 'OPENAI_API_KEY="new"\n')

    assert profile.is_symlink()
    assert str(profile.readlink()) == target.name
    assert target.read_text() == 'OPENAI_API_KEY="new"\n'
    assert target.stat().st_mode & 0o777 == 0o600


def test_permission_warning_uses_mode_from_the_inode_that_was_read(
    tmp_path, capsys,
):
    profile = tmp_path / ".env"
    profile.write_text('OPENAI_API_KEY="secret"\n')
    profile.chmod(0o644)

    snapshot = env_profiles.snapshot_profile(profile)
    # A later path lookup now sees a different mode. The snapshot must keep the
    # mode paired with the bytes it actually read and still warn.
    profile.chmod(0o600)

    env_profiles.warn_permissive_credentials(snapshot)

    assert "mode is 0644" in capsys.readouterr().err


def test_path_swap_during_fstat_still_reads_the_open_inode(tmp_path, monkeypatch):
    profile = tmp_path / ".env"
    profile.write_text('OPENAI_API_KEY="opened-inode"\n')
    profile.chmod(0o600)
    replacement = tmp_path / "replacement"
    replacement.write_text('OPENAI_API_KEY="path-after-swap"\n')
    replacement.chmod(0o600)
    real_fstat = env_profiles.os.fstat
    swapped = False

    def swap_then_fstat(fd):
        nonlocal swapped
        if not swapped:
            swapped = True
            profile.unlink()
            replacement.replace(profile)
        return real_fstat(fd)

    monkeypatch.setattr(env_profiles.os, "fstat", swap_then_fstat)

    snapshot = env_profiles.snapshot_profile(profile)

    assert snapshot.text == 'OPENAI_API_KEY="opened-inode"\n'
    assert profile.read_text() == 'OPENAI_API_KEY="path-after-swap"\n'


@pytest.mark.parametrize("argv", [["--env-file"], ["--env-file", "--help"]])
def test_env_file_without_path_is_user_error(fake_task, capsys, argv):
    assert cli.main(argv) == 1
    assert "requires a path" in capsys.readouterr().err


def test_utf8_bom_does_not_become_part_of_first_key(fake_task):
    root, seen = fake_task
    (root / ".env.windows").write_bytes(
        b"\xef\xbb\xbfRW_MODELS_CONFIG=models.gemini.yaml\r\n"
    )

    assert cli.main(["--env-file", ".env.windows", "faketask"]) == 0
    assert seen["config"] == "models.gemini.yaml"


def test_loader_records_only_keys_it_actually_inserted(fake_task, monkeypatch):
    root, _seen = fake_task
    profile = root / ".env.keys"
    profile.write_text(
        'OPENAI_API_KEY="same-secret"\nRW_MODELS_CONFIG=models.gemini.yaml\n'
    )
    profile.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "same-secret")

    cli._load_dotenv(profile)

    assert not env_profiles.loaded_from_profile(profile, "OPENAI_API_KEY")
    assert env_profiles.loaded_from_profile(profile, "RW_MODELS_CONFIG")


def test_loader_marks_profile_owned_key_when_it_inserted_it(fake_task):
    root, _seen = fake_task
    profile = root / ".env.keys"
    profile.write_text('OPENAI_API_KEY="profile-secret"\n')
    profile.chmod(0o600)

    cli._load_dotenv(profile)

    assert os.environ["OPENAI_API_KEY"] == "profile-secret"
    assert env_profiles.loaded_from_profile(profile, "OPENAI_API_KEY")


@pytest.mark.parametrize(
    "contents",
    [
        "LITELLM_API_KEY=profile-secret\nOPENAI_API_KEY=$LITELLM_API_KEY\n",
        "OPENAI_API_KEY=${LITELLM_API_KEY}\nLITELLM_API_KEY=profile-secret\n",
    ],
)
def test_reference_never_resolves_a_value_from_the_same_profile(
    fake_task, contents,
):
    root, _seen = fake_task
    profile = root / ".env.alias"
    profile.write_text(contents, encoding="utf-8")

    cli._load_dotenv(profile)

    assert os.environ["LITELLM_API_KEY"] == "profile-secret"
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize(
    "contents,diagnostic",
    [
        (b"\xff\xfe\x00", "cannot read env profile"),
        (b"RW_MODELS_CONFIG\n", "expected KEY=value"),
        (b"RW_MODELS_CONFIG=one\nRW_MODELS_CONFIG=two\n", "duplicate variable"),
        (b'RW_MODELS_CONFIG="unterminated\n', "unterminated quoted value"),
        (b"RW_MODELS_CONFIG=ok\x00bad\n", "embedded null byte"),
    ],
)
def test_malformed_explicit_profile_is_environment_error(
    fake_task, capsys, contents, diagnostic,
):
    root, _seen = fake_task
    (root / ".env.bad").write_bytes(contents)

    assert cli.main(["--env-file", ".env.bad", "faketask"]) == 2
    err = capsys.readouterr().err
    assert diagnostic in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [
        (b"OPENAI_API_KEY=must-not-leak\nBROKEN\n", "expected KEY=value"),
        (b"BAD KEY=value\n", "invalid variable name"),
        (b'OPENAI_API_KEY="unterminated\n', "unterminated quoted value"),
        (b"RW_MODELS_CONFIG=one\nRW_MODELS_CONFIG=two\n", "duplicate variable"),
    ],
)
def test_present_default_env_is_strict_and_applied_only_after_full_validation(
    fake_task, capsys, contents, diagnostic,
):
    root, _seen = fake_task
    (root / ".env").write_bytes(contents)

    assert cli.main(["faketask"]) == 2
    assert diagnostic in capsys.readouterr().err
    assert "OPENAI_API_KEY" not in os.environ
    assert "RW_MODELS_CONFIG" not in os.environ


def test_loader_warns_for_group_readable_credential_profile(fake_task, capsys):
    root, _seen = fake_task
    profile = root / ".env"
    profile.write_text('OPENAI_API_KEY="secret"\n')
    profile.chmod(0o644)

    assert cli.main(["faketask"]) == 0
    err = capsys.readouterr().err
    assert "mode is 0644" in err
    assert "chmod 600" in err
    assert os.environ["OPENAI_API_KEY"] == "secret"
