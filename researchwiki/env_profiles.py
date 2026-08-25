"""Strict dotenv profiles with origin tracking and secret-safe writes.

The CLI gives parent-shell variables precedence over a selected profile.  A
setup wizard that later edits that profile therefore needs more than value
equality to decide ownership: a shell value can be byte-for-byte identical to
the file and still win again on the next invocation.  This module records the
keys the loader itself inserted and exposes that provenance only as internal
process metadata.

Profile writes use a 0600 temporary inode from creation through atomic replace.
That closes the short 0644 window produced by a normal umask before a later
``chmod`` and makes permission failures happen before the destination changes.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import EnvironmentFailure
from .fsatomic import write_text_atomic


ACTIVE_ENV_FILE_VAR = "_RESEARCHWIKI_ENV_FILE"
DOTENV_PROVENANCE_VAR = "_RESEARCHWIKI_DOTENV_PROVENANCE"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REFERENCE_RE = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)
_RESERVED_KEYS = frozenset({ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR})
CREDENTIAL_KEYS = frozenset({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})


class EnvProfileFailure(EnvironmentFailure):
    """An env profile is malformed, unreadable, or cannot be written safely."""


@dataclass(frozen=True)
class Assignment:
    key: str
    value: str
    exported: bool


@dataclass(frozen=True)
class ProfileSnapshot:
    path: Path
    existed: bool
    data: bytes
    text: str
    mode: int | None


def parse_assignment(raw: str, *, path: Path, line_no: int) -> Assignment | None:
    """Parse one supported dotenv line or raise a path/line-rich error."""
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    exported = line.startswith("export ")
    if exported:
        line = line[len("export "):].lstrip()
    if "=" not in line:
        raise EnvProfileFailure(
            f"invalid env profile {path}:{line_no}: expected KEY=value"
        )
    key, _, value = line.partition("=")
    key = key.strip()
    if not _ENV_KEY_RE.fullmatch(key):
        raise EnvProfileFailure(
            f"invalid env profile {path}:{line_no}: invalid variable name {key!r}"
        )
    if key in _RESERVED_KEYS:
        raise EnvProfileFailure(
            f"invalid env profile {path}:{line_no}: {key} is reserved internally"
        )
    value = value.strip()
    if value.startswith(("'", '"')):
        quote = value[0]
        if len(value) < 2 or not value.endswith(quote):
            raise EnvProfileFailure(
                f"invalid env profile {path}:{line_no}: unterminated quoted value"
            )
        value = value[1:-1]
    if "\0" in value:
        raise EnvProfileFailure(
            f"invalid env profile {path}:{line_no}: embedded null byte"
        )
    return Assignment(key=key, value=value, exported=exported)


def parse_profile_text(
    text: str,
    *,
    path: Path,
    reject_duplicates: bool,
) -> list[Assignment]:
    """Validate a complete profile before any caller mutates process state."""
    assignments: list[Assignment] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        assignment = parse_assignment(raw, path=path, line_no=line_no)
        if assignment is None:
            continue
        if reject_duplicates and assignment.key in seen:
            raise EnvProfileFailure(
                f"invalid env profile {path}:{line_no}: duplicate variable "
                f"{assignment.key}"
            )
        seen.add(assignment.key)
        assignments.append(assignment)
    return assignments


def snapshot_profile(
    path: Path,
    *,
    reject_duplicates: bool = False,
) -> ProfileSnapshot:
    """Read and validate one regular-file snapshot from a single open fd.

    ``open`` followed by ``fstat`` and a read from that same descriptor binds
    the bytes and permission mode to one inode.  Separate path-level stat/read
    calls can be raced so the mode check describes a different file from the
    credentials that were loaded.  Non-blocking open also lets us reject a FIFO
    without waiting forever for a writer.  Symlinks to regular files remain
    supported; a dangling symlink is a typed failure rather than optional
    absence. UTF-8 BOM is accepted and removed in ``text``.
    """
    path = Path(path)
    fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        # `Path.exists()` hides a broken symlink. A selected path is still
        # present in that case and must not masquerade as the optional absent
        # root profile, which could silently select built-in routing instead.
        if path.is_symlink():
            raise EnvProfileFailure(
                f"cannot read env profile {path}: broken symbolic link"
            ) from exc
        return ProfileSnapshot(
            path=path, existed=False, data=b"", text="", mode=None
        )
    except OSError as exc:
        raise EnvProfileFailure(f"cannot read env profile {path}: {exc}") from exc
    try:
        inode_mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(inode_mode):
            raise EnvProfileFailure(
                f"cannot read env profile {path}: not a regular file"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            data = handle.read()
        text = data.decode("utf-8-sig")
    except EnvProfileFailure:
        raise
    except (OSError, UnicodeError) as exc:
        raise EnvProfileFailure(f"cannot read env profile {path}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    parse_profile_text(text, path=path, reject_duplicates=reject_duplicates)
    return ProfileSnapshot(
        path=path, existed=True, data=data, text=text, mode=inode_mode
    )


def credential_keys_in_text(text: str, *, path: Path) -> set[str]:
    """Literal credentials in profile text (values are never returned).

    An exact ``$NAME``/``${NAME}`` reference contains no secret bytes itself,
    so it does not require private file permissions merely because its target
    variable is a credential. The referenced shell value is resolved only at
    load time.
    """
    return {
        assignment.key
        for assignment in parse_profile_text(
            text, path=path, reject_duplicates=False
        )
        if (
            assignment.key in CREDENTIAL_KEYS
            and assignment.value
            and _ENV_REFERENCE_RE.fullmatch(assignment.value) is None
        )
    }


def credential_keys(snapshot: ProfileSnapshot) -> set[str]:
    return credential_keys_in_text(snapshot.text, path=snapshot.path)


def require_private_credentials(snapshot: ProfileSnapshot) -> None:
    """Fail closed when an existing secret profile is readable by other users."""
    if not snapshot.existed or not credential_keys(snapshot) or os.name != "posix":
        return
    if snapshot.mode is None:  # defensive: an existing snapshot always has it
        raise EnvProfileFailure(
            f"cannot inspect env profile permissions {snapshot.path}: mode unavailable"
        )
    mode = stat.S_IMODE(snapshot.mode)
    if mode & 0o077:
        raise EnvProfileFailure(
            f"env profile {snapshot.path} contains credentials but mode is "
            f"{mode:04o}; run `chmod 600 {snapshot.path}` and retry"
        )


def clear_loader_metadata() -> None:
    """Discard origin metadata before selecting a profile for this invocation."""
    os.environ.pop(ACTIVE_ENV_FILE_VAR, None)
    os.environ.pop(DOTENV_PROVENANCE_VAR, None)


def _record_provenance(path: Path, keys: set[str]) -> None:
    payload = {"path": str(path.resolve()), "keys": sorted(keys)}
    os.environ[DOTENV_PROVENANCE_VAR] = json.dumps(payload, separators=(",", ":"))


def load_profile(path: Path, *, required: bool) -> None:
    """Strictly validate then apply one profile with shell-first precedence."""
    clear_loader_metadata()
    path = Path(path)
    snapshot = snapshot_profile(path, reject_duplicates=True)
    if not snapshot.existed:
        if required:
            raise EnvProfileFailure(
                f"env profile does not exist: {path} — fix --env-file or remove it"
            )
        return
    require_private_credentials(snapshot)
    assignments = parse_profile_text(
        snapshot.text, path=path, reject_duplicates=True
    )
    # References mean "already exported by the parent process", not "loaded
    # from an earlier line in this same file". Freezing the source environment
    # prevents a public profile-local alias from laundering literal credentials.
    ambient_environment = dict(os.environ)
    inserted: dict[str, str] = {}
    try:
        for assignment in assignments:
            if assignment.key in os.environ:
                continue
            value = assignment.value
            reference = _ENV_REFERENCE_RE.fullmatch(value)
            if reference:
                source = reference.group(1) or reference.group(2)
                if source not in ambient_environment:
                    continue
                value = ambient_environment[source]
            os.environ[assignment.key] = value
            inserted[assignment.key] = value
        os.environ[ACTIVE_ENV_FILE_VAR] = str(path.resolve())
        _record_provenance(path, set(inserted))
    except (OSError, TypeError, ValueError) as exc:
        for key, value in inserted.items():
            if os.environ.get(key) == value:
                os.environ.pop(key, None)
        clear_loader_metadata()
        raise EnvProfileFailure(f"cannot apply env profile {path}: {exc}") from exc


def loaded_from_profile(path: Path, key: str) -> bool:
    """Whether this invocation's loader inserted ``key`` from exactly ``path``."""
    try:
        payload = json.loads(os.environ.get(DOTENV_PROVENANCE_VAR, ""))
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("path") == str(Path(path).resolve())
        and isinstance(payload.get("keys"), list)
        and key in payload["keys"]
    )


def update_provenance(path: Path, *, added: set[str], removed: set[str]) -> None:
    """Reflect keys the wizard just installed into or removed from its profile."""
    current = {
        key for key in added | removed
        if loaded_from_profile(path, key)
    }
    try:
        payload = json.loads(os.environ.get(DOTENV_PROVENANCE_VAR, ""))
        if payload.get("path") == str(Path(path).resolve()):
            current |= set(payload.get("keys", []))
    except (AttributeError, json.JSONDecodeError, TypeError):
        current = set()
    current.difference_update(removed)
    current.update(added)
    _record_provenance(Path(path), current)


def assignment_value(snapshot: ProfileSnapshot, key: str) -> str | None:
    """Return the first value for ``key``, matching loader precedence."""
    for assignment in parse_profile_text(
        snapshot.text, path=snapshot.path, reject_duplicates=False
    ):
        if assignment.key == key:
            return assignment.value
    return None


def effective_assignment_value(
    snapshot: ProfileSnapshot,
    key: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a profile assignment against an ambient environment snapshot."""
    value = assignment_value(snapshot, key)
    if value is None:
        return None
    reference = _ENV_REFERENCE_RE.fullmatch(value)
    if reference is None:
        return value
    source = reference.group(1) or reference.group(2)
    values = os.environ if environment is None else environment
    return values.get(source)


def edit_profile_text(
    snapshot: ProfileSnapshot,
    *,
    updates: dict[str, str] | None = None,
    removals: set[str] | None = None,
) -> tuple[str, set[str]]:
    """Render one atomic profile edit while preserving unrelated raw lines."""
    updates = dict(updates or {})
    removals = set(removals or ())
    remaining = dict(updates)
    removed: set[str] = set()
    out: list[str] = []
    for line_no, raw in enumerate(snapshot.text.splitlines(), 1):
        assignment = parse_assignment(raw, path=snapshot.path, line_no=line_no)
        if assignment and assignment.key in removals:
            removed.add(assignment.key)
            continue
        if assignment and assignment.key in updates:
            if assignment.key in remaining:
                prefix = "export " if assignment.exported else ""
                out.append(
                    f'{prefix}{assignment.key}="{remaining.pop(assignment.key)}"'
                )
            continue
        out.append(raw)
    for key, value in remaining.items():
        out.append(f'{key}="{value}"')
    text = "\n".join(out).rstrip("\n")
    return (text + "\n" if text else ""), removed


def _write_profile_bytes_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` from a 0600 inode; never expose secret bytes as 0644."""
    path = Path(path)
    tmp_path: Path | None = None
    fd = -1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        tmp_path = Path(raw_tmp)
        os.fchmod(fd, 0o600)
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
            raise OSError("temporary profile permissions are not 0600")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise EnvProfileFailure(
            f"cannot securely write env profile {path}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def write_profile_atomic(path: Path, text: str) -> None:
    _write_profile_bytes_atomic(Path(path), text.encode("utf-8"))


def restore_profile(snapshot: ProfileSnapshot) -> None:
    """Restore an exact byte snapshot, still using a secret-safe 0600 inode."""
    if snapshot.existed:
        _write_profile_bytes_atomic(snapshot.path, snapshot.data)
        return
    try:
        snapshot.path.unlink(missing_ok=True)
    except OSError as exc:
        raise EnvProfileFailure(
            f"cannot restore absent env profile {snapshot.path}: {exc}"
        ) from exc


_MISSING_ENV = object()


def _restore_config(path: Path, existed: bool, data: bytes) -> None:
    try:
        if existed:
            write_text_atomic(path, data.decode("utf-8"))
        else:
            path.unlink(missing_ok=True)
    except (OSError, UnicodeError) as exc:
        raise EnvironmentFailure(f"cannot restore {path}: {exc}") from exc


def _restore_process_env(before: dict[str, object]) -> None:
    for key, value in before.items():
        if value is _MISSING_ENV:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def commit_profile_and_config(
    *,
    config_path: Path,
    apply_config: Callable[[], bool],
    env_path: Path,
    updates: dict[str, str],
    removals: set[str],
    remove_openai_key: bool,
    replace_openai_key: bool,
    protected_credential_keys: set[str] | None = None,
) -> tuple[bool, set[str], set[str]]:
    """Commit config + profile as one rollback-capable routing transaction.

    A credential whose endpoint changes is taken out first. If the final
    profile write fails, config is rolled back before that credential is
    restored. A rollback failure leaves the profile credentialless instead of
    pairing an old secret with a newly written endpoint. The two OpenAI flags
    preserve the original call contract; ``protected_credential_keys`` extends
    the same safety boundary to other SDK-controlled endpoints.
    """
    profile_before = snapshot_profile(env_path)
    require_private_credentials(profile_before)
    final_removals = set(removals)
    if remove_openai_key:
        final_removals.add("OPENAI_API_KEY")
    final_text, removed_file = edit_profile_text(
        profile_before, updates=updates, removals=final_removals
    )
    try:
        config_existed = config_path.exists()
        config_data = config_path.read_bytes() if config_existed else b""
    except OSError as exc:
        raise EnvironmentFailure(f"cannot snapshot {config_path}: {exc}") from exc

    protected = set(protected_credential_keys or ())
    if remove_openai_key or replace_openai_key:
        protected.add("OPENAI_API_KEY")
    safe_keys = {
        key for key in protected
        if assignment_value(profile_before, key) is not None
    }
    touched = final_removals | set(updates) | safe_keys | {DOTENV_PROVENANCE_VAR}
    process_before = {
        key: os.environ.get(key, _MISSING_ENV) for key in touched
    }
    profile_mutated = False
    config_attempted = False
    try:
        if safe_keys:
            safe_text, _ = edit_profile_text(
                profile_before, removals=safe_keys
            )
            write_profile_atomic(env_path, safe_text)
            profile_mutated = True
            for key in safe_keys:
                os.environ.pop(key, None)

        config_attempted = True
        if not apply_config():
            if profile_mutated:
                restore_profile(profile_before)
            _restore_process_env(process_before)
            return False, set(), set()

        final_bytes = final_text.encode("utf-8")
        if (updates or removed_file) and (
            profile_mutated
            or not profile_before.existed
            or final_bytes != profile_before.data
        ):
            write_profile_atomic(env_path, final_text)
            profile_mutated = True

        removed_process = {
            key for key in final_removals if key in os.environ
        }
        for key in final_removals:
            os.environ.pop(key, None)
        os.environ.update(updates)
        # A protected credential may be intentionally reused at the new
        # endpoint. It was absent only during the multi-file transition, so put
        # its prior effective value back after the profile and config agree.
        for key in safe_keys - final_removals - set(updates):
            value = process_before[key]
            if value is _MISSING_ENV:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        update_provenance(env_path, added=set(updates), removed=final_removals)
        return True, removed_file, removed_process
    except BaseException as exc:
        for key in safe_keys:
            os.environ.pop(key, None)
        config_restored = not config_attempted
        try:
            if config_attempted:
                _restore_config(config_path, config_existed, config_data)
                config_restored = True
        except Exception:
            config_restored = False
        profile_restored = not profile_mutated
        if config_restored and profile_mutated:
            try:
                restore_profile(profile_before)
                profile_restored = True
            except Exception:
                profile_restored = False
        if config_restored and profile_restored:
            _restore_process_env(process_before)
            state = "the previous config and env profile were restored"
        else:
            state = (
                "rollback was incomplete; "
                f"{', '.join(sorted(safe_keys))} was left unset so the new "
                "endpoint cannot receive the old credential"
                if safe_keys
                else "rollback was incomplete; inspect config and env before retrying"
            )
        if config_restored and profile_restored and not isinstance(exc, Exception):
            raise
        raise EnvironmentFailure(
            f"provider setup transaction failed ({exc}); {state}"
        ) from exc
