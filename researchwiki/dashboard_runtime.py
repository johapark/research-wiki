"""Agent-visible ownership and authenticated control of a local dashboard.

The receipt is per repository. Status verifies the running server's identity;
stop asks that server to shut down instead of killing a possibly reused PID.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler

from .paths import wiki_root


def repository_id() -> str:
    return hashlib.sha256(str(wiki_root().resolve()).encode()).hexdigest()[:24]


def runtime_path() -> Path:
    return wiki_root() / ".researchwiki" / "dashboard.json"


def read_runtime() -> dict:
    try:
        row = json.loads(runtime_path().read_text(encoding="utf-8"))
        return row if isinstance(row, dict) else {}
    except (OSError, ValueError):
        return {}


def request_runtime(row: dict, *, stop: bool = False) -> dict:
    """Only ever contact loopback, irrespective of what's in the receipt."""
    port = row.get("port")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        return {}
    if row.get("repository_id") != repository_id():
        return {}
    route = "/api/shutdown" if stop else "/api/health"
    headers = {"X-Dashboard-Control": str(row.get("control_token", ""))} if stop else {}
    req = Request(f"http://127.0.0.1:{port}{route}", data=b"" if stop else None, headers=headers)
    try:
        # Local management never follows redirects or uses an ambient proxy.
        from urllib.request import HTTPRedirectHandler

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=2) as response:
            health = json.load(response)
    except (OSError, URLError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, OSError) and reason.errno in {errno.EACCES, errno.EPERM}:
            raise PermissionError("Local dashboard access is blocked by this execution context; use the agent's approved permission path.") from exc
        return {}
    except ValueError:
        return {}
    if not isinstance(health, dict) or health.get("server_id") != row.get("server_id") or health.get("repository_id") != repository_id():
        return {}
    return health


def status() -> dict:
    row = read_runtime()
    health = request_runtime(row)
    if not health:
        return {"running": False, "repository": wiki_root().name}
    return {"running": True, "url": f"http://127.0.0.1:{row['port']}",
            "pid": row.get("pid"), "owner_pid": row.get("owner_pid"),
            "session": row.get("session", ""), "server_id": row["server_id"],
            "repository": wiki_root().name, "started_at": row.get("started_at")}


def stop() -> bool:
    row = read_runtime()
    # Establish identity before sending a control request.
    if not request_runtime(row):
        return False
    if not request_runtime(row, stop=True):
        return False
    for _ in range(30):
        if not request_runtime(row):
            return True
        time.sleep(.1)
    return False


def owner_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is not a harmless probe on Windows.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied means it exists.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
