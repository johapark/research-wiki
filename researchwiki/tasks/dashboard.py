"""Browse the wiki in a standalone HTML dashboard, offline or on localhost.

Zero model calls. Markdown is canonical; browser bookmarks and appearance are
stored locally. --serve refreshes the dashboard when new pages arrive.
"""

from __future__ import annotations

import argparse
import json
import datetime as dt
import os
import sys
import threading
from pathlib import Path

from ..dashboard import build_dashboard, render_html
from ..fsatomic import write_text_atomic
from ..paths import wiki_dir, wiki_root

DEFAULT_OUT = Path("output/dashboard.html")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki dashboard", description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="HTML snapshot path (default: output/dashboard.html)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--serve", action="store_true", help="Run a live, read-only dashboard on localhost")
    modes.add_argument("--json", action="store_true", help="Print dashboard data instead of HTML")
    # --json remains a format flag for --status; use explicit validation below.
    parser.add_argument("--status", action="store_true", help="Report the agent-managed server's health")
    parser.add_argument("--stop", action="store_true", help="Stop this repository's managed server")
    parser.add_argument("--port", type=int, default=8765, help="Local server port (default: 8765)")
    parser.add_argument("--session", default="", help="Label the chat-agent session that owns the server")
    parser.add_argument("--owner-pid", type=int, help="Stop when this session-lifetime process exits")
    parser.add_argument("--vault", help="Obsidian vault name (default: repository directory name)")
    parser.add_argument("--open", dest="open_browser", action="store_true", help="Open the dashboard in your browser")
    args = parser.parse_args(argv)
    if args.status or args.stop:
        if args.status and args.stop or args.serve or args.stop and args.json:
            parser.error("--status and --stop are separate management commands")
        return _manage(args.status, args.json)
    if not wiki_dir().is_dir():
        print("error: wiki/ not found — run from the repository root", file=sys.stderr)
        return 2
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.owner_pid is not None and args.owner_pid <= 0:
        parser.error("--owner-pid must be a positive session-lifetime process ID")
    if args.serve:
        return _serve(args.port, args.vault, args.open_browser, args.session, args.owner_pid)
    # An HTML export must never overwrite a canonical source document.
    out = args.out.absolute()
    if out.suffix.lower() != ".html" or any(out.resolve().is_relative_to((wiki_root() / d).resolve()) for d in ("wiki", "papers", "inbox")):
        print("error: choose an .html output outside wiki/, papers/, and inbox/", file=sys.stderr)
        return 2
    data = build_dashboard(out=out, vault=args.vault)
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
        return 0
    try:
        write_text_atomic(out, render_html(data))
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out} · {len(data['pages'])} pages · {out.stat().st_size / 1024 / 1024:.1f} MB · works offline")
    if args.open_browser:
        _open(out.as_uri())
    return 0


def _open(url: str) -> None:
    import webbrowser
    if not webbrowser.open(url):
        print(f"Open {url} in your browser.")


def _manage(report: bool, as_json: bool) -> int:
    from ..dashboard_runtime import status, stop
    if report:
        row = status()
        print(json.dumps(row) if as_json else (f"Running: {row['url']} · session: {row['session'] or 'unlabelled'}" if row['running'] else "Dashboard server is not running."))
        return 0
    if not status()["running"]:
        print("Dashboard server is already stopped.")
        return 0
    if stop():
        print("Dashboard server stopped.")
        return 0
    print("error: server did not confirm shutdown", file=sys.stderr)
    return 2


def _serve(port: int, vault: str | None, open_browser: bool,
           session: str, owner_pid: int | None) -> int:
    from ..dashboard_server import make_server
    from ..dashboard_runtime import runtime_path, status, owner_alive
    from filelock import FileLock
    path = runtime_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock")), timeout=5):
        existing = status()
        if existing["running"]:
            print(f"Dashboard already running → {existing['url']} · session: {existing['session'] or 'unlabelled'}", flush=True)
            if open_browser:
                _open(existing["url"])
            return 0
        if owner_pid is not None and not owner_alive(owner_pid):
            print("error: owning process is not running", file=sys.stderr)
            return 2
        try:
            server = make_server(port, vault)
        except OSError as exc:
            print(f"error: cannot start dashboard: {exc}", file=sys.stderr)
            return 2
        record = {**server.dashboard_identity, "control_token": server.dashboard_control_token,
                  "port": server.server_port, "pid": os.getpid(), "owner_pid": owner_pid,
                  "session": session, "started_at": dt.datetime.now().astimezone().isoformat()}
        write_text_atomic(path, json.dumps(record))
        path.chmod(0o600)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Research Wiki dashboard → {url}", flush=True)
    print("Live library · read-only · Ctrl-C to stop", flush=True)
    if open_browser:
        _open(url)
    done = threading.Event()
    if owner_pid is not None:
        def watch_owner():
            while not done.wait(1):
                if not owner_alive(owner_pid):
                    server.shutdown()
                    return
        threading.Thread(target=watch_owner, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        done.set()
        server.server_close()
        # Only delete our own receipt, never a newer server's identity.
        from ..dashboard_runtime import read_runtime
        with FileLock(str(path.with_suffix(".lock")), timeout=5):
            if read_runtime().get("server_id") == server.dashboard_identity["server_id"]:
                path.unlink(missing_ok=True)
    return 0
