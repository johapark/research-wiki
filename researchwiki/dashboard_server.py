"""Loopback-only, read-only dashboard server with a changing-library snapshot.

Only wiki Markdown and source/inbox PDFs are served. A symlinked content
directory is supported; an individual file escaping that directory is refused.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .dashboard import build_dashboard, render_html
from .paths import inbox_dir, papers_dir, wiki_dir
from .dashboard_runtime import repository_id


def resolve_file(url_path: str) -> Path | None:
    """Resolve the intentionally narrow /files route, never the whole repo."""
    if not url_path.startswith("/files/"):
        return None
    raw = unquote(url_path[len("/files/"):])
    parts = raw.split("/")
    if len(parts) < 2 or any(p in {"", ".", ".."} or "\\" in p or "\x00" in p for p in parts):
        return None
    roots = {"wiki": (wiki_dir(), {".md"}), "papers": (papers_dir(), {".pdf"}),
             "inbox": (inbox_dir(), {".pdf"})}
    if parts[0] not in roots:
        return None
    root, extensions = roots[parts[0]]
    target = root.joinpath(*parts[1:]).resolve()
    if not target.is_relative_to(root.resolve()) or target.suffix.lower() not in extensions:
        return None
    return target if target.is_file() else None


class DashboardCache:
    """Rebuild only after source file identities change; no mtime-as-added-date."""

    def __init__(self, vault: str | None = None):
        self.vault = vault
        self._lock = threading.Lock()
        self._fingerprint = None
        self._data = None

    def get(self) -> tuple[dict, str]:
        with self._lock:
            items = []
            for root, pattern in [(wiki_dir(), "**/*.md"), (papers_dir(), "*.pdf"), (inbox_dir(), "*.pdf")]:
                for path in sorted(root.glob(pattern)):
                    try:
                        stat = path.stat()
                        items.append((str(path), stat.st_mtime_ns, stat.st_size))
                    except FileNotFoundError:
                        continue  # An ingest can move an inbox file during the walk.
            fingerprint = hashlib.sha256(repr(items).encode()).hexdigest()
            if fingerprint != self._fingerprint:
                self._data = build_dashboard(live=True, vault=self.vault)
                self._fingerprint = fingerprint
            return self._data, fingerprint


def make_server(port: int, vault: str | None = None) -> ThreadingHTTPServer:
    cache = DashboardCache(vault)
    server_id = secrets.token_hex(16)
    control_token = secrets.token_hex(32)
    identity = {"repository_id": repository_id(), "server_id": server_id}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args) -> None:
            pass

        def do_GET(self) -> None:
            # Refuse non-loopback Host headers (including DNS rebinding).
            host = self.headers.get("Host", "").split(":", 1)[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            try:
                self._get(path)
            except (BrokenPipeError, ConnectionResetError):
                return  # Browsers cancel requests when a query or page changes.
            except (OSError, ValueError) as exc:
                self.send_error(500, str(exc))

        def _get(self, path: str) -> None:
            if path == "/api/health":
                self._send(json.dumps(identity).encode(), "application/json")
                return
            if path == "/api/search":
                self._search()
                return
            if path in {"/", "/index.html", "/api/dashboard"}:
                data, fingerprint = cache.get()
                etag = '"' + fingerprint + '"'
                if path == "/api/dashboard" and self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    return
                body = json.dumps(data, ensure_ascii=False).encode() if path == "/api/dashboard" else render_html(data).encode()
                mime = "application/json" if path == "/api/dashboard" else "text/html"
                self._send(body, mime + "; charset=utf-8", etag=etag)
                return
            file = resolve_file(path)
            if file is None:
                self.send_error(404)
                return
            self._send(file.read_bytes(), "application/pdf" if file.suffix.lower() == ".pdf" else "text/plain; charset=utf-8")

        def _search(self) -> None:
            # Reuse the CLI's read-only API, with a fresh reader for each query
            # so incremental ingests and index replacements become visible.
            from .search import SearchBackendUnavailable, get_default_backend
            params = parse_qs(urlsplit(self.path).query)
            query = params.get("q", [""])[0].strip()
            try:
                limit = int(params.get("limit", ["1000"])[0])
                if len(query) > 512 or not 1 <= limit <= 10000:
                    raise ValueError("Search query or result limit is too large.")
                hits = get_default_backend().query(query, limit=limit) if query else []
            except SearchBackendUnavailable:
                self._send(b'{"error":"Keyword index unavailable."}', "application/json", status=503)
                return
            except ValueError as exc:
                self._send(json.dumps({"error": str(exc)}).encode(), "application/json", status=400)
                return
            self._send(json.dumps({"engine": "tantivy", "query": query,
                                   "hits": [{"id": hit.key, "score": hit.score} for hit in hits]}).encode(),
                       "application/json")

        def do_POST(self) -> None:
            host = self.headers.get("Host", "").split(":", 1)[0]
            if (host not in {"127.0.0.1", "localhost"}
                    or urlsplit(self.path).path != "/api/shutdown"
                    or not secrets.compare_digest(self.headers.get("X-Dashboard-Control", ""), control_token)):
                self.send_error(403)
                return
            self._send(json.dumps(identity).encode(), "application/json")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _send(self, body: bytes, mime: str, *, etag: str = "", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.dashboard_identity = identity
    server.dashboard_control_token = control_token
    return server
