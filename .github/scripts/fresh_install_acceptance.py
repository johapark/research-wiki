"""Hermetic fresh-install acceptance lifecycle for the beta gate.

Run from a clean checkout after installing the package::

    python .github/scripts/fresh_install_acceptance.py

The harness creates a disposable wiki and exercises the installed CLI through
subprocesses. It deliberately keeps the ML stack and external services out of
the loop: semantic indexing is disabled, authored text uses ``--stub``, and a
local HTTP sentinel asserts that stub mode never reaches a configured provider.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pdf_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_fixture_pdf(
    path: Path,
    *,
    title: str,
    author: str | None,
    year: int = 2026,
) -> None:
    """Write a tiny two-page PDF with extractable text and optional metadata."""
    page_one = "\n".join(
        [
            "BT",
            "/F1 16 Tf",
            f"72 720 Td ({_pdf_string(title)}) Tj",
            "/F1 11 Tf",
            f"0 -28 Td (Published: January {year}) Tj",
            "0 -28 Td (Abstract) Tj",
            "0 -20 Td (Acceptance pipelines preserve durable research records.) Tj",
            "0 -28 Td (Introduction) Tj",
            "0 -20 Td (This fixture exercises initialization ingest and retrieval.) Tj",
            "ET",
        ]
    ).encode("ascii")
    page_two = "\n".join(
        [
            "BT",
            "/F1 12 Tf",
            "72 720 Td (Methods) Tj",
            "0 -22 Td (A deterministic fixture validates recovery without a network.) Tj",
            "0 -28 Td (Results) Tj",
            "0 -22 Td (The acceptance lifecycle completes in a disposable wiki.) Tj",
            "0 -28 Td (Discussion) Tj",
            "0 -22 Td (Derived indexes can be deleted and rebuilt.) Tj",
            "ET",
        ]
    ).encode("ascii")

    info_parts = [f"/Title ({_pdf_string(title)})"]
    if author:
        info_parts.append(f"/Author ({_pdf_string(author)})")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(page_one)).encode("ascii") + b" >>\nstream\n"
        + page_one
        + b"\nendstream",
        b"<< /Length " + str(len(page_two)).encode("ascii") + b" >>\nstream\n"
        + page_two
        + b"\nendstream",
        ("<< " + " ".join(info_parts) + " >>").encode("ascii"),
    ]

    payload = bytearray(b"%PDF-1.4\n%RWACCEPT\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 8 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class ProviderSentinel(BaseHTTPRequestHandler):
    requests_seen = 0

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        type(self).requests_seen += 1
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":{"message":"stub mode reached provider"}}')

    def log_message(self, _format: str, *_args: object) -> None:
        return


def run_cli(
    root: Path,
    env: dict[str, str],
    *args: str,
    expected: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    print(f"+ researchwiki {' '.join(args)}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "researchwiki", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if proc.returncode not in expected:
        raise AssertionError(
            f"command returned {proc.returncode}, expected {expected}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def run_python(
    root: Path,
    env: dict[str, str],
    code: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode:
        raise AssertionError(
            f"python probe returned {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def write_openai_sentinel_config(root: Path, port: int) -> Path:
    config = root / "config" / "models.acceptance.yaml"
    config.write_text(
        f"""\
base_url: http://127.0.0.1:{port}/v1
roles:
  author:     {{provider: openai-compatible, model: acceptance-model, temperature: 0.2, max_tokens: 800}}
  critic:     {{provider: openai-compatible, model: acceptance-model, temperature: 0.2, max_tokens: 400}}
  judge:      {{provider: openai-compatible, model: acceptance-model, temperature: 0.1, max_tokens: 400}}
  classifier: {{provider: openai-compatible, model: acceptance-model, temperature: 0.0, max_tokens: 200}}
  proposer:   {{provider: openai-compatible, model: acceptance-model, temperature: 0.0, max_tokens: 200}}
  extractor:  {{provider: openai-compatible, model: acceptance-model, temperature: 0.0, max_tokens: 400}}
""",
        encoding="utf-8",
    )
    return config


def verify_provider_configs(root: Path, base_env: dict[str, str]) -> None:
    probe = """
from researchwiki.agents import model_config
from researchwiki.agents.llm import missing_provider_credentials
cfg = model_config.for_phase('author')
expected = __import__('os').environ['RW_EXPECT_PROVIDER']
assert cfg.provider == expected, (cfg.provider, expected)
assert missing_provider_credentials() == []
print(f'{cfg.provider}/{cfg.model}')
"""
    openai_env = dict(base_env)
    openai_env.update(
        RW_MODELS_CONFIG="config/models.acceptance.yaml",
        RW_EXPECT_PROVIDER="openai-compatible",
        OPENAI_API_KEY="acceptance-placeholder",
    )
    openai_env.pop("ANTHROPIC_API_KEY", None)
    run_python(root, openai_env, probe)

    anthropic_env = dict(base_env)
    anthropic_env.update(
        RW_MODELS_CONFIG="config/models.anthropic.yaml",
        RW_EXPECT_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="acceptance-placeholder",
    )
    anthropic_env.pop("OPENAI_API_KEY", None)
    run_python(root, anthropic_env, probe)


def acceptance_lifecycle(root: Path, env: dict[str, str]) -> None:
    run_cli(root, env, "init", "--scaffold-only")
    for rel in (
        "inbox",
        "papers",
        "wiki/index.md",
        "wiki/other",
        "wiki/synthesis",
        "wiki/ideas",
        "wiki/concepts",
        "wiki/references",
    ):
        assert (root / rel).exists(), rel

    single = root / "inbox" / "single.pdf"
    write_fixture_pdf(
        single,
        title="Alpha Acceptance Pipelines for Reliable Research",
        author="Ada Lovelace; Alan Turing",
    )
    run_cli(
        root,
        env,
        "agent",
        "ingest",
        str(single),
        "--stub",
        "--no-semantic",
        "--no-llm-reconcile",
        "--no-cross-link",
        "--auto-promote",
        "--max-evolve",
        "0",
        "-n",
        "1",
        "--title",
        "Alpha Acceptance Pipelines for Reliable Research",
        "--year",
        "2026",
        "--authors",
        "Ada Lovelace;Alan Turing",
    )
    single_stem = "lovelace-2026-alpha-acceptance-pipelines-for-reliable"
    assert (root / "wiki" / "other" / f"{single_stem}.md").is_file()
    assert (root / "papers" / f"{single_stem}.pdf").is_file()

    batch_ok = root / "inbox" / "batch-ok.pdf"
    batch_retry = root / "inbox" / "batch-retry.pdf"
    write_fixture_pdf(
        batch_ok,
        title="Batch Acceptance Pipelines Preserve Durable Checkpoints",
        author="Katherine Johnson; Dorothy Vaughan",
    )
    write_fixture_pdf(
        batch_retry,
        title="Resumable Acceptance Pipelines Recover Interrupted Work",
        author=None,
    )
    run_cli(
        root,
        env,
        "agent",
        "ingest",
        str(batch_ok),
        str(batch_retry),
        "--stub",
        "--no-semantic",
        "--no-llm-reconcile",
        "--no-cross-link",
        "--auto-promote",
        "--max-evolve",
        "0",
        "-n",
        "1",
        "-w",
        "1",
        expected=(1,),
    )
    [batch_dir] = list((root / ".ingest").glob("batch-*"))
    checkpoint = json.loads((batch_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["completed"]) == 1
    assert len(checkpoint["failed"]) == 1

    write_fixture_pdf(
        batch_retry,
        title="Resumable Acceptance Pipelines Recover Interrupted Work",
        author="Grace Hopper; Barbara Liskov",
    )
    run_cli(
        root,
        env,
        "agent",
        "ingest",
        "--resume",
        str(batch_dir),
        "-w",
        "1",
    )
    checkpoint = json.loads((batch_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["completed"]) == 2
    assert checkpoint["failed"] == {}

    hopper_stem = "hopper-2026-resumable-acceptance-pipelines-recover-interrupted"
    assert (root / "wiki" / "other" / f"{hopper_stem}.md").is_file()
    assert (root / "papers" / f"{hopper_stem}.pdf").is_file()

    run_cli(root, env, "db", "rebuild")
    run_cli(root, env, "db", "verify")
    run_cli(root, env, "reindex", "--no-semantic")
    search = run_cli(root, env, "search", "acceptance", "--mode", "bm25", "--json")
    hits = json.loads(search.stdout)
    assert any(hit["stem"] == single_stem for hit in hits)

    run_cli(
        root,
        env,
        "synthesize",
        "--title",
        "Acceptance pipeline reliability",
        "--papers",
        single_stem,
        hopper_stem,
    )
    synthesis = root / "wiki" / "synthesis" / "acceptance-pipeline-reliability.md"
    run_cli(root, env, "check-grounding", str(synthesis), "--quiet")
    run_cli(root, env, "grade", "synthesis", str(synthesis), "--no-semantic")

    # Recovery proof: markdown and PDFs remain canonical after deleting every
    # derived state/index artifact used by query paths.
    for db_file in root.glob("state.db*"):
        db_file.unlink()
    shutil.rmtree(root / ".tantivy-index")
    run_cli(root, env, "db", "rebuild")
    run_cli(root, env, "db", "verify")
    run_cli(root, env, "reindex", "--no-semantic")
    recovered = run_cli(
        root, env, "search", "resumable", "--mode", "bm25", "--json"
    )
    assert any(hit["stem"] == hopper_stem for hit in json.loads(recovered.stdout))


def main() -> int:
    ProviderSentinel.requests_seen = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderSentinel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="researchwiki-acceptance-") as tmp:
            root = Path(tmp)
            shutil.copytree(REPO_ROOT / "config", root / "config")
            shutil.copytree(REPO_ROOT / "prompts", root / "prompts")
            config = write_openai_sentinel_config(root, server.server_port)
            env = dict(os.environ)
            for key in (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "RW_LLM_BASE_URL",
                "RW_LLM_PROVIDER",
            ):
                env.pop(key, None)
            env.update(
                PYTHONUTF8="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                OPENAI_API_KEY="acceptance-placeholder",
                RESEARCHWIKI_DB_PATH=str(root / "state.db"),
                RW_MODELS_CONFIG=str(config),
            )

            verify_provider_configs(root, env)
            acceptance_lifecycle(root, env)

        assert ProviderSentinel.requests_seen == 0, (
            f"stub acceptance reached the configured provider "
            f"{ProviderSentinel.requests_seen} time(s)"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    print("fresh-install acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
