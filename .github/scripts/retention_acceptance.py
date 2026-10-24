"""Live opt-in capture, operator inspection and portable export with stub models."""

import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

from fresh_install_acceptance import ProviderSentinel, run_cli, write_fixture_pdf, write_openai_sentinel_config


REPO = Path(__file__).resolve().parents[2]


def exercise(root, port):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR

    shutil.copytree(REPO / "config", root / "config")
    shutil.copytree(REPO / "prompts", root / "prompts")
    config = write_openai_sentinel_config(root, port)
    with config.open("a") as stream:
        stream.write('\ningest:\n  retention:\n    mode: required\n    profile: extraction-v1\n    root: .researchwiki/records\n')
    env = dict(os.environ)
    for key in (ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR, "RW_MUTATION_JOURNAL",
                "RW_LLM_PROVIDER", "RW_LLM_BASE_URL", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "PYTHONPATH"):
        env.pop(key, None)
    env.update(RW_MODELS_CONFIG=str(config), RESEARCHWIKI_DB_PATH=str(root / "state.db"),
               OPENAI_API_KEY="synthetic-acceptance-key", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    run_cli(root, env, "init", "--scaffold-only")
    source = root / "inbox/paper.pdf"
    write_fixture_pdf(source, title="Alpha Acceptance Pipelines for Reliable Research", author="Ada Lovelace; Alan Turing")
    report = root / "result.json"
    run_cli(root, env, "add", str(source), "--stub", "--no-semantic", "--no-llm-reconcile", "--no-cross-link",
            "--auto-promote", "--max-evolve", "0", "-n", "1", "--title", "Alpha Acceptance Pipelines for Reliable Research",
            "--year", "2026", "--authors", "Ada Lovelace;Alan Turing", "--result-file", str(report))
    result = json.loads(report.read_text())
    assert result["publication"] == "published" and not source.exists()
    listed = json.loads(run_cli(root, env, "records", "list", "--json").stdout)
    assert len(listed["attempts"]) == 1 and listed["attempts"][0]["capture_state"] == "complete"
    attempt = result["attempt_id"]
    shown = json.loads(run_cli(root, env, "records", "show", attempt, "--json").stdout)
    assert shown["manifest"]["spec"]["request_id"] == result["request_id"]
    assert shown["publication_receipts"][-1]["state"] == "committed"
    portable = root.parent / "portable-archive"
    run_cli(root, env, "records", "export", attempt, "--output", str(portable), "--json")
    shutil.rmtree(root / ".researchwiki")
    shutil.rmtree(root / "papers")
    shutil.rmtree(root / "wiki")
    (root / "state.db").unlink(missing_ok=True)
    checked = json.loads(run_cli(root, env, "records", "--root", str(portable), "verify", attempt, "--json").stdout)
    assert checked["checkpoint"] == "current" and checked["capture_state"] == "complete"
    checkpoint = portable / "attempts" / attempt / "manifest.json"
    checkpoint.unlink()
    checked = json.loads(run_cli(root, env, "records", "--root", str(portable), "verify", attempt, "--json").stdout)
    assert checked["checkpoint"] == "missing" and not checkpoint.exists()
    run_cli(root, env, "records", "--root", str(portable), "verify", attempt, "--repair-checkpoint", "--json")
    restored = json.loads(run_cli(root, env, "records", "--root", str(portable), "show", attempt, "--json").stdout)
    assert restored == shown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-installed", action="store_true")
    args = parser.parse_args()
    import researchwiki
    if args.expect_installed:
        assert Path(researchwiki.__file__).is_relative_to(Path(sys.prefix)), researchwiki.__file__
        assert "site-packages" in Path(researchwiki.__file__).parts
    ProviderSentinel.requests_seen = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderSentinel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="researchwiki-retention-acceptance-") as tmp:
            root = Path(tmp) / "library"
            exercise(root, server.server_port)
        assert ProviderSentinel.requests_seen == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(json.dumps({"status": "passed", "scope": "live opt-in retention with synthetic PDF",
                      "external_provider_calls": 0, "portable_after_corpus_deletion": True}))


if __name__ == "__main__":
    main()
