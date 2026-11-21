"""Content-addressed target-claim responses, local to the ingest library."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ...fsatomic import read_json, write_private_json
from ...log import log
from ...paths import wiki_root
from .. import llm, model_config


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cache_path(source_hash, prompt, system, schema, config, inventory) -> Path | None:
    if config.provider in model_config.OPENAI_COMPAT_PROVIDERS:
        endpoint = llm.resolve_openai_endpoint().url
    elif config.provider == "anthropic":
        endpoint = llm._validated_anthropic_base_url() or "https://api.anthropic.com"
    else:
        return None  # Chat relay handles its own request reuse and fresh mode.
    key = _digest({
        "version": 1, "pdf": source_hash, "prompt": prompt, "system": system,
        "schema": schema, "routing": asdict(config), "endpoint": endpoint,
        "inventory": [asdict(entry) for entry in inventory] if inventory is not None else None,
    })
    return wiki_root() / ".ingest" / "target-claims" / f"{key}.json"


def load(path: Path) -> llm.LLMResponse | None:
    record = read_json(path)
    if not isinstance(record, dict):
        return None
    model, text = record.get("model"), record.get("text")
    if (not isinstance(model, str) or not isinstance(text, str)
            or record.get("checksum") != _digest([path.stem, model, text])):
        return None
    return llm.LLMResponse(model=model, text=text, temperature=0, input_tokens=0, output_tokens=0)


def save(path: Path, response) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_json(path, {"model": response.model, "text": response.text,
                                  "checksum": _digest([path.stem, response.model, response.text])})
    except OSError as exc:
        log(f"Could not cache target claims: {exc}", tag="agent")
