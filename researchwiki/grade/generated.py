"""Shared strict decoding, phase preflight, and model provenance for generated prose."""
from dataclasses import dataclass

import json
import re

from ..agents import llm, model_config
from ..operations._wire import WireModel, nonempty
from ..provenance import specific_author_model


def validate_prose(text):
    """Validate model prose while leaving Markdown and citations to the renderer."""
    nonempty(text, "prose")
    if (len(text) > 1500 or any(c in text for c in "\r\n[]<>`*#")
            or re.search(r"https?://|www\.", text, re.I)):
        raise ValueError("prose must be bounded plain text without links or markup")


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate model response key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def response_system(system, schema):
    """Expose the output contract to API models as well as schema-aware relays.

    llm.call's schema argument is enforced only by chat-relay. API providers
    otherwise see no schema, so the contract must also travel in instructions.
    Local decoding and source checks remain mandatory for every response.
    """
    return system + '\nReturn only JSON conforming to this response schema:\n' + json.dumps(schema)


def preflight(phases):
    problems = llm.missing_provider_credentials(phases=phases)
    for phase in phases:
        cfg = model_config.for_phase(phase)
        if cfg.provider != "chat-relay" and (not specific_author_model(cfg.model) or cfg.model.startswith("stub:")):
            problems.append(f"{phase} requires an exact non-stub model ID")
    if problems:
        raise llm.ProviderUnavailable("generated-prose provider preflight failed:\n" + "\n".join(problems))


@dataclass(frozen=True, kw_only=True)
class ModelProvenance(WireModel):
    phase: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.phase, "phase")
        if not specific_author_model(self.model) or self.model.startswith("stub:") or min(
            self.input_tokens, self.output_tokens, self.cache_read_tokens, self.cache_write_tokens,
        ) < 0:
            raise ValueError("generated prose requires attributable, non-stub model runs")
