from typing import Any

import pytest
from strands.models import BedrockModel

import agent_factory
from agent_factory import Tenant, build_agent
from cache_config import PromptCacheConfig
from cache_hooks import TurnBoundaryCachePointHook


def test_build_agent_uses_standard_bedrock_model_and_cache_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_arguments: dict[str, Any] = {}

    def capture_agent_arguments(**kwargs: Any) -> dict[str, Any]:
        agent_arguments.update(kwargs)
        return kwargs

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("BEDROCK_CLAUDE_SONNET_5_MODEL_ID", "global.anthropic.claude-sonnet-5-test")
    monkeypatch.setattr(agent_factory, "Agent", capture_agent_arguments)

    build_agent(
        Tenant("billing", "You answer billing questions."),
        past_messages=[],
        cache=PromptCacheConfig(ttl="1h"),
    )

    assert type(agent_arguments["model"]) is BedrockModel
    assert agent_arguments["model"].client.meta.region_name == "us-east-1"
    assert len(agent_arguments["hooks"]) == 1
    assert type(agent_arguments["hooks"][0]) is TurnBoundaryCachePointHook
    assert agent_arguments["hooks"][0].ttl == "1h"
