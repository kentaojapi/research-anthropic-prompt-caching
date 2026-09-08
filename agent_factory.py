"""The smallest practical multi-tenant Strands prompt-cache example."""

import os
from dataclasses import dataclass

from strands import Agent, tool
from strands.models import BedrockModel
from strands.types.content import Message

from cache_config import PromptCacheConfig
from cache_hooks import TurnBoundaryCachePointHook


@tool
def look_up_tenant_data(record_id: str) -> str:
    """Look up data using the current tenant's authenticated context."""
    return f"Tenant-scoped data for {record_id}"


# The first live run needs a cacheable prefix large enough for the selected
# model. Replace this repeat with a real policy/RAG document in production.
COMMON_PROMPT = "Follow the shared security policy. Do not expose another tenant's data. " * 220


@dataclass(frozen=True)
class Tenant:
    """A tenant changes the prompt, while tool definitions stay cache-identical."""

    tenant_id: str
    prompt: str


TENANTS = (
    Tenant("billing", "You answer billing questions."),
    Tenant("support", "You answer support questions."),
    Tenant("retail", "You answer inventory questions."),
)


def build_agent(
    tenant: Tenant,
    past_messages: list[Message],
    cache: PromptCacheConfig,
    common_prompt: str = COMMON_PROMPT,
) -> Agent:
    """Create one tenant's persistent agent and preserve its supplied history.

    Bedrock orders tools -> system -> messages. Therefore BP1 immediately
    after COMMON_PROMPT caches the fixed tools plus common instructions across
    every tenant. BP2 then adds the tenant-specific instruction. The hook
    keeps BP3 at the previous call boundary and moves BP4 to the current one.
    """
    model_id = os.environ["BEDROCK_CLAUDE_SONNET_5_MODEL_ID"]
    return Agent(
        model=BedrockModel(model_id=model_id, region_name="us-east-1", max_tokens=512),
        system_prompt=[
            {"text": common_prompt},
            {"cachePoint": {"type": "default", "ttl": cache.ttl}},
            {"text": tenant.prompt},
            {"cachePoint": {"type": "default", "ttl": cache.ttl}},
        ],
        tools=[look_up_tenant_data],
        messages=past_messages,
        hooks=[TurnBoundaryCachePointHook(cache.ttl)],
        callback_handler=None,
        load_tools_from_directory=False,
    )
