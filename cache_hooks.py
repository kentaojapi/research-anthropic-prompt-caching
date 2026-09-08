"""Add conversation cache points immediately before each model call."""

from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry
from strands.types.content import Message

from cache_config import CacheTtl


class TurnBoundaryCachePointHook(HookProvider):
    """Cache the previous and current model-call boundaries."""

    def __init__(self, ttl: CacheTtl = "5m") -> None:
        self.ttl = ttl

    def register_hooks(self, registry: HookRegistry) -> None:
        """Update the conversation history immediately before each model call."""
        registry.add_callback(BeforeModelCallEvent, self.on_before_model_call)

    def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Keep the previous call boundary and add the current call boundary."""
        messages = event.agent.messages
        previous_call_boundary: Message | None = None
        for message in messages:
            if any("cachePoint" in block for block in message["content"]):
                previous_call_boundary = message
            message["content"] = [block for block in message["content"] if "cachePoint" not in block]

        current_call_boundary = messages[-1] if messages else None
        if previous_call_boundary is not None:
            previous_call_boundary["content"].append({"cachePoint": {"type": "default", "ttl": self.ttl}})
        if current_call_boundary is not None and current_call_boundary is not previous_call_boundary:
            current_call_boundary["content"].append({"cachePoint": {"type": "default", "ttl": self.ttl}})
