"""Add request-local conversation cache points to Bedrock Converse requests."""

from copy import deepcopy
from typing import Any, ClassVar

from strands.models import BedrockModel
from strands.types.content import Messages, SystemContentBlock
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override


class ConversationHistoryCachePointBedrockModel(BedrockModel):
    """Add cache points after Bedrock has normalized request messages."""

    CACHEABLE_MODEL_ID_PREFIXES: ClassVar[tuple[str, ...]] = (
        "anthropic.claude-3-5-sonnet-",
        "anthropic.claude-3-7-sonnet-",
        "anthropic.claude-fable-5",
        "anthropic.claude-haiku-4-5",
        "anthropic.claude-mythos-5",
        "anthropic.claude-opus-4-",
        "anthropic.claude-opus-5",
        "anthropic.claude-sonnet-4-",
        "anthropic.claude-sonnet-5",
    )

    @classmethod
    def supports_conversation_history_caching(cls, model_id: str) -> bool:
        """Return whether this Converse model accepts explicit cachePoint blocks."""
        normalized_model_id = model_id.lower()
        return any(prefix in normalized_model_id for prefix in cls.CACHEABLE_MODEL_ID_PREFIXES)

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
        dynamic_trailing_blocks: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format a request and cache its final two normalized messages when supported."""
        request = super().format_request(
            messages=messages,
            tool_specs=tool_specs,
            system_prompt_content=system_prompt_content,
            tool_choice=tool_choice,
            dynamic_trailing_blocks=dynamic_trailing_blocks,
            **kwargs,
        )
        if not self.supports_conversation_history_caching(self.config["model_id"]):
            return request

        request_messages = deepcopy(request["messages"])
        for message in request_messages[-2:]:
            message["content"].append({"cachePoint": {"type": "default"}})

        request["messages"] = request_messages
        return request
