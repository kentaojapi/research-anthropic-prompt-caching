import pytest

from cache_hooks import ConversationHistoryCachePointBedrockModel


class TestConversationHistoryCachePointBedrockModel:
    def test_formats_shared_and_tenant_prefixes_with_conversation_history_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        model = ConversationHistoryCachePointBedrockModel(model_id="global.anthropic.claude-sonnet-5-test")
        request = model.format_request(
            messages=[
                {"role": "user", "content": [{"text": "first"}]},
                {"role": "assistant", "content": [{"text": "answer"}]},
                {"role": "user", "content": [{"text": "second"}]},
            ],
            tool_specs=[
                {"name": "shared_tool", "description": "shared", "inputSchema": {"json": {"type": "object"}}},
                {"name": "tenant_tool", "description": "tenant", "inputSchema": {"json": {"type": "object"}}},
            ],
            system_prompt_content=[
                {"text": "common prompt"},
                {"cachePoint": {"type": "default", "ttl": "5m"}},
                {"text": "tenant prompt"},
                {"cachePoint": {"type": "default", "ttl": "5m"}},
            ],
        )

        assert sum("cachePoint" in block for block in request["toolConfig"]["tools"]) == 0
        assert sum("cachePoint" in block for block in request["system"]) == 2
        assert sum("cachePoint" in block for message in request["messages"] for block in message["content"]) == 2
