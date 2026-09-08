from copy import deepcopy
from typing import Any

import pytest

from cache_hooks import ConversationHistoryCachePointBedrockModel


class TestConversationHistoryCachePointBedrockModel:
    @pytest.fixture
    def model(self, monkeypatch: pytest.MonkeyPatch) -> ConversationHistoryCachePointBedrockModel:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        return ConversationHistoryCachePointBedrockModel(model_id="global.anthropic.claude-sonnet-5-test")

    @staticmethod
    def _request_messages(
        model: ConversationHistoryCachePointBedrockModel, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return model.format_request(messages=messages)["messages"]

    def test_places_cache_points_only_on_the_final_two_messages(
        self, model: ConversationHistoryCachePointBedrockModel
    ) -> None:
        messages = [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "second"}]},
            {"role": "user", "content": [{"text": "third"}]},
        ]

        request_messages = self._request_messages(model, messages)

        assert request_messages[0]["content"] == [{"text": "first"}]
        assert request_messages[-2]["content"][-1] == {"cachePoint": {"type": "default"}}
        assert request_messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}

    def test_places_cache_points_on_assistant_and_user_messages(
        self, model: ConversationHistoryCachePointBedrockModel
    ) -> None:
        messages = [
            {"role": "assistant", "content": [{"text": "previous response"}]},
            {"role": "user", "content": [{"text": "current input"}]},
        ]

        request_messages = self._request_messages(model, messages)

        assert request_messages[-2]["content"][-1] == {"cachePoint": {"type": "default"}}
        assert request_messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}

    def test_places_cache_points_on_tool_use_and_tool_result_messages(
        self, model: ConversationHistoryCachePointBedrockModel
    ) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tool-1", "name": "lookup", "input": {}}}],
            },
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "tool-1", "content": [{"text": "result"}]}}],
            },
        ]

        request_messages = self._request_messages(model, messages)

        assert request_messages[-2]["content"][-1] == {"cachePoint": {"type": "default"}}
        assert request_messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}

    def test_places_a_cache_point_on_the_only_message(self, model: ConversationHistoryCachePointBedrockModel) -> None:
        request_messages = self._request_messages(model, [{"role": "user", "content": [{"text": "only"}]}])

        assert request_messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}
        assert sum("cachePoint" in block for message in request_messages for block in message["content"]) == 1

    def test_does_not_add_a_cache_point_when_no_messages_exist(
        self, model: ConversationHistoryCachePointBedrockModel
    ) -> None:
        assert self._request_messages(model, []) == []

    def test_does_not_mutate_conversation_history(self, model: ConversationHistoryCachePointBedrockModel) -> None:
        messages = [
            {"role": "assistant", "content": [{"text": "previous response"}]},
            {"role": "user", "content": [{"text": "current input"}]},
        ]
        original_messages = deepcopy(messages)

        self._request_messages(model, messages)

        assert messages == original_messages

    def test_does_not_duplicate_cache_points_across_repeated_request_builds(
        self, model: ConversationHistoryCachePointBedrockModel
    ) -> None:
        messages = [
            {"role": "assistant", "content": [{"text": "previous response"}]},
            {"role": "user", "content": [{"text": "current input"}]},
        ]

        first_request_messages = self._request_messages(model, messages)
        second_request_messages = self._request_messages(model, messages)

        assert first_request_messages == second_request_messages
        assert sum("cachePoint" in block for message in second_request_messages for block in message["content"]) == 2

    def test_does_not_cache_history_for_a_model_without_explicit_cache_point_support(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        model = ConversationHistoryCachePointBedrockModel(model_id="meta.llama3-8b-instruct-v1:0")

        request_messages = self._request_messages(model, [{"role": "user", "content": [{"text": "input"}]}])

        assert all("cachePoint" not in block for message in request_messages for block in message["content"])
