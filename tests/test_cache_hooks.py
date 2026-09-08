from types import SimpleNamespace
from typing import cast

import pytest
from strands.hooks import BeforeModelCallEvent
from strands.types.content import Message

from cache_hooks import TurnBoundaryCachePointHook


class TestTurnBoundaryCachePointHook:
    @staticmethod
    def _event(messages: list[Message]) -> BeforeModelCallEvent:
        return cast(BeforeModelCallEvent, SimpleNamespace(agent=SimpleNamespace(messages=messages)))

    def test_places_cache_points_on_the_previous_and_current_call_boundaries(self) -> None:
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "first question"}]},
        ]
        hook = TurnBoundaryCachePointHook()

        hook.on_before_model_call(self._event(messages))
        messages.extend(
            [
                {"role": "assistant", "content": [{"text": "previous response"}]},
                {"role": "user", "content": [{"text": "latest question"}]},
            ]
        )
        hook.on_before_model_call(self._event(messages))

        assert messages[0]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "5m"}}
        assert messages[1]["content"] == [{"text": "previous response"}]
        assert messages[2]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "5m"}}

    def test_places_cache_points_on_the_previous_user_input_and_tool_result(self) -> None:
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "look this up"}]},
        ]
        hook = TurnBoundaryCachePointHook("1h")

        hook.on_before_model_call(self._event(messages))
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "tool-1", "name": "lookup", "input": {}}}],
                },
                {
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": "tool-1", "content": [{"text": "result"}]}}],
                },
            ]
        )

        hook.on_before_model_call(self._event(messages))

        assert messages[0]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "1h"}}
        assert all("cachePoint" not in block for block in messages[1]["content"])
        assert messages[2]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "1h"}}

    @pytest.mark.parametrize(
        ["messages", "expected_cache_point_count"],
        [
            pytest.param([], 0, id="case1: no messages"),
            pytest.param(
                [{"role": "user", "content": [{"text": "only message"}]}],
                1,
                id="case2: one message",
            ),
        ],
    )
    def test_handles_fewer_than_two_messages(self, messages: list[Message], expected_cache_point_count: int) -> None:
        TurnBoundaryCachePointHook().on_before_model_call(self._event(messages))

        assert (
            sum("cachePoint" in block for message in messages for block in message["content"])
            == expected_cache_point_count
        )

    def test_keeps_only_the_latest_previous_boundary_without_duplication(self) -> None:
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"text": "old question"},
                    {"cachePoint": {"type": "default", "ttl": "5m"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "previous response"},
                    {"cachePoint": {"type": "default", "ttl": "5m"}},
                ],
            },
            {"role": "user", "content": [{"text": "latest question"}]},
        ]
        hook = TurnBoundaryCachePointHook()
        event = self._event(messages)

        hook.on_before_model_call(event)

        assert messages[0]["content"] == [{"text": "old question"}]
        assert messages[1]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "5m"}}
        assert messages[2]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "5m"}}
        assert sum("cachePoint" in block for message in messages for block in message["content"]) == 2

    def test_does_not_duplicate_a_boundary_when_no_message_was_added(self) -> None:
        messages: list[Message] = [{"role": "user", "content": [{"text": "question"}]}]
        hook = TurnBoundaryCachePointHook()
        event = self._event(messages)

        hook.on_before_model_call(event)
        hook.on_before_model_call(event)

        assert sum("cachePoint" in block for block in messages[0]["content"]) == 1
