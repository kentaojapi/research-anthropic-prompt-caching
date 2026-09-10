# /// script
# dependencies = ["strands-agents[anthropic]", "python-dotenv"]
# ///
"""Verify Strands preserves cache hits across 15 parallel tool calls.

The first Agent call writes one cache entry after its real user turn. The
second call instructs Claude to issue exactly 15 parallel tool calls. An
AfterToolsEvent hook marks the resulting consecutive tool_result group as a
new cache point, so the following model call must find the first cache entry
inside its 20-position lookback window.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from strands import Agent, tool
from strands.hooks import AfterToolsEvent, BeforeModelCallEvent, HookProvider, HookRegistry
from strands.models.anthropic import AnthropicModel
from strands.types.content import Message

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MODEL = "claude-sonnet-4-5"
RUN_ID = uuid.uuid4().hex[:12]
TOOL_COUNT = 15
SYSTEM_PARAGRAPH = (
    "You are a support assistant for TripleTail Analytics. "
    "Follow the caller's tool-use instructions exactly and answer concisely. "
)
SYSTEM_TEXT = (SYSTEM_PARAGRAPH * 80).strip() + f" [run:{RUN_ID}]"


@tool
def lookup_record(record_id: str) -> str:
    """Look up a fictional record by its ID."""
    return f"result for {record_id}"


class LookbackCacheHook(HookProvider):
    """Mark the warm-up turn and the complete parallel tool-result group."""

    def __init__(self) -> None:
        self.initial_turn_cached = False

    def register_hooks(self, registry: HookRegistry) -> None:
        """Add a cache point before the first call and after the tool loop."""
        registry.add_callback(BeforeModelCallEvent, self.on_before_model_call)
        registry.add_callback(AfterToolsEvent, self.on_after_tools)

    def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Write the initial cache entry after the first real user message."""
        if self.initial_turn_cached:
            return
        message = self._first_real_user_message(event.agent.messages)
        if message is None:
            raise RuntimeError("The initial user message was not present before the model call.")
        message["content"].append({"cachePoint": {"type": "default"}})
        self.initial_turn_cached = True

    def on_after_tools(self, event: AfterToolsEvent) -> None:
        """Place the second cache point after all consecutive tool results."""
        event.message["content"].append({"cachePoint": {"type": "default"}})

    @staticmethod
    def _first_real_user_message(messages: list[Message]) -> Message | None:
        for message in messages:
            content = message["content"]
            if message["role"] == "user" and not (content and all("toolResult" in block for block in content)):
                return message
        return None


class ParallelToolLookbackExperiment:
    """Run the two Agent calls and fail if Claude does not form one 15-tool group."""

    def __init__(self, api_key: str) -> None:
        self.agent = Agent(
            model=AnthropicModel(model_id=MODEL, client_args={"api_key": api_key}, max_tokens=2_048),
            system_prompt=[{"text": SYSTEM_TEXT}],
            tools=[lookup_record],
            hooks=[LookbackCacheHook()],
            callback_handler=None,
            load_tools_from_directory=False,
        )

    def run(self) -> None:
        """Warm the cache, execute the parallel group, and inspect final usage."""
        self.agent("Reply with READY. Do not use a tool.")
        self.agent(
            f"Call lookup_record exactly {TOOL_COUNT} times in parallel, once each for "
            f"record-1 through record-{TOOL_COUNT}. Do not call it serially. "
            "After every result is available, reply with DONE."
        )
        tool_use_batches, tool_result_batches = self._tool_block_batches()
        self._show_usage()
        if tool_use_batches != [TOOL_COUNT] or tool_result_batches != [TOOL_COUNT]:
            raise AssertionError(
                f"Expected one consecutive group of {TOOL_COUNT} tool uses/results, got "
                f"tool-use groups {tool_use_batches} and tool-result groups {tool_result_batches}."
            )
        final_usage = self.agent.messages[-1].get("metadata", {}).get("usage", {})
        if final_usage.get("cacheReadInputTokens", 0) == 0:
            raise AssertionError("Cache miss after the consecutive parallel tool blocks.")
        print(f"PASS: {TOOL_COUNT} parallel tool uses/results and cache_read_input_tokens > 0.")

    def _tool_block_batches(self) -> tuple[list[int], list[int]]:
        """Return tool-use/result counts per Strands message, proving one parallel group."""
        tool_use_batches = [
            sum("toolUse" in block for block in message["content"])
            for message in self.agent.messages
            if any("toolUse" in block for block in message["content"])
        ]
        tool_result_batches = [
            sum("toolResult" in block for block in message["content"])
            for message in self.agent.messages
            if any("toolResult" in block for block in message["content"])
        ]
        return tool_use_batches, tool_result_batches

    def _show_usage(self) -> None:
        """Print usage from every Anthropic response retained by Strands."""
        for index, message in enumerate(self.agent.messages, start=1):
            usage = message.get("metadata", {}).get("usage")
            if usage:
                print(f"--- Strands model response {index} ---")
                print("cache_creation_input_tokens:", usage.get("cacheWriteInputTokens", 0))
                print("cache_read_input_tokens:", usage.get("cacheReadInputTokens", 0))
                print("input_tokens:", usage.get("inputTokens", 0))


load_dotenv(dotenv_path=ENV_PATH)
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が .env から読み込めませんでした")

ParallelToolLookbackExperiment(api_key).run()
