# /// script
# dependencies = ["anthropic", "python-dotenv"]
# ///
"""Verify that consecutive parallel tool blocks do not exhaust Claude's 20-position lookback.

The first request writes a cache entry at the static system prefix. The second
request contains 15 consecutive tool_use blocks followed by 15 matching
tool_result blocks and a cache point. Anthropic documents each consecutive
group as one lookback position, so the second response should read the first
request's cache entry.
"""

import os
import uuid
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MODEL = "claude-sonnet-4-5"
RUN_ID = uuid.uuid4().hex[:12]
TOOL_COUNT = 15
TOOL = {
    "name": "lookup_record",
    "description": "Look up one fictional record.",
    "input_schema": {
        "type": "object",
        "properties": {"record_id": {"type": "string"}},
        "required": ["record_id"],
    },
}
SYSTEM_PARAGRAPH = (
    "You are a support assistant for TripleTail Analytics. "
    "Answer concisely and never disclose data from another account. "
)
SYSTEM_TEXT = (SYSTEM_PARAGRAPH * 80).strip() + f" [run:{RUN_ID}]"


def show_usage(label: str, response: anthropic.types.Message) -> None:
    """Print the three fields that distinguish a cache write, read, and miss."""
    usage = response.usage
    print(f"--- {label} ---")
    print("cache_creation_input_tokens:", usage.cache_creation_input_tokens)
    print("cache_read_input_tokens:", usage.cache_read_input_tokens)
    print("input_tokens:", usage.input_tokens)
    print()


load_dotenv(dotenv_path=ENV_PATH)
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY が .env から読み込めませんでした")

client = anthropic.Anthropic(api_key=api_key)

# Request 1 writes a unique static prefix. The same tools must be supplied to
# request 2 because tools precede system in the Anthropic cache prefix.
first_response = client.messages.create(
    model=MODEL,
    max_tokens=64,
    tools=[TOOL],
    system=[
        {
            "type": "text",
            "text": SYSTEM_TEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": "OK"}],
)
show_usage("Request 1: cache write", first_response)

tool_uses = [
    {
        "type": "tool_use",
        "id": f"toolu_{RUN_ID}_{index}",
        "name": TOOL["name"],
        "input": {"record_id": f"record-{index}"},
    }
    for index in range(TOOL_COUNT)
]
tool_results = [
    {
        "type": "tool_result",
        "tool_use_id": tool_use["id"],
        "content": f"result for {tool_use['input']['record_id']}",
    }
    for tool_use in tool_uses
]
tool_results[-1]["cache_control"] = {"type": "ephemeral"}

# The two consecutive groups count as two positions, not thirty. With the
# prior user message and the final cache point, the Request 1 system cache is
# still inside the 20-position lookback window.
second_response = client.messages.create(
    model=MODEL,
    max_tokens=64,
    tools=[TOOL],
    system=[{"type": "text", "text": SYSTEM_TEXT}],
    messages=[
        {"role": "user", "content": "OK"},
        {"role": "assistant", "content": tool_uses},
        {"role": "user", "content": tool_results},
    ],
)
show_usage(f"Request 2: {TOOL_COUNT} consecutive tool_use + tool_result blocks", second_response)

if second_response.usage.cache_read_input_tokens == 0:
    raise AssertionError(
        "Cache miss: the static prefix was not found. This contradicts the expected "
        "contiguous-tool-block lookback behavior, so inspect the request and usage."
    )

print("PASS: cache_read_input_tokens > 0. Parallel tool blocks did not exhaust the lookback window.")
