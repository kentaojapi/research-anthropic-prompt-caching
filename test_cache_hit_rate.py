"""Run three tenant conversations and write each Bedrock usage response as JSON Lines."""

import argparse
import json
import logging
from pathlib import Path

from agent_factory import TENANTS, build_agent
from cache_config import PromptCacheConfig

logger = logging.getLogger(__name__)
PROMPTS = (
    "What constraints should you follow?",
    "What tool can help with this tenant's data?",
    "Summarize the conversation in one sentence.",
)


def parse_args() -> argparse.Namespace:
    """Accept TTL, output location, and optional region-specific token prices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttl", choices=("5m", "1h"), default="5m")
    parser.add_argument("--output", type=Path, default=Path("cache_usage.jsonl"))
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cache-read-price", type=float)
    parser.add_argument("--cache-write-price", type=float)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def estimated_savings(record: dict[str, int], args: argparse.Namespace) -> float | None:
    """Calculate a dollar saving only when the caller supplies local Bedrock prices."""
    prices = (args.input_price, args.cache_read_price, args.cache_write_price)
    if any(price is None for price in prices):
        return None
    input_tokens = record["input_tokens"]
    read_tokens = record["cache_read_input_tokens"]
    write_tokens = record["cache_creation_input_tokens"]
    uncached = (input_tokens + read_tokens + write_tokens) * args.input_price
    cached = (
        input_tokens * args.input_price + read_tokens * args.cache_read_price + write_tokens * args.cache_write_price
    )
    return (uncached - cached) / 1_000_000


def main() -> None:
    """Run persistent conversations; a positive read count is an observed cache hit."""
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    cache = PromptCacheConfig(ttl=args.ttl)
    hit_count = 0
    response_count = 0

    with args.output.open("w", encoding="utf-8") as output_file:
        for tenant in TENANTS:
            agent = build_agent(tenant, past_messages=[], cache=cache)
            for turn, prompt in enumerate(PROMPTS, start=1):
                message_count = len(agent.messages)
                agent(prompt)
                for message in agent.messages[message_count:]:
                    usage = message.get("metadata", {}).get("usage") if message["role"] == "assistant" else None
                    if not usage:
                        continue
                    record = {
                        "tenant_id": tenant.tenant_id,
                        "turn": turn,
                        "cache_creation_input_tokens": usage.get("cacheWriteInputTokens", 0),
                        "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
                        "input_tokens": usage.get("inputTokens", 0),
                    }
                    record["estimated_savings_usd"] = estimated_savings(record, args)
                    output_file.write(json.dumps(record) + "\n")
                    hit_count += record["cache_read_input_tokens"] > 0
                    response_count += 1
                    logger.info("%s", json.dumps(record))

    hit_rate = hit_count / response_count if response_count else 0
    logger.info("cache_hit_rate=%.1f%% (%d/%d API responses)", hit_rate * 100, hit_count, response_count)


if __name__ == "__main__":
    main()
