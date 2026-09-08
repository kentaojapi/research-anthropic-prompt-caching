"""Verify each layer of the four-layer Bedrock prompt cache."""

import argparse
import copy
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from strands import Agent
from strands.types.content import Message
from strands.types.event_loop import Usage

from agent_factory import COMMON_PROMPT, Tenant, build_agent
from cache_config import PromptCacheConfig

CACHE_OK = "CACHE_OK"
CONTEXT_REPETITIONS = 60


@dataclass(frozen=True)
class CacheObservation:
    """Cache usage observed for one controlled model request."""

    case: str
    expected_hit: str
    input_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int


def parse_args() -> argparse.Namespace:
    """Accept the cache TTL, output path, and an optional reproducible run ID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttl", choices=("5m", "1h"), default="5m")
    parser.add_argument("--output", type=Path, default=Path("cache_layer_usage.jsonl"))
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:12])
    return parser.parse_args()


def repeated_context(label: str, run_id: str) -> str:
    """Create a visibly distinct cache layer with enough tokens to measure."""
    sentence = f"{label} cache validation context for run {run_id}. "
    return sentence * CONTEXT_REPETITIONS


def validation_prompt(label: str, run_id: str) -> str:
    """Create a unique user input for one cache-layer branch."""
    return f"{repeated_context(label, run_id)}Do not call a tool. Reply with exactly {CACHE_OK}."


def invoke(agent: Agent, prompt: str, case: str, expected_hit: str) -> CacheObservation:
    """Run exactly one model call and extract its Bedrock cache counters."""
    result = agent(prompt)
    if any("toolUse" in block for block in result.message["content"]):
        raise RuntimeError(f"{case} unexpectedly called a tool")

    usage = result.message.get("metadata", {}).get("usage")
    if usage is None:
        raise RuntimeError(f"{case} did not return usage metadata")

    typed_usage = Usage(**usage)
    return CacheObservation(
        case=case,
        expected_hit=expected_hit,
        input_tokens=typed_usage["inputTokens"],
        cache_read_input_tokens=typed_usage.get("cacheReadInputTokens", 0),
        cache_write_input_tokens=typed_usage.get("cacheWriteInputTokens", 0),
        output_tokens=typed_usage["outputTokens"],
    )


def verify_observations(observations: list[CacheObservation]) -> None:
    """Fail unless each branch reads through its intended cache boundary."""
    by_case = {observation.case: observation for observation in observations}
    seed1 = by_case["seed-1"]
    seed2 = by_case["seed-2"]
    l1 = by_case["L1"]
    l2 = by_case["L1-L2"]
    l3 = by_case["L1-L3"]
    l4 = by_case["L1-L4"]

    failures: list[str] = []
    if seed1.cache_write_input_tokens == 0:
        failures.append("seed-1 wrote no cache tokens")
    if seed2.cache_read_input_tokens == 0 or seed2.cache_write_input_tokens == 0:
        failures.append("seed-2 did not read and extend the seed cache")
    if l1.cache_read_input_tokens == 0:
        failures.append("L1 read no cached tokens")
    if l2.cache_read_input_tokens <= l1.cache_read_input_tokens:
        failures.append("L1-L2 did not extend the cached prefix beyond L1")
    if l3.cache_read_input_tokens <= l2.cache_read_input_tokens:
        failures.append("L1-L3 did not extend the cached prefix beyond L1-L2")
    if l4.cache_read_input_tokens <= l3.cache_read_input_tokens:
        failures.append("L1-L4 did not extend the cached prefix beyond L1-L3")
    for observation in (l1, l2, l3, l4):
        if observation.cache_write_input_tokens == 0:
            failures.append(f"{observation.case} wrote no cache entry for its new user message")

    if failures:
        details = "; ".join(failures)
        raise AssertionError(f"Four-layer cache verification failed: {details}")


def run_experiment(run_id: str, cache: PromptCacheConfig) -> list[CacheObservation]:
    """Seed all four boundaries, then test each boundary with an independent branch."""
    common_prompt = f"{COMMON_PROMPT}\n{repeated_context('L1 shared', run_id)}"
    tenant_a = Tenant("tenant-a", repeated_context("L2 tenant A", run_id))
    tenant_b = Tenant("tenant-b", repeated_context("L2 tenant B", run_id))

    seed_agent = build_agent(tenant_a, [], cache, common_prompt)
    observations = [
        invoke(seed_agent, validation_prompt("seed message 1", run_id), "seed-1", "create L1, L2, and L4"),
    ]
    l3_history: list[Message] = copy.deepcopy(seed_agent.messages)
    observations.append(invoke(seed_agent, validation_prompt("seed message 2", run_id), "seed-2", "create L3 and L4"))
    l4_history: list[Message] = copy.deepcopy(seed_agent.messages)

    observations.extend(
        [
            invoke(
                build_agent(tenant_b, [], cache, common_prompt),
                validation_prompt("new message 1", run_id),
                "L1",
                "Tool and common system prompt",
            ),
            invoke(
                build_agent(tenant_a, [], cache, common_prompt),
                validation_prompt("new message 2", run_id),
                "L1-L2",
                "L1 and tenant-specific system prompt",
            ),
            invoke(
                build_agent(tenant_a, l3_history, cache, common_prompt),
                validation_prompt("new message 3", run_id),
                "L1-L3",
                "L1, L2, and the previous model-call boundary",
            ),
            invoke(
                build_agent(tenant_a, l4_history, cache, common_prompt),
                validation_prompt("new message 4", run_id),
                "L1-L4",
                "L1, L2, L3, and the previous latest-user boundary",
            ),
        ]
    )
    verify_observations(observations)
    return observations


def main() -> None:
    """Run the live Bedrock experiment and persist its evidence as JSON Lines."""
    load_dotenv()
    args = parse_args()
    observations = run_experiment(args.run_id, PromptCacheConfig(ttl=args.ttl))

    with args.output.open("w", encoding="utf-8") as output_file:
        for observation in observations:
            output_file.write(json.dumps(asdict(observation)) + "\n")

    print(f"{'case':<8} {'cache read':>12} {'cache write':>12} {'uncached':>10}  expected hit")
    for observation in observations:
        print(
            f"{observation.case:<8} "
            f"{observation.cache_read_input_tokens:>12} "
            f"{observation.cache_write_input_tokens:>12} "
            f"{observation.input_tokens:>10}  "
            f"{observation.expected_hit}"
        )
    print("PASS: L1, L1-L2, L1-L3, and L1-L4 cache reuse was observed.")


if __name__ == "__main__":
    main()
