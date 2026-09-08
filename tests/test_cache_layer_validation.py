import pytest

from test_cache_hit_rate import CacheObservation, verify_observations


def observation(case: str, cache_read: int, cache_write: int) -> CacheObservation:
    return CacheObservation(
        case=case,
        expected_hit=case,
        input_tokens=100,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        output_tokens=1,
    )


def test_verify_observations_accepts_four_cumulative_cache_layers() -> None:
    verify_observations(
        [
            observation("seed-1", 0, 5_000),
            observation("seed-2", 2_000, 3_000),
            observation("L1", 2_500, 3_000),
            observation("L1-L2", 3_000, 2_000),
            observation("L1-L3", 4_000, 1_000),
            observation("L1-L4", 5_000, 1_000),
        ]
    )


def test_verify_observations_rejects_a_missing_layer() -> None:
    with pytest.raises(AssertionError, match="L2 did not extend"):
        verify_observations(
            [
                observation("seed-1", 0, 5_000),
                observation("seed-2", 2_000, 3_000),
                observation("L1", 2_500, 3_000),
                observation("L1-L2", 2_500, 2_000),
                observation("L1-L3", 4_000, 1_000),
                observation("L1-L4", 5_000, 1_000),
            ]
        )
