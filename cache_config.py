"""Keep the experiment's cache setting visible in one small object."""

from dataclasses import dataclass
from typing import Literal

CacheTtl = Literal["5m", "1h"]


@dataclass(frozen=True)
class PromptCacheConfig:
    """Use five minutes by default; change this one value for a one-hour run."""

    ttl: CacheTtl = "5m"
