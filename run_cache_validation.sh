#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f ".env" ]]; then
    printf '%s\n' "Error: .env was not found. Create it with BEDROCK_CLAUDE_SONNET_5_MODEL_ID first." >&2
    exit 1
fi

poetry run python test_cache_hit_rate.py "$@"
