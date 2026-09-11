#!/usr/bin/env bash
# OpenCode-through-Bifrost proof for DeepSeek-V4.1-Flash.
# Runs the read-tool fixture against the Bifrost-routed local service in tmux.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
VK_ENV_NAME=${DSV41_BIFROST_VK_ENV_NAME:-CHUNGLET_OPENCODE_VK}

[[ -n "${!VK_ENV_NAME:-}" ]] || { echo "$VK_ENV_NAME is required (Bifrost virtual key)" >&2; exit 1; }
tmux set-environment -g "$VK_ENV_NAME" "${!VK_ENV_NAME}"

export OPENCODE_PROOF_SLUG=dsv41-flash-bifrost
export OPENCODE_PROOF_CONFIG="$REPO_ROOT/campaigns/2026-09-10-dsv41-flash-cmp170hx/opencode.bifrost.json"
export OPENCODE_PROOF_MODEL=bifrost-dsv41/dsv41-flash-cmp/deepseek-v4.1-flash
export OPENCODE_PROOF_AGENT=dsv41-flash-proof
export OPENCODE_PROOF_BASE_URL=${DSV41_BASE_URL:-http://100.64.0.119:8090}
export OPENCODE_PROOF_LMCACHE_URL=${DSV41_LMCACHE_URL:-http://127.0.0.1:7001}
export OPENCODE_PROOF_REQUIRE_LMCACHE=${DSV41_REQUIRE_LMCACHE:-0}
export OPENCODE_PROOF_RUNTIME_ROOT=${DSV41_OPENCODE_RUNTIME_ROOT:-$REPO_ROOT/.runtime/opencode-dsv41-flash-bifrost}
export OPENCODE_PROOF_LOG_ROOT=${DSV41_OPENCODE_LOG_ROOT:-/mnt/kv/logs/dsv41/opencode-proof-bifrost}
export OPENCODE_PROOF_WAIT_TIMEOUT=${DSV41_OPENCODE_WAIT_TIMEOUT:-3600}
export OPENCODE_PROOF_BIN=${DSV41_OPENCODE_BIN:-/home/chunglet/.opencode/bin/opencode}
export OPENCODE_PROOF_RUN_ID=${OPENCODE_PROOF_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
export OPENCODE_PROOF_RUN_DIR=${OPENCODE_PROOF_RUN_DIR:-$OPENCODE_PROOF_LOG_ROOT/$OPENCODE_PROOF_RUN_ID}

"$REPO_ROOT/campaign/opencode-proof.sh"
"$REPO_ROOT/glm5.3-flash/scripts/validate-opencode-proof.py" \
  --jsonl "$OPENCODE_PROOF_RUN_DIR/opencode.jsonl" \
  --fixture "$REPO_ROOT/campaign/opencode-proof-fixture-v2.txt" \
  --output "$OPENCODE_PROOF_RUN_DIR/strict-validation.json"
{
  "$OPENCODE_PROOF_BIN" --version
  sha256sum "$OPENCODE_PROOF_CONFIG"
} >"$OPENCODE_PROOF_RUN_DIR/opencode-receipt.txt"
(cd "$OPENCODE_PROOF_RUN_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS && sha256sum --check SHA256SUMS)
printf 'PASS: DeepSeek-V4.1-Flash OpenCode-through-Bifrost proof; evidence=%s\n' "$OPENCODE_PROOF_RUN_DIR"
