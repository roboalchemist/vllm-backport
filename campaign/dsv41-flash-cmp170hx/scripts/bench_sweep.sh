#!/usr/bin/env bash
# DeepSeek-V4.1-Flash CMP170HX benchmark sweep.
# Runs vllm bench serve inside the serving container for a set of
# input lengths and concurrencies, capturing the reported summary lines.
set -euo pipefail

ISO=${ISO:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
NAME=${NAME:-dsv41-schaka}
MODEL_DIR=${MODEL_DIR:-/model}
BASE_URL=${BASE_URL:-http://127.0.0.1:8090}
OUT=${OUT:-/mnt/kv/logs/dsv41/bench}
mkdir -p "$OUT"

# Space-separated: "in_len:out_len:concurrency:num_prompts"
ARMS=${ARMS:-"1024:256:1:8 1024:256:2:8 1024:256:4:8 1024:256:8:16 1024:256:16:32"}

D() { sudo -n docker --host "$ISO" exec "$NAME" "$@"; }

for arm in $ARMS; do
  IFS=: read -r IN OUTL C N <<<"$arm"
  tag="in${IN}_out${OUTL}_c${C}"
  echo "=================== $tag ==================="
  D bash -lc "vllm bench serve --model deepseek-v4.1-flash --tokenizer $MODEL_DIR \
    --base-url $BASE_URL --dataset-name random \
    --seed $((RANDOM+1)) \
    --random-input-len $IN --random-output-len $OUTL \
    --num-prompts $N --max-concurrency $C --disable-tqdm" \
    2>&1 | tee "$OUT/bench-$tag.txt" | \
    grep -E "Request throughput|Output token throughput|Peak output|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL|Acceptance rate|Acceptance length|Peak concurrent" || true
done
