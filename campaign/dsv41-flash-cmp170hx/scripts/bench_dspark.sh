#!/usr/bin/env bash
# Repeatable DSpark / speculative-decoding benchmark for DeepSeek-V4.1-Flash
# on CMP 170HX.
#
# Runs `vllm bench serve` with a REAL text dataset (sonnet by default) so that
# DSpark acceptance reflects realistic content, not random tokens. Writes a
# machine-readable JSON summary per arm and prints a compact table.
#
# Usage:
#   SPEC_K=5 REPS=3 bash bench_dspark.sh                 # label the current server
#   DATASET=sharegpt INPUT=512 OUTPUT=128 CONC="1 4 8" bash bench_dspark.sh
#
# The DSpark arm is chosen at SERVER start (`--speculative-config`), so to
# compare k=5 vs off you run this script once per server config and compare the
# emitted SPEC_K-labelled JSON files.
set -euo pipefail

ISO=${ISO:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
NAME=${NAME:-dsv41-schaka}
MODEL_DIR=${MODEL_DIR:-/model}
BASE_URL=${BASE_URL:-http://127.0.0.1:8090}
MODEL=${MODEL:-deepseek-v4.1-flash}

DATASET=${DATASET:-custom}
DATASET_PATH=${DATASET_PATH:-/mnt/kv/bench/dsv41-realtext.jsonl}
INPUT=${INPUT:-512}
OUTPUT=${OUTPUT:-128}
CONC_LIST=${CONC:-"1 4 8"}
REPS=${REPS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-16}
SPEC_K=${SPEC_K:-unknown}
SEED=${SEED:-0}
OUT=${OUT:-/mnt/kv/logs/dsv41/bench-dspark}
mkdir -p "$OUT"

D() { sudo -n docker --host "$ISO" exec "$NAME" "$@"; }

summarize() {
  local f=$1
  python3 - "$f" <<'PY'
import re,sys,json
t=open(sys.argv[1]).read()
def g(pat, cast=float):
    m=re.search(pat+r"[^0-9\-]*([0-9.]+)", t)
    return cast(m.group(1)) if m else None
row=dict(
  output_tok_s=g(r"Output token throughput \(tok/s\):"),
  total_tok_s=g(r"Total token throughput \(tok/s\):"),
  req_s=g(r"Request throughput \(req/s\):"),
  ttft_ms=g(r"Mean TTFT \(ms\):"),
  tpot_ms=g(r"Mean TPOT \(ms\):"),
  itl_ms=g(r"Mean ITL \(ms\):"),
  accept_rate=g(r"Acceptance rate \(%\):"),
  accept_len=g(r"Acceptance length:"),
  drafts=g(r"Drafts:", int),
  draft_tokens=g(r"Draft tokens:", int),
  accepted=g(r"Accepted tokens:", int),
)
print(json.dumps(row))
PY
}

declare -a ROWS
for C in $CONC_LIST; do
  for R in $(seq 1 "$REPS"); do
    tag="dsv41_dspark_${DATASET}_in${INPUT}_out${OUTPUT}_c${C}_k${SPEC_K}_rep${R}"
    echo "=== $tag ==="
    D bash -lc "vllm bench serve --model $MODEL --tokenizer $MODEL_DIR \
      --base-url $BASE_URL --dataset-name $DATASET --seed $SEED \
      ${DATASET_PATH:+--dataset-path $DATASET_PATH} \
      --num-prompts $NUM_PROMPTS --max-concurrency $C --disable-tqdm" \
      >"$OUT/$tag.txt" 2>&1 || { echo "arm failed; see $OUT/$tag.txt"; continue; }
    js=$(summarize "$OUT/$tag.txt")
    echo "{\"tag\":\"$tag\",\"k\":\"$SPEC_K\",\"dataset\":\"$DATASET\",\"input\":$INPUT,\"output\":$OUTPUT,\"conc\":$C,\"rep\":$R,\"summary\":$js}" \
      >>"$OUT/results.jsonl"
    echo "  $js"
  done
done

echo
echo "=== summary (median over reps) : $OUT/results.jsonl ==="
python3 - "$OUT/results.jsonl" <<'PY'
import json,statistics as st,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
from collections import defaultdict
grp=defaultdict(list)
for r in rows: grp[(r["k"],r["conc"])].append(r["summary"])
for (k,c),vals in sorted(grp.items(),key=lambda x:(str(x[0][0]),x[0][1])):
    out=st.median(v["output_tok_s"] for v in vals if v["output_tok_s"])
    tpot=st.median(v["tpot_ms"] for v in vals if v["tpot_ms"])
    acc=st.median(v["accept_rate"] for v in vals if v["accept_rate"] is not None)
    alen=st.median(v["accept_len"] for v in vals if v["accept_len"] is not None)
    print(f"k={k} c={c}: out={out:.1f} tok/s  TPOT={tpot:.1f} ms  accept={acc}% len={alen}")
PY
