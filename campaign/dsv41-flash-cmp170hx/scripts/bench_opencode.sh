#!/usr/bin/env bash
# Repeatable live-OpenCode benchmark for DeepSeek-V4.1-Flash on CMP 170HX.
#
# Drives a real (headless) OpenCode session per prompt against the served
# model, then reports end-to-end wall clock, server-reported token usage, and
# output tok/s. Emits JSONL for later comparison. Nothing about the model or
# the agent is mocked: OpenCode performs its real tool-calling loop.
#
# Usage:
#   REPEAT=2 bash bench_opencode.sh
#   PROMPTS_FILE=./prompts.txt OPENCODE_PROOF_CONFIG=.../opencode.direct.json bash bench_opencode.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CONFIG=${OPENCODE_PROOF_CONFIG:-$REPO_ROOT/campaigns/2026-09-10-dsv41-flash-cmp170hx/opencode.direct.json}
MODEL=${OC_MODEL:-dsv41-direct/deepseek-v4.1-flash}
AGENT=${OC_AGENT:-dsv41-direct}
BIN=${OC_BIN:-/home/chunglet/.opencode/bin/opencode}
PROMPTS=${PROMPTS_FILE:-$REPO_ROOT/campaigns/2026-09-10-dsv41-flash-cmp170hx/scripts/opencode-bench-prompts.txt}
REPEAT=${REPEAT:-2}
LABEL=${LABEL:-baseline}
OUT=${OUT:-/mnt/kv/logs/dsv41/bench-opencode}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
export RT="$OUT/$LABEL/$RUN_ID"
mkdir -p "$RT"/{data,config,state,cache}

export XDG_DATA_HOME="$RT/data" XDG_CONFIG_HOME="$RT/config" \
       XDG_STATE_HOME="$RT/state" XDG_CACHE_HOME="$RT/cache"
export OPENCODE_CONFIG="$CONFIG" OPENCODE_DISABLE_PROJECT_CONFIG=1 \
       OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1

mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$PROMPTS")

run_one() {
  local name=$1 prompt=$2 out=$3
  local t0 t1
  t0=$(date +%s.%N)
  "$BIN" run --pure --format json --model "$MODEL" --agent "$AGENT" \
    --title "bench $name" --dir "$REPO_ROOT" "$prompt" \
    >"$out.jsonl" 2>"$out.stderr" || true
  t1=$(date +%s.%N)
  python3 - "$out.jsonl" "$name" "$t0" "$t1" "$LABEL" <<'PY'
import json,sys
f,name,t0,t1,label=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
pin=pout=preason=ptot=0; steps=0
for line in open(f):
    try: d=json.loads(line)
    except: continue
    if d.get("type")=="step_finish":
        tok=d.get("part",{}).get("tokens") or d.get("tokens") or {}
        pin+=tok.get("input",0); pout+=tok.get("output",0)
        preason+=tok.get("reasoning",0); ptot+=tok.get("total",0); steps+=1
wall=t1-t0
row=dict(label=label,prompt=name,wall_s=round(wall,3),steps=steps,
         input_tokens=pin,output_tokens=pout,reasoning_tokens=preason,total_tokens=ptot,
         output_tok_s=round(pout/wall,2) if wall>0 else None)
print(json.dumps(row))
open("/dev/stdout").flush()
PY
}

echo "== live OpenCode bench: label=$LABEL model=$MODEL reps=$REPEAT =="
for rep in $(seq 1 "$REPEAT"); do
  for line in "${LINES[@]}"; do
    name=${line%%|*}; prompt=${line#*|}
    f="$RT/${name}_rep${rep}"
    row=$(run_one "$name" "$prompt" "$f")
    echo "$row" | tee -a "$RT/results.jsonl"
  done
done

echo
echo "== summary =="
python3 - "$RT/results.jsonl" <<'PY'
import json,sys,statistics as st
from collections import defaultdict
rows=[json.loads(l) for l in open(sys.argv[1])]
g=defaultdict(list)
for r in rows: g[r["prompt"]].append(r)
print(f"{'prompt':28s} {'out_tok':>8s} {'wall_s':>8s} {'tok/s':>7s} {'reason':>7s}")
for name,rs in g.items():
    o=st.median(r["output_tokens"] for r in rs); w=st.median(r["wall_s"] for r in rs)
    t=st.median(r["output_tok_s"] for r in rs); re=st.median(r["reasoning_tokens"] for r in rs)
    print(f"{name:28s} {o:8.0f} {w:8.2f} {t:7.2f} {re:7.0f}")
allt=[r["output_tok_s"] for r in rows]
allo=[r["output_tokens"] for r in rows]
print(f"\noverall: median end-to-end {st.median(allt):.2f} tok/s, median output {st.median(allo):.0f} tok")
PY
echo "artifacts: $RT"
