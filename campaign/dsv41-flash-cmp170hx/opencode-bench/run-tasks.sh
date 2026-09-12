#!/bin/bash
# Repeatable real-OpenCode coding benchmark. Runs the bench agent on real coding
# tasks against the bundled fixture, and prints per-task wall time + output tokens.
set -u
MODEL="${BENCH_MODEL:-dsv41-bench/deepseek-v4.1-flash}"
AGENT="${BENCH_AGENT:-bench}"
REPEAT="${BENCH_REPEAT:-1}"
OUT="${BENCH_OUT:-/work/bench-results.jsonl}"
: > "$OUT"

tasks=(
  "fix_bugs|The tests in tests/test_textstats.py fail. Fix src/textstats.py so all tests pass. Run python3 -c \"import tests.test_textstats\" or read the code carefully. Do not change the tests."
  "add_feature|Add a function running_max(values) to src/textstats.py returning the running maximum, and add a test for it in tests/test_textstats.py. Keep existing behaviour."
  "docstring|Add a concise module docstring and function docstrings to every function in src/textstats.py without changing behaviour."
)

run_one() {
  local name="$1" prompt="$2" rep="$3"
  local t0 t1 f
  f="/work/.out-${name}-${rep}"
  t0=$(date +%s.%N)
  OPENCODE_CONFIG=/root/.config/opencode/opencode.json \
    opencode run --format json --model "$MODEL" --agent "$AGENT" \
      --title "bench $name" --dir /work "$prompt" >"$f.jsonl" 2>"$f.stderr" || true
  t1=$(date +%s.%N)
  python3 - "$f.jsonl" "$name" "$t0" "$t1" "$rep" <<'PY'
import json,sys
f,name,t0,t1,rep=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
pin=pout=preason=steps=0
for line in open(f):
    try: d=json.loads(line)
    except: continue
    if d.get("type")=="step_finish":
        tok=d.get("part",{}).get("tokens") or d.get("tokens") or {}
        pin+=tok.get("input",0); pout+=tok.get("output",0)
        preason+=tok.get("reasoning",0); steps+=1
wall=t1-t0
print(json.dumps(dict(task=name,rep=int(rep),wall_s=round(wall,2),steps=steps,
    input_tokens=pin,output_tokens=pout,reasoning_tokens=preason,
    output_tok_s=round(pout/wall,2) if wall>0 else None,
    total_tokens_per_s=round((pin+pout)/wall,2) if wall>0 else None)))
PY
}

echo "== real OpenCode bench: model=$MODEL agent=$AGENT repeat=$REPEAT =="
for rep in $(seq 1 "$REPEAT"); do
  for t in "${tasks[@]}"; do
    name="${t%%|*}"; prompt="${t#*|}"
    row=$(run_one "$name" "$prompt" "$rep")
    echo "$row" | tee -a "$OUT"
  done
done
echo "results: $OUT"
