#!/usr/bin/env bash
# Repeatable TP/PP arrangement benchmark for DeepSeek-V4.1-Flash on CMP 170HX.
#
# For each arrangement it recreates the serving container, waits for health,
# then runs the decode sweep (bench_sweep.sh) and the DSpark real-text bench
# (bench_dspark.sh), recording a JSON summary per arrangement.
#
# ARRANGEMENTS: space-separated "label:TP:PP:GPUS:PARTITION"
# (empty PARTITION => don't set VLLM_PP_LAYER_PARTITION)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISO=${ISO:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
NAME=${NAME:-dsv41-schaka}
OUT=${OUT:-/mnt/kv/logs/dsv41/arrangements}
mkdir -p "$OUT"

ARRANGEMENTS=${ARRANGEMENTS:-"tp8pp1:8:1:0,1,2,3,4,5,7,8: tp4pp2:4:2:0,1,2,3,4,5,7,8:20,20 tp2pp3:2:3:0,1,2,3,4,5:14,14,12"}

wait_health() {
  for i in $(seq 1 150); do
    if curl -fsS -m 5 http://127.0.0.1:8090/health >/dev/null 2>&1; then echo " healthy after ~$((i*15))s"; return 0; fi
    sleep 15
  done
  echo " TIMEOUT waiting for health" >&2; return 1
}

D() { sudo -n docker --host "$ISO" "$@"; }

for spec in $ARRANGEMENTS; do
  IFS=: read -r label tp pp gpus part <<<"$spec"
  echo "############ arrangement $label (TP$tp PP$pp GPUS=$gpus part=${part:-none}) ############"
  D rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 6
  if D ps -a --filter name="$NAME" --format '{{.Names}}' 2>/dev/null | grep -q "$NAME"; then
    sudo -n systemctl restart glm53-build-docker.service; sleep 10; D rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  PART_ENV=""
  [ -n "$part" ] && PART_ENV="PARTITION=$part"
  bash "$HERE/serve_arrangement.sh" "$tp" "$pp" "$gpus" "$part" "$NAME" >"$OUT/$label.serve.log" 2>&1 || { echo " serve failed"; continue; }
  if ! wait_health; then continue; fi

  echo "-- decode sweep --"
  ARMS="1024:256:1:4 1024:256:4:8 1024:256:8:16" OUT="$OUT/$label" \
    bash "$HERE/bench_sweep.sh" >"$OUT/$label.sweep.log" 2>&1 || true
  echo "-- dspark real-text --"
  SPEC_K="$tp$pp" CONC="1 4 8" REPS=1 OUT="$OUT/$label/dspark" \
    bash "$HERE/bench_dspark.sh" >"$OUT/$label.dspark.log" 2>&1 || true

  python3 - "$OUT/$label" "$label" <<'PY' >>"$OUT/arrangements.jsonl"
import json,sys,re,glob,statistics as st
d,label=sys.argv[1],sys.argv[2]
def med(pat,files):
    vals=[]
    for f in files:
        t=open(f).read()
        for m in re.finditer(pat+r"[^0-9\-]*([0-9.]+)",t): vals.append(float(m.group(1)))
    return round(st.median(vals),3) if vals else None
sweep=sorted(glob.glob(d+"/bench-in*.txt"))
dsp=sorted(glob.glob(d+"/dspark/dsv41_dspark_*.txt"))
print(json.dumps(dict(label=label,
  decode_tok_s=med(r"Output token throughput \(tok/s\):",sweep),
  ttft_ms=med(r"Mean TTFT \(ms\):",sweep),
  tpot_ms=med(r"Mean TPOT \(ms\):",sweep),
  dspark_out_tok_s=med(r"Output token throughput \(tok/s\):",dsp),
  dspark_accept_rate=med(r"Acceptance rate \(%\):",dsp),
  dspark_accept_len=med(r"Acceptance length:",dsp))))
PY
done

echo; echo "=== arrangements summary ($OUT/arrangements.jsonl) ==="
cat "$OUT/arrangements.jsonl" 2>/dev/null || true
