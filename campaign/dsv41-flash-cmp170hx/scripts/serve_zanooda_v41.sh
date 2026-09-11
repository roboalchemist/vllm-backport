#!/usr/bin/env bash
# Launch Zanooda's DeepSeek-V4.1-Flash PP8 stack (image zanooda/vllm-sm80-ds41f:v41-sm80)
# adapted to this host's docker daemon (no bridge -> --network host) and GPUs.
set -euo pipefail

ISO=${ISO:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
D() { sudo -n docker --host "$ISO" "$@"; }

IMG=${IMG:-zanooda/vllm-sm80-ds41f:v41-sm80}
MODEL=${MODEL:-/mnt/kv/models/DeepSeek-V4.1-Flash}
SRC=${SRC:-/mnt/kv/build/vllm-zano}
NAME=${NAME:-dsv41-zano}
GPUS=${GPUS:-0,1,2,3,4,5,7,8}
PORT=${PORT:-8090}
PARTITION=${PARTITION:-5,5,5,5,5,5,5,5}
ENGRAM=${ENGRAM:-cpu}
MAXSEQS=${MAXSEQS:-8}
MEMUTIL=${MEMUTIL:-0.95}
MAXLEN=${MAXLEN:-1048576}
# commit after patches 0001-0008 (the image's build point); diff -> files 0009-0013
IMG_COMMIT=${IMG_COMMIT:-7fea9efe393197908e74a8e75089d0eff0e9483b}

MOUNTS=()
for f in $(git -C "$SRC" diff --name-only "$IMG_COMMIT" -- 'vllm/*.py'); do
  [ -f "$SRC/$f" ] && MOUNTS+=(-v "$SRC/$f:/vllm/$f:ro")
done
echo "bind-mounting ${#MOUNTS[@]} file(s) (2 per mount)"

D rm -f "$NAME" >/dev/null 2>&1 || true

D run -d --name "$NAME" --init --restart no --runtime nvidia \
  --network host --ipc host --shm-size 16g --security-opt label=disable \
  -e "NVIDIA_VISIBLE_DEVICES=$GPUS" \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e "VLLM_PP_LAYER_PARTITION=$PARTITION" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$MODEL":/model \
  "${MOUNTS[@]}" \
  "$IMG" vllm serve /model --host 127.0.0.1 --port "$PORT" --served-model-name dsv41 \
    --pipeline-parallel-size 8 --kv-cache-dtype fp8 --block-size 128 \
    --max-model-len "$MAXLEN" --max-num-batched-tokens 4096 --trust-remote-code \
    --gpu-memory-utilization "$MEMUTIL" --max-num-seqs "$MAXSEQS" \
    --engram-config "{\"storage\":\"$ENGRAM\",\"disk_threads\":32}" \
    --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
    --enable-prompt-tokens-details --limit-mm-per-prompt '{"image":8}' \
    --no-enable-flashinfer-autotune \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
echo "launched $NAME on :$PORT"
