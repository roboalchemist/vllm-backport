#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash with Schaka's proven 170hx-journey recipe.
# TP4 x PP2 over 8 CMP 170HX (SM80), ports 8090 (host) -> 8000 (container).
set -euo pipefail

ISO=unix:///mnt/kv/runtime/glm53-build/docker.sock
D() { sudo -n docker --host "$ISO" "$@"; }

NAME=${NAME:-dsv41-schaka}
IMG=${IMG:-localhost/vllm-backport-v41:sm80}
MODEL=${MODEL:-/mnt/kv/models/DeepSeek-V4.1-Flash}
GPUS=${GPUS:-0,1,2,3,4,5,7,8}
PORT=${PORT:-8090}
PARTITION=${PARTITION:-20,20}
MAXLEN=${MAXLEN:-1048576}
UTIL=${UTIL:-0.92}
BLOCK=${BLOCK:-128}
SPEC_K=${SPEC_K:-5}
SEQS=${SEQS:-8}

CACHE=/mnt/kv/cache/dsv41
mkdir -p "$CACHE"/{vllm,flashinfer,triton,nv,humming,cupy,tilelang}

D rm -f "$NAME" 2>/dev/null || true

D run -d --name "$NAME" --init --restart no \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  --network host --ipc host --shm-size 32g --security-opt label=disable \
  -v "$MODEL":/model:ro \
  -v "$CACHE/vllm":/root/.cache/vllm \
  -v "$CACHE/flashinfer":/root/.cache/flashinfer \
  -v "$CACHE/triton":/root/.triton \
  -v "$CACHE/nv":/root/.nv \
  -v "$CACHE/humming":/root/.humming \
  -v "$CACHE/cupy":/root/.cupy \
  -v "$CACHE/tilelang":/root/.tilelang \
  "$IMG" \
  /model \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --attention-backend TRITON_MLA_SPARSE_DSV41 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size "$BLOCK" \
  --dtype bfloat16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs "$SEQS" \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --disable-custom-all-reduce \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K}}" \
  --kernel-config '{"enable_jit_warmup": false}'

echo "launched $NAME on :$PORT"
