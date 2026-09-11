#!/usr/bin/env bash
# Launch a specific TP/PP arrangement of DeepSeek-V4.1-Flash (Schaka recipe).
#   serve_arrangement.sh TP PP GPUS PARTITION [NAME]
set -euo pipefail

TP=${1:?TP}; PP=${2:?PP}; GPUS=${3:?GPUS}; PART=${4:-}; NAME=${5:-dsv41-schaka}

ISO=${ISO:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
IMG=${IMG:-localhost/vllm-backport-v41:sm80-bench}
MODEL=${MODEL:-/mnt/kv/models/DeepSeek-V4.1-Flash}
PORT=${PORT:-8090}
MAXLEN=${MAXLEN:-1048576}
UTIL=${UTIL:-0.92}
BLOCK=${BLOCK:-128}
SPEC_K=${SPEC_K:-5}
SEQS=${SEQS:-32}
EP=${EP:-1}
MAXBATCH=${MAXBATCH:-}
KVDTYPE=${KVDTYPE:-fp8_ds_mla}
KERNELEXTRA=${KERNELEXTRA:-}
D() { sudo -n docker --host "$ISO" "$@"; }

CACHE=/mnt/kv/cache/dsv41; mkdir -p "$CACHE"/{vllm,flashinfer,triton,nv,humming,cupy,tilelang}
D rm -f "$NAME" >/dev/null 2>&1 || true

ENVF=(-e OMP_NUM_THREADS=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NCCL_ALGO=${NCCL_ALGO:-Ring} -e NCCL_PROTO=${NCCL_PROTO:-Simple})
[ -n "$PART" ] && ENVF+=(-e "VLLM_PP_LAYER_PARTITION=$PART")

D run -d --name "$NAME" --init --restart no --runtime nvidia \
  --network host --ipc host --shm-size 32g --security-opt label=disable \
  -e "NVIDIA_VISIBLE_DEVICES=$GPUS" "${ENVF[@]}" \
  ${BREAKABLE:+-e VLLM_USE_BREAKABLE_CUDAGRAPH=$BREAKABLE} \
  ${PROFILER:+-e VLLM_TORCH_PROFILER_DIR=/tmp/prof} \
  ${PROFILER:+-v /mnt/kv/logs/dsv41/prof:/tmp/prof} \
  ${MARLIN_ATOMIC:+-e VLLM_MARLIN_USE_ATOMIC_ADD=$MARLIN_ATOMIC} \
  -v "$MODEL":/model:ro \
  ${OVERLAY_MOUNT:+-v $OVERLAY_MOUNT:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/utils.py:ro} \
  -v /mnt/kv/bench:/bench:ro \
  -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/flashinfer":/root/.cache/flashinfer \
  -v "$CACHE/triton":/root/.triton -v "$CACHE/nv":/root/.nv \
  -v "$CACHE/humming":/root/.humming -v "$CACHE/cupy":/root/.cupy -v "$CACHE/tilelang":/root/.tilelang \
  "$IMG" /model \
  --host 127.0.0.1 --port "$PORT" --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --attention-backend TRITON_MLA_SPARSE_DSV41 --kv-cache-dtype "$KVDTYPE" \
  --block-size "$BLOCK" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$SEQS" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  ${EP:+$([ "$EP" = 1 ] && echo --enable-expert-parallel)} \
  ${MAXBATCH:+--max-num-batched-tokens $MAXBATCH} \
  --disable-custom-all-reduce \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K}}" \
  --kernel-config "{\"enable_jit_warmup\": false${KERNELEXTRA:+, $KERNELEXTRA}}"
echo "launched $NAME TP$TP PP$PP on :$PORT"
