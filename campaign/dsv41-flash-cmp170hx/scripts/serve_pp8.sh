#!/usr/bin/env bash
# DeepSeek-V4.1-Flash PP8 on 8x CMP 170HX (SM80). Eager AR bring-up.
#
# Topology rationale (see STATUS.md): 475.25 GiB weights; each Engram table is
# 91.55 GiB (> 1 card), so Engram is pinned to host RAM via
# --engram-config cpu_offload (vLLM default). No P2P ⇒ pipeline parallelism, not
# TP. 9 cards cannot form one group; PP8 on GPUs 0-7. GPU8 unused, GPU6 is an
# unrelated workload but is included here only because PP8 needs eight cards;
# prefer 0,1,2,3,4,5,7,8 when GPU6 must be left alone.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../glm5.3-flash/scripts/docker-command.sh"

readonly IMAGE=${DSV41_IMAGE:-local/vllm-dsv41:sm80}
readonly MODEL_DIR=${DSV41_MODEL_DIR:-/mnt/kv/models/DeepSeek-V4.1-Flash}
readonly PORT=${DSV41_PORT:-8090}
readonly PP=${DSV41_PP:-1}
readonly TP=${DSV41_TP:-8}
readonly GPUS=${DSV41_GPUS:-0,1,2,3,4,5,7,8}
readonly MAX_MODEL_LEN=${DSV41_MAX_MODEL_LEN:-32768}
readonly MAX_NUM_SEQS=${DSV41_MAX_NUM_SEQS:-1}
readonly UTIL=${DSV41_UTIL:-0.90}
readonly SPEC=${DSV41_SPEC:-none}
readonly RUN_DIR=${DSV41_RUN_DIR:-/mnt/kv/logs/dsv41/$(date -u +%Y%m%dT%H%M%SZ)-pp${PP}-${SPEC}}
readonly CONTAINER=${DSV41_CONTAINER:-dsv41-pp${PP}}
readonly CACHE_ROOT=${DSV41_CACHE_ROOT:-/mnt/kv/cache/dsv41}

engram_args=()
if [[ "${DSV41_ENGRAM:-1}" == 1 ]]; then
  engram_args+=(--engram-config '{"cpu_offload": true}')
fi
spec_args=()
case "$SPEC" in
  none) ;;
  dspark) spec_args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5}') ;;
  *) echo "DSV41_SPEC must be none|dspark" >&2; exit 2 ;;
esac

# Overlay the SM80 port files (patches 0001/0002) over the image's vllm package.
# The built image may predate the later patch, so mount every changed .py.
readonly OVERLAY_SRC=${DSV41_OVERLAY_SRC:-/mnt/kv/build/vllm-dsv41}
readonly OVERLAY_LIST=${DSV41_OVERLAY_LIST:-$SCRIPT_DIR/../sm80-overlay-files.txt}
overlay_mounts=()
if [[ -f "$OVERLAY_LIST" ]]; then
  while read -r f; do
    [[ -z "$f" ]] && continue
    [[ -f "$OVERLAY_SRC/$f" ]] || { echo "overlay file missing: $OVERLAY_SRC/$f" >&2; exit 2; }
    overlay_mounts+=(--volume "$OVERLAY_SRC/$f:/usr/local/lib/python3.12/dist-packages/$f:ro")
  done < "$OVERLAY_LIST"
fi

mkdir -p "$RUN_DIR" "$CACHE_ROOT"/{tmp,huggingface,xdg,torchinductor,triton,cuda,torch-extensions,flashinfer}
{
  printf 'image=%q\nmodel_dir=%q\ntp=%q\npp=%q\ngpus=%q\nport=%q\n' "$IMAGE" "$MODEL_DIR" "$TP" "$PP" "$GPUS" "$PORT"
  printf 'max_model_len=%q\nmax_num_seqs=%q\nutil=%q\nspec=%q\n' "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$UTIL" "$SPEC"
} >"$RUN_DIR/launch.env"

exec "${docker_cmd[@]}" run -d --init --name "$CONTAINER" \
  --runtime nvidia --network host --ipc host --shm-size 32g --security-opt label=disable \
  --ulimit memlock=-1:-1 --stop-timeout 120 \
  --health-cmd "curl -fsS http://127.0.0.1:$PORT/health >/dev/null || exit 1" \
  --health-interval 15s --health-timeout 5s --health-retries 4 --health-start-period 7200s \
  --label ai.roboalch.campaign=dsv41-flash-cmp170hx \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$CACHE_ROOT:/cache" \
  "${overlay_mounts[@]}" \
  --env "NVIDIA_VISIBLE_DEVICES=$GPUS" \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env HF_HUB_OFFLINE=1 --env HF_HOME=/cache/huggingface --env XDG_CACHE_HOME=/cache/xdg --env TMPDIR=/cache/tmp \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor --env TRITON_CACHE_DIR=/cache/triton \
  --env CUDA_CACHE_PATH=/cache/cuda --env TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env "DSV41_DEBUG_WOA=${DSV41_DEBUG_WOA:-0}" \
  --env "DSV41_ZERO_ENGRAM=${DSV41_ZERO_ENGRAM:-0}" \
  --env "DSV41_HIDDEN_DEBUG=${DSV41_HIDDEN_DEBUG:-0}" \
  --env NCCL_P2P_DISABLE=1 --env NCCL_IB_DISABLE=1 \
  "$IMAGE" \
  --model /model \
  --served-model-name "${DSV41_SERVED_MODEL_NAME:-deepseek-v4.1-flash}" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --distributed-executor-backend mp \
  "${engram_args[@]}" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$UTIL" \
  --enforce-eager \
  --language-model-only \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 \
  "${spec_args[@]}"
