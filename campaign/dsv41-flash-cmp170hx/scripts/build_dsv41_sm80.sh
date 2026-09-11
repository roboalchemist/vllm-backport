#!/usr/bin/env bash
# Build the vLLM `dsv41-feat` branch as an SM80 (Ampere/CMP 170HX) image.
#
# Reuses the SSD-backed build daemon. The upstream Dockerfile accepts
# torch_cuda_arch_list / max_jobs / nvcc_threads / FINAL_BASE_IMAGE; we pin
# FINAL_BASE_IMAGE to the same NVCR CUDA base the GLM builds used (no Docker Hub).
set -euo pipefail

readonly SOURCE_DIR=${DSV41_SOURCE:-/mnt/kv/build/vllm-dsv41}
readonly EXPECTED_COMMIT=${DSV41_EXPECTED_COMMIT:-e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba}
readonly SERVING_TAG=${DSV41_TAG:-local/vllm-dsv41:sm80}
readonly BUILD_DOCKER_HOST=${DSV41_BUILD_DOCKER_HOST:-unix:///mnt/kv/runtime/glm53-build/docker.sock}
readonly FINAL_BASE_IMAGE=${DSV41_FINAL_BASE_IMAGE:-nvcr.io/nvidia/cuda@sha256:97d085a7423ee18ec483a2878b9be2c976dc4ba908aef96518beb00e1899dcc4}
readonly TORCH_ARCH=${DSV41_TORCH_ARCH:-8.0}
readonly MAX_JOBS=${DSV41_MAX_JOBS:-16}
readonly NVCC_THREADS=${DSV41_NVCC_THREADS:-2}
readonly MIN_FREE_GIB=${DSV41_MIN_FREE_GIB:-80}

output_dir=${1:-/mnt/kv/logs/dsv41/$(date -u +%Y%m%dT%H%M%SZ)-sm80-build}
[[ -d "$SOURCE_DIR/.git" ]] || { echo "not a git checkout: $SOURCE_DIR" >&2; exit 2; }
actual=$(git -C "$SOURCE_DIR" rev-parse HEAD)
[[ "$actual" == "$EXPECTED_COMMIT" ]] || { echo "source HEAD $actual != $EXPECTED_COMMIT" >&2; exit 2; }
mkdir -p "$output_dir"

docker_cmd=(sudo -n env "DOCKER_HOST=$BUILD_DOCKER_HOST" docker)
"${docker_cmd[@]}" info >/dev/null

available_kib=$(df -Pk /mnt/kv | awk 'NR==2{print $4}')
docker_free_kib=$("${docker_cmd[@]}" info --format '{{.DockerRootDir}}' >/dev/null 2>&1; df -Pk /mnt/kv | awk 'NR==2{print $4}')
[[ "$docker_free_kib" -ge $((MIN_FREE_GIB*1024*1024)) ]] || { echo "docker root free < ${MIN_FREE_GIB} GiB" >&2; exit 2; }

{
  printf 'created_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'source_dir=%q\nsource_commit=%s\n' "$SOURCE_DIR" "$actual"
  printf 'serving_tag=%q\n' "$SERVING_TAG"
  printf 'final_base_image=%q\n' "$FINAL_BASE_IMAGE"
  printf 'torch_cuda_arch_list=%q\nmax_jobs=%s\nnvcc_threads=%s\n' "$TORCH_ARCH" "$MAX_JOBS" "$NVCC_THREADS"
  printf 'build_docker_host=%q\n' "$BUILD_DOCKER_HOST"
  git -C "$SOURCE_DIR" remote -v
} >"$output_dir/build.env"

build_args=(
  --file "$SOURCE_DIR/docker/Dockerfile"
  --target vllm-openai
  --label "ai.roboalch.campaign=dsv41-flash-cmp170hx"
  --label "ai.roboalch.vllm.commit=$EXPECTED_COMMIT"
  --build-arg "VLLM_BUILD_COMMIT=$EXPECTED_COMMIT"
  --build-arg "VLLM_IMAGE_TAG=$SERVING_TAG"
  --build-arg "FINAL_BASE_IMAGE=$FINAL_BASE_IMAGE"
  --build-arg "RUN_WHEEL_CHECK=true"
  --build-arg "torch_cuda_arch_list=$TORCH_ARCH"
  --build-arg "max_jobs=$MAX_JOBS"
  --build-arg "nvcc_threads=$NVCC_THREADS"
  --build-arg "USE_SCCACHE=0"
  --tag "$SERVING_TAG"
)

"${docker_cmd[@]}" build --network host "${build_args[@]}" "$SOURCE_DIR" \
  2>&1 | tee "$output_dir/build-serving.log"

"${docker_cmd[@]}" image inspect "$SERVING_TAG" >"$output_dir/serving-image-inspect.json"
"${docker_cmd[@]}" run --rm --network none --entrypoint python3 "$SERVING_TAG" -c \
  "import vllm; print('vllm', vllm.__version__)" >"$output_dir/import-check.txt" 2>&1 || true
(cd "$output_dir" && sha256sum build.env build-serving.log serving-image-inspect.json import-check.txt > SHA256SUMS)
printf 'PASS: built %s; receipt=%s\n' "$SERVING_TAG" "$output_dir"
