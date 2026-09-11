#!/usr/bin/env bash
# Build DeepSeek-V4.1-Flash on the v0.13.0 backport base (Schaka's new kit).
set -euo pipefail
ISO=unix:///mnt/kv/runtime/glm53-build/docker.sock
D() { sudo -n docker --host "$ISO" "$@"; }
BASE=${BASE:-docker.io/lazymio/vllm-backport:v0.13.0-sm80}
TAG=${TAG:-localhost/vllm-backport-v41:sm80-v13}
WORK=${WORK:-/mnt/kv/build/schaka-v13}
KIT=${KIT:-/tmp/opencode/schaka-journey/patches/vllm-backport-v41}

rm -rf "$WORK"; mkdir -p "$WORK"
cid=$(D create --entrypoint /bin/sh "$BASE")
D cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm" "$WORK/vllm"
D rm "$cid" >/dev/null
sudo -n chown -R "$(id -u):$(id -g)" "$WORK"; chmod -R u+rwX "$WORK"
find "$WORK/vllm" -name '*.pyc' -delete
find "$WORK/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cp -r "$WORK/vllm" "$WORK/vllm.orig"

( cd "$WORK" && python3 "$KIT/fixups.py" )
cp "$KIT/pp_kv_group_relay.py" "$WORK/vllm/models/deepseek_v4_1/"

rm -rf "$WORK/ctx" "$WORK/gitdiff"; mkdir -p "$WORK/ctx" "$WORK/gitdiff"
cp -r "$WORK/vllm.orig" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git init -q . && git add -A >/dev/null 2>&1 \
    && git -c user.email=b@b -c user.name=b commit -qm base >/dev/null )
rm -rf "$WORK/gitdiff/vllm"; cp -r "$WORK/vllm" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git add -A >/dev/null 2>&1 \
    && git diff --cached --name-only | grep '^vllm/' > "$WORK/changed.txt" )
echo "changed files: $(wc -l < "$WORK/changed.txt")"
( cd "$WORK" && tar cf - -T changed.txt ) | tar xf - -C "$WORK/ctx"
cp "$KIT/Containerfile" "$WORK/ctx/Dockerfile"
D build --network host -t "$TAG" "$WORK/ctx" 2>&1 | tail -3
echo "OVERLAY READY: $WORK/ctx/vllm  TAG=$TAG"
