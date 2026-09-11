#!/usr/bin/env bash
# Adapted from Schaka/170hx-journey patches/vllm-backport-v41/build.sh
# Uses the glm53-build docker daemon instead of podman.
set -euo pipefail

ISO=unix:///mnt/kv/runtime/glm53-build/docker.sock
D() { sudo -n docker --host "$ISO" "$@"; }

BASE=docker.io/lazymio/vllm-backport:v0.12.0-sm80
VLLM_SRC=${VLLM_SRC:-/mnt/kv/build/vllm-dsv41}
WORK=${WORK:-/mnt/kv/build/schaka-v41}
KIT=${KIT:-/tmp/opencode/schaka-journey/patches/vllm-backport-v41}

rm -rf "$WORK"; mkdir -p "$WORK"

cid=$(D create --entrypoint /bin/sh "$BASE")
echo "container=$cid"
D cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm" "$WORK/vllm"
D rm "$cid" >/dev/null
sudo -n chown -R "$(id -u):$(id -g)" "$WORK"
chmod -R u+rwX "$WORK"

find "$WORK/vllm" -name '*.pyc' -delete
find "$WORK/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cp -r "$WORK/vllm" "$WORK/vllm.orig"

# 2. Apply upstream PR 56201.
MB=$( cd "$VLLM_SRC" && git merge-base upstream-main pr56201 )
N=$( cd "$VLLM_SRC" && git rev-list --count "$MB..pr56201" )
echo "PR diff: $N commits from $MB"
( cd "$VLLM_SRC" && git diff "$MB" pr56201 -- vllm ) > "$WORK/v41.patch"
( cd "$WORK" && git apply --reject --whitespace=nowarn v41.patch || true )

# 3. fixups.py
rm -rf "$WORK/prtree"; mkdir -p "$WORK/prtree"
( cd "$VLLM_SRC" && git archive pr56201 vllm ) | tar x -C "$WORK/prtree"
( cd "$WORK" && PR_TREE="$WORK/prtree" python3 "$KIT/fixups.py" )
find "$WORK/vllm" -name '*.rej' -delete

# 4. ampere shim + relay
mkdir -p "$WORK/vllm/models/deepseek_v4_1/ampere"
cp "$KIT/ampere/__init__.py" "$KIT/ampere/ampere_sparse.py" \
   "$KIT/ampere/qnorm_rope_kv_insert.py" \
   "$WORK/vllm/models/deepseek_v4_1/ampere/"
cp "$KIT/ampere/pp_kv_group_relay.py" "$WORK/vllm/models/deepseek_v4_1/"

# 5. changed files
rm -rf "$WORK/ctx" "$WORK/gitdiff"; mkdir -p "$WORK/ctx" "$WORK/gitdiff"
cp -r "$WORK/vllm.orig" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git init -q . && git add -A >/dev/null 2>&1 \
    && git -c user.email=b@b -c user.name=b commit -qm base >/dev/null )
rm -rf "$WORK/gitdiff/vllm"
cp -r "$WORK/vllm" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git add -A >/dev/null 2>&1 \
    && git diff --cached --name-only | grep '^vllm/' > "$WORK/changed.txt" )
echo "changed files: $(wc -l < "$WORK/changed.txt")"
( cd "$WORK" && tar cf - -T changed.txt ) | tar xf - -C "$WORK/ctx"
echo "OVERLAY READY: $WORK/ctx/vllm"
