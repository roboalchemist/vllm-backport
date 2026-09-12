#!/usr/bin/env python3
"""LMCache accuracy validation for DeepSeek-V4.1-Flash (PP6 + LMCacheMPConnector).

Modes:
  send <label> <nonce>   one greedy request; prints output + external-hit metric delta
The planted retrieval key makes correctness checkable: any correct run must
answer KEY-<nonce>.  A LMCache hit must reproduce the SAME text as the cold run
and register on vllm:external_prefix_cache_hits_total / external_kv_transfer.

Usage: validate_lmcache_v41.py send cold   <nonce>
       validate_lmcache_v41.py send repeat <nonce>
"""
import json
import sys
import time
import urllib.request

VLLM = "http://127.0.0.1:8090"
MODEL = "deepseek-v4.1-flash"

FILLER = (
    "Routine maintenance of the playback archive requires verifying checksum "
    "labels on every shelf container before the weekly inventory review. "
)


def build_prompt(nonce):
    head = (
        f"Archive manifest {nonce}: the retrieval key for this manifest is "
        f"KEY-{nonce}. "
    ) * 2
    filler = FILLER * 460
    return head + filler + (
        f"Question: what is the exact retrieval key of archive manifest "
        f"{nonce}? Answer with the key only."
    )


def metrics():
    with urllib.request.urlopen(f"{VLLM}/metrics", timeout=15) as r:
        body = r.read().decode()
    want = {
        "external_hits": "vllm:external_prefix_cache_hits_total",
        "external_queries": "vllm:external_prefix_cache_queries_total",
        "local_hits": "vllm:prefix_cache_hits_total",
        "ext_kv_tokens": 'vllm:prompt_tokens_by_source_total{engine="0",model_name="deepseek-v4.1-flash",source="external_kv_transfer"}',
    }
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for k, prefix in want.items():
            if line.startswith(prefix + " "):
                try:
                    out[k] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def post(path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main():
    label, nonce = sys.argv[1], sys.argv[2]
    prompt = build_prompt(nonce)

    tok = post("/tokenize", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    })["count"]

    before = metrics()
    t0 = time.time()
    out = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    dt = time.time() - t0
    after = metrics()

    text = out["choices"][0]["message"]["content"].strip()
    delta = {k: round(after.get(k, 0) - before.get(k, 0), 1) for k in after}
    ok = f"KEY-{nonce}" in text
    print(json.dumps({
        "label": label,
        "nonce": nonce,
        "prompt_tokens": tok,
        "elapsed_s": round(dt, 3),
        "answer": text,
        "planted_key_present": ok,
        "metric_delta": delta,
    }, indent=2))
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("cold", "repeat"):
        print(__doc__)
        sys.exit(2)
    main()
