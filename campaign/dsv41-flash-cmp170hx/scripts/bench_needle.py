#!/usr/bin/env python3
"""Long-context viability (needle-at-depth) for DeepSeek-V4.1-Flash.

Plants a unique fact at a chosen fraction of a long prompt, asks for it, and
checks the answer. Uses prefix caching: a filler prefix is reused across depths
so the harness measures recall, not re-prefill, as depth grows.

Usage:
  python3 bench_needle.py --base-url http://127.0.0.1:8090 \
      --prompts /mnt/kv/bench/dsv41-realtext.jsonl \
      --depths 4096 32768 131072 524288 --positions 0.15 0.5 0.85 --label dsv41
"""
import argparse, json, random, re, time, urllib.request

def _get(u, t=60):
    with urllib.request.urlopen(u, timeout=t) as r: return r.read().decode()

def post(base, payload, timeout=3600):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def build(prompts, tokens, position, fact):
    # ~3.2 chars/token
    target = max(1, int(tokens * 3.2))
    buf, tot = [], 0
    i = 0
    while tot < target:
        p = prompts[i % len(prompts)]; buf.append(p); tot += len(p); i += 1
    text = "\n\n".join(buf)[:target]
    cut = int(len(text) * position)
    # keep the fact on a clean boundary
    return text[:cut] + f"\n\nIMPORTANT NOTE: the access code is {fact}.\n\n" + text[cut:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--positions", type=float, nargs="+", default=[0.15, 0.5, 0.85])
    ap.add_argument("--label", default="needle")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/needle/results.jsonl")
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    rng = random.Random(0)
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True); fh = open(a.out, "a")
    ok = tot = 0
    for depth in a.depths:
        for pos in a.positions:
            fact = f"NEEDLE-{rng.randrange(10**6):06d}-X"
            ctx = build(prompts, depth, pos, fact)
            q = ("\n\nQuestion: what is the access code stated in the IMPORTANT NOTE above? "
                 "Reply with only the code.")
            t0 = time.time()
            try:
                d = post(a.base_url, {"model": a.model,
                        "messages": [{"role": "user", "content": ctx + q}],
                        "max_tokens": 300, "temperature": 0.0})
                ans = d["choices"][0]["message"].get("content") or ""
            except Exception as e:
                ans = f"<error {e}>"
            hit = fact in ans
            tot += 1; ok += hit
            row = dict(label=a.label, depth=depth, position=pos, fact=fact,
                       hit=hit, wall_s=round(time.time()-t0,2), answer=ans.strip()[:120])
            print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()
    print(f"\n== recall {ok}/{tot} ==")

if __name__ == "__main__": main()
