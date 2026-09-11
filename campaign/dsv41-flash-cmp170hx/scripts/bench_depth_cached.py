#!/usr/bin/env python3
"""Prefix-cached deep-context decode benchmark for DeepSeek-V4.1-Flash.

A 512k prefill costs ~5 min, so measuring aggregate decode at 512k depth by
prefilling per request is impractical. Instead:

  1. build one shared prefix of ~N tokens (real prose repeated),
  2. send a single warmup request to populate the prefix cache,
  3. fire a concurrency sweep of requests that REUSE that prefix with short
     outputs, so each request skips the 512k prefill and measures pure
     decode-at-depth.

Reports server-reported output tok/s, TPOT, and DSpark acceptance (from
vllm:spec_decode_* counters) per concurrency.

Usage:
  python3 bench_depth_cached.py --base-url http://127.0.0.1:8090 \
      --prompts /mnt/kv/bench/dsv41-realtext.jsonl \
      --context 524288 --output-len 64 --conc 4 8 16 --reps 2 --label 512k
"""
import argparse, json, re, time, statistics, urllib.request, concurrent.futures as cf

def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r: return r.read().decode()

def counters(base):
    t = _get(base + "/metrics")
    out = {}
    for n in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(r'vllm:spec_decode_%s_total\{[^}]*\}\s+([0-9.eE+]+)' % n, t)
        out[n] = float(m.group(1)) if m else 0.0
    return out

def post(base, payload, timeout=3600):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def build_prefix(prompts, target_tokens):
    # ~3.2 chars/token for this tokenizer; repeat prose until we reach target,
    # then append a cue that forces a long continuation.
    buf, total = [], 0
    i = 0
    while total < target_tokens:
        p = prompts[i % len(prompts)]
        buf.append(p); total += len(p) // 3
        i += 1
    text = "\n\n".join(buf)
    cue = ("\n\nGiven the long document above, write a detailed new section of at "
           "least several hundred words that summarizes the engineering tradeoffs "
           "discussed, in your own words. Do not stop early.\n\nSection:\n")
    return text[: max(1, target_tokens * 32 // 10)] + cue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--context", type=int, default=524288)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--conc", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="depth")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/bench-depth/cached.jsonl")
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    prefix = build_prefix(prompts, a.context)
    print(f"prefix chars={len(prefix)} target_tokens~{a.context}")

    # warmup: populate prefix cache (single request, long enough to prefill).
    for w in range(a.warmup):
        t0 = time.time()
        d = post(a.base_url, {"model": a.model, "prompt": prefix, "max_tokens": 8,
                              "temperature": 0.0})
        u = (d.get("usage") or {})
        print(f"warmup{w+1}: prompt_tokens={u.get('prompt_tokens')} wall={time.time()-t0:.1f}s")

    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fh = open(a.out, "a")
    for c in a.conc:
        for rep in range(a.reps):
            b0 = counters(a.base_url)
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                rs = list(ex.map(lambda _: post(a.base_url, {"model": a.model, "prompt": prefix,
                        "max_tokens": a.output_len, "temperature": 0.0, "ignore_eos": True}), range(c)))
            wall = time.time() - t0
            b1 = counters(a.base_url)
            comp = sum((d.get("usage") or {}).get("completion_tokens", 0) for d in rs)
            pre = sum((d.get("usage") or {}).get("prompt_tokens", 0) for d in rs)
            dd, dt, da = (b1["num_drafts"]-b0["num_drafts"],
                          b1["num_draft_tokens"]-b0["num_draft_tokens"],
                          b1["num_accepted_tokens"]-b0["num_accepted_tokens"])
            row = dict(label=a.label, context=a.context, conc=c, rep=rep, wall_s=round(wall,3),
                       completion_tokens=comp, prompt_tokens=pre,
                       output_tok_s=round(comp/wall,2) if wall else None,
                       tpot_ms=round(wall*1000/comp,2) if comp else None,
                       drafts=int(dd), draft_tokens=int(dt), accepted=int(da),
                       accept_rate_pct=round(100*da/dt,2) if dt else None,
                       accept_len=round(da/dd,3) if dd else None)
            print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()

    rows=[json.loads(l) for l in open(a.out) if json.loads(l)["label"]==a.label]
    print("\n== summary label=%s ctx=%d ==" % (a.label, a.context))
    for c in a.conc:
        rs=[r for r in rows if r["conc"]==c]
        if not rs: continue
        def med(k):
            vals=[r[k] for r in rs if r[k] is not None]
            return statistics.median(vals) if vals else None
        print(f"c{c}: out={med('output_tok_s')} tok/s  TPOT={med('tpot_ms')} ms  accept={med('accept_rate_pct')}%  len={med('accept_len')}")

if __name__=="__main__": main()
