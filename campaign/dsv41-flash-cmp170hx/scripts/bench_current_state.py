#!/usr/bin/env python3
"""Current-state sweep: prefill and decode across context x concurrency.

Decode: shared prefix (prefix-cached), ignore_eos, fixed output -> aggregate
and per-stream tok/s, plus DSpark acceptance from the engine counters.
Prefill: one unique prompt per arm at the target context -> TTFT and prefill tps.

Usage:
  python3 bench_current_state.py --base-url http://127.0.0.1:8090 --model deepseek-v4.1-flash \
      --prompts /mnt/kv/bench/dsv41-realtext.jsonl \
      --ctx 4096 32768 131072 524288 --conc 1 2 4 8 16 --label tp8pp1
"""
import argparse, json, os, re, statistics, time, urllib.request, concurrent.futures as cf

def _get(u, t=60):
    with urllib.request.urlopen(u, timeout=t) as r: return r.read().decode()

def counters(base):
    t = _get(base + "/metrics"); o = {}
    for n in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(r'vllm:spec_decode_%s_total\{[^}]*\}\s+([0-9.eE+]+)' % n, t)
        o[n] = float(m.group(1)) if m else 0.0
    return o

def post(base, payload, timeout=7200):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def build_prefix(prompts, tokens):
    buf, tot, i = [], 0, 0
    while tot < tokens:
        p = prompts[i % len(prompts)]; buf.append(p); tot += len(p)//3; i += 1
    cue = ("\n\nGiven the long document above, write a detailed new section of at least "
           "several hundred words summarizing the engineering tradeoffs. Do not stop early.\n\nSection:\n")
    return "\n\n".join(buf)[: max(1, tokens*32//10)] + cue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768, 131072, 524288])
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--decode-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/current-state/sweep.jsonl")
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fh = open(a.out, "a")

    print("# DECODE (prefix-cached, ignore_eos, %d tok out)" % a.decode_len)
    for ctx in a.ctx:
        prefix = build_prefix(prompts, ctx)
        # warm the prefix cache
        d = post(a.base_url, {"model": a.model, "prompt": prefix, "max_tokens": 8, "temperature": 0.0})
        ptok = (d.get("usage") or {}).get("prompt_tokens")
        print(f"## ctx={ctx} (warmup prompt_tokens={ptok})")
        for c in a.conc:
            for rep in range(a.reps):
                b0 = counters(a.base_url); t0 = time.time()
                with cf.ThreadPoolExecutor(max_workers=c) as ex:
                    rs = list(ex.map(lambda _: post(a.base_url, {"model": a.model, "prompt": prefix,
                            "max_tokens": a.decode_len, "temperature": 0.0, "ignore_eos": True}), range(c)))
                wall = time.time() - t0; b1 = counters(a.base_url)
                comp = sum((x.get("usage") or {}).get("completion_tokens", 0) for x in rs)
                dd, dt, da = (b1["num_drafts"]-b0["num_drafts"], b1["num_draft_tokens"]-b0["num_draft_tokens"],
                              b1["num_accepted_tokens"]-b0["num_accepted_tokens"])
                row = dict(label=a.label, mode="decode", ctx=ctx, conc=c, rep=rep,
                           agg_tok_s=round(comp/wall, 2) if wall else None,
                           per_stream_tok_s=round(comp/wall/c, 2) if wall else None,
                           tpot_ms=round(wall*1000/(comp/c), 2) if comp else None,
                           accept_pct=round(100*da/dt, 2) if dt else None, accept_len=round(da/dd, 3) if dd else None)
                print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()

    print("# PREFILL (unique prompt, 8 tok out)")
    for ctx in a.ctx:
        for c in a.conc:
            for rep in range(a.reps):
                # unique per, avoid prefix-cache reuse
                pls = [build_prefix(prompts, ctx) + f"\n\n[run {rep}-{c}-{i}]" for i in range(c)]
                t0 = time.time()
                try:
                    with cf.ThreadPoolExecutor(max_workers=c) as ex:
                        rs = list(ex.map(lambda p: post(a.base_url, {"model": a.model, "prompt": p,
                                "max_tokens": 8, "temperature": 0.0}), pls))
                    wall = time.time() - t0
                    pt = sum((x.get("usage") or {}).get("prompt_tokens", 0) for x in rs)
                    row = dict(label=a.label, mode="prefill", ctx=ctx, conc=c, rep=rep,
                               ttft_s=round(wall, 2), agg_prefill_tps=round(pt/wall, 1) if wall else None,
                               per_req_tps=round(pt/wall/c, 1) if wall else None)
                except Exception as e:
                    row = dict(label=a.label, mode="prefill", ctx=ctx, conc=c, rep=rep, error=str(e)[:120])
                print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()

    print("\n== summary ==")
    rows = [json.loads(l) for l in open(a.out) if json.loads(l)["label"] == a.label]
    for mode in ("decode", "prefill"):
        mr = [r for r in rows if r["mode"] == mode]
        if not mr: continue
        print(f"\n{mode}:")
        hdr = "ctx      " + "".join(f"c{c:<9}" for c in a.conc); print(hdr)
        for ctx in a.ctx:
            line = f"{ctx:<9}"
            for c in a.conc:
                v = [r for r in mr if r["ctx"] == ctx and r["conc"] == c]
                if mode == "decode":
                    line += f"{statistics.median([x['agg_tok_s'] for x in v]):<10.1f}" if v else "-         "
                else:
                    line += f"{statistics.median([x.get('agg_prefill_tps', 0) for x in v]):<10.1f}" if v else "-         "
            print(line)

if __name__ == "__main__": main()
