#!/usr/bin/env python3
"""Repeatable DSpark measurement for DeepSeek-V4.1-Flash on CMP 170HX.

Measures speculative-decoding acceptance directly from the engine's Prometheus
counters (vllm:spec_decode_num_*), around a controlled workload sent to
/v1/completions (raw text, so no chat template is needed). Reports:

  accepted/draft_tokens  -> acceptance rate
  accepted/drafts        -> acceptance length (tokens accepted per draft step)
  output tok/s           -> server-reported completion throughput
  decode TPOT            -> per-output-token latency

Usage:
  python3 bench_dspark_metrics.py --base-url http://127.0.0.1:8090 \
      --prompts /mnt/kv/bench/dsv41-realtext.jsonl \
      --conc 1 4 8 --output-len 128 --reps 2 --label k5
"""
import argparse, json, re, time, statistics, urllib.request, concurrent.futures as cf

def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()

def spec_counters(base):
    t = _get(base + "/metrics", timeout=30)
    out = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(r'vllm:spec_decode_%s_(?:total|created)\{[^}]*\}\s+([0-9.eE+]+)' % name, t)
        out[name] = float(m.group(1)) if m else None
    # gauge vs counter: prefer *_total
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(r'vllm:spec_decode_%s_total\{[^}]*\}\s+([0-9.eE+]+)' % name, t)
        if m: out[name] = float(m.group(1))
    return out

def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def one_request(base, model, prompt, output_len):
    t0 = time.time()
    d = post_json(base + "/v1/completions",
                  {"model": model, "prompt": prompt, "max_tokens": output_len,
                   "temperature": 0.0, "stream": False})
    dt = time.time() - t0
    u = d.get("usage") or {}
    return dict(seconds=dt, prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="k?")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/bench-dspark/metrics.jsonl")
    a = ap.parse_args()

    prompts = []
    for line in open(a.prompts):
        line = line.strip()
        if not line: continue
        try: prompts.append(json.loads(line)["prompt"])
        except Exception: pass
    if not prompts: raise SystemExit("no prompts")
    prompts = prompts[:a.num_prompts]

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fh = open(a.out, "a")
    for c in a.conc:
        for rep in range(a.reps):
            b0 = spec_counters(a.base_url)
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=c) as ex:
                results = list(ex.map(lambda p: one_request(a.base_url, a.model, p, a.output_len), prompts))
            wall = time.time() - t0
            b1 = spec_counters(a.base_url)
            d_drafts = b1["num_drafts"] - b0["num_drafts"]
            d_dtok = b1["num_draft_tokens"] - b0["num_draft_tokens"]
            d_acc = b1["num_accepted_tokens"] - b0["num_accepted_tokens"]
            comp = sum(r["completion_tokens"] for r in results)
            pre = sum(r["prompt_tokens"] for r in results)
            row = dict(label=a.label, conc=c, rep=rep, wall_s=round(wall, 3),
                       prompts=len(prompts), output_len=a.output_len,
                       completion_tokens=comp, prompt_tokens=pre,
                       output_tok_s=round(comp / wall, 2) if wall else None,
                       tpot_ms=round(wall * 1000 / comp, 2) if comp else None,
                       drafts=int(d_drafts), draft_tokens=int(d_dtok), accepted=int(d_acc),
                       accept_rate_pct=round(100 * d_acc / d_dtok, 2) if d_dtok else None,
                       accept_len=round(d_acc / d_drafts, 3) if d_drafts else None)
            print(json.dumps(row)); fh.write(json.dumps(row) + "\n"); fh.flush()

    rows = [json.loads(l) for l in open(a.out) if json.loads(l)["label"] == a.label]
    print("\n== summary label=%s ==" % a.label)
    for c in a.conc:
        rs = [r for r in rows if r["conc"] == c]
        if not rs: continue
        med = lambda k: statistics.median([r[k] for r in rs if r[k] is not None])
        print(f"c{c}: out={med('output_tok_s')} tok/s  TPOT={med('tpot_ms')} ms  "
              f"accept={med('accept_rate_pct')}%  len={med('accept_len')}  "
              f"drafts={med('drafts')} draft_tok={med('draft_tokens')} accepted={med('accepted')}")

if __name__ == "__main__":
    main()
