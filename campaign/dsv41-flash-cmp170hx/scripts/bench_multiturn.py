#!/usr/bin/env python3
"""Accumulated multi-turn prefix recall for DeepSeek-V4.1-Flash.

Builds an accumulating chat history: turn 1 plants a fact, then N filler turns
each add substantial text, and the final turn asks for the early fact. Verifies
recall and reports per-turn latency (which should drop once the shared prefix is
cached). This exercises prefix caching on a growing conversation prefix.

Usage:
  python3 bench_multiturn.py --base-url http://127.0.0.1:8090 \
      --prompts /mnt/kv/bench/dsv41-realtext.jsonl --turns 8 --label dsv41
"""
import argparse, json, random, time, urllib.request

def post(base, payload, timeout=3600):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--filler-chars", type=int, default=12000)
    ap.add_argument("--label", default="dsv41")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/multiturn/results.jsonl")
    a = ap.parse_args()

    prose = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    rng = random.Random(0)
    fact = f"MULTITURN-{rng.randrange(10**6):06d}-K"
    msgs = [{"role": "user", "content":
             f"Remember this access code for later: {fact}. Acknowledge in one word."}]
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True); fh = open(a.out, "a")
    turn_rows = []
    # turn 1
    t0 = time.time(); d = post(a.base_url, {"model": a.model, "messages": msgs,
                            "max_tokens": 200, "temperature": 0.0})
    msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""})
    turn_rows.append(dict(turn=1, wall_s=round(time.time()-t0,2), role="plant"))
    # filler turns
    for k in range(2, a.turns + 1):
        chunk = "\n\n".join(prose[(k*3) % max(1,len(prose)-8):][:8])[: a.filler_chars]
        msgs.append({"role": "user", "content":
                     f"Here is more reference material:\n{chunk}\n\nIn one sentence, what is this about?"})
        t0 = time.time(); d = post(a.base_url, {"model": a.model, "messages": msgs,
                                "max_tokens": 200, "temperature": 0.0})
        msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""})
        turn_rows.append(dict(turn=k, wall_s=round(time.time()-t0,2), role="filler"))
    # final recall
    msgs.append({"role": "user", "content":
                 "What was the access code I asked you to remember at the very start? Reply with only the code."})
    t0 = time.time(); d = post(a.base_url, {"model": a.model, "messages": msgs,
                            "max_tokens": 400, "temperature": 0.0})
    ans = d["choices"][0]["message"].get("content") or ""
    hit = fact in ans
    turn_rows.append(dict(turn=a.turns+1, wall_s=round(time.time()-t0,2), role="recall"))
    ctx_tokens = sum(len(m["content"]) // 3 for m in msgs)
    row = dict(label=a.label, turns=a.turns, fact=fact, hit=hit,
               approx_context_tokens=ctx_tokens, answer=ans.strip()[:120],
               per_turn=turn_rows)
    print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()
    print("\nper-turn wall:", " ".join(f"t{r['turn']}={r['wall_s']}s" for r in turn_rows))

if __name__ == "__main__": main()
