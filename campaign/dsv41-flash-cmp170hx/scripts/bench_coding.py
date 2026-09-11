#!/usr/bin/env python3
"""Coding-correctness eval for DeepSeek-V4.1-Flash (run-the-tests, not text match).

For each task: ask the model for a Python function, extract the code block,
execute it in a subprocess with the task's assertions, and record pass/fail.
Deterministic (temperature 0).

Usage:
  python3 bench_coding.py --base-url http://127.0.0.1:8090 --label dsv41
"""
import argparse, json, re, subprocess, sys, tempfile, os, time, urllib.request

TASKS = [
    ("merge_intervals",
     "Write a Python function `merge_intervals(intervals)` that merges overlapping "
     "closed intervals given as a list of [start, end] pairs, sorted output. "
     "Return only one fenced python code block.",
     "from solution import merge_intervals as f\n"
     "assert f([]) == []\n"
     "assert f([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert f([[1,4],[4,5]]) == [[1,5]]\n"
     "assert f([[5,6],[1,2]]) == [[1,2],[5,6]]\n"),
    ("lru_cache",
     "Write a Python class `LRUCache(capacity)` with `get(key)` and `put(key,value)` "
     "returning -1 for a miss, evicting least-recently-used on overflow. "
     "Return only one fenced python code block.",
     "from solution import LRUCache\n"
     "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3);\n"
     "assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4\n"),
    ("balanced",
     "Write a Python function `is_balanced(s)` returning True iff brackets ()[]{} are "
     "balanced and correctly nested. Return only one fenced python code block.",
     "from solution import is_balanced as f\n"
     "assert f('()[]{}') is True\n"
     "assert f('(]') is False\n"
     "assert f('([)]') is False\n"
     "assert f('{[]}') is True\n"
     "assert f('') is True\n"),
    ("top_k_freq",
     "Write a Python function `top_k(nums, k)` returning the k most frequent values, "
     "ties broken by smaller value first. Return only one fenced python code block.",
     "from solution import top_k as f\n"
     "assert f([1,1,1,2,2,3],2) == [1,2]\n"
     "assert f([1,2],2) == [1,2]\n"
     "assert f([4,4,4,4,3,3,2],1) == [4]\n"),
    ("level_order",
     "Write a Python function `level_order(root)` for a binary tree given as nested "
     "tuples (val, left, right) or None, returning values level by level. "
     "Return only one fenced python code block.",
     "from solution import level_order as f\n"
     "t=(3,(9,None,None),(20,(15,None,None),(7,None,None)))\n"
     "assert f(t) == [[3],[9,20],[15,7]]\n"
     "assert f(None) == []\n"),
]

def post(base, payload, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def extract(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1) if m else text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--label", default="dsv41")
    ap.add_argument("--out", default="/mnt/kv/logs/dsv41/coding/results.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fh = open(a.out, "a")
    passed = 0
    for name, prompt, tests in TASKS:
        t0 = time.time()
        try:
            d = post(a.base_url, {"model": a.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500, "temperature": 0.0})
            text = d["choices"][0]["message"].get("content") or ""
            code = extract(text)
        except Exception as e:
            text, code = f"<error {e}>", ""
        ok = False; err = ""
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "solution.py"), "w").write(code)
            open(os.path.join(td, "test_it.py"), "w").write(tests)
            try:
                r = subprocess.run([sys.executable, "test_it.py"], cwd=td,
                                   capture_output=True, text=True, timeout=60)
                ok = r.returncode == 0
                err = (r.stderr or "")[-200:]
            except Exception as e:
                err = str(e)[:200]
        passed += ok
        row = dict(label=a.label, task=name, pass_=ok, wall_s=round(time.time()-t0,2),
                   code_len=len(code), error=err)
        print(json.dumps(row)); fh.write(json.dumps(row)+"\n"); fh.flush()
    print(f"\n== coding pass {passed}/{len(TASKS)} ==")

if __name__ == "__main__": main()
