#!/usr/bin/env python3
"""32K master table harness (c1/c4/c8) for DeepSeek-V4.1-Flash on CMP 170HX.

One command produces the standard test: for each concurrency cell it runs
`vllm bench serve` (random dataset -> ignore_eos, fixed output length) inside the
serving container, parses the server/tokenizer-confirmed metrics, writes a JSON
result row set to the results DB, and renders a markdown table with a config
header (flags, KV dtype, KV token capacity).

Usage (host):
  python3 master_table.py --container dsv41-pp8 --label pp8-hb32 \
      --kv-dtype fp8_ds_mla --kv-capacity 9332296 \
      --flags "TP1xPP8 5x8 SEQS=32 UTIL=0.96 HEAD_BLOCK_SIZE=32" \
      --conc 1 4 8 --num-prompts 16 --input-len 32768 --output-len 256
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

ISO = os.environ.get("DOCKER_ISO", "unix:///mnt/kv/runtime/glm53-build/docker.sock")
DB = os.environ.get("MASTER_TABLE_DB", "/mnt/kv/logs/dsv41/master-table")

# label -> metric key
METRICS = {
    "Successful requests": "ok",
    "Failed requests": "fail",
    "Request throughput (req/s)": "req_s",
    "Output token throughput (tok/s)": "gen_tok_s",
    "Peak output token throughput (tok/s)": "gen_tok_s_peak",
    "Total token throughput (tok/s)": "total_tok_s",
    "Peak concurrent requests": "peak_conc",
    "Mean TTFT (ms)": "ttft_mean",
    "Median TTFT (ms)": "ttft_med",
    "P99 TTFT (ms)": "ttft_p99",
    "Mean TPOT (ms)": "tpot_mean",
    "Median TPOT (ms)": "tpot_med",
    "P99 TPOT (ms)": "tpot_p99",
    "Mean ITL (ms)": "itl_mean",
    "Median ITL (ms)": "itl_med",
    "P99 ITL (ms)": "itl_p99",
    "Acceptance rate (%)": "accept_pct",
    "Acceptance length": "accept_len",
    "Mean E2EL (ms)": "e2el_mean",
    "P99 E2EL (ms)": "e2el_p99",
}


def bench_cell(a, conc, rep=0):
    """Run one vllm bench serve cell; return parsed metrics dict.

    The seed is fixed by (base, conc, rep) so that every config is measured on
    the same prompt sets (comparable A/B) while each rep uses a *different*
    prompt set (no prefix-cache carry-over -> cold measurement).
    """
    seed = a.seed + conc + rep * 7919
    cmd = (
        f"vllm bench serve --model {a.model} --tokenizer {a.tokenizer} "
        f"--base-url {a.base_url} --dataset-name {a.dataset} --seed {seed} "
        + (f"--dataset-path {a.dataset_path} " if a.dataset_path else "")
        + (f"--custom-output-len {a.output_len} " if a.dataset == "custom" else "")
        + "--ignore-eos --skip-chat-template "
        + f"--input-len {a.input_len} --output-len {a.output_len} "
        f"--num-prompts {a.num_prompts} --max-concurrency {conc} --disable-tqdm"
    )
    t0 = time.time()
    p = subprocess.run(
        ["sudo", "-n", "docker", "--host", ISO, "exec", a.container, "bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=a.timeout,
    )
    out = p.stdout + p.stderr
    row = {"conc": conc, "wall_s": round(time.time() - t0, 1)}
    for label, key in METRICS.items():
        m = re.search(re.escape(label) + r"\s*:\s*([0-9.]+)", out)
        if m:
            row[key] = float(m.group(1))
    row["fail"] = int(row.get("fail", -1))
    return row, out


def render(a, rows, best):
    lines = []
    lines.append(f"### 32K master table — `{a.label}`")
    lines.append("")
    lines.append(f"- config: `{a.flags}`")
    lines.append(f"- KV dtype `{a.kv_dtype}`, KV token capacity **{a.kv_capacity:,}**"
                 if a.kv_capacity else f"- KV dtype `{a.kv_dtype}`")
    lines.append(f"- grid: input {a.input_len} x c{a.conc}, {a.num_prompts} prompts, "
                 f"{a.output_len} output tokens, ignore_eos, dataset `{a.dataset}`")
    lines.append("")
    lines.append("| conc | Gen tok/s (med [min-max]) | total tok/s | TTFT mean/med/p99 ms | "
                 "TPOT med/p99 ms | ITL p99 ms | Accept %/len | peak conc | fail |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        spread = ""
        if "gen_tok_s_min" in r:
            spread = " [{:.1f}-{:.1f}]".format(r["gen_tok_s_min"], r["gen_tok_s_max"])
        lines.append(
            "| c{} | **{:.1f}**{} | {:.0f} | {:.0f}/{:.0f}/{:.0f} | {:.1f}/{:.1f} | "
            "{:.1f} | {:.1f}/{:.2f} | {:.0f} | {} |".format(
                r["conc"], r.get("gen_tok_s", 0), spread, r.get("total_tok_s", 0),
                r.get("ttft_mean", 0), r.get("ttft_med", 0), r.get("ttft_p99", 0),
                r.get("tpot_med", 0), r.get("tpot_p99", 0), r.get("itl_p99", 0),
                r.get("accept_pct", 0), r.get("accept_len", 0),
                r.get("peak_conc", 0), r["fail"],
            )
        )
    if best:
        c1 = next((r for r in rows if r["conc"] == 1), None)
        c8 = next((r for r in rows if r["conc"] == max(x["conc"] for x in rows)), None)
        if c1 and c8 and c1.get("gen_tok_s"):
            ratio = c8.get("gen_tok_s", 0) / c1["gen_tok_s"]
            lines.append("")
            lines.append(f"- c{int(c8['conc'])}/c1 scaling ratio: **{ratio:.2f}x**")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--tokenizer", default="/model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--kv-dtype", default="fp8_ds_mla")
    ap.add_argument("--kv-capacity", type=int, default=0)
    ap.add_argument("--dataset", default="custom",
                    help="vllm bench dataset; custom = raw real-text prompts, no chat template")
    ap.add_argument("--dataset-path", default="/bench/dsv41-master.jsonl")
    ap.add_argument("--input-len", type=int, default=32768)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed seed so cells are comparable across configs")
    ap.add_argument("--reps", type=int, default=1,
                    help="repeat each cell; Gen tok/s reported as median over reps")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded warmup runs per cell before measuring")
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = {
        "ts": ts, "label": a.label, "flags": a.flags, "kv_dtype": a.kv_dtype,
        "kv_capacity": a.kv_capacity, "input_len": a.input_len,
        "output_len": a.output_len, "num_prompts": a.num_prompts, "rows": [],
    }
    for c in a.conc:
        for w in range(a.warmup):
            print(f"== {a.label} 32K c{c} warmup{w} (discarded) ==", flush=True)
            bench_cell(a, c, 1000 + w)
        reps = []
        for r in range(a.reps):
            print(f"== {a.label} 32K c{c} rep{r} ==", flush=True)
            row, out = bench_cell(a, c, r)
            reps.append(row)
            print(json.dumps(row), flush=True)
        agg = {"conc": c, "reps": a.reps}
        keys = [k for k in reps[0] if k not in ("conc", "reps", "wall_s", "fail")]
        for k in keys:
            vals = sorted(x[k] for x in reps if isinstance(x.get(k), (int, float)))
            if vals:
                agg[k] = vals[len(vals) // 2]
                agg[k + "_min"] = vals[0]
                agg[k + "_max"] = vals[-1]
        agg["fail"] = max(x["fail"] for x in reps)
        agg["raw"] = [{k: x.get(k) for k in ("gen_tok_s", "ttft_med", "tpot_med", "itl_p99")}
                      for x in reps]
        rec["rows"].append(agg)

    os.makedirs(DB, exist_ok=True)
    path = os.path.join(DB, f"{ts}-{a.label}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    table = render(a, rec["rows"], best=True)
    mpath = os.path.join(DB, f"{ts}-{a.label}.md")
    with open(mpath, "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nresults: {path}\nmarkdown: {mpath}")


if __name__ == "__main__":
    main()
