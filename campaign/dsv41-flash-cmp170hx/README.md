# DeepSeek-V4.1-Flash on CMP 170HX (sm_80) — working serving recipe

A working **DeepSeek-V4.1-Flash** serving configuration on NVIDIA CMP 170HX
(GA100, `sm_80`, PCIe Gen2 x4), built as a Python overlay on
`wtdcode/vllm-backport`. Deployed as **TP1 × PP6** (pipeline parallel only; no
tensor parallelism on this restrictive PCIe topology).

Everything below was measured on this host. Author-reported third-party numbers
are marked as such.

## Provenance

- Base image: `lazymio/vllm-backport:v0.13.0-sm80` (`bfe237ee0886`).
- Ampere shims + PP KV-group relay: adopted from
  [`Schaka/170hx-journey`](https://github.com/Schaka/170hx-journey)
  (`patches/vllm-backport-v41/`, head `a5bc5c1d`), including the indexer-K
  ownership fix (`_build_index_k_replica`).
- Model support: upstream vLLM PR #56201 ("[Model] Support
  DeepSeek-V4.1-Flash").
- Our additions: serve/benchmark harnesses, the current-state sweeps, the
  real-OpenCode c1 harness, the LMCache bring-up, and the lever audits.

Build: `scripts/build_schaka_v13.sh` -> `localhost/vllm-backport-v41:sm80-v13`.

## Deployment configuration (default)

```
Image:     localhost/vllm-backport-v41:sm80-v13
Layout:    TP1 x PP6, VLLM_PP_LAYER_PARTITION=7,7,7,7,7,5
Cache:     --kv-cache-dtype fp8_ds_mla --block-size 128
Attention: --attention-backend TRITON_MLA_SPARSE_DSV41
Spec:      --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
Context:   --max-model-len 1048576 --enable-prefix-caching
Serving:   --gpu-memory-utilization 0.96 --max-num-seqs 8
Tools:     --enable-auto-tool-choice --tool-call-parser deepseek_v41
           --reasoning-parser deepseek_v41
```

Launch: `scripts/serve_arrangement.sh 1 6 0,1,2,3,4,5 7,7,7,7,7,5 dsv41-schaka`
with `IMG=localhost/vllm-backport-v41:sm80-v13 EP=1 SEQS=8 UTIL=0.96`.

## Measured results (this host)

**c1 (single stream).** Real OpenCode coding tasks in a repeatable Docker
harness (`opencode-bench/`): median **~30 tok/s end-to-end** (range 8-48),
dominated by reasoning-token volume and tool-call steps. Raw 512-token
generation (thinking off, unpredictable content): **65 tok/s**.

**Prefill (c1).** 32k TTFT 5.8 s (~5.7k tok/s); 128k 21 s (~6.2k); 512k 117 s
(~4.5k); **1M 342 s (~3.1k tok/s)**.

**Decode at depth (shared prefix, pure decode).** Aggregate output tok/s:

| ctx | c1 | c2 | c4 | c8 | c12/c16 |
|---|---|---|---|---|---|
| 128k | 36.5 | 56.9 | 79.7 | 106.4 | 113.8 |
| 512k | 25.1 | 38.1 | 44.1 | 52.3 | 54.6 |

Aggregate decode saturates with concurrency and scales down with depth: the
per-step cost grows ~linearly with batch at long context.

**KV memory for 1M tokens.** Whole-model pool is the min over PP ranks:
- util 0.95 -> 2,888,012 tokens (4.67 GiB/rank), max concurrency @1M 2.75x
- util 0.96 -> 3,404,072 tokens (5.31 GiB/rank), 3.25x  **(default)**
- util 0.985 -> 4,694,119 tokens (6.89 GiB/rank), 4.48x, but a 512k prefill
  **OOMs** (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` / `aten::new_empty`)

1M KV is ~30 GiB total across 6 ranks (~11,214 B/token) — ~12.6x the model
card's nominal 890 B/token, because the port uses fp8 (not FP4) main KV plus
uncompressed indexer/compressor caches.

**Long context.** Needle-at-depth recall: 9/9 at 32k/128k/512k (positions
0.15/0.5/0.85) and **3/3 at 1M** — exact answers.

**Live OpenCode.** TUI in tmux against the served model: PASS (read-tool call,
exact final marker, tool-calls/stop termination, no error events).

## LMCache

Validated end to end with **`LMCacheMPConnector`** (the HMA-capable external
connector) + the lmcache MP server (`lmcache server`, run **with GPU access** so
it can open the workers' CUDA IPC KV handles) + `--prefix-cache-retention-interval
256`. Accuracy: after a full vLLM restart, 9,728 / 9,737 prompt tokens restored
from LMCache with output identical to the cold run.

It costs ~11% c1 decode (58.0 vs 65.1 tok/s) because vLLM's in-GPU prefix cache
already covers single-task reuse, so LMCache is **not** the c1 default; it is
kept for cross-restart / evicted-prefix reuse. See `STATUS.md`.

## The 500 tok/s @512k target

Not met (measured ceiling ~55 tok/s aggregate at 512k). The gap is a
kernel/hardware problem, not a configuration one:

- SM80 sparse-MLA decode at depth scales poorly with batch (per-step cost grows
  ~linearly with concurrency).
- No 4-bit KV path exists for this model on SM80: `nvfp4_ds_mla` is Hopper+,
  `indexer_kv_dtype=mxfp4` is Blackwell-only (explicitly rejects pre-sm_10x),
  and TurboQuant is a separate non-MLA backend.
- The Triton SM80 attention kernel is already occupancy-aware
  (`num_compute_units` + split autotune); indexer decode query-sharding requires
  TP>1, which this host does not allow.

Full evidence, negative results, and method notes are in
[`STATUS.md`](STATUS.md).
