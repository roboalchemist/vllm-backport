# DeepSeek-V4.1-Flash on CMP 170HX (sm_80) — working serving recipe

This branch publishes a **working DeepSeek-V4.1-Flash serving configuration on
eight NVIDIA CMP 170HX** cards (GA100, `sm_80`, PCIe Gen2 x4). It is the first
coherent CMP-170HX V4.1 serving we know of; it is a vLLM layer over the
`wtdcode/vllm-backport` fork plus upstream vLLM PR #56201.

Everything here was measured on that hardware. Third-party numbers are marked
author-reported.

## Provenance

- Base image: `lazymio/vllm-backport:v0.12.0-sm80` (wtdcode's sm_80 backport).
- Model support: upstream **vLLM PR #56201** ("[Model] Support
  DeepSeek-V4.1-Flash", head `79a7108d9a`).
- Ampere shims + PP relay: adopted from
  [`Schaka/170hx-journey`](https://github.com/Schaka/170hx-journey)
  (`patches/vllm-backport-v41/`).
- Our additions: a KV-layout intersection fix for PP, a GQA guard, and the
  serve/benchmark harnesses.

`campaign/dsv41-flash-cmp170hx/overlay/` holds the exact changed
`vllm/*.py` files (94 files) relative to the base image's tree; `kit/` is the
build kit (fixups + Containerfile); `scripts/` are the serve + benchmark
harnesses; `reports/` holds the written reports and `STATUS.md`.

## Serving

```
Image:   localhost/vllm-backport-v41:sm80   (base + overlay; Python-only, no rebuild)
Model:   deepseek-ai/DeepSeek-V4.1-Flash    (475 GiB, 48 shards)
Config:  TP4xPP2 (aggregate)  or  TP8xPP1 (c1)
         TRITON_MLA_SPARSE_DSV41, fp8_ds_mla, block-128, DSpark k=5,
         max-model-len 1,048,576, CUDA graphs on
```

Note: the build daemon here runs `"bridge":"none"`, so containers must use
`--network host`; `-p` port maps do not work.

### Launch

```
# aggregate-optimal (max 512k throughput)
EP=1 SEQS=32 bash scripts/serve_arrangement.sh 4 2 0,1,2,3,4,5,7,8 20,20 dsv41-schaka

# c1-optimal (best single-stream)
EP=1 SEQS=32 bash scripts/serve_arrangement.sh 8 1 0,1,2,3,4,5,7,8 "" dsv41-schaka
```

## Validation (all local)

| Gate | Result |
|---|---|
| Coherence + planted fact | `Berlin/Paris/Canberra` + `7391-MARLIN-42` exact |
| Long-context recall | 12/12 at 4k–512k, 1/1 at the full 1M tokens |
| Accumulated multi-turn prefix | 8-turn chat, fact from turn 1 recalled on turn 9 |
| Coding (execute the tests) | 5/5 |
| OpenCode headless (tmux) | PASS |
| OpenCode live TUI | PASS (`TUI-7391-MARLIN-OK`, 16.8 s) |
| OpenCode via Bifrost | PASS (`BIFROST-OPENCODE-OK`) |

## Performance (server-reported, `ignore_eos`, prefix-cached)

Two layouts are used. **TP8×PP1** wins single-stream latency; **TP4×PP2** wins
prefill and higher concurrency. All numbers below are server-reported; decode
arms share a cached prefix and use `ignore_eos`, so tokens are exact.

### c1 / c2 / c4 decode (aggregate output tok/s)

The c-numbers are concurrent 256-token generations behind a shared,
prefix-cached context; `per-stream` is aggregate ÷ concurrency.

**TP8×PP1** (c1-optimal)

| context | c1 | c2 | c4 |
|---|---|---|---|
| 4k | 46.9 | 83.6 | 107.3 |
| 128k | 36.6 | 66.9 | 88.1 |

**TP4×PP2** (aggregate/prefill-optimal)

| context | c1 | c2 | c4 |
|---|---|---|---|
| 4k | 44.2 | 77.5 | **124.9** |
| 128k | 35.6 | 61.6 | **92.6** |

Single-stream pure decode (short prompt, no meaningful prefill):
**~101 tok/s** on TP8×PP1 (`100.8 / 97.6 / 89.9` at 8 / 4k / 32k tokens of
prompt) versus **~89 tok/s** on TP4×PP2 (`89.4 / 87.6 / 54.1`). TP8×PP1 is much
better at depth because it has zero pipeline hops.

### Prefill (TTFT and prefill tok/s)

Random prompts, 8 output tokens.

**TP8×PP1**

| input | c1 | c2 | c4 |
|---|---|---|---|
| 4k | 5.87 s · 697 t/s | 5.13 s · 798 t/s | 7.25 s · 565 t/s |
| 128k | 184.6 s · 710 t/s | 140.0 s · 937 t/s | 231.4 s · 566 t/s |

**TP4×PP2**

| input | c1 | c2 | c4 |
|---|---|---|---|
| 4k | 3.37 s · 1,215 t/s | 2.61 s · 1,570 t/s | 4.64 s · 883 t/s |
| 128k | 86.6 s · 1,513 t/s | 66.2 s · 1,981 t/s | 108.7 s · 1,205 t/s |

**TP4×PP2 prefills ~2× faster than TP8×PP1** — prefill is link-bound by the
all-reduce, and 8-way tensor parallelism moves twice the traffic over the same
Gen2 x4 fabric as 4-way. This is the real cost of the c1-optimal layout.

### Aggregate decode at higher concurrency

| context | c8 | c16 | c32 | c64 |
|---|---|---|---|---|
| 4k | 157 | 188 | **303** | 296 |
| 32k | 160 | — | **305** | 289 |
| 128k | 140 | 180 | 214 | 225 |
| 512k | — | 112 | — | **130** |

(These are TP4×PP2 arms; PP layouts lose aggregate at depth.)

### PP8 / pipeline-parallel layouts

"PP8" means **TP1×PP8** — one pipeline stage per card, so there is no
all-reduce at all (`VLLM_PP_LAYER_PARTITION=5,5,5,5,5,5,5,5`). It is the layout
Schaka and Zanooda both run. Two things must be true for it to work: the KV
layout lists of every rank must be intersected (they legitimately differ under
PP), and the pipeline hop must relay the compressed KV, the indexer-K cache and
the candidate blocks / top-k indices. This repo's `overlay/` carries the first
fix; the relay is upstream work.

Measured on this hardware:

| layout | 512k decode c8 | c16 | c64 | correct? |
|---|---|---|---|---|
| **Zanooda PP8** (TP1×PP8, his 13-patch stack) | 77.6 | 82.8 | 92.0 | ✅ needle 3/3 |
| Schaka PP6 (TP1×PP6) | — | 23.2 | 74.9 | ✅ needle 3/3 |
| Schaka PP8 (TP1×PP8, this repo's kit + the KV fix) | — | — | — | ❌ needle 0/3 |

Notable measured facts:

- **The no-all-reduce layout prefills ~2×.** Zanooda's PP8 and Schaka's PP6 both
  roughly double prefill versus TP4×PP2, matching the link analysis: at TP4 each
  layer moves ~16 MB per 2,048-token chunk over a ~1.6 GB/s Gen2 x4 link, which
  caps prefill near 1,700 tok/s. A pipeline stage per card removes that traffic.
- **But deep-context decode is lower than TP4×PP2** (92 vs 130 tok/s at c64):
  more pipeline hops and no tensor split of the KV.
- **Schaka's PP8 relay is not yet correct here.** With the KV-layout
  intersection fix it boots, but it fails the accuracy gate (needle 0/3, coding
  1/5 with garbled identifiers) — the relay does not reproduce the indexer-K
  write on every stage. Rejected. (His own PP8 profile is marked "no measured
  numbers yet".)
- **Author-reported PP8 numbers are higher than what we measure wall-clock.**
  Zanooda reports 117 tok/s single-stream, 532 tok/s aggregate @8 streams
  (105k), 6,066 tok/s prefill, 6.17M-token KV pool. His single-stream figure is
  tokens/(last−first token) and the aggregate comes from *staggered* arrivals
  (overlapping in-flight micro-batches); measured wall-clock on this box gives
  43–81 tok/s decode. Schaka's `dsv416pp` (TP1×PP6) reports 44.4 / 108.8 / 163.4
  tok/s at 1/4/8 streams and 5,078 tok/s prefill.

### PP6 vs PP8 — measured 2026-09-12 (PP6 adopted)

Both layouts were rebuilt and accuracy-gated. **PP6 (TP1×PP6, partition
`7,7,7,7,7,5`, util 0.95)** is the one we serve; it uses Schaka's v0.13.0-based
kit including the relay fix that lets the kv-group relay own the source
indexer-K cache (`language_model.model.layers.20.attn.indexer.k_cache`).

| metric | PP6 (TP1×PP6) | PP8 (TP1×PP8) |
|---|---|---|
| **pure single-stream decode** (8-tok prompt, best of 4–8) | **96.9–105 tok/s** | 43.2 |
| single-stream @512 / 4k / 32k | 87.1 / 103.8 / 49.4 | 40.9 / 32.2 / 27.6 |
| decode aggregate c8 (4k / 32k / 128k) | 146 / 159 / 164 | 155 / 137 / 128 |
| decode aggregate c16 (512k) | 88.3 | 86.8 |
| prefill c1 (4k / 128k / 512k) | 2,258 / 7,716 / 5,267 | 1,350 / 9,209 / 7,306 |
| prefill c16 (4k / 128k / 512k) | 9,501 / 23,780 / 14,630 | 7,067 / 30,058 / 24,151 |
| KV pool | 2,888,012 tok | 6,170,576 tok |
| **KV bytes/token (whole model)** | **11,214 B** | 23,617 B |
| **KV for 1M tokens** | **10.95 GiB** | 23.06 GiB |
| correct | needle 9/9 + 1M 1/1 + multi-needle @44k 5/5, coding 5/5 | needle 3/3 |

**Why PP6 wins c1:** it has 5 pipeline hops vs PP8's 7, so a single token crosses
fewer serial stages. PP8's per-token KV is also ~2× because its `pp_share` relay
replicates the compressed/indexer caches across ranks. PP6 c1 is within noise of
the TP8×PP1 ceiling (100.8) while remaining Tensor-Parallel-free.

**KV-cache benchmark (asked):** the engine reports a **2,888,012-token pool in
30.16 GiB** for PP6 → **11,214 bytes/token → 1M tokens = 10.95 GiB**. The model
card advertises ~890 B/token for its FP4 global KV; the sm_80 path is ~13× that
because it runs `fp8_ds_mla` (FP4 KV is rejected for MLA backends) **and** keeps
the uncompressed indexer-K caches for the eight indexer layers alongside the four
compressed KV groups. On PP8 the same measurement gives 23.06 GiB/1M (135.72 GiB
pool), the extra being the relay's cache replication.

**Full context×concurrency sweep (PP6, decode aggregate tok/s):**

| ctx | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| 4k | 49.2 | 91.1 | 108.9 | 146.4 | 141.5 |
| 32k | 45.1 | 80.5 | 104.2 | 159.0 | 159.6 |
| 128k | 42.6 | 65.9 | 100.4 | 164.5 | 150.0 |
| 512k | 30.0 | 46.1 | 54.0 | 84.9 | 88.3 |
| 1M | 15.9 | 22.4 | — | — | — |

At 1M only c1/c2 fit (one 1M request ≈ one third of the pool). Prefill at 1M is
TTFT 342.5 s (~3.1k tok/s).

## The 500 tok/s @ 512k target — not met, and why

Best measured is **130 tok/s at 512k c64** (~26% of target). `aggregate ≈
concurrency / per-stream TPOT`, and per-stream TPOT rises ~5× from 4k to 512k
(106 → 504 ms): the wall is the **sparse-attention KV gather at depth**, which
is memory-bound. Confirmed across three independently-built correct stacks
(this TP4xPP2 130, Zanooda PP8 92, Schaka PP6 75). Every config lever was
measured and is neutral or worse (max-num-seqs, max-num-batched-tokens,
EP on/off, NCCL algos, Marlin atomic-add, KV dtype; `moe_backend=triton` fails
on MXFP4/SM80; `nvfp4` KV is rejected for MLA). Reaching 500 needs kernel work
(a correct TP-split relay and/or a faster sparse-MLA decode / Marlin MoE GEMV).

## c1 levers tested (all closed)

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` is slower here; `nvidia-smi -lgc` is a no-op
(CMP clamps it); under load the cards draw ~100 W of a 250 W limit, so c1 is
not power-bound; TP8 is the minimum single-stage width (475 GiB model); DSpark
k=5 is pinned by the checkpoint (`dspark_block_size=5`).

## Layout

```
campaign/dsv41-flash-cmp170hx/
  README.md          this file
  STATUS.md          full campaign log (all measurements, negative results)
  overlay/vllm/...   the 94 changed vllm/*.py files
  kit/               build kit (fixups.py, Containerfile, ampere shims)
  scripts/           serve + benchmark harnesses (repeatable)
  reports/           written reports
```

## License

The base project and this overlay are Apache-2.0 (see LICENSE).
