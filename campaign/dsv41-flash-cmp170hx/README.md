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

Aggregate decode (tok/s):

| context | c8 | c16 | c32 | c64 |
|---|---|---|---|---|
| 4k | 157 | 188 | **303** | 296 |
| 32k | 160 | — | **305** | 289 |
| 128k | 140 | 180 | 214 | 225 |
| 512k | — | 112 | — | **130** |

Single-stream (c1): **~105 tok/s** shallow; **100.8 / 97.6 / 89.9** at
8 / 4k / 32k tokens of prompt on TP8×PP1. Prefill 1.6–2.9k tok/s (TP4xPP2);
2–3× higher on PP layouts (no all-reduce).

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
