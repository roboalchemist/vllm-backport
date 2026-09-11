# DeepSeek-V4.1-Flash on CMP 170HX — what changed since launch

**Report window:** 2026-09-10T02:17Z (HF launch) → 2026-09-11T00:15Z
**Prepared:** 2026-09-11 (CMP 170HX campaign, `junk-tokens`)
**Scope:** upstream/forks, HuggingFace, Reddit, and the `cmpunlocker` Discord corpus.

---

## Headline

**A working, coherent DeepSeek-V4.1-Flash on 8× CMP 170HX (SM80) now exists — published by Schaka since launch.** It is the first and only CMP-170HX/SM80 V4.1 result found anywhere. His repo ships the exact SM80 fix kit, and its root causes match the correctness bug we have been chasing all night. Discord corroborates the run ("working on 8 cards at 27 tps"; "DSpark works too").

Our own engine (vLLM `dsv41-feat` @ `e47aa780`) still boots and serves but emits incoherent text. Schaka's `170hx-journey` is now the reference implementation to port/serve.

---

## 1. The breakthrough: Schaka/170hx-journey (8× CMP 170HX, SM80)

- Repo: https://github.com/Schaka/170hx-journey
- 8-GPU commit: "Serve DeepSeek-V4.1-Flash on all 8 GPUs" — `0fc8ff90e` (2026-09-10T18:36:47Z)
- 6-GPU commit: "Serve DeepSeek-V4.1-Flash on 6 GPUs" — `53a61e64a` (2026-09-10T22:48:29Z)
- DSpark acceptance: `cd83672b8` (18:43:16Z)
- Pacth kit: `patches/vllm-backport-v41/` (`fixups.py`, `ampere/ampere_sparse.py`, `ampere/qnorm_rope_kv_insert.py`, `ampere/pp_kv_group_relay.py`, `Containerfile`, `build.sh`)

**Recipe:** `lazymio/vllm-backport:v0.12.0-sm80` base + upstream PR #56201 + `fixups.py` + `ampere/` shims. Python-only overlay.
**Layout:** TP4 × PP2 (`VLLM_PP_LAYER_PARTITION=20,20`), `fp8_ds_mla`, `TRITON_MLA_SPARSE_DSV41`, block-128, DSpark k=5, full 1,048,576 ctx.
**KV pool:** 14,058,003 tokens. **Prefill:** 1,584 / 1,650 / 1,389 tok/s at 39k/119k/319k. **Decode:** 43.8 / 72.9 / 104.2 tok/s at c1/c4/c8.

### The SM80 gaps it closes (directly relevant to our bug)
- **`wo_a` wrong shape:** "Marlin packs 4 fp8 values into one int32, so the `wo_a` weight arrives with the wrong shape. The kit routes that one layer to the emulation kernel." → our symptom is exactly an `_o_proj` output collapse.
- **Silent double Q-norm:** "The compiled `_C` operator `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` carries the V4.0 signature and takes 9 arguments. V4.1 passes a tenth, `apply_q_norm`. **Dropping it normalizes Q twice and gives wrong output with no error.**" → Triton replacement in `ampere/qnorm_rope_kv_insert.py`.
- **CuteDSL SM gate:** `dequantize_and_gather_k_cache` picks its path with `has_cutedsl()`; sm_80 must use `is_cutedsl_supported()` (checks capability).
- **fp8 software encode/decode** (`vllm/v1/attention/ops/fp8_sm80.py`) — we already carry this.
- **Compressor circular buffers:** the KV-group code asserts `[MLA, *SWA]` and drops 3 compressor buffers; the kit gives them their own group.
- **DSpark embed table under PP:** the drafter aliases the target table, which only exists on stage 0; the kit builds it on the last stage too.
- **PP is possible after all:** `pp_kv_group_relay.py` relays the layer-20 cache/index group, overriding vLLM's "PP splits inside a v4.1 kv-sharing group are not supported". This is the piece we concluded was impossible.

### Discord corroboration (dsum `cmpunlocker`)
- **#testing-general, 2026-09-10T16:31:28.803Z**, `schakaqt` (msg `1547645700236378212`): *"Anyone got numbers for 4.1 yet? I have it working on 8 cards at 27 tps so far, no dspark yet"* — https://discord.com/channels/1518634379604136178/1518695289278566450/1547645700236378212
- **#development-general, 2026-09-10T20:20:51.797Z**, `schakaqt` (msg `1547703426417168514`): *"4.1 flash? So far seems fine. But I'Ve only gotten it to work on 8 cards. DSpark works too."* — https://discord.com/channels/1518634379604136178/1518695328008638516/1547703426417168514
- Snapshot: dsum archive meta `cmpunlocker`, refreshed 2026-09-11T00:03:16Z (69,393 msgs, 53 channels).
- Caveat: author-reported; no raw token logs checked into the repo. No second user corroborates. No coherence transcript published.

---

## 2. Timeline (checked APIs, UTC)

| time | event | source |
|---|---|---|
| 02:17 | HF `initial commit` (launch) | HF |
| 07:17 | HF `df42c109` | HF |
| 08:18 | HF main → `dba1be0a` (still current) | HF |
| 12:16 | **vLLM #56228 merged to `main`** ("[Model] DeepSeek-V4.1-Flash Model Definitions") | GH |
| 12:59 | vLLM **#56120 "[DS-V4][SM80] portable Triton fallbacks…Ampere/Ada"** opened — **still open, merge-conflict, not merged** | GH |
| 13:28 | Discord `cmdrwaka324`: *"Anyone working on getting DSv4.1 running with ngram offload?"* | dsum |
| 13:32 | Discord `schakaqt`: V4.1 PP-split analysis of the layers-20–39 group; 286.1 GiB resident vs 361.3 GiB on 6 cards | dsum |
| 17:39 | **wtdcode/vllm-backport `a3507666`** — sm8x Ampere Triton sparse-MLA route; README now claims V4.1 "Fully Supported (v0.13.0+)" | GH |
| 18:36 | **Schaka 8-GPU V4.1 run** `0fc8ff90e` | GH |
| 18:43 | Schaka DSpark acceptance `cd83672b8` | GH |
| 20:20 | Discord `schakaqt`: *"So far seems fine… DSpark works too"* | dsum |
| 20:58–23:57 | vLLM perf/bug tail: #56342 (ROCm mHC), #56344 (Attention MegaKernel), #56347 (ROCm segfault 3rd decode token), #56349 (ROCm breakable graphs), #56357 (Engram mmap prefetch) | GH |
| 21:54–22:57 | SGLang merges #38946/#38947/#38949/#38951/#38952 (V4.1 aliases, DSpark PD, structural tag) | GH |
| 22:48 | Schaka 6-GPU run (TP2×PP3) `53a61e64a` | GH |
| 23:40:58 | **artlair/vllm-backport `dsv41-flash` `7ff0b2b7`** — 7 commits after `369f4ccc` (real_quality harness, engram-mmap, PP-KV relay, KV rebalance) | GH |
| 23:45–23:47 | llama.cpp **#28696** force-pushed to `a5cb238b2` "convert : add DeepSeek V4.1" — still open, not merged | GH |
| 23:57 | `vcruz305/DeepSeek-V4.1-Flash-GGUF` updated | HF |

---

## 3. GitHub forks / upstream heads

| repo | branch | SHA | time | relevance |
|---|---|---|---|---|
| Schaka/170hx-journey | main | `53a61e64a` | 22:48 | **8× CMP170HX SM80 — WORKING (author-reported)** |
| artlair/vllm-backport | dsv41-flash | `7ff0b2b7` | 23:40 | sm86 (3090) two-node TP4×PP5; coherent, quality harness `tools/dsv41/real_quality.py`; greedy not bit-stable |
| wtdcode/vllm-backport | master (v0.13.0) | `a3507666` | 17:39 | sm80/sm86 image; "first request working"; **no published coherence/quality evidence** |
| vllm-project/vllm | v41-megakernel | `8c2d92c935f2` | 20:09 | SM100 only; contains the **weight-permutation root cause** of garbage output under the `DeepseekV41ForConditionalGeneration` wrapper |
| vllm-project/vllm | dsv41-feat | `e47aa780` | 07:23 | our engine base (PR #56214, still open/unmerged) |
| choiceoh/stkernel | main | `88db5be3c6ac` | 23:43 | DGX Spark sm_121a; "prepared, not yet bootable" |
| PixelML/sm80vllm | — | `f8ea5bb1` | 08-01 | no V4.1 work |

**Upstream vLLM root causes documented in-window (SM100 reference, portable as hypotheses):** bad `Q_b`/`Wv` load-time permutation when the attention prefix is `language_model.model.layers.N.attn` → garbage; and stale index sentinels in a reused `out=` workspace → NaN-argmax / 4096-token garbage (fixed by writing `-1` past `combined_lens`).

---

## 4. HuggingFace

30 model repos match `DeepSeek-V4.1-Flash` updated on/after 2026-09-10; **none advertise an SM80/A100/Ampere/CMP path** (grep for `sm80|sm_80|a100|a800|ampere|170hx|cmp170` = 0 hits). Notable:
- `vcruz305/DeepSeek-V4.1-Flash-GGUF` (23:57, 13 likes)
- `rapid-mlx/…REAP-2bit-MLX` (23:34), `pipenetwork/…MLX-mixed-4_8bit` (13:09/15:21/19:12/19:47)
- `AtomicChat/…NVFP4-nvidia` (22:46) + metrics dataset (KL-divergence; original checkpoint's own two runs differ 0.016 KL / 4% top-1)
- `msuiche/…NVFP4`: "published ahead of runtime support … not been run end-to-end"
- `caiovicentino1/…HLWQ-Engram-Q4/Q5`: needle 5/5 at 3.4k–52k; 79% token identity to FP8 vs 88% FP8-self
- Official repo discussions: **zero** hits for `sm80/a100/170hx/ampere`; #12 is an H100 indexer-cache correctness patch.

**No EXL3 HF artifact exists** (recipe-only: `sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`, sm121).

---

## 5. Reddit / web

- **r/LocalLLaMA 2026-09-10T20:00:56Z** — "CPU Only Experimental Sloppy Deepseek V4.1 Flash" — "~30 TPS PP and ~6 TPS TG on a xeon with the n-gram table offloaded … this works." Author-reported, self-described "vibecoded by Opus 5.0, unreviewed". https://reddit.com/r/LocalLLaMA/comments/1wcu3fw/
- **r/LocalLLM 19:05Z** — "…510 GB but only about 150 has to be in memory…" (Engram on SSD, PCIe-bound). https://reddit.com/r/LocalLLM/comments/1wcsj7v/
- **r/DeepSeek 12:29Z** — "DeepSeek Flash 4.1, the worst version … a total step backward" (quality complaint). https://reddit.com/r/DeepSeek/comments/1wcht7m/
- **r/LocalLLaMA 08:27Z** — "V4.1 Flash is 748B, not 552B" (backbone 551.6B + Engram 196.9B + DSpark 14.2B). https://reddit.com/r/LocalLLaMA/comments/1wcd4rx/
- No Reddit post claims a working SM80/A100/CMP-170HX V4.1 run. `170HX`/`CMP` searches surface only V4-**Flash**-0731 170HX material.

---

## 6. Our campaign status

- Engine built (`local/vllm-dsv41:sm80`, `dsv41-feat` @ `e47aa780`); **TP8 boot achieved** — the first known SM80 V4.1 boot (init engine 137–143 s), healthy on :8090.
- Correctness still failing: attention `o` absmax ≈ 2.0 but `_o_proj` output ≈ 0.013 (~150× shrink); residual explodes (pre-norm 3.5e29) with bursts at the KV-source layers (8/14/15/20).
- Our `modelopt.py` already carries the MXFP8 block-scale expansion ([N/32,K/32] → [N,K/32]) and an `is_bmm` → `init_mxfp8_linear_kernel` route; verified checkpoint `wo_a` = [8192,4096] F8_E4M3 with scale [256,128] F8_E8M0.
- **Next:** Adopt Schaka's `170hx-journey` recipe (base `lazymio/vllm-backport:v0.12.0-sm80` + PR #56201 + fixups + `ampere/` + PP relay), which is the proven-working path and also unlocks TP4×PP2.

### Evidence quality
- **Locally measured:** our boot, our probe numbers.
- **Author-reported (unverified):** all Schaka/artlair/wtdcode throughput and quality figures.
- **Checked APIs:** HF/GitHub/Docker Hub/Discord timestamps above.
