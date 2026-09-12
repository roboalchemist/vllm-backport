# DeepSeek-V4.1-Flash on CMP 170HX — campaign status

**Started:** 2026-09-10 UTC. **Goal:** serve
`deepseek-ai/DeepSeek-V4.1-Flash` on this 9× CMP 170HX node as efficiently as
possible, validate in OpenCode (tmux), and register on the bifrost gateway
(vdevserve) + gateway@gateway.

All evidence classes follow `AGENTS.md`.

## 1. Model facts (HF rev `df42c109f1defefcbfcedbe7d905718a12266e40`)

- `DeepseekV41ForCausalLM`, `model_type=deepseek_v41`. 40 backbone layers
  (20 causal encoder + 20 decoder), **CSA2** (Compressed Sparse Attention 2)
  with static Full/Reindex/Reuse modes + a 128-token SWA in every layer;
  hierarchical sparse indexer (`index_topk=512`, 16,384 candidate positions).
- MoE: 384 routed experts top-6 + 1 shared; 552B backbone + 196B Engram.
  8B active/token prefill, 16B decode. Vocab 129,280. Context 1M (YaRN×16).
- **Engram** (conditional n-gram memory): `engram_layer_ids=[1,14]`, 196B params,
  two embedding tables.
- Quant: dense FP8 E4M3 block-32×32 ue8m0; **routed experts MXFP4** (E2M1,
  block-32); BF16 embeddings/norms; FP32 sinks/mHC.
- **Checkpoint: 475.25 GiB, 48 safetensors shards.** The two Engram tensors
  `layers.{1,14}.engram.embed.weight` are **91.55 GiB each** (single tensors in
  shards 47/48) — larger than one 64 GiB card.
- DSpark spec decode: `num_nextn_predict_layers=3`, `dspark_block_size=5`,
  target layers [37,38,39]. Reference impl does not run it.

### Authors' published inference material
- The HF repo ships only a **TileLang/Torch reference** (`inference/`), invoked
  with `torchrun --nproc-per-node $MP generate.py`, MP=8, plain AR (no spec
  loop, no HTTP server, no batching). `convert.py --model-parallel` shards by TP.
- **No vLLM/SGLang recipe was published by DeepSeek.** Production kernels named
  in the tech report (FlashMLA, DeepGEMM, TileKernels, DeepSelect) are
  Hopper/Blackwell-class; no SM80 statement.
- Reference `inference/kernel.py:sparse_attn` is TileLang bf16 (sm80-capable)
  and is the **semantic oracle** for CSA2, not runtime code.

## 2. Engine

- **vLLM branch `dsv41-feat` @ `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`**
  (PR #56214, open; model-defs split #56228; frontends #56208 merged).
  Local clone: `/mnt/kv/build/vllm-dsv41`.
- Official image `vllm/vllm-openai:deepseekv41-flash-0909` (CUDA 13.0.1) exists
  but is sm90/sm100/sm120 — **not usable as-is on sm80**.
- SGLang branch `dsv4.1` (`1aa0e962…`) + `lmsysorg/sglang:dev-dsv41` preview —
  Hopper/Blackwell/AMD only.
- Community: `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` (sm121, TP4 +
  "Engram-on-disk" patch). Author-reported, no serving numbers yet.

## 3. Fit analysis (9× 64 GiB, Gen2, no P2P)

- Weights 475.25 GiB → 8 cards = 59.4 GiB/card (no headroom); 9 cards = 52.8.
- **9 cannot form a single TP/PP group** (384 experts / 128 MTP ⇒ MP ∈ {1,2,4,8};
  `o_groups=8` caps TP at 8). Single replica = 8 cards.
- **Engram is the binding constraint**: each table > 64 GiB and is one
  unsplittable tensor per shard. vLLM solves it natively:
  `EngramConfig.cpu_offload=True` (default) stores each rank's assigned hash
  heads in pinned host RAM (UVA lookup). Host has 472 GiB RAM (410 available).
- With Engram off HBM: PP8 ≈ 35.8 GiB weights/rank avg (last stage +MTP ≈ 42) →
  ~22 GiB headroom. KV is tiny (890 B/token ⇒ ≤0.89 GiB @1M).
- `inference/convert.py` is NOT usable here (peak ~475 GiB host RAM ⇒ OOM) and
  targets the reference stack, not vLLM.

## 4. Decisive blocker + port plan

**No SM80 attention path.** `_select_dsv4_attn_cls`
(`vllm/models/deepseek_v4_1/nvidia/model.py:114-149`) only yields FlashMLA
(SM90) or FlashInfer (SM100/SM120); `sparse_mla.py:123` gates major [9,10],
`flashinfer_sparse.py:132` gates [10,12]. CMP 170HX is **major 8**.

Key insight: the v4.1 branch already ships a **platform-neutral ROCm/Triton
sparse-MLA implementation** (`deepseek_v4_1/amd/rocm.py`, 905 lines) that was the
CUDA-SM8x path in v4.0. The port is mostly wiring + pre-SM89 fp8 shims, reusing
the v4.0 SM80 tree at `/mnt/sdb/src/vllm-dsv4` and the repo's `ds4/vllm-overlay/`.

Staged port:
1. **Stage 0 scaffolding** — copy `fp8_sm80.py`; add
   `models/deepseek_v4_1/ampere/ampere_sparse.py` (subclass
   `DeepseekV41ROCMAiterMLAAttention` + backend with
   `supports_compute_capability -> major==8`);
   register `TRITON_MLA_SPARSE_DSV41`; add the `major==8` branch to
   `_select_dsv4_attn_cls`.
2. **Stage 1 eager AR** — gate fp8 encode/decode shims so Triton kernels JIT on
   sm80 (`rocm_aiter_mla_sparse.py` decode, `cache_utils.py` gather,
   `fused_compress_quant_cache.py`/`indexer_k_store.py`/`fused_indexer_q.py`
   encode); run SWA-only, short ctx, eager, seqs=1; token-exact vs reference.
3. **Stage 2 compressed path** — port `mqa_logits_triton.py` (578 lines) and the
   torch prefill-topk fallback into `sparse_attn_indexer.py`; validate c1a/c2a.
4. **Stage 3 capacity** — long context, concurrency, graphs, then DSpark last.
5. **MoE/quant** — confirm Marlin fallbacks for dense FP8
   (`MarlinFP8ScaledMMLinearKernel`) and MXFP4 experts (Marlin oracle backend).

## 5. Acquisition + build

- Model → `/mnt/kv/models/DeepSeek-V4.1-Flash` (475 GiB; `/mnt/kv` 1.3T free;
  `/mnt/sdb` has 24 GiB — unusable). Download running (Xet high-performance).
- Build: source tree `/mnt/kv/build/vllm-dsv41`; sm80 build via the existing
  SSD-backed build daemon (`glm53-build-docker`), `TORCH_CUDA_ARCH_LIST=8.0`.

## 6. Open risks

- sm80 sparse-attention numerics vs the reference oracle (must gate exactly).
- Marlin MXFP4 on these shapes (384 experts, inter 2304) — benchmark vs Humming.
- `is_uva_available()` for Engram cpu_offload on CMP 170HX.
- PP8 + CSA2 cross-stage KV/index sharing is draft-only upstream
  (#56221–#56223).
- No P2P: TP collectives are expensive; PP is the intended topology.

---

# 2026-09-10 runtime update — PP is not viable; TP8 is the topology

## Build
- `local/vllm-dsv41:sm80` built from `dsv41-feat@e47aa780bccf` with
  `torch_cuda_arch_list=8.0` (receipt `/mnt/kv/logs/dsv41/20260910T100024Z-sm80-build`).
  `import vllm` → `0.28.1rc1.dev392+ge47aa780b.d20260910`.
- The build daemon needed an `nvidia` runtime entry added
  (`glm5.3-flash/deploy/build-dockerd-ssd.json`).

## Model
- 475.24 GiB / 48 shards downloaded; index `total_size=510,286,023,000` and both
  94 GiB Engram shards SHA-256 verified (see `MODEL-RECEIPT.md`).

## Decisive finding: PP splits are rejected inside a v4.1 KV-sharing group
`attention.py:284` sets `kv_source_layer_id = max(s in kv_source_layer_ids, s <= layer)`.
With `kv_source_layer_ids=[2,8,14,20]` and 40 backbone layers, **layers 20–39 form a
single KV-sharing group** (all reuse layer 20's compressed cache). vLLM raises
`NotImplementedError("Compressed-KV source … not found on this rank; PP splits inside a
v4.1 kv-sharing group are not supported")` unless the source module is co-located with
every consumer. A naive PP8 `[5,5,5,5,5,5,5,5]` splits `[8-13]`, `[14-19]`, and `[20-39]`.

Keeping `[20-39]` intact means ~20 layers (~138 GiB non-Engram) on one rank — impossible
on 64 GiB. The cross-stage KV/index transports that would relax this are still draft
upstream (#56221–#56223).

**Therefore the topology is TP8**, matching the upstream/community recipes
(`--tp 8 --ep-size 8`, DGX-Spark `tp4`). All 40 layers live on every rank; the weights
shard across 8 ranks (~34.5 GiB/rank), and the two Engram tables are **hash-head sharded**
across the TP group via `--engram-config cpu_offload` (11.8 GiB/rank/table → ~23.6 GiB/rank
in pinned host RAM; measured: "Engram table offloaded to pinned host memory: 11.80 GiB per
rank").

## Runtime bring-up (in progress)
`serve_pp8.sh` now takes `DSV41_TP`/`DSV41_PP` (default TP8/PP1). First TP8 boot:
NCCL backend, MarlinMxfp8LinearKernel + Marlin MXFP4 MoE, `fp8_ds_mla` KV format, no
errors; ~49 GiB/GPU resident after weights; weight loading (incl. Engram) is slow but
progressing. Raw logs: `/mnt/kv/logs/dsv41/`.

---

# 2026-09-10 — FIRST SM80 BOOT (milestone)

`deepseek-v4.1-flash` reached **healthy** on 8x CMP 170HX (TP8, eager, BF16... fp8_ds_mla KV):
`init engine took 143.51 s`, `Application startup complete`, `/v1/models` live on :8090.
As of the 2026-09-10 ecosystem survey, this is the **first known DeepSeek-V4.1-Flash boot on
SM80 / CMP 170HX** (no upstream/community sm80 V4.1 run existed).

Blockers cleared to get here (24-file SM80 runtime overlay; patches 0001-0004):
1. No SM80 attention class -> added `TRITON_MLA_SPARSE_DSV41` Ampere class (reuses the
   ROCm Triton sparse-MLA path).
2. Marlin MXFP8 repacked `wo_a` -> set `wo_a.is_bmm = True` (Marlin skips) + raw-weight stash.
3. CUTLASS-DSL (`fused_indexer_q`) NVVM fails on sm80 -> gated `has_cutedsl()` off for major<9.
4. Native Triton fp8 (`fp8e4nv`) in indexer/fused_indexer_q/cache/engram -> fp8_sm80 shims
   (uint8 encode/decode) across all v4/v4_1 sites (parity with artlair/vllm-backport@dsv41-flash).
5. `_flashmla_C` required by the SWA tile-scheduler -> gate on `_is_flashmla_available()`.

**Open blocker: correctness.** Greedy output is incoherent (random multilingual tokens);
the model answers but the forward pass is numerically wrong. This is the current gate —
performance/MTP/gateway are downstream of it.

## 2026-09-10 correction debug log

Boot works; output incoherent. Isolations performed (evidence in /mnt/kv/logs/dsv41):
- **Tokenization correct** — "The capital of France is" -> 5 tokens [671,6102,294,8760,344].
- **Raw completion is also garbage** (not a chat-template issue); the exact
  `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>` frame is also garbage.
- **Engram ruled out** — `DSV41_ZERO_ENGRAM=1` still garbage.
- **wo_a fixed** — `wo_a.weight.shape=(1024,4096)` raw (correct) with `is_bmm=True`.
- **Backend confirmed** — `TRITON_MLA_SPARSE_DSV41`, Marlin MXFP8 dense, Marlin MXFP4 MoE,
  `fp8_ds_mla` KV all active; no missing/unexpected weight keys surfaced.
- **Nondeterministic** across identical greedy requests; teacher-forced logprobs are finite
  but wrong ("Germany is" -> ' is' below "'s").
- Marlin warns of thread-tile padding for some shapes (performance note; correctness TBD).

Remaining suspects: the v4.1-specific CSA2 attention/compressor, the hierarchical indexer,
and the mHC hyper-connection kernels. Reference oracle is infeasible on sm80 (HF reference
uses TileLang FP8/FP4 GEMMs that do not compile below SM89), and no community sm80/sm86
V4.1 correctness result exists (artlair's harness states dummy weights "produce garbage text").
Next: per-kernel differential tests vs torch for CSA2/indexer/mHC.

## 2026-09-10 — v4.0 differential abandoned; v4.1 core is the suspect

Attempted to load DeepSeek-V4-Flash-0731 (`/mnt/kv/models/deepseek-v4-flash-0731`,
154 GiB, `DeepseekV4ForCausalLM`) on the same engine as a shared-code control.
It loads (19.78 GiB/rank) but dies at engine init: v4.0's dense FP8 block-128 layout
routes to the **DeepGEMM** backend (SM90), which is absent on sm80. The v4.1 MXFP8
(block-32) layout correctly routes to Marlin, which is why v4.1 boots. Making v4.0 a
control therefore needs its own DeepGEMM->Marlin port; not worth a cycle here.

Consequences: the shared `deepseek_v4` Triton path cannot be quickly cleared as a
suspect. Remaining v4.1-forward correctness suspects, in priority order:
1. CSA2 attention: compressor cache write/read + the sparse MLA Triton kernels.
2. Hierarchical indexer: logits + top-k selection + candidate-block masking.
3. mHC hyper-connection kernels (TileLang/Triton) — a v4.1-only feature.
4. Marlin MXFP8 dense "thread-tile padding" (warning present; equality unproven).

Plan: per-kernel differential tests (torch reference) for each, starting with the
CSA2 compressor cache parity (encode-on-write vs decode-on-read) and the mHC path,
since those are v4.1-only and least battle-tested.

## 2026-09-10 — hidden-state probe: residual stream explodes

Added a per-layer hidden-state probe (`DSV41_HIDDEN_DEBUG=1`, `model.py` forward loop;
`[HIDDEN] layer N: absmax=.. norm=.. nan=..`). One greedy request, TP0:

```
embed     absmax=0.617
layer 0   5.03      layer 1   21.4      layer 2   19.6     layer 3   17.1
layer 6   1.35      layer 8   124       layer 10  492      layer 13  16.5
layer 14  164       layer 15  5952      layer 20  19200    layer 21  13632
layer 28  22912     layer 32  90        layer 36  9280     layer 39  1680
pre-norm  333824    post-norm 2.047
```

Interpretation: the residual stream is numerically unstable — absmax grows ~5e5x over 40 layers and
bursts at specific layers (8, 10, 14/15, 20+). Values stay finite (no NaN/Inf), so it is a
systematic amplification, not an overflow. `hc_collapse_triton` + the final `mhc_post` take
1680 -> 333824 at the tail. mHC (`hc_mult=4`, `hc_sinkhorn_iters=20`) is shared with v4.0
(also `hc_mult=4`), so the amplifier is most likely in the CSA2 attention / compressor / indexer
outputs or a dequant magnitude error feeding them, not the mHC math itself.

Next: component differentials. Cheapest decisive probes, in order:
1. Attention-only vs FFN-only magnitude at one layer (zero the other branch) to see which amplifier
   it is.
2. Marlin MXFP8 dense dequant magnitude vs a torch dequant of one real weight.
3. CSA2 compressor cache encode/decode parity (write in `fused_compress_quant_cache`, read in the
   Triton sparse-MLA decode) — a scale mismatch there would inflate attention scores.

---

# FINAL STATUS (2026-09-11) — SERVED, VALIDATED

**`deepseek-v4.1-flash` is live and coherent on this CMP 170HX node.**
This is the first known coherent DeepSeek-V4.1-Flash serving on CMP 170HX
(SM80). The server-side `dsv41-feat` engine never produced coherent output;
the working path is the Schaka/170hx-journey recipe (see `SCHAKA-RECIPE.md`).

## Service

- Container `dsv41-schaka`, image `localhost/vllm-backport-v41:sm80`,
  TP4×PP2, `TRITON_MLA_SPARSE_DSV41`, `fp8_ds_mla`, block-128, DSpark k=5,
  max-model-len 1,048,576, CUDA graphs on (PIECEWISE + FULL).
- Reachable `127.0.0.1:8090` and Tailscale `100.64.0.119:8090`.
- Bifrost provider `dsv41-flash-cmp` / `deepseek-v4.1-flash` live; route tested.
- **Networking:** daemon is `"bridge":"none"` → use `--network host`, not `-p`.

## Validation

| Check | Result |
|---|---|
| Coherence | capitals correct; planted `7391-MARLIN-42` exact |
| OpenCode headless (tmux) | PASS |
| OpenCode live TUI | PASS — `TUI-7391-MARLIN-OK` (16.8 s) |
| Long-context recall | 12/12 facts, 4k–512k × 15/50/85% depth |
| Coding (run tests) | 4/5 |
| Bifrost | PASS — `BIFROST-OK-7391` |

## Performance (Benchmarks #1–#5, shared to Mattermost)

- c1 real-text 43.1 tok/s; c8 153.4 tok/s; 128k c32 214.8 tok/s.
- **512k aggregate ~130 tok/s (c64)** — 26% of the 500 tok/s target.
- Prefill 1.6–2.9k tok/s; cold 512k ≈ 5.4 min. DSpark acceptance ~41% real-text.

## 500 tok/s @ 512k gap (honest)

Per-token cost is flat with depth; aggregate saturates ~110–130 tok/s at 512k.
`--max-num-seqs` 8→32 helped at 128k but 32→128 did nothing at 512k. EP-off
stalls after weight load on this engine. Next: vLLM #56367 (DS V4.1 Triton
K-cache gather vectorization), PP-heavy arrangement at depth, scheduler study.

## Open items

- vdevserve Bifrost host not resolvable from this box (gateway Bifrost configured).
- Gateway landing `app.py` has another agent's uncommitted edits; add the
  DSV4.1 service entry after that lands.

## Tuning arms at 512k (all measured here, prefix-cached c-arms, ignore_eos)

| Config | c16 | c32 | c64 | c128 |
|---|---|---|---|---|
| seqs 32 (best) | 103.3 | 111.7 | **129.6** | — |
| seqs 128 | — | — | 108.6 | 114.6 |
| seqs 32, max-num-batched-tokens 8192 | 89.6 | — | 113.5 | — |
| EP off | *stalls after weight load* | | | |

Conclusion: 512k aggregate decode is saturated at ~110–130 tok/s on this
box/engine. `--max-num-seqs` and `--max-num-batched-tokens` do not move it;
EP-off is not viable. Per-stream TPOT is flat with depth, so the limit is
sparse-attention KV-gather bandwidth under concurrency, not per-token compute.
Reaching 500 tok/s @512k needs a kernel-level change (candidate: vLLM #56367,
vectorize the DS V4.1 Triton K-cache gather) or more/faster interconnect.

## Full-context validation added

- **1,048,576-token prompt** with a needle at 50% depth → recalled exactly
  (`NEEDLE-885440-X`, wall 1113 s, prefill-dominated). Long-context recall is
  now 12/12 at 4k–512k plus 1/1 at the full 1M context.

- **Accumulated multi-turn prefix**: 8-turn accumulating chat (~13k tokens),
  fact planted turn 1 recalled exactly on turn 9; recall turn 1.4 s vs ~6 s
  filler turns (prefix caching serving the shared prefix).

## Arrangement at 512k

| Arrangement | 512k c16 | 512k c64 |
|---|---|---|
| TP4×PP2 (best) | 103.3 | **129.6** |
| TP2×PP3 (6 GPU) | 84.5 | 107.6 |
| TP8×PP1 | (not re-run at 512k) | |

TP4×PP2 is the best layout for deep-context aggregate decode on this box.
Combined with the tuning table above, **~130 tok/s at 512k is the measured
ceiling** for this engine on this hardware; 500 tok/s requires a kernel-level
change (KV-gather vectorization, e.g. vLLM #56367) that is not portable to the
current wtdcode backport without a rewrite.

## Arrangement exploration (2026-09-11 late)

Motivated by Schaka's link analysis — at TP4 every all-reduce crosses a Gen2
x4 link (~1.6 GB/s), which he computes caps prefill near 1,700 tok/s (we
measure 1.6–2.9k), so removing the all-reduce should raise prefill:

| Layout | Partition | Outcome |
|---|---|---|
| TP4×PP2 | 20,20 | **works, best (512k c64 129.6)** |
| TP2×PP3 (6 GPU) | 14,14,12 | works, slightly worse at 512k (107.6) |
| TP8×PP1 | — | works |
| **TP1×PP8** | 5,5,5,5,5,5,5,5 | **crashes**: "Workers disagree on supported KV cache layouts" (stage 3 reports 6 layouts vs 2) — Schaka's PP8 profile is marked untested |
| **TP2×PP4** | 10,10,10,10 | **crashes**: relay/loader `'NoneType' object has no attribute 'prefix'` |

So the no-all-reduce layout still needs engine work (the relay does not yet
support arbitrary kv-group splits). TP4×PP2 remains the deployment config.

## Gateway path

- Bifrost provider `dsv41-flash-cmp` / `deepseek-v4.1-flash` on the gateway host
  is live and validated two ways: raw curl (`BIFROST-ROUTE-OK`) and **OpenCode
  driven through Bifrost** (`BIFROST-OPENCODE-OK`). The provider accepts the
  virtual key via `Authorization: Bearer` as well as the `x-bf-vk` header; the
  tmux proof harness must propagate `CHUNGLET_OPENCODE_VK` to its worker.
- The vdevserve Bifrost host (`vdevserve` / `vdev-gateway.namespace.so`) does
  not resolve from this box, so only the gateway-host Bifrost is registered.

## Official inference docs (model repo, read 2026-09-11)

- `inference/README.md`: reference implementation only ("readable, not a
  production serving engine"). `convert.py --model-parallel 8 --expert-dtype
  fp4`; `generate.py --interactive`; `model.py` self-test checks "shapes and
  kernel plumbing, not numerics". No SM80 path (tilelang FP8/FP4 kernels).
- Top README: **CSA2** (Full/Reindex/Reuse static modes + hierarchical sparse
  indexer), **FP4 main KV cache** (E2M1, one E4M3 scale per 16 channels) →
  **890 bytes/token** (~1/4 of V4-Flash), Engram 196B, DSpark, Mega-mHC.
  Pretrained 45T tokens; sparse attention at 64K, extended to 1M.
- The model is officially evaluated with the **OpenCode** harness (1M ctx,
  temperature 1.0, top_p 0.95) — matching our validation path.
- **Lead:** the checkpoint's intended main-KV is FP4, but the sm_80 sparse-MLA
  path we serve uses `fp8_ds_mla` (Schaka: block-128 is the only size that
  kernel takes). If a FP4 KV path exists for SM80 it would cut KV read traffic
  ~2x and directly attack the 512k decode ceiling. Unverified whether the
  backport exposes it.

## FP4-KV lead closed

`--kv-cache-dtype nvfp4` (the model's designed main-KV format) is rejected by
the engine: "nvfp4 KV cache is not supported with MLA backends". The valid
packed option for this MLA path is `fp8_ds_mla` only, so we cannot halve KV
read traffic that way. This closes that route to the 512k decode ceiling.

## Tuning levers exhausted at 512k

Every engine-level knob recommended by the engine or by Schaka was measured.
None moves the 512k aggregate above ~130 tok/s:

| Lever | Result |
|---|---|
| `--max-num-seqs` 8 / 32 / 128 | 32 best (helps 128k only) |
| `--max-num-batched-tokens` 8192 | neutral/worse |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | coherence OK, throughput neutral (89.7/113.4) |
| `--kv-cache-dtype nvfp4` | rejected (not supported with MLA) |
| `--enable-expert-parallel` off | load stall |
| TP1×PP8, TP2×PP4 | engine crashes |
| CUDA graphs | already on |

**Conclusion:** on this box the backport saturates at ~130 tok/s aggregate
decode at 512k. The remaining path to 500 is a kernel rewrite (sparse-attention
KV gather / fused MoE) — the upstream MegaKernel and shi3z/ferrite work are not
portable to this sm80 backport cheaply. Service stays on the measured-best
config (TP4×PP2, seqs 32, DSpark k5, fp8_ds_mla).

## PP8: fast but INCORRECT — rejected (accuracy gate)

Schaka's TP1×PP8 profile crashes upstream on a KV-layout assert. Two patches
made it boot:
1. `resolve_kv_cache_layout`: intersect per-rank layout lists instead of
   asserting equality (correct under PP, where ranks own different layers).
2. `pp_kv_group_relay.write`: skip the indexer-K write when the index cache
   is unbound (warmup), mirroring the existing profile-run guard.

Result: **prefill is ~3.7x faster** (32k TTFT 5.7 s → 5,738 tok/s; 128k TTFT
15.9 s → 8,264 tok/s vs 1.6–2.9k at TP4×PP2) — confirming Schaka's link
analysis that the TP4 all-reduce over Gen2 x4 was the prefill bottleneck.

But it is **numerically wrong** and was rejected by the accuracy gate:
- needle recall 0/3 at 32k (TP4×PP2: 12/12 through 512k),
- coding 1/5 with garbled identifiers (`sorted(intervals, flesh=...)`,
  `for char in senn`).

Root cause: the relay does not correctly reproduce the indexer-K cache write on
every stage, so the shared sparse index is stale → wrong attention selection.
PP8 therefore needs a real relay fix (not the warmup guard) before it can be
used. Service reverted to the correct TP4×PP2 config.

## Restored-config re-validation (accuracy gate)

After reverting from PP8, the TP4×PP2 config was re-run through the full gate:

| Gate | TP4×PP2 (served) | PP8 (rejected) |
|---|---|---|
| needle recall @32k | **3/3** | 0/3 |
| coding (execute tests) | **5/5** | 1/5 (garbled identifiers) |

So the accuracy gate correctly rejects the fast-but-corrupt PP8 and confirms the
served config. Coding is 5/5 (up from 4/5 earlier — the tree level-order task
now emits a code block).

## PP relay: Schaka's fix adopted; full arrangement matrix (2026-09-11 PM)

Schaka released the relay correctness fix (`topk_indices` on the PP hop + an
`index_k_cache is None` guard) and a TP1×PP6 profile. Rebuilt the overlay from
his updated kit (+ the KV-layout intersection patch) → 94 files. Results on
our box:

| layout | prefill 32k | prefill 128k | decode c1 | decode c8 | 512k c64 | needle | correct? |
|---|---|---|---|---|---|---|---|
| **TP4×PP2 (served)** | 12.4 s (2.6k t/s) | 53 s (2.5k) | **43.1** | **153.4** | **129.6** | **3/3–12/12** | ✅ |
| TP1×PP6 (util .95) | **5.8 s (5.6k)** | **18.1 s (7.2k)** | 22.5 | 116.8 | 74.9 | **3/3** | ✅ |
| TP2×PP4 | — | — | — | — | — | **0/3** | ❌ |
| TP1×PP8 | fast | fast | — | — | — | 0/3 | ❌ |

**Findings.**
- Confirmed Schaka's link analysis: removing the TP all-reduce makes prefill
  ~2.5–3× faster. The PP hop is cheaper than the all-reduce.
- But decode is *slower* on PP6 at every concurrency (more pipeline-hop
  latency, weaker KV split): c1 22.5 vs 43.1, c8 116.8 vs 153.4, 512k c64
  74.9 vs 129.6.
- The relay fix is only correct for TP1×PP6; TP2×PP4 and TP1×PP8 stay corrupt
  (needle 0/3) — the relay does not yet handle a TP split of the relayed caches.

**Decision:** keep TP4×PP2 as the served config (best for the 512k decode
target and validated 12/12 long-context). TP1×PP6 is the prefill-optimized
alternative (2.5–3× prefill, correct) selectable via
`serve_arrangement.sh 1 6 0,1,2,3,4,5 7,7,7,7,7,5` with the pp6 image.

## NCCL algorithms

`NCCL_ALGO=Ring NCCL_PROTO=Simple` is the only working pair on this box. Both
`Tree+LL128` and `Tree+Simple` fail engine init with "NCCL error: invalid usage"
(the PEX/PCIe topology does not support the tree algorithm here). Decode stays
on Ring; no gain available from NCCL tuning.

## Gateways — done

- **Bifrost (gateway host):** provider `dsv41-flash-cmp` / `deepseek-v4.1-flash`
  registered; validated by raw curl (`BIFROST-ROUTE-OK`) and by OpenCode driven
  through Bifrost (`BIFROST-OPENCODE-OK`).
- **gateway@gateway (landing page):** `DeepSeek-V4.1-Flash (CMP 170HX)` added to
  the AI services list in `~/gitea/gateway/app.py`, container rebuilt, page
  serves it, repo pushed (`5ce4547..cdbe70f`). Also fixed a pre-existing broken
  `Odysseus` entry (missing `category`) that was 500ing the page.
- vdevserve (`vdevserve` / `vdev-gateway.namespace.so`) does not resolve from
  this box; only the gateway-host Bifrost is registered.

## Zanooda PP8 stack evaluated on this box (2026-09-11 PM)

Pulled `zanooda/vllm-sm80-ds41f:v41-sm80` (41.4 GB) and built a matching
checkout: vLLM PR #56201 head `79a7108d9a` + `patches/v41/0001-0013` applied
cleanly (`git am`, my tree `620072dd` vs his documented `70aaaced` — a
provenance discrepancy, but it runs). Launched TP1×PP8 with his flags
(`--max-num-batched-tokens 4096`, `--max-num-seqs 8`, util 0.95, engram cpu,
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) plus the 14 files of patches 0009-0013
bind-mounted.

Result on our box: **correct** (coherence, needle 32k 3/3) but the advertised
concurrency is **not reproduced**:

| 512k decode (prefix-cached, ignore_eos) | TP4×PP2 (served) | Zanooda PP8 |
|---|---|---|
| c8 | 103.3 | 77.6 |
| c16 | 111.7 | 82.8 |
| c64 | **129.6** | 92.0 |

Zanooda's own method measures decode as tokens/(last−first token) and reports
117 single-stream / 532 @8 streams (from *staggered* arrivals). His harness on
our box gives 43–81 tok/s decode; his 532 is not a like-for-like wall-clock
aggregate. **Conclusion: TP4×PP2 stays the decode-optimal, validated config.**
His PP8 does prefill ~2× (matching the no-all-reduce finding) and carries extra
optimizations (candidate-only indexer, engram-cpu, row-chunk) that are worth
porting to TP if a TP-split relay is ever made correct.

## Corrected decode measurements (ignore_eos, prefix-cached) — 2026-09-11 PM

Earlier per-concurrency decode numbers were understated: the `bench_dspark_metrics.py`
arms did not set `ignore_eos`, so completions stopped early and per-request overhead
dominated. Re-measured with `ignore_eos` + a shared cached prefix
(`bench_depth_cached.py`), depth is the only real variable:

| context | c4 | c8 | c16 | c32 | c64 |
|---|---|---|---|---|---|
| 4k | 103.8 | 157.0 | 188.2 | **302.8** | 295.5 |
| 128k | — | — | 180.1 | 214.8 | — |
| 512k | — | 77.6* | 111.7 | — | **129.6** |

\* 512k c8 on Zanooda's stack; TP4×PP2 at 512k: c16 111.7, c64 129.6.

Single-stream (ignore_eos, prompt≈8 tokens): **~105 tok/s** pure decode — matching
Zanooda's 117 claim, not the 43 the EOS-truncated harness showed.

**So the corrected picture:** aggregate decode reaches ~300 tok/s at shallow context
and falls to ~130 at 512k. The 500 tok/s @512k target is a *depth-scaling* problem
(sparse-attention / indexer cost growing with context), not a concurrency problem.

### Correction: per-stream TPOT vs depth

An earlier note said "per-stream TPOT is flat with depth". That used the
harness's `tpot_ms` field, which is wall/total-tokens (an AGGREGATE figure).
Re-derived per-stream TPOT = wall * concurrency / completion_tokens:

| ctx | c | aggregate tok/s | per-stream tpot | per-stream tok/s |
|---|---|---|---|---|
| 4k | 32 | 302.8 | 106 ms | 9.5 |
| 32k | 32 | 305.4 | 105 ms | 9.5 |
| 128k | 32 | 214.0 | 150 ms | 6.7 |
| 512k | 32 | 108.8 | 294 ms | 3.4 |
| 512k | 64 | 126.9 | 504 ms | 2.0 |

So per-stream decode DOES slow with depth (106 -> 504 ms), ~5x from 4k to 512k.
Both effects matter: depth raises per-token cost and concurrency amortises it;
the aggregate at 512k is ~c / per_stream_tpot. Reaching 500 tok/s @512k needs
either ~5x lower per-token cost at depth or ~4x the concurrency headroom.

## v0.13.0 backport base evaluated (2026-09-11 late)

Schaka bumped his kit to `lazymio/vllm-backport:v0.13.0-sm80` (V4.1 model +
Ampere sparse-MLA + software-fp8 baked into the base; his kit now only adds the
PP relay + DSpark embed + reasoning alias, and `fixups.py`). Pulled the base
(digest bfe237ee0886) and rebuilt the overlay (10 changed files, image
`localhost/vllm-backport-v41:sm80-v13`).

Same TP4×PP2 config, measured here:

| arm | v0.12.0 base (served) | v0.13.0 base |
|---|---|---|
| 4k c8 | 157.0 | 130.3 |
| 4k c32 | 302.8 | 270.2 |
| 4k c64 | 295.5 | 304.8 |
| 512k c16 | 111.7 | 101.4 |
| 512k c64 | 129.6 | 123.2 |
| needle 32k | 3/3 | 2/2 |
| coherence | OK | OK |

No gain: the newer base is equivalent-to-slightly-slower on this box, within
run-to-run noise. Kept the v0.12.0-based image (validated 12/12 long-context,
5/5 coding) as the served config.

## c1-first optimization: TP8×PP1 adopted (2026-09-11 late)

Per the "prioritize c1 speed first" directive, measured single-stream decode
(max_tokens=256, `ignore_eos`, best of 3) across arrangements:

| prompt tokens | TP4×PP2 | **TP8×PP1** |
|---|---|---|
| 8 | 89.4 | **98.1** |
| 512 | 84.2 | **95.0** |
| 4096 | 87.6 | **92.6** |
| 32768 | 54.1 | **88.4** |

TP8×PP1 (zero pipeline hops) wins c1 at every depth — decisively so at 32k
(88.4 vs 54.1), because TP4×PP2 pays two serial PP-hop latencies per token.
Accuracy gate on TP8×PP1: coherence ✓, needle 32k/128k 4/4, coding 5/5.

Tradeoff: TP8×PP1 loses aggregate at 512k (c16 83.9 / c64 113.4 vs
111.7 / 129.6). Since c1 is the current priority, **TP8×PP1 is the served
config** (`serve_arrangement.sh 8 1 0,1,2,3,4,5,7,8 ""`); switch back to
TP4×PP2 for max aggregate.

## c1 kernel levers tried (2026-09-11 late)

- `moe_backend: triton` — **fails engine init** on this MXFP4/SM80 path (worker
  init exception). Marlin is the only working MXFP4 MoE backend here.
- `max-num-seqs 1` — no c1 gain observed before the above crash; left at 32.
- DSpark draft count is fixed by the checkpoint (`dspark_block_size=5`); k=3
  was measured by Schaka as the same spread, so no c1 headroom there.
- So c1 headroom past 98 tok/s lives in the Marlin MXFP4 MoE GEMV and the
  sm_80 sparse-MLA decode kernel, not in config.

## c1 final (TP8×PP1) + clock experiment

After attempting `nvidia-smi -lgc 1695,1695` (persistence mode on) on the eight
serving GPUs: the clocks stayed at 1470–1485 MHz — the CMP 170HX clamps the
lock, so it did not take, and it was reverted with `-rgc`. Under c1 load the
cards draw ~100 W of 250 W, so c1 is not power/clock-bound.

Best-of-4 single-stream decode (warm server, `ignore_eos`, 256 tokens out):

| prompt tokens | c1 tok/s (TP8×PP1) |
|---|---|
| 8 | **100.8** |
| 4,096 | **97.6** |
| 32,768 | **89.9** |

(Run-to-run spread is large — 2.5–5.0 s walls for the same arm — so single
draws are not comparable; use best-of-N or medians.)

### VLLM_USE_BREAKABLE_CUDAGRAPH=1 — worse on this box

Tested (Zanooda's PP8 stack sets it). On our TP8×PP1 stack it degrades c1:
62.7 tok/s @8-tok and 44.0 @4k (vs 100.8 / 97.6 without it), with a very wide
spread (up to 40 s walls). Reverted. The default FULL_AND_PIECEWISE capture
(which we already get) is the faster path here.

### Profiling unavailable on this image

`VLLM_TORCH_PROFILER_DIR` is an unknown env var in this vLLM build (logged
"Unknown vLLM environment variable detected") and `/start_profile` returns 404,
so the built-in torch profile path is not available. Per-kernel attribution of
the c1 ~10 ms/token would need an out-of-band profiler (nsys/py-spy) inside the
container. c1 config work is therefore complete; further gains are kernel edits.

## KV cache size for 1M tokens (measured, PP8, 2026-09-12)

Engine-reported accounting, Zanooda PP8 stack (TP1xPP8, fp8_ds_mla, 8 ranks):

- GPU KV cache size: **6,170,576 tokens** (`Maximum concurrency for 1,048,576
  tokens per request: 5.88x`).
- Per-rank KV memory: PP0 17.89, PP1 19.67, PP2 19.28, PP3 20.95, PP4 16.88,
  PP5 17.28, PP6 17.29, PP7 6.48 GiB -> **total 135.72 GiB**.
- **23,617 bytes/token** whole-model (135.72 GiB / 6,170,576).
- **1,048,576 tokens ≈ 23.06 GiB** (≈2.9 GiB per rank).

The model card advertises ~890 B/token for its FP4 global KV; our sm_80 path is
~26x that because it runs `fp8_ds_mla` (FP4 KV is rejected for MLA here) and
stores the uncompressed indexer-K caches for the 8 indexer layers in addition to
the 4 compressed KV groups. The pool is limited by the two binding ranks
(PP7 6.48 GiB is the smallest KV budget; it also carries lm_head + drafter).

## PP6 vs PP8 (measured, 2026-09-12) — PP6 wins c1

Both correct (verified by the accuracy gate). PP6 uses Schaka's v0.13.0-based
kit (TP1xPP6, partition 7,7,7,7,7,5, util 0.95); PP8 uses Zanooda's 13-patch
stack (TP1xPP8, 5x8, util 0.95).

| metric | PP6 (TP1xPP6) | PP8 (TP1xPP8) |
|---|---|---|
| pure single-stream decode (8-tok prompt) | **96.9 tok/s** | 43.2 |
| pure single-stream @512 / 4k / 32k | 87.1 / 103.8 / 49.4 | 40.9 / 32.2 / 27.6 |
| decode aggregate c8 (4k / 32k / 128k) | 146 / 159 / 164 | 155 / 137 / 128 |
| decode aggregate c16 512k | 88.3 | 86.8 |
| prefill c1 (4k / 128k / 512k) | 2,258 / 7,716 / 5,267 | 1,350 / 9,209 / 7,306 |
| prefill c16 (4k / 128k / 512k) | 9,501 / 23,780 / 14,630 | 7,067 / 30,058 / 24,151 |
| KV pool | 2,888,012 tok | 6,170,576 tok |
| KV bytes/token (whole model) | **11,214 B** | 23,617 B |
| KV for 1M tokens | **10.95 GiB** | 23.06 GiB |
| accurate? | yes (needle 6/6) | yes (needle 3/3) |

**PP6 is strictly better for our box at c1 and KV efficiency**, at the cost of a
smaller total KV pool (6 cards vs 8). PP8's per-token KV is ~2x PP6's because its
`pp_share` relay replicates the compressed/indexer caches across ranks. PP6 c1
(96.9) is within 4% of the TP8xPP1 best (100.8) while remaining TP-free.

## PP6 adopted as the PP default (2026-09-12)

Per the c1-first + PP-only scope, **PP6 (TP1×PP6)** — Schaka's v0.13.0-based kit,
partition 7,7,7,7,7,5, util 0.95, seqs 8 — is the served PP config:

- **c1 single-stream: 96.9 tok/s (best-of-4), 107.6 tok/s with 93.8% DSpark
  acceptance** on predictable content. Within noise of TP8×PP1 (100.8) and ~2.3×
  faster than PP8 (43.2).
- Validation: coherence ✓, needle 32k/128k 6/6, needle 512k 3/3, coding 5/5.
- KV: 2,888,012-token pool (30.16 GiB over 6 ranks) -> 11,214 B/token ->
  **1M tokens = 10.95 GiB**.
- Prefill: 4k c1 2,258 / c16 9,501; 128k c1 7,716 / c16 23,780; 512k c1 5,267 /
  c16 14,630 tok/s.
- Decode aggregate (tok/s): 4k c8 146 / c16 142; 32k c8 159 / c16 160; 128k c8
  164 / c16 150; 512k c8 85 / c16 88.

Command: `IMG=localhost/vllm-backport-v41:sm80-v13 EP=1 SEQS=8 UTIL=0.95 \
  serve_arrangement.sh 1 6 0,1,2,3,4,5 7,7,7,7,7,5 dsv41-schaka`

## PP6 full validation (2026-09-12)

- coherence ✓; needle 32k/128k 6/6; needle 512k 3/3; **needle 1M (1,048,576)
  1/1** (exact recall, wall 288 s); coding 5/5.
- OpenCode headless proof PASS; live TUI answered correctly server-side
  (`PP6-TUI-7391-OK`, finish=stop) — the pane just failed to repaint.
- c1 96.9-107.6 tok/s; KV 11,214 B/token -> 1M = 10.95 GiB.

## PP6 relay fix adopted (Schaka c1b0907b, 2026-09-12)

Rebuilt the PP6 overlay from Schaka's latest kit (14 changed files; the relay
now adds `_build_index_k_replica` and "owns the indexer K cache
language_model.model.layers.20.attn.indexer.k_cache"). This targets the
prompt-corruption path he found (his 44k-token context check was 5/10 wrong
before the fix, 0/10 after).

Validation of our rebuilt PP6 (TP1xPP6, v13 image, 7,7,7,7,7,5, util 0.95):

| check | result |
|---|---|
| relay log line | "kv-group relay owns the indexer K cache ... layers.20.attn.indexer.k_cache" |
| coherence | Berlin/Paris/Canberra + `7391-MARLIN-42` |
| needle 128k (15/50/85%) | 3/3 |
| **multi-needle @44k (5 planted codes)** | **5/5** |
| coding | 5/5 |
| pure c1 (8 / 4k / 32k prompt) | 105.1 / 76.4 / 102.0 tok/s |

PP6 stays the served PP config, now with the corrected relay.

## PP6 sweep to the full 1M context (2026-09-12)

Decode (prefix-cached, 128-256 tok out, `ignore_eos`), aggregate tok/s:

| ctx | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| 4k | 49.2 | 91.1 | 108.9 | 146.4 | 141.5 |
| 32k | 45.1 | 80.5 | 104.2 | 159.0 | 159.6 |
| 128k | 42.6 | 65.9 | 100.4 | 164.5 | 150.0 |
| 512k | 30.0 | 46.1 | 54.0 | 84.9 | 88.3 |
| 1M | 15.9 | 22.4 | n/a | n/a | n/a |

At 1M only c1/c2 fit: a single 1M request consumes ~1M of the 2,888,012-token
pool (max concurrency 2.75x). DSpark acceptance rises to 51.7% at 1M.

Prefill (unique random, TTFT -> tok/s): 4k c1 2,258 / c16 9,501; 128k c1 7,716 /
c16 23,780; 512k c1 5,267 / c16 14,630; **1M c1 TTFT 342.5 s (~3.1k tok/s)**.

Single-stream (pure, short prompt): 105.1 tok/s (8-tok), 102.0 (32k).

## Gateways + OpenCode re-verified on PP6 (2026-09-12)

- Service healthy on 127.0.0.1:8090 and via Tailscale 100.64.0.119:8090.
- Bifrost provider `dsv41-flash-cmp` / `deepseek-v4.1-flash`: raw route returns
  `BIFROST-PP6-OK`; gateway landing page HTTP 200 and lists the model.
- **OpenCode through Bifrost returns `BIFROST-PP6-OPENCODE-OK`**; OpenCode direct
  (headless) proof passed earlier on PP6.
- So all four paths are live on the PP6 stack: local, Tailscale, Bifrost raw,
  and OpenCode->Bifrost.

## c1 lever: NCCL_P2P_LEVEL=SYS on PP6 (2026-09-12)

Zanooda's scripts set `NCCL_P2P_LEVEL=SYS` (BAR1 P2P enabled box). Tested on PP6:

| prompt | without | with P2P_LEVEL=SYS |
|---|---|---|
| 8 | 105.1 | 94.3 |
| 4k | 76.4 | 92.2 |
| 32k | 102.0 | 88.3 |

Neutral within the large run-to-run spread (single draws vary 2.5-10 s). `nvidia-smi
topo -p2p r` reports `GNS` for every pair (no direct peer path advertised), so the
PP hops do not benefit measurably. Not a c1 win; c1 stays ~95-105 tok/s and its
remaining headroom is the Marlin MXFP4 MoE GEMV / sm_80 sparse-MLA decode kernel.

## PP6 validation complete (2026-09-12)

| gate | result |
|---|---|
| coherence + planted fact | Berlin/Paris/Canberra + `7391-MARLIN-42` |
| needle 32k/128k (15/50/85%) | 6/6 |
| multi-needle @44k (5 codes) | 5/5 |
| needle 512k | 3/3 |
| needle 1M (1,048,576) | 1/1 |
| accumulated multi-turn prefix (8 turns, ~13k tok) | HIT (`MULTITURN-885440-K`); recall turn 1.27 s vs 4.5 s fillers (prefix cache) |
| coding (execute tests) | 5/5 |
| OpenCode headless / TUI / through-Bifrost | PASS / answer correct / `BIFROST-PP6-OPENCODE-OK` |
| gateways (local, Tailscale, Bifrost, landing) | all live |

PP6 is the served, fully validated PP config with the corrected relay.

## Third-party depth: zebgop-ops REAP-272E (2026-09-12)

Reviewed `zebgop-ops/dsv41reap-pp` (new today). It serves
`LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E` (272/384 routed experts, native MXFP4/FP8)
on **4× CMP 170HX, PP4 (partition 10,10,10,10)**.

- REAP prunes experts so each layer's experts fall from 6.72 to 4.76 GiB; that lets a
  PP4 partition keep **every expert on the GPUs**. The unpruned model on 4 cards must
  CPU-offload nine layers' experts, capping decode at 14-22 tok/s and prefill ~100.
- Author-reported REAP numbers: free prose ~30, code 60-110 / fresh ~85 tok/s, prefill
  ~3k tok/s; cost +4.1% text perplexity; greedy code-edit byte-identical to unpruned.
- Their own FINDINGS: decode step ~29-37 ms of GPU time across four ranks + ~5-7 ms
  round trip, "within ~15% of the hardware's ceiling with PP4 over PCIe Gen2". Prefill
  is compute-bound in the Marlin MXFP4 expert GEMMs (~400-500 ms per 2048-token chunk).

**Relevance to us:** REAP's win is for CPU-offload setups. Our PP6 already keeps all
experts on-GPU, and our single-stream (96.9-105 tok/s) is ~3x their REAP PP4 prose rate.
So the REAP path is not a c1 lever here. Their repo does carry a useful debug toolkit
(`DSV41_DEBUG_TIMING/PROFILE/TRACE`, `torch.profiler` on the 41st decode step,
per-layer wall time) that is a porting reference for attributing our c1 ~10 ms/token.

## c1 bottleneck diagnostic: SM-occupancy-bound, NOT DRAM-bound (2026-09-12)

`nvidia-smi dmon -s um` sampled during a sustained c1 decode (600-token, PP6):

| GPU | sm % | mem % |
|---|---|---|
| 0-3 | ~99-100 % | 4-5 % |
| 4-5 | 29-88 % | 4-14 % |

The SM utilization is saturated while the **DRAM/memory-controller utilization
is only ~4-5%**. So the c1 ~10 ms/token is **latency / SM-occupancy bound, not
memory-bandwidth bound** (an earlier note guessed bandwidth — corrected here).
At batch-1 the Marlin MXFP4 MoE GEMV and the sparse-MLA decode launch too few
warps to hide latency, so the SMs sit busy-but-stalled and DRAM is idle.

Consequence: c1 headroom is a **parallelism** problem inside the batch-1
kernels (more warps / split-K / a fused batch-1 GEMV), not a bandwidth problem —
which is also why aggregate decode scales so well with concurrency. This is the
concrete target for any kernel edit; no configuration lever addresses it.

## Real-OpenCode c1 benchmark (repeatable Docker harness) — 2026-09-12

Per the "c1 is measured via real OpenCode usage" requirement, built
`opencode-bench/`: a repeatable Docker image (`localhost/dsv41-opencode-bench`,
Ubuntu 24.04 + opencode 1.18.30 + python3, `--network host` to reach the model)
that runs the real OpenCode agent on three real coding tasks against a bundled
fixture (`src/textstats.py` with two live bugs + a test file): fix_bugs,
add_feature, docstring. Each arm reports wall time, steps, server-reported
input/output/reasoning tokens, and end-to-end output tok/s.

Measured on PP6 (`deepseek-v4.1-flash`, DSpark k=5):

| task | wall s | steps | in / out tok (reasoning) | end-to-end out tok/s |
|---|---|---|---|---|
| fix_bugs r1 | 15.5 | 5 | 1,853 / 581 (110) | 37.4 |
| add_feature r1 | 20.6 | 6 | 1,949 / 828 (34) | 40.1 |
| docstring r1 | 16.9 | 6 | 1,557 / 597 (82) | 35.3 |
| fix_bugs r2 | 88.0 | 12 | 10,810 / 1,287 (3,199) | 14.6 |
| add_feature r2 | 14.1 | 5 | 1,685 / 373 (126) | 26.4 |
| docstring r2 | 8.2 | 2 | 534 / 67 (126) | 8.2 |

**Real code-usage c1 ≈ 30 tok/s end-to-end (median), range 8-48.** The spread is
dominated by reasoning-token volume and step count, not by raw decode: the
drafting is DSpark rate-limited (~100 tok/s single-stream), while an 8.8k-token
prompt plus tool-call turns adds seconds of prefill per task. This is the honest
real-world c1 number to optimize against.

## LMCache blocker root-caused: wrong connector class, fix = MP connector (2026-09-12)

Arm: PP6 + `--kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}`
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (the first rejection:
the in-process connector refuses `expandable_segments:True`). Weights loaded, the
relay owned the indexer K cache, then ~12 min in the engine died:

```
WARNING [vllm.py:1899] Turning off hybrid kv cache manager because
  `--kv-transfer-config` selects a KV connector that does not support it.
...
WARNING [kv_cache_utils.py:1895] Hybrid KV cache manager is disabled for this
  hybrid model ...
ERROR [core.py:1382] [...] kv_cache_utils.py:2206 in get_kv_cache_groups
  -> kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))
  -> kv_cache_utils.py:1841 raise ValueError(
       "Failed to promote local KV cache specs to one unified type.")
```

Cause chain, verified in source (`/mnt/kv/build/schaka-v13/vllm`):

1. `LMCacheConnectorV1` does **not** subclass `SupportsHMA`, so
   `config/vllm.py:1897` sets `disable_hybrid_kv_cache_manager=True`.
2. With HMA off, `get_kv_cache_groups` (kv_cache_utils.py:2205) calls
   `unify_hybrid_kv_cache_specs` -> `_promote_local_kv_cache_specs`.
3. V4.1 emits two MLA-family specs that do not merge after promotion:
   - `MLAAttentionSpec` (main/indexer; `state_content_bytes=584`, packed
     448B NoPE + 128B RoPE + 8B fp8 scale per token) —
     `models/deepseek_v4/attention.py:1050,1095`.
   - `SlidingWindowMLASpec` (compressor) —
     `models/deepseek_v4/compressor.py:179`, promoted to `MLAAttentionSpec`
     but with a different page size, so `is_kv_cache_spec_uniform` and
     `UniformTypeKVCacheSpecs.is_uniform_type` both return False -> raise.

The in-process `LMCacheConnectorV1` adapter
(`lmcache/integration/vllm/vllm_v1_adapter.py`, `LMCacheConnectorV1Impl`) has no
`kv_cache_config`, no `kv_layer_groups`, and no group index anywhere: it is
**single-group by construction** and cannot serve V4.1's multi-group layout.

The correct path is already shipped and is HMA-capable:
`lmcache/integration/vllm/lmcache_mp_connector.py` declares
`class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA)` and implements
`request_finished_all_groups`. It validates the real group set at startup
(`validate_mamba_step_alignment`, `validate_kv_cache_groups`,
`get_group_tokens_per_block`). Its group model matches V4.1 exactly —
`lmcache/v1/kv_layer_groups.py` keys groups on
`(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype,
engine_kv_format)` and explicitly calls out "a rank-5 K/V group alongside a
rank-3 key-only indexer cache" (DeepSeek-V4).

`lmcache` + `lmcache server` (ZMQ MP server; `--l1-size-gb`, `--eviction-policy`)
are both installed in the image. So LMCache-for-V4.1 = `LMCacheMPConnector`
(+ server), **not** `LMCacheConnectorV1`. Per AGENTS.md the MP server must be
restarted before the vLLM service that attaches to it.

### RESOLVED: LMCache serving V4.1 (2026-09-12)

The MP path works. Exact recipe, each step forced by a concrete error:

1. Start the LMCache MP server **with GPU access** (it opens the vLLM workers'
   CUDA IPC KV handles, so it must see the same GPU UUIDs):
   ```
   docker run -d --name lmcache-mp --init --runtime nvidia \
     --network host --ipc host --shm-size 32g --security-opt label=disable \
     --entrypoint /usr/local/bin/lmcache \
     -e LMCACHE_DISABLE_BANNER=1 -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5 \
     localhost/vllm-backport-v41:sm80-v13 \
     server --host 127.0.0.1 --port 6667 --l1-size-gb 32 --eviction-policy LRU
   ```
   Without `--runtime nvidia`: `RuntimeError: Device UUID ... not found in the
   discovered devices. Please make sure the process can see all the accelerator
   devices` -> workers time out with `LMCache server did not respond to
   register_kv_caches within 300.0s`.
2. Launch vLLM with the MP connector + a retention interval that is a divisor of
   the 256-token LMCache chunk:
   ```
   --prefix-cache-retention-interval 256 \
   --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
     "kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1",
     "lmcache.mp.port":6667}}'
   ```
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Without the retention
   interval: `ValueError: This model has recurrent or sliding-window KV cache
   groups; LMCache needs a state checkpoint at every chunk boundary. Set
   --prefix-cache-retention-interval to a divisor of the LMCache chunk size
   (256, e.g. 256), got 0.`

Startup evidence:
```
Creating v1 connector with name: LMCacheMPConnector and engine_id: ...
Resolved LMCache MP geometry: group_tokens_per_block=[32, 32, 32, 32, 128, 0],
  scheduler_block_size=128, cache_model_name=/model
KV cache group edits applied: {'unified-attention-view': 10}
Excluding non-prefix-cacheable engine group 5 (UniformTypeKVCacheSpecs, 1 layers)
Application startup complete.
```
Note the exclude at group 5 — this is the "disposable KV group" exclusion the
qwen3.8 LMCache patch documented; here LMCache's own connector does it.

`serve_arrangement.sh` gained an optional `RETENTION` knob for this.

Cold/repeat/restart-replay accuracy numbers: see the LMCache accuracy section
below (and `scripts/validate_lmcache_v41.py`, logs in `/mnt/kv/logs/dsv41/`).

### LMCache accuracy validation — PASS (2026-09-12)

Harness `scripts/validate_lmcache_v41.py`: a 9.7k-token prompt with a planted
retrieval key (`KEY-<nonce>`), greedy (temperature 0, seed 0), thinking off.
Any correct run must answer `KEY-<nonce>`. Metrics are read from
`vllm:external_prefix_cache_hits_total` and
`vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}`.

| arm | prompt tok | elapsed s | answer | external_kv_transfer tok |
|---|---|---|---|---|
| cold, fresh nonce | 9,767 | 2.515 | `KEY-T8M2P1051` | 0 |
| repeat, same server | 9,767 | 0.485 | `KEY-T8M2P1051` | 128 |
| **after full vLLM restart** | 9,737 | 0.861 | `KEY-N7C41K1037` (== its cold answer) | **9,728** |

* **Cold vs full-restore equality:** after restarting only vLLM (GPU KV wiped;
  LMCache server untouched), the identical request restored **9,728 of 9,737
  tokens from LMCache** and returned the exact same answer as its cold run.
* **Same-server determinism:** repeat output is byte-identical to cold.
* **Artifact ruled out:** an earlier run with `max_tokens=24` produced empty /
  divergent continuations; that reproduced on a *fresh cold* run too
  (`external_kv_transfer=0`), proving it was tight-token/reasoning truncation,
  not a cache fault. With thinking disabled and 64 tokens the outputs are exact.

So "100% output accuracy with LMCache" holds for full LMCache restore on this
retrieval workload. Remaining LMCache work (not yet done): sustained multi-turn
prefix accumulation, mixed concurrent hit/miss semantic equality, eviction/
reload, and the L2 (disk) tier; and re-measuring the real-OpenCode c1 bench with
LMCache attached to quantify the end-to-end prefix-reuse gain.
