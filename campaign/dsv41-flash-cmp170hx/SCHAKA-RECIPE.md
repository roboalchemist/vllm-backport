# DeepSeek-V4.1-Flash via Schaka/170hx-journey (the working recipe)

Pivot after our `dsv41-feat` engine produced incoherent output: we adopted the
one proven SM80/CMP-170HX V4.1 recipe, from
[Schaka/170hx-journey](https://github.com/Schaka/170hx-journey)
(`patches/vllm-backport-v41/`).

## Provenance
- Base image: `lazymio/vllm-backport:v0.12.0-sm80` (29.5 GB, `bb1f7777193e`)
- Overlay: upstream PR #56201 + `fixups.py` + `ampere/` shims + `pp_kv_group_relay.py`
- Built as `localhost/vllm-backport-v41:sm80` (thin Python overlay, no native rebuild)
- 92 changed files. Built by `scripts/build_schaka_v41.sh`.

## Serve (TP4xPP2)
`scripts/serve_schaka_v41.sh` — TP4, PP2, `VLLM_PP_LAYER_PARTITION=20,20`,
`TRITON_MLA_SPARSE_DSV41`, `fp8_ds_mla`, block-128, DSpark k=5, max-model-len
1048576, DSpark, `--enable-expert-parallel`, `--network host`, port 8090.

Note: the build daemon runs `"bridge": "none", "iptables": false`, so `-p`
port mappings do NOT work. Use `--network host`.

## Local evidence (2026-09-11)
- `init engine (profile, create kv cache, warmup model) took 217.74 s`;
  graph capture 77 s; `Application startup complete.`
- Coherent chat completion (temperature 0), via `docker exec`:

  > "We need answer. Need comply. User asks: 'What is the capital of Germany?
  > ...' We can do: The capital of Germany is Berlin. The capitals of France
  > and Australia are Paris and Canberra."

  Berlin / Paris / Canberra all correct.

This is the first known coherent DeepSeek-V4.1-Flash generation on CMP 170HX
(SM80). Full validation (planted-fact determinism, OpenCode, tmux, Bifrost,
gateway) follows on the host-networked reload.
