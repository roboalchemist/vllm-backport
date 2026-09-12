# DeepSeek-V4.1-Flash model receipt

- Repo: `deepseek-ai/DeepSeek-V4.1-Flash`
- Revision: `df42c109f1defefcbfcedbe7d905718a12266e40`
- Local path: `/mnt/kv/models/DeepSeek-V4.1-Flash` (`/mnt/kv`, ZFS kvpool/cache)
- Shards: 48 × `model-*-of-00048.safetensors`
- `model.safetensors.index.json` `metadata.total_size`: **510,286,023,000 B = 475.24 GiB**
  (SHA-256 `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`)
- Downloaded with `hf download --revision df42c109... --local-dir ...` (Xet high-performance).

## Pinned large-shard hashes (verified locally)

| shard | bytes | sha256 |
|---|---:|---|
| `model-00047-of-00048.safetensors` | 101,535,150,272 | `824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f` |
| `model-00048-of-00048.safetensors` | 101,537,925,968 | `976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed` |

Shard 47 holds `layers.1.engram.embed.weight` (F8_E4M3, `[384006168,256]`, 91.554 GiB);
shard 48 holds `layers.14.engram.embed.weight` (F8_E4M3, `[384016682,256]`, 91.557 GiB).
These are the two single-tensor Engram tables that exceed one 64 GiB card and are
pinned to host RAM via `--engram-config cpu_offload` at serve time.

Raw hash receipt: `/mnt/kv/logs/dsv41/shard-hashes.txt`.
