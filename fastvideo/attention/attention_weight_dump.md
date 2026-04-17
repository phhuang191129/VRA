# `attention_weight_dump.py`

## Purpose

Optional runtime hook used by FlashAttention and SDPA backends to save **one** dense softmax attention matrix **\(L \times L\)** per inference run, when a specific **(denoise step, block, block index, head, CFG pass)** matches environment filters.

This exists for analysis and plotting. Full-sequence \(L\) for video DiTs is large; use `FASTVIDEO_ATTENTION_DUMP_SLICE_LEN` to cap sequence length when memory is tight.

## Behavior

- Reads configuration from **`FASTVIDEO_ATTENTION_DUMP_*`** environment variables (see below).
- Runs only when **`FASTVIDEO_ATTENTION_DUMP_DIR`** is set **and** all required **target** variables are present.
- Skips cross-attention / varlen cases where `query` and `key` sequence lengths differ.
- Skips GQA layouts where global heads do not shard evenly across sequence-parallel ranks.
- For sequence parallelism (`sp_world > 1`), only the rank that owns the **target global head** computes and saves.
- Saves **`weights`**: shape **`[L, L]`**, `float16`, plus JSON **`meta`** in a compressed `.npz` file.

## Required environment variables

| Variable | Description |
|----------|-------------|
| `FASTVIDEO_ATTENTION_DUMP_DIR` | Directory to write `.npz` files. |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_STEP` | Denoise loop index `i` (matches `set_forward_context(current_timestep=i, ...)`) |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD` | Global attention head index |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK` | `double_blocks` or `single_blocks` |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX` | Integer index within that block list |

## Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTVIDEO_ATTENTION_DUMP_SLICE_LEN` | `0` | If `> 0`, only the first `L` tokens along the sequence are used. If `0`, uses full `query` length (high memory). |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_CFG` | `any` | `pos`, `neg`, or `any` — which classifier-free pass to dump |

## Output filename pattern

```
step{timestep}_{block}_{block_index:03d}_head{global_head:03d}_{pos|neg}.npz
```

## Helper functions

- **`estimate_attention_matrix_bytes(seq_len, ...)`** — rough stored size and peak GPU estimate for one \(L \times L\) matrix.
- **`estimate_hunyuan_image_tokens_from_pixels(...)`** — rough **image** token count from pixel height/width/frames (VAE + patch assumptions); joint attention also includes text tokens, so add a buffer for total \(L\).

## Integration

Invoked from:

- `fastvideo/attention/backends/flash_attn.py` — end of `FlashAttentionImpl.forward`
- `fastvideo/attention/backends/sdpa.py` — end of `SDPAImpl.forward`

See also: `fastvideo/envs.py` for lazy env var documentation.
