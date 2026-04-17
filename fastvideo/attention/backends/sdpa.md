# `sdpa.py`

## Purpose

Implements the **Torch scaled dot-product attention (SDPA)** path for FastVideo: `torch.nn.functional.scaled_dot_product_attention`. It is exposed as attention backend name **`SDPA`** and is selected when the platform or environment requests **`TORCH_SDPA`** (see `fastvideo/platforms/` and `FASTVIDEO_ATTENTION_BACKEND`).

Use this backend when FlashAttention is unavailable or unsuitable (e.g. some ROCm/NPU setups), or for debugging against a reference PyTorch implementation.

## Components

### `SDPABackend`

- Subclasses `AttentionBackend`.
- **`get_name()`** → `"SDPA"`.
- **`get_supported_head_sizes()`** — same head-dimension list as other in-tree backends (32 … 256).
- **`get_impl_cls()`** → `SDPAImpl`.

### `SDPAMetadata`

- `AttentionMetadata` with:
  - **`current_timestep`** — diffusion step index (for forward context / optional tooling).
  - **`attn_mask`** — optional mask passed through to SDPA (same semantics as PyTorch: additive mask where valid positions are not masked).

### `SDPAMetadataBuilder`

- **`build(current_timestep, attn_mask)`** → `SDPAMetadata`.

### `SDPAImpl`

- **`forward(query, key, value, attn_metadata)`**
  - **Input layout**: `[batch, seq_len, num_heads, head_dim]` (same convention as other `AttentionImpl` classes).
  - Transposes to **`[batch, heads, seq, dim]`** for `scaled_dot_product_attention`, then transposes the output back.
  - **`softmax_scale`**: passed as `scale=` to SDPA (typically `head_dim ** -0.5`).
  - **`causal`**: passed as `is_causal`.
  - **`dropout`**: from `extra_impl_args["dropout_p"]` (default `0.0`).
  - **Cross-attention / GQA**: if query sequence length ≠ key sequence length, sets **`enable_gqa=True`** on the SDPA call (PyTorch GQA path).
  - After the main attention, optionally calls **`maybe_dump_attention_weights`** (see `fastvideo/attention/attention_weight_dump.md`) when `FASTVIDEO_ATTENTION_DUMP_*` env vars are set — same optional dump hook as FlashAttention.

## Usage

### Selecting this backend

Set the attention backend to the enum value that resolves to this module (environment variable name is **`TORCH_SDPA`**, not `SDPA`):

```bash
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

Model configs that list `AttentionBackendEnum.TORCH_SDPA` in `_supported_attention_backends` can use SDPA when selected by the platform or the override above.

### When SDPA is chosen automatically

Platform code (e.g. `fastvideo/platforms/rocm.py`) may pick SDPA when FlashAttention is not the default for that device class. See platform-specific docs and logs.

### Attention weight dumps

If you enable **`FASTVIDEO_ATTENTION_DUMP_DIR`** and the required **`FASTVIDEO_ATTENTION_DUMP_TARGET_*`** variables, this backend will attempt the same **optional** dense softmax dump as FlashAttention, after SDPA returns. Requirements and memory caveats are documented in **`attention_weight_dump.md`**.

## Related files

- `fastvideo/attention/backends/flash_attn.py` — FlashAttention variant.
- `fastvideo/attention/layer.py` — `LocalAttention` / `DistributedAttention` wiring to backends.
- `fastvideo/attention/selector.py` — backend resolution from env and hardware.
