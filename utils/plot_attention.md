# `plot_attention.py`

## Purpose

Loads `.npz` attention dumps produced by **`attention_weight_dump`** (see `fastvideo/attention/attention_weight_dump.md`) and generates Matplotlib figures for inspection.

Supports:

- **New format**: `weights` with shape **`[L, L]`** (single head per file).
- **Legacy format**: `weights` with shape **`[H, L, L]`** (multiple heads per file), for older experiments.

## Dependencies

- `numpy`
- `matplotlib`

Run from the repository root (or any path where `utils/plot_attention.py` is visible):

```bash
python utils/plot_attention.py [options]
```

## Modes

### `grid` (default)

Builds a 2D heatmap: **denoise step** × **layer** (block type + index), where each cell is a **scalar summary** of the attention matrix for the chosen `--head` and `--metric`.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--dump-dir` | Directory containing `*.npz` files (required) |
| `--head` | Head index when using legacy `[H,L,L]` tensors (default `0`) |
| `--block` | Filter: `double_blocks`, `single_blocks`, or `all` |
| `--cfg` | `pos`, `neg`, or `both` |
| `--metric` | `mean`, `entropy`, or `diag` |
| `--out` | Output image path (default `attention_plot.png`) |

**Example:**

```bash
python utils/plot_attention.py \
  --dump-dir ./attn_dumps \
  --mode grid \
  --head 0 \
  --block double_blocks \
  --cfg pos \
  --metric mean \
  --out grid.png
```

### `heatmap`

Renders a single **\(L \times L\)** heatmap from **one** `.npz` file.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--npz` | Path to a file, or a glob that resolves to **exactly one** file (required) |
| `--head` | Used for legacy multi-head arrays; new single-head dumps use `meta["global_head"]` when present |
| `--out` | Output image path |

**Example:**

```bash
python utils/plot_attention.py \
  --mode heatmap \
  --npz ./attn_dumps/step0010_double_blocks_005_head003_pos.npz \
  --out one_layer.png
```

## Filename patterns

- **New**: `step{N}_{block}_{idx}_head{H}_{pos|neg}.npz`
- **Legacy**: `step{N}_{...}_{hash}_{pos|neg}_sp{R}.npz` (see script regex in `_parse_fname`)
