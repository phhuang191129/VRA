# Hunyuan + attention dump on Modal

This flow runs **video inference only** on [Modal](https://modal.com); use **`utils/plot_attention.py` on your laptop** after you download `.npz` files from a Modal Volume.

Design follows Modal’s [LTX-Video example](https://modal.com/docs/examples/ltx) (GPU class, lifecycle) and [storing model weights on Volumes](https://modal.com/docs/guide/model-weights) (`HF_HOME` on a Volume so the Hunyuan checkpoint is not re-downloaded every cold start).

## Files

| File | Role |
|------|------|
| `scripts/inference/modal_attention_hunyuan.py` | Modal `App`, `Image`, Volumes, `HunyuanAttentionRunner` |
| `scripts/inference/attention_hunyuan.sh` | Local wrapper: `.../attention_hunyuan.sh --modal` → `modal run ...` |

## One-time local setup

```bash
pip install modal
modal setup
```

Run all commands from the **FastVideo repository root**.

## Volumes

| Volume name | Mount in container | Contents |
|-------------|-------------------|----------|
| `fastvideo-hf-hunyuan` | `/models` | Hugging Face cache (`HF_HOME`) |
| `fastvideo-hunyuan-outputs` | `/outputs` | Generated video artifacts under `hunyuan_run/` |
| `fastvideo-hunyuan-attn` | `/attn_dumps` | Attention `.npz` in container dir `/attn_dumps/dump/` → volume path **`dump/`** (when enabled) |

First run downloads the model into the HF Volume; later runs reuse it.

## Run inference on Modal

```bash
bash scripts/inference/attention_hunyuan.sh --modal
```

Equivalent:

```bash
modal run scripts/inference/modal_attention_hunyuan.py
```

### CLI overrides (`modal run` / `main` parameters)

Examples:

```bash
modal run scripts/inference/modal_attention_hunyuan.py \
  --prompt "A cat on a skateboard" \
  --dump-slice-len 2048 \
  --target-step 5 \
  --attention-backend TORCH_SDPA
```

Default attention backend on Modal is **`TORCH_SDPA`** so the image does not need to compile **flash-attn** (attention dumps still run; see `sdpa.py`). To try FlashAttention you must extend the Modal `Image` with a matching prebuilt `flash-attn` wheel or build steps.

### GPU choice (cost vs VRAM)

Set **before** `modal run` (read when the app module loads):

```bash
export MODAL_HUNYUAN_GPU=H100   # faster, higher $/s
export MODAL_HUNYUAN_GPU=A100-80GB   # default; usually enough for 544×960×75
```

Other types exist on Modal (e.g. L40S, A100 40GB) but may **OOM** for Hunyuan at this resolution or with large `dump_slice_len`.

### Gated Hugging Face models

If the hub requires a token, create a Modal Secret (e.g. name `huggingface`) containing **`HF_TOKEN`**, then:

```bash
export MODAL_USE_HF_SECRET=1
# optional: export MODAL_HF_SECRET_NAME=my-hf-secret
modal run scripts/inference/modal_attention_hunyuan.py
```

## Download attention dumps for local plotting

After a successful run:

```bash
modal volume get fastvideo-hunyuan-attn dump ./local_attn
```

Then on your machine (no GPU required for plotting):

```bash
python utils/plot_attention.py --mode heatmap \
  --npz './local_attn/step0000_double_blocks_000_head000_pos.npz' \
  --out heatmap.png
```

Use `modal volume ls fastvideo-hunyuan-attn` to inspect paths. See `modal volume --help` for the exact CLI on your Modal version.

## Price estimate (compute only)

Modal publishes **per-second** GPU rates on [modal.com/pricing](https://modal.com/pricing) (subject to change; Starter includes **$30/month** credits).

Indicative **GPU-only** rates from that page:

| GPU | ~$/second | ~$/hour |
|-----|-----------|---------|
| H100 | ~$0.00110 | ~$3.95 |
| A100 80GB | ~$0.00069 | ~$2.50 |
| A100 40GB | ~$0.00058 | ~$2.10 |
| L40S | ~$0.00054 | ~$1.95 |

**Example (A100 80GB):** one run ≈ **20–40 minutes** wall clock (first run often longer: image build + `pip install -e .` on the Volume-mounted repo). At ~$0.00069/s, **30 minutes** ≈ **$1.25** GPU time, **before** cold-start overhead and Volume/storage fees. **H100** is roughly **~1.6×** that GPU cost per second.

Add CPU/memory billing (small vs GPU for this workload) and any [Volume storage](https://modal.com/pricing) charges.

## Troubleshooting

**`IndexError` on `parents[2]`** — fixed in `modal_attention_hunyuan.py` by detecting Modal’s copy of the app at `/root/modal_attention_hunyuan.py`. If you see other import errors, run `modal run` from the repo root and keep the script under `scripts/inference/`.

## Related

- `scripts/inference/attention_hunyuan.md` — local shell script + env vars
- `fastvideo/attention/attention_weight_dump.md` — dump semantics
- `utils/plot_attention.md` — local plotting
