# SPDX-License-Identifier: Apache-2.0
"""Optional dump of dense attention softmax for **one** (head, layer, step).

Requires ``FASTVIDEO_ATTENTION_DUMP_DIR`` plus **all** of:

* ``FASTVIDEO_ATTENTION_DUMP_TARGET_STEP`` — denoise loop index ``i`` (same as
  ``set_forward_context(current_timestep=i, ...)``).
* ``FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD`` — global attention head index.
* ``FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK`` — ``double_blocks`` or
  ``single_blocks``.
* ``FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX`` — block index within that
  group.

Optional:

* ``FASTVIDEO_ATTENTION_DUMP_TARGET_CFG`` — ``pos``, ``neg``, or ``any``
  (default ``any``).
* ``FASTVIDEO_ATTENTION_DUMP_SLICE_LEN`` — max sequence length to materialize;
  **0** means full ``query`` sequence length (can be very large).

Use **one GPU** (``sp_size=1``) so the target head’s Q/K live on that process.

Memory (order of magnitude): one matrix stored as float16 is about
``2 * L^2`` bytes; peak GPU memory during ``softmax(QK^T)`` is often **2–3×**
that for the score tensor and intermediates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_VARLEN_SKIP_LOGGED = False
_GQA_SKIP_LOGGED = False
_INCOMPLETE_TARGET_LOGGED = False


@dataclass(frozen=True)
class _DumpTarget:
    dump_dir: str
    slice_len: int  # 0 => full sequence
    target_step: int
    target_head: int
    target_block: str
    target_block_index: int
    target_cfg: str  # pos | neg | any


def _dump_target() -> _DumpTarget | None:
    raw = os.environ.get("FASTVIDEO_ATTENTION_DUMP_DIR")
    if not raw:
        return None
    dump_dir = os.path.expanduser(raw)
    try:
        target_step = int(os.environ["FASTVIDEO_ATTENTION_DUMP_TARGET_STEP"])
        target_head = int(os.environ["FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD"])
        target_block = os.environ["FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK"]
        target_block_index = int(
            os.environ["FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX"])
    except KeyError as e:
        global _INCOMPLETE_TARGET_LOGGED
        if not _INCOMPLETE_TARGET_LOGGED:
            logger.warning(
                "FASTVIDEO_ATTENTION_DUMP_DIR is set but required "
                "FASTVIDEO_ATTENTION_DUMP_TARGET_* variable missing: %s. "
                "Set TARGET_STEP, TARGET_HEAD, TARGET_BLOCK, "
                "TARGET_BLOCK_INDEX.", e)
            _INCOMPLETE_TARGET_LOGGED = True
        return None

    slice_raw = os.environ.get("FASTVIDEO_ATTENTION_DUMP_SLICE_LEN", "0")
    slice_len = int(slice_raw)
    target_cfg = os.environ.get("FASTVIDEO_ATTENTION_DUMP_TARGET_CFG",
                                 "any").lower()
    if target_cfg not in ("pos", "neg", "any"):
        raise ValueError(
            "FASTVIDEO_ATTENTION_DUMP_TARGET_CFG must be pos, neg, or any")
    if target_block not in ("double_blocks", "single_blocks"):
        raise ValueError(
            "FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK must be "
            "double_blocks or single_blocks")
    return _DumpTarget(
        dump_dir=dump_dir,
        slice_len=slice_len,
        target_step=target_step,
        target_head=target_head,
        target_block=target_block,
        target_block_index=target_block_index,
        target_cfg=target_cfg,
    )


def _layer_tag_from_prefix(prefix: str) -> tuple[str, int]:
    parts = prefix.split(".")
    for i, p in enumerate(parts):
        if p in ("double_blocks", "single_blocks") and i + 1 < len(parts):
            try:
                return p, int(parts[i + 1])
            except ValueError:
                continue
    return "other", -1


def _sp_rank_world() -> tuple[int, int]:
    try:
        from fastvideo.distributed.parallel_state import (
            get_sp_parallel_rank,
            get_sp_world_size,
        )

        return get_sp_parallel_rank(), get_sp_world_size()
    except Exception:
        return 0, 1


def estimate_attention_matrix_bytes(
    seq_len: int,
    *,
    dtype_bytes: int = 2,
    peak_factor: float = 2.5,
) -> tuple[int, int]:
    """Return (stored_npz_bytes, rough_peak_gpu_bytes) for one L×L matrix."""
    l = int(seq_len)
    stored = dtype_bytes * l * l
    peak = int(stored * peak_factor)
    return stored, peak


def estimate_hunyuan_image_tokens_from_pixels(
    height_px: int,
    width_px: int,
    num_frames: int,
    *,
    spatial_compression: int = 8,
    temporal_compression: int = 4,
    patch_t: int = 1,
    patch_hw: int = 2,
) -> int:
    """Rough **image** token count after VAE + patch embed (DiT grid).

    Joint attention also attends to **text** tokens; add ~200–400 to L for
    order-of-magnitude full-sequence cost when ``txt`` is concatenated.
    """
    latent_t = (int(num_frames) - 1) // int(temporal_compression) + 1
    latent_h = int(height_px) // int(spatial_compression)
    latent_w = int(width_px) // int(spatial_compression)
    return (latent_t // patch_t) * (latent_h // patch_hw) * (latent_w //
                                                             patch_hw)


def maybe_dump_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_metadata: Any,
    softmax_scale: float,
    prefix: str,
    *,
    num_heads: int,
    causal: bool,
) -> None:
    """Save softmax weights ``[L, L]`` for one global head if (step,layer,cfg).

    ``num_heads``: **global** head count for this layer (from module config).
    ``query``/``key``: ``[B, S, H_local, D]`` after sequence-parallel shard.
    """
    global _VARLEN_SKIP_LOGGED
    tgt = _dump_target()
    if tgt is None:
        return
    if query.shape[1] != key.shape[1]:
        if not _VARLEN_SKIP_LOGGED:
            logger.warning(
                "Attention dump skipped: query/key seq len differ "
                "(varlen / cross-attn).")
            _VARLEN_SKIP_LOGGED = True
        return
    h_local_total = query.shape[2]
    h_global_total = int(num_heads)
    sp_rank, sp_world = _sp_rank_world()
    heads_per_rank = h_global_total // sp_world if sp_world > 1 else h_global_total
    if sp_world > 1 and heads_per_rank * sp_world != h_global_total:
        global _GQA_SKIP_LOGGED
        if not _GQA_SKIP_LOGGED:
            logger.warning(
                "Attention dump skipped: num_heads=%s not divisible by "
                "sp_world=%s.",
                h_global_total,
                sp_world,
            )
            _GQA_SKIP_LOGGED = True
        return

    owner_rank = (tgt.target_head //
                  heads_per_rank) if sp_world > 1 else 0
    if sp_rank != owner_rank:
        return
    h_local = (tgt.target_head -
               owner_rank * heads_per_rank) if sp_world > 1 else tgt.target_head
    if h_local < 0 or h_local >= h_local_total:
        return

    ts = 0
    if attn_metadata is not None and hasattr(attn_metadata, "current_timestep"):
        ts = int(attn_metadata.current_timestep)
    else:
        try:
            from fastvideo.forward_context import get_forward_context

            ts = int(get_forward_context().current_timestep)
        except Exception:
            ts = 0
    if ts != tgt.target_step:
        return

    block, block_idx = _layer_tag_from_prefix(prefix)
    if block != tgt.target_block or block_idx != tgt.target_block_index:
        return

    cfg_tag = "pos"
    try:
        from fastvideo.forward_context import get_forward_context

        fb = get_forward_context().forward_batch
        if fb is not None and getattr(fb, "is_cfg_negative", False):
            cfg_tag = "neg"
    except Exception:
        pass
    if tgt.target_cfg == "pos" and cfg_tag != "pos":
        return
    if tgt.target_cfg == "neg" and cfg_tag != "neg":
        return

    seq_len = query.shape[1]
    if tgt.slice_len <= 0:
        L = seq_len
    else:
        L = min(tgt.slice_len, seq_len)
    if L <= 0:
        return

    stored_b, peak_b = estimate_attention_matrix_bytes(L)
    logger.info(
        "Attention dump: writing one L×L matrix L=%d (~%.2f MiB stored, "
        "~%.2f MiB peak est.) head=%d step=%d %s[%d] cfg=%s",
        L,
        stored_b / (1024 * 1024),
        peak_b / (1024 * 1024),
        tgt.target_head,
        ts,
        tgt.target_block,
        tgt.target_block_index,
        cfg_tag,
    )

    os.makedirs(tgt.dump_dir, exist_ok=True)

    with torch.no_grad():
        q = query[:, :L, h_local:h_local + 1, :].float()
        k = key[:, :L, h_local:h_local + 1, :].float()
        scores = torch.einsum("bihd,bjhd->bhij", q, k) * float(softmax_scale)
        if causal:
            scores = scores.masked_fill(
                torch.triu(
                    torch.ones(L, L, device=scores.device, dtype=torch.bool),
                    diagonal=1,
                ).view(1, 1, L, L),
                float("-inf"),
            )
        attn = F.softmax(scores, dim=-1)
        w = attn.mean(dim=0)[0].half().cpu().numpy()

    fname = (f"step{ts:04d}_{tgt.target_block}_{tgt.target_block_index:03d}_"
             f"head{tgt.target_head:03d}_{cfg_tag}.npz")
    path = os.path.join(tgt.dump_dir, fname)
    meta = {
        "timestep_index": ts,
        "block": tgt.target_block,
        "block_index": tgt.target_block_index,
        "global_head": tgt.target_head,
        "prefix": prefix,
        "seq_len_materialized": L,
        "cfg": cfg_tag,
        "sp_rank": sp_rank,
    }
    np.savez_compressed(path, weights=w, meta=json.dumps(meta))
    logger.info("Wrote attention dump %s", path)
