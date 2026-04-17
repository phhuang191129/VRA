#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local-window attention recall from ``attention_weight_dump`` .npz files.

For each **video** query row, computes the fraction of softmax mass on **video**
key positions whose (t, h, w) patch-grid coordinates lie inside an axis-aligned
box of sizes ``(wt, wh, ww)`` centered on the query token (clipped at grid
bounds). Matches the usual "recall@local window" style scalar (one number per
layer/step/head), comparable to a single cell in step×layer grids.

Sequence layout matches Hunyuan ``DistributedAttention``: **image tokens first**,
then text. Flat index for video tokens follows ``unpatchify`` / Conv3D patch
order: ``idx = t * (th*tw) + h * tw + w`` with ``w`` fastest.

Examples::

    python utils/attention_local_recall.py \\
        --npz local_attn/step0000_double_blocks_000_head000_pos.npz \\
        --num-frames 75 --height 544 --width 960 \\
        --window 12,24,24

    python utils/attention_local_recall.py --dump-dir ./attn --window 12,24,24 \\
        --num-frames 75 --height 544 --width 960 --pattern 'step*_double_*_pos.npz'
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np


def _load_npz(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    w = np.asarray(z["weights"], dtype=np.float64)
    meta = json.loads(str(z["meta"].item()))
    return w, meta


def hunyuan_video_token_grid_dims(
    num_frames: int,
    height_px: int,
    width_px: int,
    *,
    spatial_compression: int = 8,
    temporal_compression: int = 4,
    patch_t: int = 1,
    patch_hw: int = 2,
) -> tuple[int, int, int, int]:
    """Return ``(tt, th, tw, n_img)`` patch-grid sizes and video token count."""
    latent_t = (int(num_frames) - 1) // int(temporal_compression) + 1
    latent_h = int(height_px) // int(spatial_compression)
    latent_w = int(width_px) // int(spatial_compression)
    tt = latent_t // patch_t
    th = latent_h // patch_hw
    tw = latent_w // patch_hw
    n_img = tt * th * tw
    return tt, th, tw, n_img


def _flat_to_thw(flat: int, tw: int, th: int) -> tuple[int, int, int]:
    w_i = flat % tw
    h_i = (flat // tw) % th
    t_i = flat // (th * tw)
    return t_i, h_i, w_i


def _window_axis_bounds(center: int, window: int, dim: int) -> tuple[int, int]:
    """Inclusive low, exclusive high along one axis."""
    left = (window - 1) // 2
    right = window // 2
    lo = max(0, center - left)
    hi = min(dim, center + right + 1)
    return lo, hi


def local_recall_for_matrix(
    weights: np.ndarray,
    *,
    tt: int,
    th: int,
    tw: int,
    n_img: int,
    window_t: int,
    window_h: int,
    window_w: int,
    only_video_queries: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-query recall vector (video query rows only) and summary stats."""
    if weights.ndim != 2:
        raise ValueError(f"Expected weights [L,L], got shape {weights.shape}")
    l = weights.shape[0]
    if weights.shape[1] != l:
        raise ValueError("Expected square attention matrix")
    if l == 0:
        raise ValueError("Empty attention matrix")

    n_img_eff = min(n_img, l)
    q_hi = n_img_eff if only_video_queries else l
    recalls: list[float] = []

    for qi in range(q_hi):
        row = weights[qi]
        tq, hq, wq = _flat_to_thw(qi, tw, th)
        t0, t1 = _window_axis_bounds(tq, window_t, tt)
        h0, h1 = _window_axis_bounds(hq, window_h, th)
        w0, w1 = _window_axis_bounds(wq, window_w, tw)

        s = 0.0
        for tk in range(t0, t1):
            base_t = tk * (th * tw)
            for hk in range(h0, h1):
                row_h = base_t + hk * tw
                for wk in range(w0, w1):
                    kj = row_h + wk
                    if kj < n_img_eff:
                        s += float(row[kj])
        recalls.append(s)

    arr = np.asarray(recalls, dtype=np.float64)
    summary = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_queries": int(len(arr)),
        "L": int(l),
        "n_img_tokens_grid": int(n_img),
        "n_img_effective": int(n_img_eff),
        "window": [window_t, window_h, window_w],
        "grid_tt_th_tw": [tt, th, tw],
    }
    return arr, summary


def _parse_fname(path: str) -> dict[str, Any] | None:
    base = os.path.basename(path)
    m = re.match(
        r"step(\d+)_(double_blocks|single_blocks)_(\d+)_head(\d+)_(pos|neg)\.npz",
        base,
    )
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "block": m.group(2),
        "block_index": int(m.group(3)),
        "head": int(m.group(4)),
        "cfg": m.group(5),
        "path": path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attention mass in (T,H,W) local window (Hunyuan dumps).")
    ap.add_argument("--npz", type=str, default="", help="Single .npz path")
    ap.add_argument("--dump-dir", type=str, default="")
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern under --dump-dir (fnmatch on basename)",
    )
    ap.add_argument(
        "--window",
        type=str,
        default="12,24,24",
        help="Comma-separated window sizes along patch (t,h,w)",
    )
    ap.add_argument("--num-frames", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument(
        "--n-img-tokens",
        type=int,
        default=0,
        help="Override video token count (0 = from resolution formula)",
    )
    ap.add_argument(
        "--include-text-queries",
        action="store_true",
        help="Also score text query rows (window undefined; uses same box in "
        "video index space — usually not meaningful)",
    )
    ap.add_argument(
        "--per-query-npz",
        action="store_true",
        help="When scanning a directory, write <stem>_recall.npz per file",
    )
    args = ap.parse_args()

    try:
        wt, wh, ww = (int(x.strip()) for x in args.window.split(","))
    except ValueError as e:
        raise SystemExit("--window must be like 12,24,24") from e

    tt, th, tw, n_img = hunyuan_video_token_grid_dims(
        args.num_frames,
        args.height,
        args.width,
    )
    if args.n_img_tokens > 0:
        n_img = int(args.n_img_tokens)

    only_video = not args.include_text_queries

    def one_file(path: str) -> None:
        w, meta = _load_npz(path)
        arr, summary = local_recall_for_matrix(
            w,
            tt=tt,
            th=th,
            tw=tw,
            n_img=n_img,
            window_t=wt,
            window_h=wh,
            window_w=ww,
            only_video_queries=only_video,
        )
        summary["file"] = path
        summary["meta"] = meta
        print(json.dumps(summary, indent=2))
        if args.per_query_npz:
            out = path.replace(".npz", "_recall.npz")
            if out == path:
                out = path + ".recall.npz"
            np.savez_compressed(
                out,
                per_query_recall=arr.astype(np.float32),
                summary=json.dumps(summary),
            )
            print(f"wrote {out}", file=sys.stderr)

    if args.npz:
        if not os.path.isfile(args.npz):
            raise SystemExit(f"not a file: {args.npz!r}")
        one_file(args.npz)
        return

    if not args.dump_dir:
        raise SystemExit("Provide --npz or --dump-dir")

    paths = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    paths = [p for p in paths if fnmatch.fnmatch(os.path.basename(p), args.pattern)]
    if not paths:
        raise SystemExit("No matching .npz files")

    for p in paths:
        parsed = _parse_fname(p)
        label = parsed["path"] if parsed else p
        print(f"=== {label} ===", file=sys.stderr)
        one_file(p)


if __name__ == "__main__":
    main()
