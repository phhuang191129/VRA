#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot attention dumps written by ``attention_weight_dump`` (see
``FASTVIDEO_ATTENTION_DUMP_DIR``).

Examples::

    python utils/plot_attention.py --dump-dir outputs/attn \\
        --mode grid --head 0 --block double_blocks --cfg pos \\
        --out attention_mean_grid.png

    python utils/plot_attention.py --mode heatmap \\
        --npz outputs/attn/step0000_double_blocks_000_head000_pos.npz \\
        --out one_step.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    w = z["weights"]
    meta = json.loads(str(z["meta"].item()))
    return w, meta


def _entropy_rows(p: np.ndarray, eps: float = 1e-12) -> float:
    """Mean entropy over query rows (distribution over keys)."""
    p = np.clip(p, eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1)
    return float(np.mean(ent))


def _scalar(w: np.ndarray, head: int, metric: str) -> float:
    if w.ndim == 2:
        m = w
    else:
        h = min(head, w.shape[0] - 1)
        m = w[h]
    if metric == "mean":
        return float(np.mean(m))
    if metric == "entropy":
        return _entropy_rows(m)
    if metric == "diag":
        d = min(m.shape[0], m.shape[1])
        return float(np.mean(np.diag(m[:d, :d])))
    raise ValueError(f"Unknown metric {metric!r}")


def _parse_fname(path: str) -> dict[str, Any] | None:
    base = os.path.basename(path)
    m = re.match(
        r"step(\d+)_(double_blocks|single_blocks)_(\d+)_head(\d+)_(pos|neg)\.npz",
        base,
    )
    if m:
        return {
            "step": int(m.group(1)),
            "block": m.group(2),
            "block_index": int(m.group(3)),
            "head": int(m.group(4)),
            "cfg": m.group(5),
            "path": path,
        }
    m = re.match(
        r"step(\d+)_(double_blocks|single_blocks|other)_(\d+)_([0-9a-f]{8})_"
        r"(pos|neg)_sp(\d+)\.npz",
        base,
    )
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "block": m.group(2),
        "block_index": int(m.group(3)),
        "head": None,
        "cfg": m.group(5),
        "path": path,
    }


def run_grid(
    dump_dir: str,
    head: int,
    block: str,
    cfg: str,
    metric: str,
    out_path: str,
) -> None:
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    rows: list[tuple[int, int, str, float]] = []
    for p in paths:
        parsed = _parse_fname(p)
        if parsed is None:
            continue
        if block != "all" and parsed["block"] != block:
            continue
        if cfg != "both" and parsed["cfg"] != cfg:
            continue
        w, meta = _load_npz(p)
        bname = str(meta.get("block", parsed["block"]))
        bidx = int(meta.get("block_index", parsed["block_index"]))
        file_head = parsed.get("head")
        use_head = head if file_head is None else int(file_head)
        rows.append((parsed["step"], bidx, bname, _scalar(w, use_head, metric)))
    if not rows:
        raise SystemExit("No matching .npz files (check --dump-dir, --block, --cfg).")

    steps = sorted({r[0] for r in rows})
    layer_keys = sorted({(r[2], r[1]) for r in rows})
    step_index = {s: i for i, s in enumerate(steps)}
    layer_index = {k: j for j, k in enumerate(layer_keys)}
    grid = np.full((len(steps), len(layer_keys)), np.nan, dtype=np.float64)
    for step, bidx, bname, val in rows:
        grid[step_index[step], layer_index[(bname, bidx)]] = val

    fig, ax = plt.subplots(figsize=(max(8, len(layer_keys) * 0.35), 6))
    im = ax.imshow(grid.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Denoise step index")
    ax.set_ylabel("Layer (block_type, index)")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=7)
    ylab = [f"{a},{b}" for a, b in layer_keys]
    ax.set_yticks(range(len(layer_keys)))
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_title(f"{metric} (head={head})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_heatmap(npz_path: str, head: int, out_path: str) -> None:
    if any(ch in npz_path for ch in "*?["):
        paths = sorted(glob.glob(npz_path))
    else:
        paths = [npz_path] if os.path.isfile(npz_path) else []
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one .npz for --npz, got {paths!r}")
    w, meta = _load_npz(paths[0])
    if w.ndim == 2:
        mat = w
        h_disp = int(meta.get("global_head", head))
    else:
        h_disp = min(head, w.shape[0] - 1)
        mat = w[h_disp]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    ax.set_title(
        f"L×L attention head={h_disp} step={meta.get('timestep_index')} "
        f"{meta.get('block')}#{meta.get('block_index')}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=str, default="")
    ap.add_argument("--mode", choices=("grid", "heatmap"), default="grid")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--block", type=str, default="double_blocks")
    ap.add_argument("--cfg", type=str, default="pos", help="pos, neg, or both")
    ap.add_argument(
        "--metric",
        type=str,
        default="mean",
        choices=("mean", "entropy", "diag"),
        help="Scalar per (step, layer) for grid mode",
    )
    ap.add_argument("--npz", type=str, default="", help="Glob for one file (heatmap)")
    ap.add_argument("--out", type=str, default="attention_plot.png")
    args = ap.parse_args()

    if args.mode == "heatmap":
        if not args.npz:
            raise SystemExit("--npz is required for heatmap mode")
        run_heatmap(args.npz, args.head, args.out)
    else:
        if not args.dump_dir:
            raise SystemExit("--dump-dir is required for grid mode")
        run_grid(args.dump_dir, args.head, args.block, args.cfg, args.metric,
                 args.out)


if __name__ == "__main__":
    main()