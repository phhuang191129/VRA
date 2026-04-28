#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_ROOT))
PYTHON_ROOT = KERNEL_ROOT / "python"
if os.environ.get("FASTVIDEO_KERNEL_USE_INSTALLED") != "1" and str(
        PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from tests.support_vra import parse_shape, parse_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one VRA H100 kernel call under an external profiler.")
    parser.add_argument("--seq-shape", default="30x48x80")
    parser.add_argument("--tile-shape", default="6x8x8")
    parser.add_argument("--dense-radius", default="1,1,1")
    parser.add_argument("--mid-radius", default="2,2,3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--target",
        choices=["mixed", "fused_stride3", "packed_stride3"],
        default="mixed",
    )
    return parser.parse_args()


def profiler_start(torch) -> None:
    if os.environ.get("FASTVIDEO_VRA_DISABLE_CUDA_PROFILER_API") == "1":
        return
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        torch.cuda.profiler.start()


def profiler_stop(torch) -> None:
    if os.environ.get("FASTVIDEO_VRA_DISABLE_CUDA_PROFILER_API") == "1":
        return
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        torch.cuda.profiler.stop()


def main() -> None:
    import torch

    from fastvideo_kernel.ops import (
        _pack_kv_by_stride,
        mixed_vra_attention_h100,
        packed_attn_h100,
        stride3_attn_h100,
    )

    args = parse_args()
    shape = parse_shape(args.seq_shape, args.tile_shape)
    dense_radius = parse_3d(args.dense_radius)
    mid_radius = parse_3d(args.mid_radius)

    torch.manual_seed(0)
    q = torch.randn(
        args.batch_size,
        args.num_heads,
        shape.img_tokens,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if args.target == "mixed":
        def fn():
            return mixed_vra_attention_h100(
                q,
                k,
                v,
                dense_radius=dense_radius,
                mid_radius=mid_radius,
                text_length=0,
                has_text=False,
                seq_shape=args.seq_shape,
            )
    elif args.target == "fused_stride3":
        def fn():
            return stride3_attn_h100(q, k, v)
    else:
        k_packed = _pack_kv_by_stride(k, shape.img_tokens, 0, 3)
        v_packed = _pack_kv_by_stride(v, shape.img_tokens, 0, 3)

        def fn():
            return packed_attn_h100(q, k_packed, v_packed)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    profiler_start(torch)
    out = fn()
    torch.cuda.synchronize()
    profiler_stop(torch)

    print(
        f"profiled target={args.target} shape={args.seq_shape} "
        f"device={torch.cuda.get_device_name(0)} checksum={out.float().mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
