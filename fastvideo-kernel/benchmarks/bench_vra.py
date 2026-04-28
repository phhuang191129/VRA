#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from tests.support_vra import count_vra_pattern, parse_3d, parse_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VRA pattern counter and benchmark scaffold.")
    parser.add_argument(
        "--seq-shape",
        nargs="+",
        default=["30x48x80", "18x48x80"],
        help="Video token canvas shape(s), e.g. 30x48x80 or 18x48x80.",
    )
    parser.add_argument(
        "--tile-shape",
        default="6x8x8",
        help="Tile shape used by the kernel layout.",
    )
    parser.add_argument(
        "--dense-radius",
        default="1,1,1",
        help="Tile-distance radius for dense gray zone.",
    )
    parser.add_argument(
        "--mid-radius",
        default="2,2,3",
        help="Tile-distance radius for stride-2 yellow zone.",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=0,
        help="Text token count. Text is counted as dense by default.",
    )
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--include-sdpa", action="store_true")
    parser.add_argument(
        "--include-packed-stride3-sdpa",
        action="store_true",
        help="Benchmark q against K/V packed to every third image token. "
        "This is an upper-bound baseline for the all-stride-3 VRA path.",
    )
    parser.add_argument(
        "--include-packed-stride3-h100",
        action="store_true",
        help="Benchmark the experimental TK/H100 dense attention kernel over packed stride-3 K/V.",
    )
    parser.add_argument(
        "--include-fused-stride3-h100",
        action="store_true",
        help="Benchmark the experimental TK/H100 kernel that gathers stride-3 K/V rows inside the producer.",
    )
    parser.add_argument(
        "--include-mixed-vra-h100",
        action="store_true",
        help="Benchmark the experimental TK/H100 kernel that uses the mixed dense/stride-2/stride-3 VRA schedule.",
    )
    parser.add_argument(
        "--include-pack-timing",
        action="store_true",
        help="When packed stride-3 SDPA is enabled, also time K/V packing. Uses the native VRA pack extension when available, otherwise torch index_select.",
    )
    parser.add_argument("--max-sdpa-tokens", type=int, default=4096)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def attention_tflop_pairs(pairs: int, batch: int, heads: int,
                          head_dim: int) -> float:
    # QK and PV each cost roughly 2 FLOPs per multiply-add.
    return 4.0 * batch * heads * head_dim * pairs / 1e12


def format_int(value: int) -> str:
    return f"{value:,}"


def cuda_bench_ms(fn, warmup: int, rep: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def run_gpu_bench(args: argparse.Namespace, seq_shape: str,
                  dense_radius: tuple[int, int, int],
                  mid_radius: tuple[int, int, int],
                  row: dict[str, object]) -> None:
    import torch
    import torch.nn.functional as F

    from fastvideo_kernel import (
        has_native_vra_pack,
        has_mixed_vra_attn_h100,
        has_packed_attn_h100,
        has_stride3_attn_h100,
        variable_rate_attention,
    )
    from fastvideo_kernel.ops import _pack_kv_by_stride

    shape = parse_shape(seq_shape, args.tile_shape)
    seq_len = shape.img_tokens + args.text_length
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_name = torch.cuda.get_device_name(0)
    print(f"gpu_device={device_name}")
    print(f"native_vra_pack={has_native_vra_pack()}")
    print(f"packed_attn_h100={has_packed_attn_h100()}")
    print(f"stride3_attn_h100={has_stride3_attn_h100()}")
    print(f"mixed_vra_attn_h100={has_mixed_vra_attn_h100()}")

    torch.manual_seed(0)
    q = torch.randn(args.batch_size,
                    args.num_heads,
                    seq_len,
                    args.head_dim,
                    device="cuda",
                    dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def vra_fn():
        return variable_rate_attention(
            q,
            k,
            v,
            dense_radius=dense_radius,
            mid_radius=mid_radius,
            text_length=args.text_length,
            has_text=args.text_length > 0,
            seq_shape=seq_shape,
        )

    vra_ms = cuda_bench_ms(vra_fn, args.warmup, args.rep)
    selected_tflop = float(row["selected_tflop"])
    dense_equiv_tflop = float(row["dense_equiv_tflop"])
    row["vra_triton_ms"] = vra_ms
    row["vra_selected_tflops_per_s"] = selected_tflop / (vra_ms / 1e3)
    row["vra_dense_equiv_tflops_per_s"] = dense_equiv_tflop / (vra_ms / 1e3)
    print(
        "vra_triton: "
        f"{vra_ms:.3f} ms, "
        f"selected={row['vra_selected_tflops_per_s']:.2f} TFLOP/s, "
        f"dense_equiv={row['vra_dense_equiv_tflops_per_s']:.2f} TFLOP/s")

    if args.include_sdpa:
        if seq_len > args.max_sdpa_tokens:
            print(
                f"sdpa: skipped because seq_len={seq_len} > max_sdpa_tokens={args.max_sdpa_tokens}"
            )
        else:
            def sdpa_fn():
                return F.scaled_dot_product_attention(q, k, v)

            sdpa_ms = cuda_bench_ms(sdpa_fn, args.warmup, args.rep)
            row["sdpa_ms"] = sdpa_ms
            row["speedup_vs_sdpa"] = sdpa_ms / vra_ms
            print(
                f"sdpa: {sdpa_ms:.3f} ms, speedup_vs_sdpa={sdpa_ms / vra_ms:.2f}x"
            )

    if args.include_mixed_vra_h100:
        from fastvideo_kernel.ops import mixed_vra_attention_h100

        if not has_mixed_vra_attn_h100() and args.text_length == 0:
            print("mixed_vra_h100: skipped because extension is unavailable")
        else:
            if not has_mixed_vra_attn_h100():
                print(
                    "mixed_vra_h100: extension unavailable; using Triton text fallback"
                )

            def mixed_h100_fn():
                return mixed_vra_attention_h100(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    dense_radius=dense_radius,
                    mid_radius=mid_radius,
                    text_length=args.text_length,
                    has_text=args.text_length > 0,
                    seq_shape=seq_shape,
                )

            mixed_ms = cuda_bench_ms(mixed_h100_fn, args.warmup, args.rep)
            row["mixed_vra_h100_ms"] = mixed_ms
            row["mixed_vra_h100_tflops_per_s"] = selected_tflop / (
                mixed_ms / 1e3)
            h100_text_mode = os.environ.get("FASTVIDEO_VRA_H100_TEXT_MODE",
                                            "auto")
            if args.text_length:
                row["mixed_vra_h100_text_mode"] = h100_text_mode
            if args.text_length == 0:
                label = "mixed_vra_h100"
            elif has_mixed_vra_attn_h100():
                label = "mixed_vra_h100_text"
            else:
                label = "mixed_vra_h100_fallback_triton_text"
            print(
                f"{label}: "
                f"{mixed_ms:.3f} ms, "
                f"throughput={row['mixed_vra_h100_tflops_per_s']:.2f} TFLOP/s, "
                f"speedup_vs_vra_triton={vra_ms / mixed_ms:.2f}x"
                + (f", text_mode={h100_text_mode}" if args.text_length else ""))

    if (args.include_packed_stride3_sdpa or args.include_packed_stride3_h100
            or args.include_fused_stride3_h100):
        packed_text_length = args.text_length
        packed_kv_tokens = (shape.img_tokens + 2) // 3 + packed_text_length
        k_packed = _pack_kv_by_stride(k.contiguous(), shape.img_tokens,
                                      packed_text_length, 3)
        v_packed = _pack_kv_by_stride(v.contiguous(), shape.img_tokens,
                                      packed_text_length, 3)
        packed_pairs = shape.img_tokens * packed_kv_tokens
        if args.text_length:
            packed_pairs += args.text_length * seq_len
        packed_tflop = attention_tflop_pairs(
            packed_pairs,
            args.batch_size,
            args.num_heads,
            args.head_dim,
        )

        row["packed_stride3_kv_tokens"] = packed_kv_tokens
        row["native_vra_pack"] = has_native_vra_pack()

        if args.include_packed_stride3_sdpa:
            def packed_sdpa_fn():
                if args.text_length:
                    image_out = F.scaled_dot_product_attention(
                        q[:, :, :shape.img_tokens], k_packed, v_packed)
                    text_out = F.scaled_dot_product_attention(
                        q[:, :, shape.img_tokens:], k, v)
                    return torch.cat([image_out, text_out], dim=2)
                return F.scaled_dot_product_attention(q, k_packed, v_packed)

            packed_ms = cuda_bench_ms(packed_sdpa_fn, args.warmup, args.rep)
            row["packed_stride3_sdpa_ms"] = packed_ms
            row["packed_stride3_tflops_per_s"] = packed_tflop / (packed_ms / 1e3)
            print(
                "packed_stride3_sdpa: "
                f"{packed_ms:.3f} ms, "
                f"kv_tokens={packed_kv_tokens}, "
                f"throughput={row['packed_stride3_tflops_per_s']:.2f} TFLOP/s, "
                f"speedup_vs_vra_triton={vra_ms / packed_ms:.2f}x")

        if args.include_packed_stride3_h100:
            from fastvideo_kernel.ops import packed_attn_h100

            if packed_attn_h100 is None:
                print("packed_stride3_h100: skipped because extension is unavailable")
            elif args.text_length:
                print("packed_stride3_h100: skipped because text path is not implemented")
            else:
                def packed_h100_fn():
                    return packed_attn_h100(q.contiguous(), k_packed, v_packed)

                h100_ms = cuda_bench_ms(packed_h100_fn, args.warmup, args.rep)
                row["packed_stride3_h100_ms"] = h100_ms
                row["packed_stride3_h100_tflops_per_s"] = packed_tflop / (
                    h100_ms / 1e3)
                print(
                    "packed_stride3_h100: "
                    f"{h100_ms:.3f} ms, "
                    f"throughput={row['packed_stride3_h100_tflops_per_s']:.2f} TFLOP/s, "
                    f"speedup_vs_vra_triton={vra_ms / h100_ms:.2f}x")

        if args.include_fused_stride3_h100:
            from fastvideo_kernel.ops import stride3_attn_h100

            if stride3_attn_h100 is None:
                print("fused_stride3_h100: skipped because extension is unavailable")
            elif args.text_length:
                print("fused_stride3_h100: skipped because text path is not implemented")
            else:
                def fused_h100_fn():
                    return stride3_attn_h100(q.contiguous(), k.contiguous(),
                                             v.contiguous())

                fused_ms = cuda_bench_ms(fused_h100_fn, args.warmup, args.rep)
                row["fused_stride3_h100_ms"] = fused_ms
                row["fused_stride3_h100_tflops_per_s"] = packed_tflop / (
                    fused_ms / 1e3)
                print(
                    "fused_stride3_h100: "
                    f"{fused_ms:.3f} ms, "
                    f"throughput={row['fused_stride3_h100_tflops_per_s']:.2f} TFLOP/s, "
                    f"speedup_vs_vra_triton={vra_ms / fused_ms:.2f}x")

        if args.include_pack_timing:
            def pack_fn():
                kp = _pack_kv_by_stride(k.contiguous(), shape.img_tokens,
                                        packed_text_length, 3)
                vp = _pack_kv_by_stride(v.contiguous(), shape.img_tokens,
                                        packed_text_length, 3)
                return kp, vp

            pack_ms = cuda_bench_ms(pack_fn, args.warmup, args.rep)
            row["packed_stride3_pack_ms"] = pack_ms
            if "packed_ms" in locals():
                row["packed_stride3_pack_plus_sdpa_ms"] = pack_ms + packed_ms
                print(
                    "packed_stride3_pack: "
                    f"{pack_ms:.3f} ms, "
                    f"pack_plus_sdpa={pack_ms + packed_ms:.3f} ms")
            else:
                print(f"packed_stride3_pack: {pack_ms:.3f} ms")

    del q, k, v
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    dense_radius = parse_3d(args.dense_radius)
    mid_radius = parse_3d(args.mid_radius)

    rows: list[dict[str, object]] = []
    print("VRA Pattern Counter")
    print(f"tile_shape={args.tile_shape}")
    print(f"dense_radius={dense_radius}, mid_radius={mid_radius}")
    print(f"text_length={args.text_length}, phase={args.phase}")
    print(
        f"batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}"
    )

    for seq_shape in args.seq_shape:
        shape = parse_shape(seq_shape, args.tile_shape)
        count = count_vra_pattern(
            shape,
            dense_radius=dense_radius,
            mid_radius=mid_radius,
            text_length=args.text_length,
            phase=args.phase,
        )
        total_ratio = count.total_pairs / max(1, count.total_dense_pairs)
        dense_tflop = attention_tflop_pairs(
            count.total_dense_pairs,
            args.batch_size,
            args.num_heads,
            args.head_dim,
        )
        selected_tflop = attention_tflop_pairs(
            count.total_pairs,
            args.batch_size,
            args.num_heads,
            args.head_dim,
        )

        row = {
            "seq_shape": seq_shape,
            "tile_grid": "x".join(map(str, shape.tile_grid)),
            "img_tokens": shape.img_tokens,
            "tiles": shape.num_tiles,
            "avg_dense_tiles_per_query": count.dense_tiles /
            count.query_tiles,
            "avg_mid_tiles_per_query": count.mid_tiles / count.query_tiles,
            "avg_far_tiles_per_query": count.far_tiles / count.query_tiles,
            "avg_selected_img_rows_per_query":
            count.selected_image_rows_per_query_avg,
            "image_selected_ratio": count.image_selected_ratio,
            "total_selected_ratio": total_ratio,
            "dense_equiv_tflop": dense_tflop,
            "selected_tflop": selected_tflop,
            "theoretical_pair_speedup": 1.0 / max(total_ratio, 1e-12),
        }
        rows.append(row)

        print("\n" + "=" * 88)
        print(f"shape={seq_shape}  tile_grid={row['tile_grid']}")
        print(f"image_tokens={format_int(shape.img_tokens)}  tiles={shape.num_tiles}")
        print(
            "avg tiles/query: "
            f"dense={row['avg_dense_tiles_per_query']:.2f}, "
            f"mid={row['avg_mid_tiles_per_query']:.2f}, "
            f"far={row['avg_far_tiles_per_query']:.2f}")
        print(
            "avg selected image K rows/query: "
            f"{row['avg_selected_img_rows_per_query']:.1f} / {shape.img_tokens}"
        )
        print(
            "selected ratio: "
            f"image={row['image_selected_ratio']:.4f}, "
            f"total_with_text={row['total_selected_ratio']:.4f}")
        print(
            "attention FLOPs: "
            f"dense_equiv={row['dense_equiv_tflop']:.2f} TFLOP, "
            f"selected={row['selected_tflop']:.2f} TFLOP, "
            f"pair_speedup_bound={row['theoretical_pair_speedup']:.2f}x")
        if args.run_gpu:
            run_gpu_bench(args, seq_shape, dense_radius, mid_radius, row)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
