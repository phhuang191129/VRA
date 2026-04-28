import math
import os
import torch
import torch.nn.functional as F
from .block_sparse_attn import block_sparse_attn
from .triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward
from .triton_kernels.st_attn_triton import sliding_tile_attention_triton
from .triton_kernels.vra_attn_triton import sliding_variable_rate_attention_triton
try:
    from .triton_kernels.vra_attn_triton import variable_rate_attention_triton
except ImportError:
    variable_rate_attention_triton = None
from .triton_kernels.index import map_to_index

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    sta_fwd = getattr(fastvideo_kernel_ops, "sta_fwd", None)
    block_sparse_fwd = getattr(fastvideo_kernel_ops, "block_sparse_fwd", None)
    block_sparse_bwd = getattr(fastvideo_kernel_ops, "block_sparse_bwd", None)
    vra_pack_kv = getattr(fastvideo_kernel_ops, "vra_pack_kv", None)
    packed_attn_h100 = getattr(fastvideo_kernel_ops, "packed_attn_h100", None)
    dense_attn_h100_valid_kv = getattr(fastvideo_kernel_ops,
                                       "dense_attn_h100_valid_kv", None)
    stride3_attn_h100 = getattr(fastvideo_kernel_ops, "stride3_attn_h100", None)
    mixed_vra_attn_h100 = getattr(fastvideo_kernel_ops, "mixed_vra_attn_h100",
                                  None)
except ImportError:
    sta_fwd = None
    block_sparse_fwd = None
    block_sparse_bwd = None
    vra_pack_kv = None
    packed_attn_h100 = None
    dense_attn_h100_valid_kv = None
    stride3_attn_h100 = None
    mixed_vra_attn_h100 = None

def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: list,
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    try:
        from .block_sparse_attn import _is_sm90
        is_sm90 = _is_sm90()
    except ImportError:
        is_sm90 = False

    force_triton = os.environ.get("FASTVIDEO_STA_FORCE_TRITON", "0") == "1"

    # Check if the specific op is available
    if force_triton or sta_fwd is None or not is_sm90:
        return sliding_tile_attention_triton(
            q, k, v, window_size, text_length, has_text, seq_shape
        )

    seq_length = q.shape[2]
    shape_map = {"30x48x80": 1, "36x48x48": 2, "18x48x80": 3}

    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)

    output = torch.empty_like(q)
    flag = shape_map[seq_shape]

    for head_idx, (t, h, w) in enumerate(window_size):
        # Per-head slices are not contiguous in the batch dimension when batch>1
        # (they keep the original head-stride). The TK kernel assumes contiguous
        # [B, H, S, D] layout, so we materialize a contiguous [B,1,S,D] view.
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        sta_fwd(
            q_h, k_h,
            v_h, o_h,
            t, h, w, text_length, False, has_text, flag
        )
        output[:, head_idx:head_idx + 1] = o_h

    if has_text:
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output, 3, 3, 3, text_length, True, True, flag)

    return output[:, :, :seq_length]


def video_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: torch.Tensor = None,
) -> torch.Tensor:
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if v.shape[2] != kv_seq_len:
        raise ValueError(
            f"Expected k and v to have the same sequence length, got "
            f"k.shape[2]={kv_seq_len}, v.shape[2]={v.shape[2]}"
        )
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("Expected q/k/v to have the same batch and head dimensions.")

    if q_seq_len % block_elements != 0 or kv_seq_len % block_elements != 0:
        raise ValueError(
            f"q_seq_len and kv_seq_len must be divisible by block_elements={block_elements}, "
            f"got q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}"
        )
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements

    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length kv_num_blocks={kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )

    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length q_num_blocks={q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )

    # Compression branch
    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)

    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(
        q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(
        v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)

    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements,
                         1).view(batch, heads, q_seq_len, dim)

    # Sparse branch
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores,
                            dtype=torch.bool).scatter_(-1, topk_idx, True)

    idx, num = map_to_index(mask)

    if block_sparse_fwd is not None:
        # Use autograd-enabled wrapper so backward works (and still uses SM90 kernel when available)
        out_s = block_sparse_attn(q, k, v, mask, variable_block_sizes)[0]
    else:
        # Triton-only forward (kept for environments without the wrapper deps)
        out_s, _ = triton_block_sparse_attn_forward(q, k, v, idx, num, variable_block_sizes)

    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s
    return out_c + out_s


def variable_rate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: list | None = None,
    text_length: int = 0,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
    dense_radius: list[int] | tuple[int, int, int] | None = None,
    mid_radius: list[int] | tuple[int, int, int] | None = None,
) -> torch.Tensor:
    if window_size is not None:
        return sliding_variable_rate_attention_triton(
            q, k, v, window_size, text_length, has_text, seq_shape)
    if dense_radius is None or mid_radius is None:
        raise ValueError(
            "variable_rate_attention requires either window_size or "
            "dense_radius/mid_radius")
    if variable_rate_attention_triton is None:
        raise RuntimeError(
            "dense_radius/mid_radius VRA Triton fallback is unavailable; "
            "use the window_size API or native H100 path")
    return variable_rate_attention_triton(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        has_text=has_text,
        seq_shape=seq_shape,
    )


def has_native_vra_pack() -> bool:
    return vra_pack_kv is not None


def has_packed_attn_h100() -> bool:
    return packed_attn_h100 is not None


def has_stride3_attn_h100() -> bool:
    return stride3_attn_h100 is not None


def has_mixed_vra_attn_h100() -> bool:
    return mixed_vra_attn_h100 is not None


def _pack_kv_by_stride(
    x: torch.Tensor,
    img_seq_len: int,
    text_length: int,
    stride: int,
) -> torch.Tensor:
    if vra_pack_kv is not None and x.is_cuda and x.is_contiguous():
        return vra_pack_kv(x, img_seq_len, text_length, stride)

    image_idx = torch.arange(0, img_seq_len, stride, device=x.device)
    if text_length:
        text_idx = torch.arange(img_seq_len,
                                img_seq_len + text_length,
                                device=x.device)
        kv_idx = torch.cat([image_idx, text_idx])
    else:
        kv_idx = image_idx
    return x.index_select(2, kv_idx).contiguous()


def packed_stride_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    stride: int = 3,
    text_length: int = 0,
    has_text: bool = False,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    shape = tuple(int(part) for part in seq_shape.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"seq_shape must be T x H x W, got {seq_shape!r}")
    img_seq_len = shape[0] * shape[1] * shape[2]
    expected_seq_len = img_seq_len + (text_length if has_text else 0)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.shape[2] != expected_seq_len:
        raise ValueError(
            f"expected seq_len={expected_seq_len}, got {q.shape[2]}")

    packed_text_length = text_length if has_text else 0
    k_packed = _pack_kv_by_stride(k.contiguous(), img_seq_len,
                                  packed_text_length, stride)
    v_packed = _pack_kv_by_stride(v.contiguous(), img_seq_len,
                                  packed_text_length, stride)
    if not has_text or text_length == 0:
        return F.scaled_dot_product_attention(q, k_packed, v_packed)

    image_out = F.scaled_dot_product_attention(q[:, :, :img_seq_len],
                                               k_packed, v_packed)
    text_out = F.scaled_dot_product_attention(q[:, :, img_seq_len:], k, v)
    return torch.cat([image_out, text_out], dim=2)


def packed_stride_attention_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    stride: int = 3,
    text_length: int = 0,
    has_text: bool = False,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    if packed_attn_h100 is None:
        raise RuntimeError("packed_attn_h100 extension is not available")
    if has_text or text_length:
        raise NotImplementedError(
            "packed_stride_attention_h100 currently supports image-only inputs")

    shape = tuple(int(part) for part in seq_shape.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"seq_shape must be T x H x W, got {seq_shape!r}")
    img_seq_len = shape[0] * shape[1] * shape[2]
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.shape[2] != img_seq_len:
        raise ValueError(
            f"expected image seq_len={img_seq_len}, got {q.shape[2]}")

    k_packed = _pack_kv_by_stride(k.contiguous(), img_seq_len, 0, stride)
    v_packed = _pack_kv_by_stride(v.contiguous(), img_seq_len, 0, stride)
    return packed_attn_h100(q.contiguous(), k_packed, v_packed)


def fused_stride3_attention_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    text_length: int = 0,
    has_text: bool = False,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    if stride3_attn_h100 is None:
        raise RuntimeError("stride3_attn_h100 extension is not available")
    if has_text or text_length:
        raise NotImplementedError(
            "fused_stride3_attention_h100 currently supports image-only inputs")

    shape = tuple(int(part) for part in seq_shape.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"seq_shape must be T x H x W, got {seq_shape!r}")
    img_seq_len = shape[0] * shape[1] * shape[2]
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.shape[2] != img_seq_len:
        raise ValueError(
            f"expected image seq_len={img_seq_len}, got {q.shape[2]}")

    return stride3_attn_h100(q.contiguous(), k.contiguous(), v.contiguous())


def _parse_vra_seq_shape(seq_shape: str) -> tuple[int, int, int]:
    shape = tuple(int(part) for part in seq_shape.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"seq_shape must be T x H x W, got {seq_shape!r}")
    return shape  # type: ignore[return-value]


def _tile_grid_for_seq_shape(seq_shape: str) -> tuple[int, int, int]:
    t, h, w = _parse_vra_seq_shape(seq_shape)
    if t % 6 or h % 8 or w % 8:
        raise ValueError(
            f"seq_shape={seq_shape!r} must be divisible by tile shape 6x8x8")
    return t // 6, h // 8, w // 8


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_sequence_rows(x: torch.Tensor, target_rows: int) -> torch.Tensor:
    if x.shape[2] == target_rows:
        return x.contiguous()
    if x.shape[2] > target_rows:
        raise ValueError(
            f"cannot pad sequence with rows={x.shape[2]} down to {target_rows}")
    pad_rows = target_rows - x.shape[2]
    padding = x.new_zeros(*x.shape[:2], pad_rows, x.shape[3])
    return torch.cat([x.contiguous(), padding], dim=2)


def _h100_text_mode(grid_t: int) -> str:
    mode = os.environ.get("FASTVIDEO_VRA_H100_TEXT_MODE", "auto").lower()
    if mode not in ("auto", "fused", "separate"):
        raise ValueError(
            "FASTVIDEO_VRA_H100_TEXT_MODE must be one of auto/fused/separate")
    if mode != "auto":
        return mode
    if grid_t >= 5 and dense_attn_h100_valid_kv is not None:
        return "separate"
    return "fused"


def mixed_vra_attention_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_radius: list[int] | tuple[int, int, int],
    mid_radius: list[int] | tuple[int, int, int],
    text_length: int = 0,
    has_text: bool = False,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    if has_text or text_length:
        if mixed_vra_attn_h100 is None:
            raise RuntimeError("mixed_vra_attn_h100 extension is not available")
        if len(dense_radius) != 3 or len(mid_radius) != 3:
            raise ValueError(
                "dense_radius and mid_radius must be length-3 values")

        shape = _parse_vra_seq_shape(seq_shape)
        img_seq_len = shape[0] * shape[1] * shape[2]
        expected_seq_len = img_seq_len + text_length
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical shapes")
        if q.shape[2] != expected_seq_len:
            raise ValueError(
                f"expected seq_len={expected_seq_len}, got {q.shape[2]}")

        grid_t, grid_h, grid_w = _tile_grid_for_seq_shape(seq_shape)
        padded_text_len = _ceil_to_multiple(text_length, 128)
        padded_kv_rows = img_seq_len + padded_text_len
        k_padded = _pad_sequence_rows(k, padded_kv_rows)
        v_padded = _pad_sequence_rows(v, padded_kv_rows)

        text_mode = _h100_text_mode(grid_t)
        if text_mode == "separate":
            if dense_attn_h100_valid_kv is None:
                raise RuntimeError(
                    "FASTVIDEO_VRA_H100_TEXT_MODE=separate requires "
                    "dense_attn_h100_valid_kv")
            image_out = mixed_vra_attn_h100(
                q[:, :, :img_seq_len].contiguous(),
                k_padded,
                v_padded,
                int(dense_radius[0]),
                int(dense_radius[1]),
                int(dense_radius[2]),
                int(mid_radius[0]),
                int(mid_radius[1]),
                int(mid_radius[2]),
                grid_t,
                grid_h,
                grid_w,
                int(text_length),
            )
            padded_text_q_rows = _ceil_to_multiple(text_length, 64)
            q_text_padded = _pad_sequence_rows(
                q[:, :, img_seq_len:].contiguous(), padded_text_q_rows)
            text_out_padded = dense_attn_h100_valid_kv(
                q_text_padded,
                k_padded,
                v_padded,
                expected_seq_len,
            )
            return torch.cat(
                [image_out, text_out_padded[:, :, :text_length]], dim=2)

        padded_q_rows = img_seq_len + _ceil_to_multiple(text_length, 64)
        q_padded = _pad_sequence_rows(q, padded_q_rows)
        out_padded = mixed_vra_attn_h100(
            q_padded,
            k_padded,
            v_padded,
            int(dense_radius[0]),
            int(dense_radius[1]),
            int(dense_radius[2]),
            int(mid_radius[0]),
            int(mid_radius[1]),
            int(mid_radius[2]),
            grid_t,
            grid_h,
            grid_w,
            int(text_length),
        )
        return out_padded[:, :, :expected_seq_len].contiguous()

    if mixed_vra_attn_h100 is None:
        raise RuntimeError("mixed_vra_attn_h100 extension is not available")
    if len(dense_radius) != 3 or len(mid_radius) != 3:
        raise ValueError("dense_radius and mid_radius must be length-3 values")

    shape = _parse_vra_seq_shape(seq_shape)
    img_seq_len = shape[0] * shape[1] * shape[2]
    grid_t, grid_h, grid_w = _tile_grid_for_seq_shape(seq_shape)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.shape[2] != img_seq_len:
        raise ValueError(
            f"expected image seq_len={img_seq_len}, got {q.shape[2]}")

    return mixed_vra_attn_h100(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        int(dense_radius[0]),
        int(dense_radius[1]),
        int(dense_radius[2]),
        int(mid_radius[0]),
        int(mid_radius[1]),
        int(mid_radius[2]),
        grid_t,
        grid_h,
        grid_w,
        0,
    )
