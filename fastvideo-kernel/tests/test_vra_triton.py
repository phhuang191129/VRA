from __future__ import annotations

import pytest

from .support_vra import build_vra_bool_mask, parse_shape, pytorch_vra_attention


def _make_qkv(torch, seq_len: int, heads: int = 2, dim: int = 32):
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq_len, dim, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn(1, heads, seq_len, dim, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn(1, heads, seq_len, dim, device="cuda",
                    dtype=torch.bfloat16)
    return q, k, v


@pytest.mark.parametrize("text_length", [0, 7])
def test_vra_triton_matches_pytorch_reference(text_length: int):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton VRA")

    from fastvideo_kernel import variable_rate_attention

    seq_shape = "6x8x16"
    shape = parse_shape(seq_shape)
    seq_len = shape.img_tokens + text_length
    q, k, v = _make_qkv(torch, seq_len)
    dense_radius = (0, 0, 0)
    mid_radius = (-1, -1, -1)
    mask = build_vra_bool_mask(
        shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
    )

    ref = pytorch_vra_attention(q, k, v, mask)
    out = variable_rate_attention(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        has_text=text_length > 0,
        seq_shape=seq_shape,
    )

    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1


def test_mixed_h100_text_fallback_does_not_require_extension(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton VRA")

    import fastvideo_kernel.ops as ops

    seq_shape = "6x8x16"
    text_length = 7
    shape = parse_shape(seq_shape)
    seq_len = shape.img_tokens + text_length
    q, k, v = _make_qkv(torch, seq_len, heads=2, dim=128)
    dense_radius = (0, 0, 0)
    mid_radius = (-1, -1, -1)

    monkeypatch.setattr(ops, "mixed_vra_attn_h100", None)
    expected = ops.variable_rate_attention(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        has_text=True,
        seq_shape=seq_shape,
    )
    actual = ops.mixed_vra_attention_h100(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        has_text=True,
        seq_shape=seq_shape,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("text_length", [0, 7])
def test_packed_stride_attention_matches_all_stride3_reference(
        text_length: int):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for packed stride attention")

    from fastvideo_kernel import packed_stride_attention

    seq_shape = "6x8x16"
    shape = parse_shape(seq_shape)
    seq_len = shape.img_tokens + text_length
    q, k, v = _make_qkv(torch, seq_len)
    dense_radius = (-1, -1, -1)
    mid_radius = (-1, -1, -1)
    mask = build_vra_bool_mask(
        shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
    )

    ref = pytorch_vra_attention(q, k, v, mask)
    out = packed_stride_attention(
        q,
        k,
        v,
        stride=3,
        text_length=text_length,
        has_text=text_length > 0,
        seq_shape=seq_shape,
    )

    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1


def test_native_vra_pack_matches_index_select():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native VRA pack")

    import fastvideo_kernel.ops as ops

    if not ops.has_native_vra_pack():
        pytest.skip("native VRA pack extension is not available")

    seq_shape = "6x8x16"
    text_length = 7
    stride = 3
    shape = parse_shape(seq_shape)
    seq_len = shape.img_tokens + text_length
    x = torch.randn(2, 3, seq_len, 32, device="cuda", dtype=torch.bfloat16)

    image_idx = torch.arange(0, shape.img_tokens, stride, device="cuda")
    text_idx = torch.arange(shape.img_tokens, seq_len, device="cuda")
    expected = x.index_select(2, torch.cat([image_idx, text_idx])).contiguous()
    actual = ops._pack_kv_by_stride(x.contiguous(), shape.img_tokens,
                                    text_length, stride)
    assert torch.equal(actual, expected)


def test_packed_attn_h100_matches_packed_sdpa():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for packed H100 attention")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("packed_attn_h100 requires Hopper")

    import torch.nn.functional as F
    import fastvideo_kernel.ops as ops

    if not ops.has_packed_attn_h100():
        pytest.skip("packed_attn_h100 extension is not available")

    seq_shape = "6x8x16"
    stride = 3
    shape = parse_shape(seq_shape)
    q, k, v = _make_qkv(torch, shape.img_tokens, heads=2, dim=128)
    k_packed = ops._pack_kv_by_stride(k.contiguous(), shape.img_tokens, 0,
                                      stride)
    v_packed = ops._pack_kv_by_stride(v.contiguous(), shape.img_tokens, 0,
                                      stride)

    ref = F.scaled_dot_product_attention(q, k_packed, v_packed)
    out = ops.packed_attn_h100(q.contiguous(), k_packed, v_packed)
    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1


def test_stride3_attn_h100_matches_packed_sdpa():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused stride-3 H100 attention")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("stride3_attn_h100 requires Hopper")

    import torch.nn.functional as F
    import fastvideo_kernel.ops as ops

    if not ops.has_stride3_attn_h100():
        pytest.skip("stride3_attn_h100 extension is not available")

    seq_shape = "6x8x16"
    stride = 3
    shape = parse_shape(seq_shape)
    q, k, v = _make_qkv(torch, shape.img_tokens, heads=2, dim=128)
    k_packed = ops._pack_kv_by_stride(k.contiguous(), shape.img_tokens, 0,
                                      stride)
    v_packed = ops._pack_kv_by_stride(v.contiguous(), shape.img_tokens, 0,
                                      stride)

    ref = F.scaled_dot_product_attention(q, k_packed, v_packed)
    out = ops.stride3_attn_h100(q.contiguous(), k.contiguous(), v.contiguous())
    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1


@pytest.mark.parametrize(
    ("dense_radius", "mid_radius"),
    [
        ((0, 0, 0), (-1, -1, -1)),
        ((-1, -1, -1), (9, 9, 9)),
    ],
)
def test_mixed_vra_attn_h100_matches_reference(dense_radius, mid_radius):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mixed VRA H100 attention")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("mixed_vra_attn_h100 requires Hopper")

    import fastvideo_kernel.ops as ops

    if not ops.has_mixed_vra_attn_h100():
        pytest.skip("mixed_vra_attn_h100 extension is not available")

    seq_shape = "6x8x16"
    shape = parse_shape(seq_shape)
    q, k, v = _make_qkv(torch, shape.img_tokens, heads=2, dim=128)
    mask = build_vra_bool_mask(
        shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=0,
    )

    ref = pytorch_vra_attention(q, k, v, mask)
    out = ops.mixed_vra_attention_h100(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=0,
        has_text=False,
        seq_shape=seq_shape,
    )
    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1


def test_mixed_vra_attn_h100_with_text_matches_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mixed VRA H100 attention")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("mixed_vra_attn_h100 requires Hopper")

    import fastvideo_kernel.ops as ops

    if not ops.has_mixed_vra_attn_h100():
        pytest.skip("mixed_vra_attn_h100 extension is not available")

    seq_shape = "6x8x16"
    text_length = 7
    shape = parse_shape(seq_shape)
    seq_len = shape.img_tokens + text_length
    q, k, v = _make_qkv(torch, seq_len, heads=2, dim=128)
    dense_radius = (0, 0, 0)
    mid_radius = (-1, -1, -1)
    mask = build_vra_bool_mask(
        shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
    )

    ref = pytorch_vra_attention(q, k, v, mask)
    out = ops.mixed_vra_attention_h100(
        q,
        k,
        v,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        has_text=True,
        seq_shape=seq_shape,
    )
    diff = (ref.float() - out.float()).abs()
    assert diff.mean().item() < 2e-2
    assert diff.max().item() < 2e-1
