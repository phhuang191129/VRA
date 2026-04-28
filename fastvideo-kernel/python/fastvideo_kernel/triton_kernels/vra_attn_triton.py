import math
import torch
import triton
import triton.language as tl


def get_common_autotune_config():
    supported_num_staged = [1, 2, 3, 4]
    configs = [
        triton.Config({'BLOCK_Q': BLOCK_Q, 'BLOCK_KV': BLOCK_KV},
                      num_stages=s,
                      num_warps=w)
        for BLOCK_Q in [32, 64, 128]
        for BLOCK_KV in [32, 64, 128]
        for s in supported_num_staged
        for w in [4, 8]
    ]
    return configs


def get_autotune_config():
    return get_common_autotune_config()


@triton.jit
def clamp_int(value, min_val, max_val):
    ret = tl.where(value > max_val, max_val, value)
    ret = tl.where(ret < min_val, min_val, ret)
    return ret


@triton.jit
def _attn_fwd_loop(
    q, k, v, kv_mask, m, l, acc, sm_scale,
    MASK_KV: tl.constexpr,
):
    scores = tl.dot(q, k.T)
    scores = scores * sm_scale
    if MASK_KV:
        scores = tl.where(kv_mask[None, :], scores, -float('inf'))

    current_m = tl.max(scores, axis=1)
    new_m = tl.maximum(m, current_m)
    exp_scores = tl.math.exp2(scores - new_m[:, None])
    current_l = tl.sum(exp_scores, axis=1)

    alpha = tl.math.exp2(m - new_m)
    l = l * alpha + current_l
    m = new_m

    acc = (acc * alpha[:, None] +
           tl.dot(exp_scores.to(v.type.element_ty), v))
    return m, l, acc


@triton.autotune(
    configs=get_autotune_config(),
    key=['head_dim'],
)
@triton.jit
def triton_vra_kernel(
    Q, K, V, output,
    batch_size: int, num_heads: int, seq_len: int, head_dim: int,
    img_seq_len: int,
    text_length: int,
    canvas_t: int, canvas_h: int, canvas_w: int,
    # All core_*, mid_*, outer_* are RADII (half-extents in tile units).
    # core window spans |rel| <= core_r in each dim (dense).
    # mid  window spans |rel| <= mid_r  in each dim (stride-sampled).
    # outer window spans |rel| <= outer_r in each dim (stride-sampled).
    # mid_r  > core_r and outer_r >= mid_r must hold.
    core_t: int, core_h: int, core_w: int,
    mid_t: int, mid_h: int, mid_w: int,
    outer_t: int, outer_h: int, outer_w: int,
    mid_stride: int, outer_stride: int,
    tile_t: int, tile_h: int, tile_w: int,
    scale: float,
    has_text: tl.constexpr,
    text_q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    total_tile_size = tile_t * tile_h * tile_w
    q_block_per_tile = (total_tile_size + BLOCK_Q - 1) // BLOCK_Q

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if text_q:
        q_block_idx = tl.program_id(2)
    else:
        q_tile_flat = tl.program_id(2) // q_block_per_tile
        q_block_idx = tl.program_id(2) % q_block_per_tile

    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DIM), dtype=tl.float32)

    q_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim
    if text_q:
        q_base_idx = img_seq_len + q_block_idx * BLOCK_Q
    else:
        q_base_idx = q_tile_flat * total_tile_size + q_block_idx * BLOCK_Q

    q_offset_in_tile = tl.arange(0, BLOCK_Q)
    q_idx = q_base_idx + q_offset_in_tile
    q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < total_tile_size

    q = tl.load(
        Q + q_offset + q_idx[:, None] * head_dim +
        tl.arange(0, BLOCK_DIM)[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )

    sm_scale = scale * 1.4426950408889634

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    tiles_per_hw = num_tiles_h * num_tiles_w

    # ------------------------------------------------------------------ #
    #  Compute KV tile iteration range                                     #
    # ------------------------------------------------------------------ #
    if text_q:
        # Text queries attend to ALL video tiles (text-sink: full attention).
        kv_tile_start_t = 0
        kv_tile_end_t = num_tiles_t
        kv_tile_start_h = 0
        kv_tile_end_h = num_tiles_h
        kv_tile_start_w = 0
        kv_tile_end_w = num_tiles_w
    else:
        # Decompose flat tile index into 3D tile coordinates.
        q_tile_t = q_tile_flat // tiles_per_hw
        remaining = q_tile_flat % tiles_per_hw
        q_tile_h = remaining // num_tiles_w
        q_tile_w = remaining % num_tiles_w

        # Clamp the "center" used for the outer-window iteration range so
        # boundary queries still get a full-radius window.
        # (The in_core / in_mid predicates still use q_tile_* for relative
        #  distance, not this clamped center.)
        max_center_t = tl.maximum(num_tiles_t - 1 - outer_t, outer_t)
        kernel_center_t = clamp_int(q_tile_t, outer_t, max_center_t)
        max_center_h = tl.maximum(num_tiles_h - 1 - outer_h, outer_h)
        kernel_center_h = clamp_int(q_tile_h, outer_h, max_center_h)
        max_center_w = tl.maximum(num_tiles_w - 1 - outer_w, outer_w)
        kernel_center_w = clamp_int(q_tile_w, outer_w, max_center_w)

        # Outer window range (clamped to grid bounds).
        kv_tile_start_t = kernel_center_t - outer_t
        kv_tile_start_t = tl.where(kv_tile_start_t < 0, 0, kv_tile_start_t)
        kv_tile_end_t = kernel_center_t + outer_t + 1
        kv_tile_end_t = tl.where(kv_tile_end_t > num_tiles_t,
                                  num_tiles_t, kv_tile_end_t)

        kv_tile_start_h = kernel_center_h - outer_h
        kv_tile_start_h = tl.where(kv_tile_start_h < 0, 0, kv_tile_start_h)
        kv_tile_end_h = kernel_center_h + outer_h + 1
        kv_tile_end_h = tl.where(kv_tile_end_h > num_tiles_h,
                                  num_tiles_h, kv_tile_end_h)

        kv_tile_start_w = kernel_center_w - outer_w
        kv_tile_start_w = tl.where(kv_tile_start_w < 0, 0, kv_tile_start_w)
        kv_tile_end_w = kernel_center_w + outer_w + 1
        kv_tile_end_w = tl.where(kv_tile_end_w > num_tiles_w,
                                  num_tiles_w, kv_tile_end_w)

    # ------------------------------------------------------------------ #
    #  Main KV-tile loop — VRA masking                                    #
    # ------------------------------------------------------------------ #
    for kv_tile_t in tl.range(kv_tile_start_t, kv_tile_end_t):
        if not text_q:
            # Relative temporal distance from the query tile.
            rel_t = kv_tile_t - q_tile_t
            abs_rel_t = tl.abs(rel_t)
        for kv_tile_h in tl.range(kv_tile_start_h, kv_tile_end_h):
            if not text_q:
                rel_h = kv_tile_h - q_tile_h
                abs_rel_h = tl.abs(rel_h)
            for kv_tile_w in tl.range(kv_tile_start_w, kv_tile_end_w):
                compute = True
                if not text_q:
                    rel_w = kv_tile_w - q_tile_w
                    abs_rel_w = tl.abs(rel_w)

                    # Concentric region membership (using RADII).
                    in_core = ((abs_rel_t <= core_t) and
                               (abs_rel_h <= core_h) and
                               (abs_rel_w <= core_w))
                    in_mid = ((abs_rel_t <= mid_t) and
                              (abs_rel_h <= mid_h) and
                              (abs_rel_w <= mid_w))

                    # Per-query relative stride filter.
                    # A tile passes if ALL relative coords are multiples of
                    # the stride (i.e., sample at ±stride, ±2*stride, …)
                    if not in_core:
                        if in_mid:
                            # Mid ring: AND semantics — skip if ANY dim fails stride.
                            if mid_stride > 1:
                                if ((abs_rel_t % mid_stride != 0) or
                                        (abs_rel_h % mid_stride != 0) or
                                        (abs_rel_w % mid_stride != 0)):
                                    compute = False
                        else:
                            # Outer ring: AND semantics — skip if ANY dim fails stride.
                            if outer_stride > 1:
                                if ((abs_rel_t % outer_stride != 0) or
                                        (abs_rel_h % outer_stride != 0) or
                                        (abs_rel_w % outer_stride != 0)):
                                    compute = False

                if compute:
                    kv_base_idx = (
                        (kv_tile_t * num_tiles_h * num_tiles_w +
                         kv_tile_h * num_tiles_w +
                         kv_tile_w) * total_tile_size
                    )

                    for kv_block_idx in tl.range(0, total_tile_size, BLOCK_KV):
                        kv_offset_in_block = tl.arange(0, BLOCK_KV)
                        kv_idx = kv_base_idx + kv_block_idx + kv_offset_in_block
                        kv_mask = ((kv_block_idx + tl.arange(0, BLOCK_KV)) <
                                   total_tile_size)

                        kv_offset = (
                            (batch_idx * num_heads + head_idx) * seq_len * head_dim
                        )

                        k = tl.load(
                            K + kv_offset + kv_idx[:, None] * head_dim +
                            tl.arange(0, BLOCK_DIM)[None, :],
                            mask=kv_mask[:, None],
                            other=0.0,
                        )
                        v = tl.load(
                            V + kv_offset + kv_idx[:, None] * head_dim +
                            tl.arange(0, BLOCK_DIM)[None, :],
                            mask=kv_mask[:, None],
                            other=0.0,
                        )

                        m, l, acc = _attn_fwd_loop(
                            q, k, v, kv_mask, m, l, acc, sm_scale, False)

    # ------------------------------------------------------------------ #
    #  Text KV tokens: every query (image + text) attends to all text KV  #
    # ------------------------------------------------------------------ #
    if has_text:
        kv_base_idx = img_seq_len
        for kv_block_idx in tl.range(0, text_length, BLOCK_KV):
            kv_offset_in_block = tl.arange(0, BLOCK_KV)
            kv_idx = kv_base_idx + kv_block_idx + kv_offset_in_block
            kv_mask = (kv_block_idx + tl.arange(0, BLOCK_KV)) < text_length

            kv_offset = (
                (batch_idx * num_heads + head_idx) * seq_len * head_dim
            )

            k = tl.load(
                K + kv_offset + kv_idx[:, None] * head_dim +
                tl.arange(0, BLOCK_DIM)[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                V + kv_offset + kv_idx[:, None] * head_dim +
                tl.arange(0, BLOCK_DIM)[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            )

            m, l, acc = _attn_fwd_loop(
                q, k, v, kv_mask, m, l, acc, sm_scale, True)

    output_acc = acc / l[:, None]
    tl.store(
        output + q_offset + q_idx[:, None] * head_dim +
        tl.arange(0, BLOCK_DIM)[None, :],
        output_acc,
        mask=q_mask[:, None],
    )


def sliding_variable_rate_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size,
    text_length: int,
    has_text: bool = True,
    dit_seq_shape: str = '30x48x80',
) -> torch.Tensor:
    # window_size: list of per-head tuples
    #   (core_rt, core_rh, core_rw,
    #    mid_rt,  mid_rh,  mid_rw,
    #    outer_rt, outer_rh, outer_rw,
    #    mid_stride, outer_stride)
    # All *_r* values are RADII in tile units.

    # Bug-fix: initialise pad_size before the conditional so the final
    # trim at the bottom of the function is always safe.
    pad_size = 0

    seq_length = q.shape[2]
    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)

    assert q.shape[1] == len(window_size), \
        "Number of heads must match the number of window sizes"

    batch_size, num_heads, seq_len, head_dim = q.shape

    if dit_seq_shape == '30x48x80':
        canvas_t, canvas_h, canvas_w = 30, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '36x48x48':
        canvas_t, canvas_h, canvas_w = 36, 48, 48
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '18x48x80':
        canvas_t, canvas_h, canvas_w = 18, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8
    else:
        raise ValueError(f"Unknown dit_seq_shape: {dit_seq_shape}")

    img_seq_len = canvas_t * canvas_h * canvas_w
    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    num_tiles = num_tiles_t * num_tiles_h * num_tiles_w
    total_tile_size = tile_t * tile_h * tile_w
    BLOCK_DIM = head_dim

    output = torch.empty_like(q)

    # ---- Image-token pass (one sub-kernel per head × batch) ----
    for head_index, (
        core_rt, core_rh, core_rw,
        mid_rt, mid_rh, mid_rw,
        outer_rt, outer_rh, outer_rw,
        mstride, ostride,
    ) in enumerate(window_size):
        for batch in range(batch_size):
            q_head = q[batch:batch + 1, head_index:head_index + 1]
            k_head = k[batch:batch + 1, head_index:head_index + 1]
            v_head = v[batch:batch + 1, head_index:head_index + 1]
            o_head = output[batch:batch + 1, head_index:head_index + 1]

            grid = lambda META: (
                1, 1,
                num_tiles * triton.cdiv(total_tile_size, META['BLOCK_Q']),
            )
            triton_vra_kernel[grid](
                q_head, k_head, v_head, o_head,
                1, 1, seq_len, head_dim,
                img_seq_len,
                text_length,
                canvas_t, canvas_h, canvas_w,
                core_rt, core_rh, core_rw,
                mid_rt, mid_rh, mid_rw,
                outer_rt, outer_rh, outer_rw,
                mstride, ostride,
                tile_t, tile_h, tile_w,
                scale=1.0 / (head_dim ** 0.5),
                has_text=has_text,
                text_q=False,
                BLOCK_DIM=BLOCK_DIM,
            )

    # ---- Text-token pass: text queries attend to everything ----
    if has_text:
        dummy_win = window_size[0]
        # Bug-fix: grid must cover text_length tokens, not total_tile_size.
        grid = lambda META: (
            batch_size, num_heads,
            triton.cdiv(text_length, META['BLOCK_Q']),
        )
        triton_vra_kernel[grid](
            q, k, v, output,
            batch_size, num_heads, seq_len, head_dim,
            img_seq_len,
            text_length,
            canvas_t, canvas_h, canvas_w,
            dummy_win[0], dummy_win[1], dummy_win[2],
            dummy_win[3], dummy_win[4], dummy_win[5],
            dummy_win[6], dummy_win[7], dummy_win[8],
            dummy_win[9], dummy_win[10],
            tile_t, tile_h, tile_w,
            scale=1.0 / (head_dim ** 0.5),
            has_text=has_text,
            text_q=True,
            BLOCK_DIM=BLOCK_DIM,
        )

    # Trim any padding that was added for alignment.
    if has_text and pad_size > 0:
        output = output[:, :, :seq_length]

    return output
