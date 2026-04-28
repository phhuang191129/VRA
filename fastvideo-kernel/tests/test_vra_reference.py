from __future__ import annotations

from .support_vra import (
    build_vra_bool_mask,
    classify_stride,
    count_vra_pattern,
    parse_shape,
    selected_kv_indices_for_query_tile,
    selected_rows_per_tile,
)


def test_selected_rows_per_tile_for_fastvideo_tile():
    tile_volume = 6 * 8 * 8
    assert selected_rows_per_tile(tile_volume, 1) == 384
    assert selected_rows_per_tile(tile_volume, 2) == 192
    assert selected_rows_per_tile(tile_volume, 3) == 128


def test_stride_classification():
    q_tile = (2, 3, 4)
    assert classify_stride(q_tile, (2, 3, 4), (1, 1, 1),
                           (2, 2, 2)) == 1
    assert classify_stride(q_tile, (4, 3, 4), (1, 1, 1),
                           (2, 2, 2)) == 2
    assert classify_stride(q_tile, (5, 3, 4), (1, 1, 1),
                           (2, 2, 2)) == 3


def test_dense_degenerate_count_matches_full_attention():
    shape = parse_shape("12x16x16")
    count = count_vra_pattern(shape, dense_radius=(99, 99, 99),
                              mid_radius=(99, 99, 99))
    assert count.image_attention_pairs == count.image_dense_pairs
    assert count.image_selected_ratio == 1.0
    assert count.mid_tiles == 0
    assert count.far_tiles == 0


def test_all_stride3_count_is_one_third_for_384_token_tiles():
    shape = parse_shape("12x16x16")
    count = count_vra_pattern(shape, dense_radius=(-1, -1, -1),
                              mid_radius=(-1, -1, -1))
    assert count.dense_tiles == 0
    assert count.mid_tiles == 0
    assert count.far_tiles == shape.num_tiles * shape.num_tiles
    assert count.image_selected_ratio == 1.0 / 3.0


def test_selected_indices_are_tile_major_and_sorted_by_kv_tile():
    shape = parse_shape("12x16x16")
    selected = selected_kv_indices_for_query_tile(
        shape,
        q_tile=(0, 0, 0),
        dense_radius=(0, 0, 0),
        mid_radius=(0, 0, 1),
        phase=0,
    )
    assert selected[:4] == [0, 1, 2, 3]

    # The first tile is dense. The next tile in tile-major order is mid-zone
    # for this query and therefore starts with even local rows.
    first_mid_base = shape.tile_volume
    mid_slice = selected[shape.tile_volume:shape.tile_volume + 4]
    assert mid_slice == [
        first_mid_base,
        first_mid_base + 2,
        first_mid_base + 4,
        first_mid_base + 6,
    ]


def test_small_bool_mask_shape_and_text_policy():
    import pytest

    pytest.importorskip("torch")

    shape = parse_shape("6x8x8")
    text_length = 5
    mask = build_vra_bool_mask(
        shape,
        dense_radius=(0, 0, 0),
        mid_radius=(0, 0, 0),
        text_length=text_length,
    )
    seq_len = shape.img_tokens + text_length
    assert tuple(mask.shape) == (seq_len, seq_len)
    assert mask[:shape.img_tokens, :shape.img_tokens].all()
    assert mask[:shape.img_tokens, shape.img_tokens:].all()
    assert mask[shape.img_tokens:, :].all()


def test_hunyuan_default_count_is_sparse_but_nonzero():
    shape = parse_shape("30x48x80")
    count = count_vra_pattern(shape, dense_radius=(1, 1, 1),
                              mid_radius=(2, 2, 3), text_length=256)
    assert 0 < count.image_selected_ratio < 1
    assert count.dense_tiles > 0
    assert count.mid_tiles > 0
    assert count.far_tiles > 0
    assert count.text_pairs > 0
