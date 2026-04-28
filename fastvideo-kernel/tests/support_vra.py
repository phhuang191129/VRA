from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class VraShape:
    canvas_t: int
    canvas_h: int
    canvas_w: int
    tile_t: int = 6
    tile_h: int = 8
    tile_w: int = 8

    @property
    def tile_volume(self) -> int:
        return self.tile_t * self.tile_h * self.tile_w

    @property
    def img_tokens(self) -> int:
        return self.canvas_t * self.canvas_h * self.canvas_w

    @property
    def tile_grid(self) -> tuple[int, int, int]:
        if (self.canvas_t % self.tile_t or self.canvas_h % self.tile_h
                or self.canvas_w % self.tile_w):
            raise ValueError(
                "canvas shape must be divisible by tile shape, got "
                f"canvas=({self.canvas_t},{self.canvas_h},{self.canvas_w}) "
                f"tile=({self.tile_t},{self.tile_h},{self.tile_w})")
        return (
            self.canvas_t // self.tile_t,
            self.canvas_h // self.tile_h,
            self.canvas_w // self.tile_w,
        )

    @property
    def num_tiles(self) -> int:
        gt, gh, gw = self.tile_grid
        return gt * gh * gw


@dataclass(frozen=True)
class VraCount:
    query_tiles: int
    kv_tiles: int
    img_query_rows: int
    img_kv_rows: int
    dense_tiles: int
    mid_tiles: int
    far_tiles: int
    dense_rows: int
    mid_rows: int
    far_rows: int
    image_attention_pairs: int
    image_dense_pairs: int
    text_pairs: int

    @property
    def selected_image_rows_per_query_avg(self) -> float:
        return (self.dense_rows + self.mid_rows + self.far_rows) / max(
            1, self.query_tiles)

    @property
    def image_selected_ratio(self) -> float:
        return self.image_attention_pairs / max(1, self.image_dense_pairs)

    @property
    def total_pairs(self) -> int:
        return self.image_attention_pairs + self.text_pairs

    @property
    def total_dense_pairs(self) -> int:
        return self.image_dense_pairs + self.text_pairs


def parse_3d(value: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = value.lower().replace(",", "x").split("x")
        if len(parts) != 3:
            raise ValueError(f"expected 3D value, got {value!r}")
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    if len(value) != 3:
        raise ValueError(f"expected 3D value, got {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


def parse_shape(seq_shape: str,
                tile_shape: str | Sequence[int] = (6, 8, 8)) -> VraShape:
    canvas_t, canvas_h, canvas_w = parse_3d(seq_shape)
    tile_t, tile_h, tile_w = parse_3d(tile_shape)
    shape = VraShape(canvas_t, canvas_h, canvas_w, tile_t, tile_h, tile_w)
    _ = shape.tile_grid
    return shape


def iter_tiles(shape: VraShape) -> Iterable[tuple[int, int, int]]:
    gt, gh, gw = shape.tile_grid
    for tile_t in range(gt):
        for tile_h in range(gh):
            for tile_w in range(gw):
                yield tile_t, tile_h, tile_w


def tile_id(shape: VraShape, tile: tuple[int, int, int]) -> int:
    _, gh, gw = shape.tile_grid
    tile_t, tile_h, tile_w = tile
    return (tile_t * gh + tile_h) * gw + tile_w


def tile_from_id(shape: VraShape, idx: int) -> tuple[int, int, int]:
    _, gh, gw = shape.tile_grid
    tile_t = idx // (gh * gw)
    rem = idx % (gh * gw)
    return tile_t, rem // gw, rem % gw


def token_tile(shape: VraShape, token_idx: int) -> tuple[int, int, int]:
    if token_idx < 0 or token_idx >= shape.img_tokens:
        raise ValueError(
            f"token_idx must be in image token range [0, {shape.img_tokens})")
    return tile_from_id(shape, token_idx // shape.tile_volume)


def classify_stride(
    q_tile: tuple[int, int, int],
    kv_tile: tuple[int, int, int],
    dense_radius: Sequence[int],
    mid_radius: Sequence[int],
) -> int:
    dense = parse_3d(dense_radius)
    mid = parse_3d(mid_radius)
    dist = tuple(abs(q_tile[i] - kv_tile[i]) for i in range(3))
    if all(dist[i] <= dense[i] for i in range(3)):
        return 1
    if all(dist[i] <= mid[i] for i in range(3)):
        return 2
    return 3


def selected_rows_per_tile(tile_volume: int, stride: int,
                           phase: int = 0) -> int:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if phase < 0 or phase >= stride:
        raise ValueError(f"phase must be in [0, stride), got {phase}")
    if phase >= tile_volume:
        return 0
    return (tile_volume - 1 - phase) // stride + 1


def selected_local_rows(tile_volume: int,
                        stride: int,
                        phase: int = 0) -> list[int]:
    return list(range(phase, tile_volume, stride))


def selected_kv_indices_for_query_tile(
    shape: VraShape,
    q_tile: tuple[int, int, int],
    dense_radius: Sequence[int],
    mid_radius: Sequence[int],
    phase: int = 0,
) -> list[int]:
    selected: list[int] = []
    for kv_tile in iter_tiles(shape):
        stride = classify_stride(q_tile, kv_tile, dense_radius, mid_radius)
        base = tile_id(shape, kv_tile) * shape.tile_volume
        selected.extend(base + row
                        for row in selected_local_rows(shape.tile_volume,
                                                       stride, phase))
    return selected


def count_vra_pattern(
    shape: VraShape,
    dense_radius: Sequence[int],
    mid_radius: Sequence[int],
    text_length: int = 0,
    phase: int = 0,
) -> VraCount:
    dense_tiles = mid_tiles = far_tiles = 0
    dense_rows = mid_rows = far_rows = 0

    for q_tile in iter_tiles(shape):
        for kv_tile in iter_tiles(shape):
            stride = classify_stride(q_tile, kv_tile, dense_radius, mid_radius)
            rows = selected_rows_per_tile(shape.tile_volume, stride, phase)
            if stride == 1:
                dense_tiles += 1
                dense_rows += rows
            elif stride == 2:
                mid_tiles += 1
                mid_rows += rows
            else:
                far_tiles += 1
                far_rows += rows

    img_query_rows = shape.img_tokens
    img_kv_rows = shape.img_tokens
    selected_rows_total = dense_rows + mid_rows + far_rows
    image_attention_pairs = selected_rows_total * shape.tile_volume
    image_dense_pairs = img_query_rows * img_kv_rows

    # Conservative text policy for experiments:
    # image queries attend all text K/V, and text queries attend full image+text.
    text_pairs = 0
    if text_length:
        text_pairs += img_query_rows * text_length
        text_pairs += text_length * (img_kv_rows + text_length)

    return VraCount(
        query_tiles=shape.num_tiles,
        kv_tiles=shape.num_tiles,
        img_query_rows=img_query_rows,
        img_kv_rows=img_kv_rows,
        dense_tiles=dense_tiles,
        mid_tiles=mid_tiles,
        far_tiles=far_tiles,
        dense_rows=dense_rows,
        mid_rows=mid_rows,
        far_rows=far_rows,
        image_attention_pairs=image_attention_pairs,
        image_dense_pairs=image_dense_pairs,
        text_pairs=text_pairs,
    )


def build_vra_bool_mask(
    shape: VraShape,
    dense_radius: Sequence[int],
    mid_radius: Sequence[int],
    text_length: int = 0,
    phase: int = 0,
):
    import torch

    seq_len = shape.img_tokens + text_length
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)

    for q_tile in iter_tiles(shape):
        q_base = tile_id(shape, q_tile) * shape.tile_volume
        kv_indices = selected_kv_indices_for_query_tile(
            shape, q_tile, dense_radius, mid_radius, phase)
        if text_length:
            kv_indices.extend(range(shape.img_tokens, seq_len))
        q_rows = slice(q_base, q_base + shape.tile_volume)
        mask[q_rows, kv_indices] = True

    if text_length:
        mask[shape.img_tokens:seq_len, :seq_len] = True

    return mask


def pytorch_vra_attention(q, k, v, mask):
    import torch

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores / (q.shape[-1]**0.5)
    scores = scores.masked_fill(~mask.to(q.device).view(1, 1, *mask.shape),
                                float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)
