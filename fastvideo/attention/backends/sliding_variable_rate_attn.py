# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from einops import rearrange
from fastvideo_kernel.ops import variable_rate_attention

import fastvideo.envs as envs
from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.forward_context import get_forward_context
from fastvideo.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Gap-mask helper (module-level, cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _compute_gap_mask(
    full_window_size: tuple,
    vra_config: tuple,
) -> torch.Tensor:
    """Return a (nt, nh, nw) bool tensor: True = gap tile.

    A tile at absolute position (ti, hi, wi) is a gap tile when, seen from a
    query placed at the grid centre, it sits inside the outer window but is
    skipped by the OR-stride filter in every ring that covers it.

    The approximation uses a single centred query; tiles outside the outer
    window are also flagged so the fill pass can propagate border context.
    """
    nt, nh, nw = full_window_size
    (core_rt, core_rh, core_rw,
     mid_rt, mid_rh, mid_rw,
     outer_rt, outer_rh, outer_rw,
     mid_stride, outer_stride) = vra_config

    ct, ch, cw = nt // 2, nh // 2, nw // 2
    mask = torch.zeros(nt, nh, nw, dtype=torch.bool)

    for ti in range(nt):
        for hi in range(nh):
            for wi in range(nw):
                dt = abs(ti - ct)
                dh = abs(hi - ch)
                dw = abs(wi - cw)

                in_core = (dt <= core_rt and dh <= core_rh
                           and dw <= core_rw)
                if in_core:
                    continue  # always sampled

                in_mid = (dt <= mid_rt and dh <= mid_rh
                          and dw <= mid_rw)
                if in_mid:
                    # OR: skip only when ALL dims fail the stride test
                    skip = (mid_stride > 1 and
                            dt % mid_stride != 0 and
                            dh % mid_stride != 0 and
                            dw % mid_stride != 0)
                else:
                    in_outer = (dt <= outer_rt and dh <= outer_rh
                                and dw <= outer_rw)
                    if in_outer:
                        skip = (outer_stride > 1 and
                                dt % outer_stride != 0 and
                                dh % outer_stride != 0 and
                                dw % outer_stride != 0)
                    else:
                        skip = True  # beyond outer window

                mask[ti, hi, wi] = skip
    return mask


class RangeDict(dict):
    def __getitem__(self, item: int) -> str:
        for key in self.keys():
            if isinstance(key, tuple):
                low, high = key
                if low <= item <= high:
                    return str(super().__getitem__(key))
            elif key == item:
                return str(super().__getitem__(key))
        raise KeyError(f"seq_len {item} not supported for VRA")


class VariableRateAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SLIDING_VARIABLE_RATE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VariableRateAttentionImpl"]:
        return VariableRateAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VariableRateAttentionMetadata"]:
        return VariableRateAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VariableRateAttentionMetadataBuilder"]:
        return VariableRateAttentionMetadataBuilder


@dataclass
class VariableRateAttentionMetadata(AttentionMetadata):
    current_timestep: int
    VRA_param: list[tuple[int, ...]]
    sparsity_fraction: float


class VariableRateAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int = 0,
        VRA_param: list[tuple[int, ...]] | None = None,
        sparsity_fraction: float = 0.0,
        **kwargs: dict[str, Any],
    ) -> VariableRateAttentionMetadata:
        return VariableRateAttentionMetadata(
            current_timestep=current_timestep,
            VRA_param=VRA_param if VRA_param is not None else [],
            sparsity_fraction=sparsity_fraction,
        )


class VariableRateAttentionImpl(AttentionImpl):

    # Config tuple layout (all units are tile-grid counts):
    #   (core_rt, core_rh, core_rw,   -- dense core RADII
    #    mid_rt,  mid_rh,  mid_rw,    -- mid-ring RADII  (must be > core_r)
    #    outer_rt, outer_rh, outer_rw,-- outer-ring RADII (must be >= mid_r)
    #    mid_stride, outer_stride)    -- stride in each dim for ring sampling
    #
    # A KV tile at relative position (dt, dh, dw) from the query tile is:
    #   - always computed  if |dt|<=core_r,  |dh|<=core_r,  |dw|<=core_r
    #   - stride-sampled   if in mid ring:   abs(dX) % mid_stride == 0 for all X
    #   - stride-sampled   if in outer ring: abs(dX) % outer_stride == 0 for all X
    #   - skipped otherwise
    #
    # 30x48x80 tile grid 5x6x10 = 300 tiles, tile size (6,8,8).
    # The "58" preset is close to 58% overall sparsity for 50 denoising steps
    # when the first 12 steps are full attention. Sparse timesteps are sparser
    # because boundary query tiles see fewer neighbors than a centered query.
    # "sta58" matches STA's sparse-step density: ~125 / 300 active tiles.
    # 91% target (AND semantics): tiny core (0,0,1); very sparse mid and outer rings
    _VRA_CONFIGS = {
        "58": (2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3),
        "sta58": (1, 4, 5, 2, 4, 5, 2, 5, 9, 3, 3),
        "91": (0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 2),  # 91.0% sparsity
    }

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        self.STA_base_tile_size = [6, 8, 8]
        self.dit_seq_shape_mapping = RangeDict({
            (115200, 115456): '30x48x80',
            82944: '36x48x48',
            69120: '18x48x80',
        })
        # full_window_size = number of tiles in each dimension
        self.full_window_mapping = {
            '30x48x80': [5, 6, 10],
            '36x48x48': [6, 6, 6],
            '18x48x80': [3, 6, 10],
        }

        preset = str(envs.FASTVIDEO_VRA_SPARSITY)
        if preset not in self._VRA_CONFIGS:
            logger.warning(
                f"[VRA] Unknown FASTVIDEO_VRA_SPARSITY={preset!r}, "
                f"falling back to '58' preset.")
            preset = "58"
        self.vra_config = self._VRA_CONFIGS[preset]
        self._logged_sparsity = False

    # ------------------------------------------------------------------ #
    #  Tile / untile helpers (identical to STA)                            #
    # ------------------------------------------------------------------ #
    def tile(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2],
        )

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2],
        )

    # ------------------------------------------------------------------ #
    #  Sparsity estimator (AND-semantics, exact brute-force count)          #
    # ------------------------------------------------------------------ #
    def calculate_ideal_sparsity(self):
        """Estimate sparse-step tile sparsity using the same rules as the kernel.

        This averages over every query tile. A centered-query estimate is
        misleading for VRA because boundary tiles naturally see fewer in-range
        neighbors.
        """
        nt, nh, nw = self.full_window_size
        total = nt * nh * nw

        (core_rt, core_rh, core_rw,
         mid_rt, mid_rh, mid_rw,
         outer_rt, outer_rh, outer_rw,
         mid_stride, outer_stride) = self.vra_config

        computed_counts = []
        for qt in range(nt):
            for qh in range(nh):
                for qw in range(nw):
                    max_ct = max(nt - 1 - outer_rt, outer_rt)
                    max_ch = max(nh - 1 - outer_rh, outer_rh)
                    max_cw = max(nw - 1 - outer_rw, outer_rw)
                    center_t = min(max(qt, outer_rt), max_ct)
                    center_h = min(max(qh, outer_rh), max_ch)
                    center_w = min(max(qw, outer_rw), max_cw)

                    t_start = max(0, center_t - outer_rt)
                    t_end = min(nt, center_t + outer_rt + 1)
                    h_start = max(0, center_h - outer_rh)
                    h_end = min(nh, center_h + outer_rh + 1)
                    w_start = max(0, center_w - outer_rw)
                    w_end = min(nw, center_w + outer_rw + 1)

                    computed = 0
                    for ti in range(t_start, t_end):
                        for hi in range(h_start, h_end):
                            for wi in range(w_start, w_end):
                                dt = abs(ti - qt)
                                dh = abs(hi - qh)
                                dw = abs(wi - qw)
                                in_core = (dt <= core_rt and dh <= core_rh
                                           and dw <= core_rw)
                                in_mid = (dt <= mid_rt and dh <= mid_rh
                                          and dw <= mid_rw)
                                if in_core:
                                    computed += 1
                                elif in_mid:
                                    if (dt % mid_stride == 0 and
                                            dh % mid_stride == 0 and
                                            dw % mid_stride == 0):
                                        computed += 1
                                elif (dt % outer_stride == 0 and
                                      dh % outer_stride == 0 and
                                      dw % outer_stride == 0):
                                    computed += 1
                    computed_counts.append(computed)

        avg_computed = sum(computed_counts) / len(computed_counts)
        sparsity = 1.0 - avg_computed / total
        return sparsity, avg_computed, total, min(computed_counts), max(
            computed_counts)

    # ------------------------------------------------------------------ #
    #  Main attention interface                                            #
    # ------------------------------------------------------------------ #
    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VariableRateAttentionMetadata,
    ) -> torch.Tensor:
        img_sequence_length = qkv.shape[1]
        self.dit_seq_shape_str = self.dit_seq_shape_mapping[img_sequence_length]
        self.full_window_size = self.full_window_mapping[self.dit_seq_shape_str]
        self.dit_seq_shape_int = list(
            map(int, self.dit_seq_shape_str.split('x')))
        self.img_seq_length = (self.dit_seq_shape_int[0] *
                               self.dit_seq_shape_int[1] *
                               self.dit_seq_shape_int[2])

        if not self._logged_sparsity:
            sparsity, comp, tot, min_comp, max_comp = (
                self.calculate_ideal_sparsity())
            logger.info(f"[VRA] Config: {self.vra_config}")
            logger.info(
                f"[VRA] Sparse-step average sparsity ~{sparsity * 100:.1f}% "
                f"({comp:.1f} active / {tot} total tiles, "
                f"range {min_comp}-{max_comp})")
            self._logged_sparsity = True

        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VariableRateAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: VariableRateAttentionMetadata,
    ) -> torch.Tensor:
        text_length = q.shape[1] - self.img_seq_length
        has_text = text_length > 0

        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()

        head_num = query.size(1)

        current_timestep = (
            attn_metadata.current_timestep if attn_metadata is not None else
            get_forward_context().current_timestep)

        if current_timestep < 12:
            # Match STA: keep full attention for early diffusion steps.
            nt, nh, nw = self.full_window_size
            full_window = (nt, nh, nw, nt, nh, nw, nt, nh, nw, 1, 1)
            windows = [full_window for _ in range(head_num)]
        else:
            # Broadcast the same VRA config to all heads uniformly for later steps.
            windows = [self.vra_config for _ in range(head_num)]

        hidden_states = variable_rate_attention(
            query, key, value, windows,
            text_length, has_text,
            self.dit_seq_shape_str,
        ).transpose(1, 2)

        return hidden_states
