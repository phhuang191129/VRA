#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torch_helpers.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cassert>
#include <iostream>
#include <pybind11/pybind11.h>
#include <type_traits>
#include <torch/extension.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

namespace {

template <ducks::st::all ST, int ROW_STRIDE>
__host__ static inline void create_row_strided_tensor_map(
    CUtensorMap* tma_map,
    const typename ST::dtype* src,
    int batch,
    int depth,
    int dense_rows,
    int cols) {
    using dtype = typename ST::dtype;

    constexpr uint32_t tma_dim = 5;
    void* global_addr = reinterpret_cast<void*>(const_cast<dtype*>(src));

    constexpr CUtensorMapDataType tma_format =
        (std::is_same_v<dtype, bf16>    ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
         : std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
         : std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                                        : CUtensorMapDataType(-1));
    constexpr CUtensorMapInterleave tma_interleave =
        CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion tma_l2Promotion =
        CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill =
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle tma_swizzle =
        (ST::swizzle_bytes == 32    ? CU_TENSOR_MAP_SWIZZLE_32B
         : ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B
         : ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                                    : CU_TENSOR_MAP_SWIZZLE_NONE);

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
    constexpr uint64_t shared_tile_height = ST::rows;
    constexpr uint64_t shared_tile_width = ST::cols;
    int packed_rows = dense_rows / ROW_STRIDE;

    uint64_t gmem_shape[5] = {
        swizzle_elements,
        static_cast<uint64_t>(packed_rows),
        static_cast<uint64_t>((cols + swizzle_elements - 1) /
                              swizzle_elements),
        static_cast<uint64_t>(depth),
        static_cast<uint64_t>(batch),
    };
    uint64_t gmem_stride[4] = {
        static_cast<uint64_t>(ROW_STRIDE * cols) * sizeof(dtype),
        ST::swizzle_bytes,
        static_cast<uint64_t>(dense_rows) * cols * sizeof(dtype),
        static_cast<uint64_t>(depth) * dense_rows * cols * sizeof(dtype),
    };
    uint32_t smem_shape[5] = {
        swizzle_elements,
        shared_tile_height,
        shared_tile_width / swizzle_elements,
        1,
        1,
    };
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    assert(gmem_stride[0] % 16 == 0);
    assert(gmem_stride[1] % 16 == 0);
    assert(gmem_stride[2] % 16 == 0);
    assert(gmem_stride[3] % 16 == 0);

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, tma_format, tma_dim, global_addr, gmem_shape, gmem_stride,
        smem_shape, smem_stride, tma_interleave, tma_swizzle,
        tma_l2Promotion, tma_oobFill);
    if (result != CUDA_SUCCESS) {
        const char* error_string = nullptr;
        cuGetErrorString(result, &error_string);
        std::cerr << "Error in row-strided TMA descriptor creation: "
                  << (error_string ? error_string : "unknown") << std::endl;
    }
}

template <int D, int ROW_STRIDE>
struct row_strided_kv_global {
    using identifier = ducks::gl::identifier;
    using dtype = bf16;
    using tile = st_bf<128, D>;

    bf16* raw_ptr;
    int batch_internal;
    int depth_internal;
    int dense_rows_internal;
    int packed_rows_internal;
    CUtensorMap tma_desc;

    __host__ inline row_strided_kv_global(bf16* data,
                                          unsigned int batch,
                                          unsigned int depth,
                                          unsigned int dense_rows,
                                          std::nullptr_t)
        : raw_ptr(data),
          batch_internal(static_cast<int>(batch)),
          depth_internal(static_cast<int>(depth)),
          dense_rows_internal(static_cast<int>(dense_rows)),
          packed_rows_internal(static_cast<int>(dense_rows / ROW_STRIDE)) {
        create_row_strided_tensor_map<tile, ROW_STRIDE>(
            &tma_desc, raw_ptr, batch_internal, depth_internal,
            dense_rows_internal, D);
    }

    __host__ __device__ inline row_strided_kv_global(
        const row_strided_kv_global& other)
        : raw_ptr(other.raw_ptr),
          batch_internal(other.batch_internal),
          depth_internal(other.depth_internal),
          dense_rows_internal(other.dense_rows_internal),
          packed_rows_internal(other.packed_rows_internal),
          tma_desc(other.tma_desc) {}

    __device__ __host__ inline int batch() const { return batch_internal; }
    __device__ __host__ inline int depth() const { return depth_internal; }
    __device__ __host__ inline int rows() const { return packed_rows_internal; }
    __device__ __host__ static constexpr int cols() { return D; }

    template <typename U, int axis>
    __device__ inline const CUtensorMap* get_tma() const {
        static_assert(std::is_same_v<U, tile>,
                      "row_strided_kv_global only stores its KV tile TMA map");
        static_assert(axis == dim::ROW,
                      "row_strided_kv_global only supports row-axis TMA");
        return &tma_desc;
    }

    __device__ inline bf16& operator[](const coord<ducks::default_type>& idx)
        const {
        return raw_ptr[((idx.b * depth() + idx.d) * dense_rows_internal +
                        idx.r) *
                           cols() +
                       idx.c];
    }

    template <int axis>
    __device__ inline size_t shape() const {
        static_assert(axis >= 0 && axis <= 3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis == 0) {
            return size_t(batch());
        } else if constexpr (axis == 1) {
            return size_t(depth());
        } else if constexpr (axis == 2) {
            return size_t(rows());
        } else {
            return size_t(cols());
        }
    }

    template <int axis>
    __device__ inline size_t stride() const {
        static_assert(axis >= 0 && axis <= 3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis == 0) {
            return size_t(depth()) * dense_rows_internal * cols();
        } else if constexpr (axis == 1) {
            return size_t(dense_rows_internal) * cols();
        } else if constexpr (axis == 2) {
            return size_t(ROW_STRIDE * cols());
        } else {
            return 1;
        }
    }
};

template <int D>
using stride3_kv_global = row_strided_kv_global<D, 3>;

template <int D, int NUM_WORKERS, bool GATHER_STRIDE3>
struct packed_attn_fwd_layout {
    using qo_tile = st_bf<64, D>;
    using kv_tile = st_bf<128, D>;
    using qo_global = gl<bf16, -1, -1, -1, D, qo_tile>;
    using packed_kv_global = gl<bf16, -1, -1, -1, D, kv_tile>;
    using kv_global = std::conditional_t<GATHER_STRIDE3, stride3_kv_global<D>,
                                         packed_kv_global>;

    struct globals {
        qo_global o;
        qo_global q;
        kv_global k;
        kv_global v;
        int valid_kv_rows;
    };

    struct input_block {
        kv_tile k;
        kv_tile v;
    };

    struct scratch_block {
        qo_tile q[NUM_WORKERS];
    };

    struct common_state {
        int batch;
        int head;
        int seq;
        int kv_rows;
    };

    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec;
        col_vec<rt_fl<16, kv_tile::rows>> norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};

template <int D, bool GATHER_STRIDE3>
struct packed_attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12;
    static constexpr int NUM_WORKERS = NUM_CONSUMER_WARPS / 4;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY - 2048;

    using layout = packed_attn_fwd_layout<D, NUM_WORKERS, GATHER_STRIDE3>;

    __device__ static inline void common_setup(
        common_setup_args<layout> args) {
        int task_id = gridDim.x * args.task_iter + blockIdx.x;
        int seq_q = (args.globals.q.rows() +
                     NUM_WORKERS * layout::qo_tile::rows - 1) /
                    (NUM_WORKERS * layout::qo_tile::rows);
        int total_tasks = args.globals.q.batch() * args.globals.q.depth() * seq_q;
        if (task_id >= total_tasks) {
            args.num_iters = -1;
            return;
        }

        args.common.batch = task_id / (seq_q * args.globals.q.depth());
        task_id -= args.common.batch * seq_q * args.globals.q.depth();
        args.common.head = task_id / seq_q;
        task_id -= args.common.head * seq_q;
        args.common.seq = task_id;
        args.common.kv_rows = args.globals.valid_kv_rows;
        args.num_iters = (args.common.kv_rows + layout::kv_tile::rows - 1) /
                         layout::kv_tile::rows;
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }

        __device__ static inline void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                tma::load_async(args.input.k, args.globals.k,
                                {args.common.batch, args.common.head,
                                 args.iter, 0},
                                args.inputs_arrived);
                tma::load_async(args.input.v, args.globals.v,
                                {args.common.batch, args.common.head,
                                 args.iter, 0},
                                args.inputs_arrived);
            } else if (laneid() == 0) {
                arrive(args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            int q_block = args.common.seq * NUM_WORKERS + warpgroup::groupid();
            if (q_block * layout::qo_tile::rows < args.globals.q.rows()) {
                warpgroup::load(args.scratch.q[warpgroup::groupid()],
                                args.globals.q,
                                {args.common.batch, args.common.head, q_block,
                                 0});
            }
            args.state.o_reg = 0.f;
            args.state.norm_vec = 0.f;
            args.state.max_vec = base_types::constants<float>::neg_infty();
            warpgroup::sync(warpgroup::groupid());
        }

        __device__ static inline void compute(
            consumer_compute_args<layout> args) {
            constexpr float TEMPERATURE_SCALE =
                (D == 128) ? 0.08838834764f * 1.44269504089f
                           : 0.125f * 1.44269504089f;

            warpgroup::mm<transpose::N, transpose::T>(
                args.state.att_block,
                args.scratch.q[warpgroup::groupid()],
                args.input.k);
            args.state.max_vec_last_scaled =
                args.state.max_vec * TEMPERATURE_SCALE;
            warpgroup::mma_async_wait();

            right_fill(args.state.att_block, args.state.att_block,
                       args.common.kv_rows - args.iter * layout::kv_tile::rows,
                       base_types::constants<float>::neg_infty());
            args.state.max_vec =
                max<axis::COL>(args.state.att_block, args.state.max_vec);
            args.state.max_vec_scaled =
                args.state.max_vec * TEMPERATURE_SCALE;
            args.state.att_block =
                exp2((args.state.att_block * TEMPERATURE_SCALE) -
                     args.state.max_vec_scaled);
            args.state.max_vec_last_scaled =
                exp2(args.state.max_vec_last_scaled -
                     args.state.max_vec_scaled);
            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec =
                sum<axis::COL>(args.state.att_block, args.state.norm_vec);
            args.state.o_reg *= args.state.max_vec_last_scaled;
            args.state.att_block_mma = args.state.att_block;

            warpgroup::mma<transpose::N, transpose::N>(
                args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();

            if (laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static inline void finish(
            consumer_finish_args<layout> args) {
            int q_block = args.common.seq * NUM_WORKERS + warpgroup::groupid();
            if (q_block * layout::qo_tile::rows < args.globals.q.rows()) {
                args.state.o_reg /= args.state.norm_vec;
                auto& o_smem =
                    reinterpret_cast<typename layout::qo_tile&>(
                        args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if (warpgroup::warpid() == 0) {
                    tma::store_async(args.globals.o, o_smem,
                                     {args.common.batch, args.common.head,
                                      q_block, 0});
                }
                tma::store_async_read_wait();
            }
            __syncwarp();
            if (laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

struct vra_segment {
    int stride;
    int source_row;
    int valid_rows;
    int source_row_b;
    int valid_rows_b;
};

__device__ static inline int tile_distance_stride_from_coords(int q_t,
                                                              int q_h,
                                                              int q_w,
                                                              int kv_t,
                                                              int kv_h,
                                                              int kv_w,
                                                              int dense_t,
                                                              int dense_h,
                                                              int dense_w,
                                                              int mid_t,
                                                              int mid_h,
                                                              int mid_w) {
    int dt = abs(q_t - kv_t);
    int dh = abs(q_h - kv_h);
    int dw = abs(q_w - kv_w);

    if (dense_t >= 0 && dense_h >= 0 && dense_w >= 0 &&
        dt <= dense_t && dh <= dense_h && dw <= dense_w) {
        return 1;
    }
    if (mid_t >= 0 && mid_h >= 0 && mid_w >= 0 && dt <= mid_t &&
        dh <= mid_h && dw <= mid_w) {
        return 2;
    }
    return 3;
}

__device__ static inline int clipped_axis_count(int q,
                                                int radius,
                                                int extent) {
    if (radius < 0) {
        return 0;
    }
    int lo = q - radius;
    int hi = q + radius;
    if (lo < 0) {
        lo = 0;
    }
    if (hi >= extent) {
        hi = extent - 1;
    }
    return hi - lo + 1;
}

__device__ static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ static inline int clipped_box_count(int q_t,
                                               int q_h,
                                               int q_w,
                                               int radius_t,
                                               int radius_h,
                                               int radius_w,
                                               int grid_t,
                                               int grid_h,
                                               int grid_w) {
    if (radius_t < 0 || radius_h < 0 || radius_w < 0) {
        return 0;
    }
    return clipped_axis_count(q_t, radius_t, grid_t) *
           clipped_axis_count(q_h, radius_h, grid_h) *
           clipped_axis_count(q_w, radius_w, grid_w);
}

template <typename globals_t>
__device__ static inline int mixed_vra_num_iters(const globals_t& globals,
                                                 int q_t,
                                                 int q_h,
                                                 int q_w) {
    int dense_count = clipped_box_count(
        q_t, q_h, q_w, globals.dense_t, globals.dense_h, globals.dense_w,
        globals.grid_t, globals.grid_h, globals.grid_w);
    int mid_box_count = clipped_box_count(
        q_t, q_h, q_w, globals.mid_t, globals.mid_h, globals.mid_w,
        globals.grid_t, globals.grid_h, globals.grid_w);
    int intersection_count = 0;
    if (globals.dense_t >= 0 && globals.dense_h >= 0 &&
        globals.dense_w >= 0 && globals.mid_t >= 0 &&
        globals.mid_h >= 0 && globals.mid_w >= 0) {
        intersection_count = clipped_box_count(
            q_t, q_h, q_w, min_int(globals.dense_t, globals.mid_t),
            min_int(globals.dense_h, globals.mid_h),
            min_int(globals.dense_w, globals.mid_w), globals.grid_t,
            globals.grid_h, globals.grid_w);
    }
    int mid_count = mid_box_count - intersection_count;
    int num_tiles = globals.grid_t * globals.grid_h * globals.grid_w;
    int far_count = num_tiles - dense_count - mid_count;
    int mid_tail_segments = (mid_count + 1) / 2;
    return dense_count * 3 + mid_count + mid_tail_segments + far_count;
}

template <int DENSE_T,
          int DENSE_H,
          int DENSE_W,
          int MID_T,
          int MID_H,
          int MID_W,
          int GRID_T,
          int GRID_H,
          int GRID_W>
__device__ static inline int mixed_vra_num_iters_fixed(int q_t,
                                                       int q_h,
                                                       int q_w) {
    int dense_count = clipped_box_count(
        q_t, q_h, q_w, DENSE_T, DENSE_H, DENSE_W, GRID_T, GRID_H, GRID_W);
    int mid_box_count = clipped_box_count(
        q_t, q_h, q_w, MID_T, MID_H, MID_W, GRID_T, GRID_H, GRID_W);
    int intersection_count = 0;
    if constexpr (DENSE_T >= 0 && DENSE_H >= 0 && DENSE_W >= 0 &&
                  MID_T >= 0 && MID_H >= 0 && MID_W >= 0) {
        intersection_count = clipped_box_count(
            q_t, q_h, q_w, min_int(DENSE_T, MID_T),
            min_int(DENSE_H, MID_H), min_int(DENSE_W, MID_W), GRID_T,
            GRID_H, GRID_W);
    }
    int mid_count = mid_box_count - intersection_count;
    int far_count = GRID_T * GRID_H * GRID_W - dense_count - mid_count;
    return dense_count * 3 + mid_count + ((mid_count + 1) / 2) + far_count;
}

template <int DENSE_T,
          int DENSE_H,
          int DENSE_W,
          int MID_T,
          int MID_H,
          int MID_W>
__device__ static inline int tile_distance_stride_fixed(int q_t,
                                                        int q_h,
                                                        int q_w,
                                                        int kv_t,
                                                        int kv_h,
                                                        int kv_w) {
    int dt = abs(q_t - kv_t);
    int dh = abs(q_h - kv_h);
    int dw = abs(q_w - kv_w);

    if constexpr (DENSE_T >= 0 && DENSE_H >= 0 && DENSE_W >= 0) {
        if (dt <= DENSE_T && dh <= DENSE_H && dw <= DENSE_W) {
            return 1;
        }
    }
    if constexpr (MID_T >= 0 && MID_H >= 0 && MID_W >= 0) {
        if (dt <= MID_T && dh <= MID_H && dw <= MID_W) {
            return 2;
        }
    }
    return 3;
}

template <typename schedule_state_t>
__device__ static inline void mixed_vra_schedule_reset(
    schedule_state_t& state) {
    state.schedule_kv_tile = 0;
    state.schedule_kv_t = 0;
    state.schedule_kv_h = 0;
    state.schedule_kv_w = 0;
    state.schedule_tile_phase = 0;
    state.current_stride = 3;
    state.pending_tail_source_row = -1;
}

template <typename schedule_state_t>
__device__ static inline void mixed_vra_schedule_reset_fixed(
    schedule_state_t& state) {
    state.schedule_kv_tile = 0;
    state.schedule_tile_phase = 0;
    state.current_stride = 3;
    state.pending_tail_source_row = -1;
}

template <typename globals_t, typename schedule_state_t>
__device__ static inline void mixed_vra_advance_kv_tile(
    const globals_t& globals,
    schedule_state_t& state) {
    state.schedule_tile_phase = 0;
    state.schedule_kv_tile += 1;
    state.schedule_kv_w += 1;
    if (state.schedule_kv_w >= globals.grid_w) {
        state.schedule_kv_w = 0;
        state.schedule_kv_h += 1;
        if (state.schedule_kv_h >= globals.grid_h) {
            state.schedule_kv_h = 0;
            state.schedule_kv_t += 1;
        }
    }
}

template <int GRID_H, int GRID_W, typename schedule_state_t>
__device__ static inline void mixed_vra_advance_kv_tile_fixed(
    schedule_state_t& state) {
    state.schedule_tile_phase = 0;
    state.schedule_kv_tile += 1;
}

template <typename globals_t, typename schedule_state_t>
__device__ static inline vra_segment mixed_vra_next_segment(
    const globals_t& globals,
    int q_t,
    int q_h,
    int q_w,
    schedule_state_t& state) {
    constexpr int TILE_ROWS = 384;
    int num_tiles = globals.grid_t * globals.grid_h * globals.grid_w;

    while (true) {
        if (state.schedule_tile_phase > 0) {
            int kv_tile_id = state.schedule_kv_tile;
            if (state.current_stride == 1) {
                int phase = state.schedule_tile_phase;
                state.schedule_tile_phase += 1;
                if (state.schedule_tile_phase >= 3) {
                    mixed_vra_advance_kv_tile(globals, state);
                }
                return vra_segment{1, kv_tile_id * TILE_ROWS + phase * 128,
                                   128, 0, 0};
            }

            int tail_source_row = kv_tile_id * TILE_ROWS + 256;
            mixed_vra_advance_kv_tile(globals, state);
            if (state.pending_tail_source_row >= 0) {
                int first_tail = state.pending_tail_source_row;
                state.pending_tail_source_row = -1;
                return vra_segment{2, first_tail, 64, tail_source_row, 64};
            }
            state.pending_tail_source_row = tail_source_row;
            continue;
        }

        if (state.schedule_kv_tile >= num_tiles) {
            if (state.pending_tail_source_row >= 0) {
                int first_tail = state.pending_tail_source_row;
                state.pending_tail_source_row = -1;
                return vra_segment{2, first_tail, 64, 0, 0};
            }
            return vra_segment{3, 0, 0, 0, 0};
        }

        int kv_tile_id = state.schedule_kv_tile;
        state.current_stride = tile_distance_stride_from_coords(
            q_t, q_h, q_w, state.schedule_kv_t, state.schedule_kv_h,
            state.schedule_kv_w, globals.dense_t, globals.dense_h,
            globals.dense_w, globals.mid_t, globals.mid_h, globals.mid_w);
        if (state.current_stride == 1) {
            state.schedule_tile_phase = 1;
            return vra_segment{1, kv_tile_id * TILE_ROWS, 128, 0, 0};
        }
        if (state.current_stride == 2) {
            state.schedule_tile_phase = 1;
            return vra_segment{2, kv_tile_id * TILE_ROWS, 128, 0, 0};
        }

        mixed_vra_advance_kv_tile(globals, state);
        return vra_segment{3, kv_tile_id * TILE_ROWS, 128, 0, 0};
    }
}

template <int DENSE_T,
          int DENSE_H,
          int DENSE_W,
          int MID_T,
          int MID_H,
          int MID_W,
          int GRID_T,
          int GRID_H,
          int GRID_W,
          typename schedule_state_t>
__device__ static inline vra_segment mixed_vra_next_segment_fixed(
    int q_t,
    int q_h,
    int q_w,
    schedule_state_t& state) {
    constexpr int TILE_ROWS = 384;
    constexpr int NUM_TILES = GRID_T * GRID_H * GRID_W;

    while (true) {
        if (state.schedule_tile_phase > 0) {
            int kv_tile_id = state.schedule_kv_tile;
            if (state.current_stride == 1) {
                int phase = state.schedule_tile_phase;
                state.schedule_tile_phase += 1;
                if (state.schedule_tile_phase >= 3) {
                    mixed_vra_advance_kv_tile_fixed<GRID_H, GRID_W>(state);
                }
                return vra_segment{1, kv_tile_id * TILE_ROWS + phase * 128,
                                   128, 0, 0};
            }

            int tail_source_row = kv_tile_id * TILE_ROWS + 256;
            mixed_vra_advance_kv_tile_fixed<GRID_H, GRID_W>(state);
            if (state.pending_tail_source_row >= 0) {
                int first_tail = state.pending_tail_source_row;
                state.pending_tail_source_row = -1;
                return vra_segment{2, first_tail, 64, tail_source_row, 64};
            }
            state.pending_tail_source_row = tail_source_row;
            continue;
        }

        if (state.schedule_kv_tile >= NUM_TILES) {
            if (state.pending_tail_source_row >= 0) {
                int first_tail = state.pending_tail_source_row;
                state.pending_tail_source_row = -1;
                return vra_segment{2, first_tail, 64, 0, 0};
            }
            return vra_segment{3, 0, 0, 0, 0};
        }

        int kv_tile_id = state.schedule_kv_tile;
        int kv_t = kv_tile_id / (GRID_H * GRID_W);
        int kv_rem = kv_tile_id - kv_t * GRID_H * GRID_W;
        int kv_h = kv_rem / GRID_W;
        int kv_w = kv_rem - kv_h * GRID_W;
        state.current_stride = tile_distance_stride_fixed<
            DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W>(
            q_t, q_h, q_w, kv_t, kv_h, kv_w);
        if (state.current_stride == 1) {
            state.schedule_tile_phase = 1;
            return vra_segment{1, kv_tile_id * TILE_ROWS, 128, 0, 0};
        }
        if (state.current_stride == 2) {
            state.schedule_tile_phase = 1;
            return vra_segment{2, kv_tile_id * TILE_ROWS, 128, 0, 0};
        }

        mixed_vra_advance_kv_tile_fixed<GRID_H, GRID_W>(state);
        return vra_segment{3, kv_tile_id * TILE_ROWS, 128, 0, 0};
    }
}

template <typename tile, typename global, int NUM_PRODUCER_WARPS>
__device__ static inline void load_strided_rows_cp_async(tile& dst,
                                                        const global& src,
                                                        int batch,
                                                        int head,
                                                        int source_row,
                                                        int stride,
                                                        int valid_rows,
                                                        int source_row_b,
                                                        int valid_rows_b) {
    using T = typename tile::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int memcpy_per_row = tile::cols / elem_per_memcpy;
    constexpr int total_tile_calls = tile::rows * memcpy_per_row;
    int producer_warp = warpgroup::warpid();
    int producer_tid = producer_warp * kittens::WARP_THREADS + laneid();
    uint32_t dst_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    for (int i = producer_tid;
         i < total_tile_calls;
         i += NUM_PRODUCER_WARPS * kittens::WARP_THREADS) {
        int row = i / memcpy_per_row;
        int col = (i % memcpy_per_row) * elem_per_memcpy;
        if (row < valid_rows) {
            int src_row = source_row + row * stride;
            auto src_coord = coord<>{batch, head, src_row, col};
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                         ::"r"(dst.idx(dst_ptr, {row, col})),
                         "l"(reinterpret_cast<uint64_t>(&src[src_coord]))
                         : "memory");
        } else if (row < valid_rows + valid_rows_b) {
            int second_row = row - valid_rows;
            int src_row = source_row_b + second_row * stride;
            auto src_coord = coord<>{batch, head, src_row, col};
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                         ::"r"(dst.idx(dst_ptr, {row, col})),
                         "l"(reinterpret_cast<uint64_t>(&src[src_coord]))
                         : "memory");
        } else {
            float4 zeros = {0.f, 0.f, 0.f, 0.f};
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int VALID_ROWS,
          int VALID_ROWS_B,
          typename tile,
          typename global,
          int NUM_PRODUCER_WARPS>
__device__ static inline void load_stride2_rows_cp_async_fixed(
    tile& dst,
    const global& src,
    int batch,
    int head,
    int source_row,
    int source_row_b) {
    static_assert(VALID_ROWS == 64 || VALID_ROWS == 128,
                  "stride-2 segments are either full or 64-row tails");
    static_assert(VALID_ROWS_B == 0 || VALID_ROWS_B == 64,
                  "paired stride-2 tails carry 64 rows from the second tile");
    static_assert(VALID_ROWS + VALID_ROWS_B <= tile::rows);
    using T = typename tile::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int memcpy_per_row = tile::cols / elem_per_memcpy;
    constexpr int total_tile_calls = tile::rows * memcpy_per_row;
    int producer_warp = warpgroup::warpid();
    int producer_tid = producer_warp * kittens::WARP_THREADS + laneid();
    uint32_t dst_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    for (int i = producer_tid;
         i < total_tile_calls;
         i += NUM_PRODUCER_WARPS * kittens::WARP_THREADS) {
        int row = i / memcpy_per_row;
        int col = (i % memcpy_per_row) * elem_per_memcpy;
        if (row < VALID_ROWS) {
            int src_row = source_row + row * 2;
            auto src_coord = coord<>{batch, head, src_row, col};
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                         ::"r"(dst.idx(dst_ptr, {row, col})),
                         "l"(reinterpret_cast<uint64_t>(&src[src_coord]))
                         : "memory");
        } else if constexpr (VALID_ROWS_B > 0) {
            if (row < VALID_ROWS + VALID_ROWS_B) {
                int second_row = row - VALID_ROWS;
                int src_row = source_row_b + second_row * 2;
                auto src_coord = coord<>{batch, head, src_row, col};
                asm volatile(
                    "cp.async.cg.shared::cta.global [%0], [%1], 16;\n" ::"r"(
                        dst.idx(dst_ptr, {row, col})),
                    "l"(reinterpret_cast<uint64_t>(&src[src_coord]))
                    : "memory");
            }
        } else if constexpr (VALID_ROWS < tile::rows) {
            float4 zeros = {0.f, 0.f, 0.f, 0.f};
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <bool FUSE_TEXT_QUERY>
struct mixed_vra_common_state {
    int batch;
    int head;
    int seq;
    int q_tile_id;
    int q_t;
    int q_h;
    int q_w;
    int image_num_iters;
};

template <>
struct mixed_vra_common_state<true> {
    int batch;
    int head;
    int seq;
    int q_tile_id;
    int q_t;
    int q_h;
    int q_w;
    int image_num_iters;
    int is_text_query;
    int valid_kv_rows;
};

template <bool FIXED_PATTERN>
struct mixed_vra_schedule_state {
    int schedule_kv_tile;
    int schedule_tile_phase;
    int current_stride;
    int pending_tail_source_row;
};

template <>
struct mixed_vra_schedule_state<false> {
    int schedule_kv_tile;
    int schedule_kv_t;
    int schedule_kv_h;
    int schedule_kv_w;
    int schedule_tile_phase;
    int current_stride;
    int pending_tail_source_row;
};

template <int D, int NUM_WORKERS, bool FUSE_TEXT_QUERY, bool FIXED_PATTERN>
struct mixed_vra_attn_fwd_layout {
    using qo_tile = st_bf<64, D>;
    using kv_tile = st_bf<128, D>;
    using qo_global = gl<bf16, -1, -1, -1, D, qo_tile>;
    using dense_kv_global = gl<bf16, -1, -1, -1, D, kv_tile>;
    using stride2_kv_global = row_strided_kv_global<D, 2>;
    using stride3_kv_global = row_strided_kv_global<D, 3>;

    struct globals {
        qo_global o;
        qo_global q;
        dense_kv_global k_dense;
        dense_kv_global v_dense;
        stride2_kv_global k_stride2;
        stride2_kv_global v_stride2;
        stride3_kv_global k_stride3;
        stride3_kv_global v_stride3;
        int dense_t;
        int dense_h;
        int dense_w;
        int mid_t;
        int mid_h;
        int mid_w;
        int grid_t;
        int grid_h;
        int grid_w;
        int image_rows;
        int text_length;
    };

    struct input_block {
        kv_tile k;
        kv_tile v;
    };

    struct scratch_block {
        qo_tile q[NUM_WORKERS];
    };

    using common_state = mixed_vra_common_state<FUSE_TEXT_QUERY>;

    using producer_state = mixed_vra_schedule_state<FIXED_PATTERN>;

    struct consumer_state : mixed_vra_schedule_state<FIXED_PATTERN> {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec;
        col_vec<rt_fl<16, kv_tile::rows>> norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};

template <int D,
          bool HAS_TEXT = false,
          bool FUSE_TEXT_QUERY = false,
          bool FIXED_PATTERN = false,
          int DENSE_T = -1,
          int DENSE_H = -1,
          int DENSE_W = -1,
          int MID_T = -1,
          int MID_H = -1,
          int MID_W = -1,
          int GRID_T = -1,
          int GRID_H = -1,
          int GRID_W = -1>
struct mixed_vra_attn_fwd_template {
    static_assert(!FUSE_TEXT_QUERY || HAS_TEXT,
                  "FUSE_TEXT_QUERY requires HAS_TEXT");
    static constexpr int NUM_CONSUMER_WARPS = 12;
    static constexpr int NUM_WORKERS = NUM_CONSUMER_WARPS / 4;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY - 2048;

    using layout = mixed_vra_attn_fwd_layout<D, NUM_WORKERS, FUSE_TEXT_QUERY,
                                             FIXED_PATTERN>;

    __device__ static inline void common_setup(
        common_setup_args<layout> args) {
        int task_id = gridDim.x * args.task_iter + blockIdx.x;
        int seq_q = (args.globals.q.rows() +
                     NUM_WORKERS * layout::qo_tile::rows - 1) /
                    (NUM_WORKERS * layout::qo_tile::rows);
        int total_tasks = args.globals.q.batch() * args.globals.q.depth() * seq_q;
        if (task_id >= total_tasks) {
            args.num_iters = -1;
            return;
        }

        args.common.batch = task_id / (seq_q * args.globals.q.depth());
        task_id -= args.common.batch * seq_q * args.globals.q.depth();
        args.common.head = task_id / seq_q;
        task_id -= args.common.head * seq_q;
        args.common.seq = task_id;
        if constexpr (FUSE_TEXT_QUERY) {
            args.common.is_text_query = 0;
            args.common.valid_kv_rows =
                args.globals.image_rows + args.globals.text_length;
            int image_seq_tasks =
                (args.globals.image_rows +
                 NUM_WORKERS * layout::qo_tile::rows - 1) /
                (NUM_WORKERS * layout::qo_tile::rows);
            if (args.common.seq >= image_seq_tasks) {
                args.common.q_tile_id = -1;
                args.common.q_t = 0;
                args.common.q_h = 0;
                args.common.q_w = 0;
                args.common.image_num_iters = 0;
                args.common.is_text_query = 1;
                args.num_iters = (args.common.valid_kv_rows + 127) / 128;
                return;
            }
        }
        args.common.q_tile_id = args.common.seq / 2;
        if constexpr (FIXED_PATTERN) {
            args.common.q_t = args.common.q_tile_id / (GRID_H * GRID_W);
            int q_rem = args.common.q_tile_id -
                        args.common.q_t * GRID_H * GRID_W;
            args.common.q_h = q_rem / GRID_W;
            args.common.q_w = q_rem - args.common.q_h * GRID_W;
            args.common.image_num_iters = mixed_vra_num_iters_fixed<
                DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W, GRID_T,
                GRID_H, GRID_W>(args.common.q_t, args.common.q_h,
                                args.common.q_w);
        } else {
            args.common.q_t = args.common.q_tile_id /
                              (args.globals.grid_h * args.globals.grid_w);
            int q_rem = args.common.q_tile_id -
                        args.common.q_t * args.globals.grid_h *
                            args.globals.grid_w;
            args.common.q_h = q_rem / args.globals.grid_w;
            args.common.q_w = q_rem - args.common.q_h * args.globals.grid_w;
            args.common.image_num_iters =
                mixed_vra_num_iters(args.globals, args.common.q_t,
                                    args.common.q_h, args.common.q_w);
        }
        if constexpr (HAS_TEXT) {
            args.num_iters = args.common.image_num_iters +
                             ((args.globals.text_length + 127) / 128);
        } else {
            args.num_iters = args.common.image_num_iters;
        }
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            if constexpr (FIXED_PATTERN) {
                mixed_vra_schedule_reset_fixed(args.state);
            } else {
                mixed_vra_schedule_reset(args.state);
            }
        }

        __device__ static inline void load(producer_load_args<layout> args) {
            vra_segment segment;
            if constexpr (FUSE_TEXT_QUERY) {
                if (args.common.is_text_query) {
                    int source_row = args.iter * 128;
                    int valid_rows = args.common.valid_kv_rows - source_row;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{1, source_row, valid_rows, 0, 0};
                } else if (args.iter >= args.common.image_num_iters) {
                    int text_iter = args.iter - args.common.image_num_iters;
                    int text_offset = text_iter * 128;
                    int valid_rows = args.globals.text_length - text_offset;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{
                        1, args.globals.image_rows + text_offset, valid_rows,
                        0, 0};
                } else {
                    if constexpr (FIXED_PATTERN) {
                        segment = mixed_vra_next_segment_fixed<
                            DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                            GRID_T, GRID_H, GRID_W>(
                            args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    } else {
                        segment = mixed_vra_next_segment(
                            args.globals, args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    }
                }
            } else if constexpr (HAS_TEXT) {
                if (args.iter >= args.common.image_num_iters) {
                    int text_iter = args.iter - args.common.image_num_iters;
                    int text_offset = text_iter * 128;
                    int valid_rows = args.globals.text_length - text_offset;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{
                        1, args.globals.image_rows + text_offset, valid_rows,
                        0, 0};
                } else {
                    if constexpr (FIXED_PATTERN) {
                        segment = mixed_vra_next_segment_fixed<
                            DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                            GRID_T, GRID_H, GRID_W>(
                            args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    } else {
                        segment = mixed_vra_next_segment(
                            args.globals, args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    }
                }
            } else {
                if constexpr (FIXED_PATTERN) {
                    segment = mixed_vra_next_segment_fixed<
                        DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                        GRID_T, GRID_H, GRID_W>(
                        args.common.q_t, args.common.q_h, args.common.q_w,
                        args.state);
                } else {
                    segment = mixed_vra_next_segment(
                        args.globals, args.common.q_t, args.common.q_h,
                        args.common.q_w, args.state);
                }
            }
            if (segment.stride == 2) {
                if (segment.valid_rows == 128) {
                    if (warpgroup::warpid() == 0) {
                        tma::expect(args.inputs_arrived, args.input);
                        coord<> src_coord{args.common.batch,
                                          args.common.head,
                                          segment.source_row / 2, 0};
                        tma::load_async(args.input.k, args.globals.k_stride2,
                                        src_coord, args.inputs_arrived);
                        tma::load_async(args.input.v, args.globals.v_stride2,
                                        src_coord, args.inputs_arrived);
                    } else if (laneid() == 0) {
                        arrive(args.inputs_arrived);
                    }
                } else if (segment.valid_rows_b > 0) {
                    load_stride2_rows_cp_async_fixed<
                        64, 64, typename layout::kv_tile,
                        typename layout::dense_kv_global, NUM_PRODUCER_WARPS>(
                        args.input.k, args.globals.k_dense, args.common.batch,
                        args.common.head, segment.source_row,
                        segment.source_row_b);
                    load_stride2_rows_cp_async_fixed<
                        64, 64, typename layout::kv_tile,
                        typename layout::dense_kv_global, NUM_PRODUCER_WARPS>(
                        args.input.v, args.globals.v_dense, args.common.batch,
                        args.common.head, segment.source_row,
                        segment.source_row_b);
                } else {
                    load_stride2_rows_cp_async_fixed<
                        64, 0, typename layout::kv_tile,
                        typename layout::dense_kv_global, NUM_PRODUCER_WARPS>(
                        args.input.k, args.globals.k_dense, args.common.batch,
                        args.common.head, segment.source_row,
                        segment.source_row_b);
                    load_stride2_rows_cp_async_fixed<
                        64, 0, typename layout::kv_tile,
                        typename layout::dense_kv_global, NUM_PRODUCER_WARPS>(
                        args.input.v, args.globals.v_dense, args.common.batch,
                        args.common.head, segment.source_row,
                        segment.source_row_b);
                }
                if (segment.valid_rows < 128) {
                    asm volatile("cp.async.wait_all;\n" ::: "memory");
                    __syncwarp();
                    warpgroup::sync(12);
                    if (laneid() == 0) {
                        arrive(args.inputs_arrived);
                    }
                }
                return;
            }

            if (warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                if (segment.stride == 1) {
                    coord<> src_coord{args.common.batch, args.common.head,
                                      segment.source_row, 0};
                    tma::load_async(args.input.k, args.globals.k_dense,
                                    src_coord, args.inputs_arrived);
                    tma::load_async(args.input.v, args.globals.v_dense,
                                    src_coord, args.inputs_arrived);
                } else {
                    coord<> src_coord{args.common.batch, args.common.head,
                                      segment.source_row / 3, 0};
                    tma::load_async(args.input.k, args.globals.k_stride3,
                                    src_coord, args.inputs_arrived);
                    tma::load_async(args.input.v, args.globals.v_stride3,
                                    src_coord, args.inputs_arrived);
                }
            } else if (laneid() == 0) {
                arrive(args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            int q_block = args.common.seq * NUM_WORKERS + warpgroup::groupid();
            if (q_block * layout::qo_tile::rows < args.globals.q.rows()) {
                warpgroup::load(args.scratch.q[warpgroup::groupid()],
                                args.globals.q,
                                {args.common.batch, args.common.head, q_block,
                                 0});
            }
            args.state.o_reg = 0.f;
            args.state.norm_vec = 0.f;
            args.state.max_vec = base_types::constants<float>::neg_infty();
            if constexpr (FIXED_PATTERN) {
                mixed_vra_schedule_reset_fixed(args.state);
            } else {
                mixed_vra_schedule_reset(args.state);
            }
            warpgroup::sync(warpgroup::groupid());
        }

        __device__ static inline void compute(
            consumer_compute_args<layout> args) {
            constexpr float TEMPERATURE_SCALE =
                (D == 128) ? 0.08838834764f * 1.44269504089f
                           : 0.125f * 1.44269504089f;
            vra_segment segment;
            if constexpr (FUSE_TEXT_QUERY) {
                if (args.common.is_text_query) {
                    int source_row = args.iter * 128;
                    int valid_rows = args.common.valid_kv_rows - source_row;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{1, source_row, valid_rows, 0, 0};
                } else if (args.iter >= args.common.image_num_iters) {
                    int text_iter = args.iter - args.common.image_num_iters;
                    int text_offset = text_iter * 128;
                    int valid_rows = args.globals.text_length - text_offset;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{
                        1, args.globals.image_rows + text_offset, valid_rows,
                        0, 0};
                } else {
                    if constexpr (FIXED_PATTERN) {
                        segment = mixed_vra_next_segment_fixed<
                            DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                            GRID_T, GRID_H, GRID_W>(
                            args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    } else {
                        segment = mixed_vra_next_segment(
                            args.globals, args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    }
                }
            } else if constexpr (HAS_TEXT) {
                if (args.iter >= args.common.image_num_iters) {
                    int text_iter = args.iter - args.common.image_num_iters;
                    int text_offset = text_iter * 128;
                    int valid_rows = args.globals.text_length - text_offset;
                    if (valid_rows > 128) {
                        valid_rows = 128;
                    }
                    segment = vra_segment{
                        1, args.globals.image_rows + text_offset, valid_rows,
                        0, 0};
                } else {
                    if constexpr (FIXED_PATTERN) {
                        segment = mixed_vra_next_segment_fixed<
                            DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                            GRID_T, GRID_H, GRID_W>(
                            args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    } else {
                        segment = mixed_vra_next_segment(
                            args.globals, args.common.q_t, args.common.q_h,
                            args.common.q_w, args.state);
                    }
                }
            } else {
                if constexpr (FIXED_PATTERN) {
                    segment = mixed_vra_next_segment_fixed<
                        DENSE_T, DENSE_H, DENSE_W, MID_T, MID_H, MID_W,
                        GRID_T, GRID_H, GRID_W>(
                        args.common.q_t, args.common.q_h, args.common.q_w,
                        args.state);
                } else {
                    segment = mixed_vra_next_segment(
                        args.globals, args.common.q_t, args.common.q_h,
                        args.common.q_w, args.state);
                }
            }

            warpgroup::mm<transpose::N, transpose::T>(
                args.state.att_block,
                args.scratch.q[warpgroup::groupid()],
                args.input.k);
            args.state.max_vec_last_scaled =
                args.state.max_vec * TEMPERATURE_SCALE;
            warpgroup::mma_async_wait();

            right_fill(args.state.att_block, args.state.att_block,
                       segment.valid_rows + segment.valid_rows_b,
                       base_types::constants<float>::neg_infty());
            args.state.max_vec =
                max<axis::COL>(args.state.att_block, args.state.max_vec);
            args.state.max_vec_scaled =
                args.state.max_vec * TEMPERATURE_SCALE;
            args.state.att_block =
                exp2((args.state.att_block * TEMPERATURE_SCALE) -
                     args.state.max_vec_scaled);
            args.state.max_vec_last_scaled =
                exp2(args.state.max_vec_last_scaled -
                     args.state.max_vec_scaled);
            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec =
                sum<axis::COL>(args.state.att_block, args.state.norm_vec);
            args.state.o_reg *= args.state.max_vec_last_scaled;
            args.state.att_block_mma = args.state.att_block;

            warpgroup::mma<transpose::N, transpose::N>(
                args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();

            if (laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static inline void finish(
            consumer_finish_args<layout> args) {
            int q_block = args.common.seq * NUM_WORKERS + warpgroup::groupid();
            if (q_block * layout::qo_tile::rows < args.globals.q.rows()) {
                args.state.o_reg /= args.state.norm_vec;
                auto& o_smem =
                    reinterpret_cast<typename layout::qo_tile&>(
                        args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if (warpgroup::warpid() == 0) {
                    tma::store_async(args.globals.o, o_smem,
                                     {args.common.batch, args.common.head,
                                      q_block, 0});
                }
                tma::store_async_read_wait();
            }
            __syncwarp();
            if (laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

template <int D, bool GATHER_STRIDE3>
void launch_attn(torch::Tensor q,
                 torch::Tensor k,
                 torch::Tensor v,
                 torch::Tensor o,
                 int valid_kv_rows,
                 cudaStream_t stream) {
    using ker_template = packed_attn_fwd_template<D, GATHER_STRIDE3>;
    using q_global = typename ker_template::layout::qo_global;
    using k_global = typename ker_template::layout::kv_global;
    using v_global = typename ker_template::layout::kv_global;
    using o_global = typename ker_template::layout::qo_global;

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto q_seq_len = q.size(2);
    auto kv_seq_len = k.size(2);

    bf16* d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());

    q_global qg{d_q, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(q_seq_len), nullptr};
    k_global kg{d_k, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(kv_seq_len), nullptr};
    v_global vg{d_v, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(kv_seq_len), nullptr};
    o_global og{d_o, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(q_seq_len), nullptr};

    typename ker_template::layout::globals globals{og, qg, kg, vg,
                                                   valid_kv_rows};
    constexpr int mem_size = ker_template::MAX_SHARED_MEMORY;
    constexpr int block_size = prototype::detail::NUM_THREADS_v<ker_template>;

    int q_blocks_per_task = ker_template::NUM_WORKERS * 64;
    int tasks = static_cast<int>(
        batch * heads * ((q_seq_len + q_blocks_per_task - 1) /
                         q_blocks_per_task));

    int device = q.get_device();
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    int grid_x = std::min(tasks, prop.multiProcessorCount);

    cudaFuncSetAttribute(prototype::lcf::kernel<ker_template>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         mem_size);
    prototype::lcf::kernel<ker_template>
        <<<grid_x, block_size, mem_size, stream>>>(globals);
}

template <int D,
          bool HAS_TEXT = false,
          bool FUSE_TEXT_QUERY = false,
          bool FIXED_PATTERN = false,
          int DENSE_T = -1,
          int DENSE_H = -1,
          int DENSE_W = -1,
          int MID_T = -1,
          int MID_H = -1,
          int MID_W = -1,
          int GRID_T = -1,
          int GRID_H = -1,
          int GRID_W = -1>
void launch_mixed_vra_attn(torch::Tensor q,
                           torch::Tensor k,
                           torch::Tensor v,
                           torch::Tensor o,
                           int dense_t,
                           int dense_h,
                           int dense_w,
                           int mid_t,
                           int mid_h,
                           int mid_w,
                           int grid_t,
                           int grid_h,
                           int grid_w,
                           int text_length,
                           cudaStream_t stream) {
    using ker_template = mixed_vra_attn_fwd_template<
        D, HAS_TEXT, FUSE_TEXT_QUERY, FIXED_PATTERN, DENSE_T, DENSE_H,
        DENSE_W, MID_T, MID_H, MID_W, GRID_T, GRID_H, GRID_W>;
    using q_global = typename ker_template::layout::qo_global;
    using dense_kv_global = typename ker_template::layout::dense_kv_global;
    using stride2_kv_global = typename ker_template::layout::stride2_kv_global;
    using stride3_kv_global = typename ker_template::layout::stride3_kv_global;
    using o_global = typename ker_template::layout::qo_global;

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto q_seq_len = q.size(2);
    auto kv_seq_len = k.size(2);
    int image_rows = grid_t * grid_h * grid_w * 384;

    bf16* d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());

    q_global qg{d_q, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(q_seq_len), nullptr};
    dense_kv_global kg_dense{d_k, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(heads),
                             static_cast<unsigned int>(kv_seq_len), nullptr};
    dense_kv_global vg_dense{d_v, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(heads),
                             static_cast<unsigned int>(kv_seq_len), nullptr};
    stride2_kv_global kg_stride2{d_k, static_cast<unsigned int>(batch),
                                 static_cast<unsigned int>(heads),
                                 static_cast<unsigned int>(kv_seq_len),
                                 nullptr};
    stride2_kv_global vg_stride2{d_v, static_cast<unsigned int>(batch),
                                 static_cast<unsigned int>(heads),
                                 static_cast<unsigned int>(kv_seq_len),
                                 nullptr};
    stride3_kv_global kg_stride3{d_k, static_cast<unsigned int>(batch),
                                 static_cast<unsigned int>(heads),
                                 static_cast<unsigned int>(kv_seq_len),
                                 nullptr};
    stride3_kv_global vg_stride3{d_v, static_cast<unsigned int>(batch),
                                 static_cast<unsigned int>(heads),
                                 static_cast<unsigned int>(kv_seq_len),
                                 nullptr};
    o_global og{d_o, static_cast<unsigned int>(batch),
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>(q_seq_len), nullptr};

    typename ker_template::layout::globals globals{
        og, qg, kg_dense, vg_dense, kg_stride2, vg_stride2, kg_stride3, vg_stride3,
        dense_t, dense_h, dense_w, mid_t, mid_h, mid_w, grid_t, grid_h,
        grid_w, image_rows, text_length};
    constexpr int mem_size = ker_template::MAX_SHARED_MEMORY;
    constexpr int block_size = prototype::detail::NUM_THREADS_v<ker_template>;

    int q_blocks_per_task = ker_template::NUM_WORKERS * 64;
    int tasks = static_cast<int>(
        batch * heads * ((q_seq_len + q_blocks_per_task - 1) /
                         q_blocks_per_task));

    int device = q.get_device();
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    int grid_x = std::min(tasks, prop.multiProcessorCount);

    cudaFuncSetAttribute(prototype::lcf::kernel<ker_template>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         mem_size);
    prototype::lcf::kernel<ker_template>
        <<<grid_x, block_size, mem_size, stream>>>(globals);
}

template <int D>
void dispatch_mixed_vra_attn(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             torch::Tensor o,
                             int dense_t,
                             int dense_h,
                             int dense_w,
                             int mid_t,
                             int mid_h,
                             int mid_w,
                             int grid_t,
                             int grid_h,
                             int grid_w,
                             int text_length,
                             cudaStream_t stream) {
    int64_t image_rows = static_cast<int64_t>(grid_t) * grid_h * grid_w * 384;
    bool fuse_text_query = text_length > 0 && q.size(2) > image_rows;
    if (dense_t == 1 && dense_h == 1 && dense_w == 1 && mid_t == 2 &&
        mid_h == 2 && mid_w == 3 && grid_t == 3 && grid_h == 6 &&
        grid_w == 10) {
        if (text_length > 0) {
            if (fuse_text_query) {
                launch_mixed_vra_attn<D, true, true, true, 1, 1, 1, 2, 2, 3,
                                      3, 6, 10>(
                    q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h,
                    mid_w, grid_t, grid_h, grid_w, text_length, stream);
            } else {
                launch_mixed_vra_attn<D, true, false, true, 1, 1, 1, 2, 2,
                                      3, 3, 6, 10>(
                    q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h,
                    mid_w, grid_t, grid_h, grid_w, text_length, stream);
            }
        } else {
            launch_mixed_vra_attn<D, false, false, true, 1, 1, 1, 2, 2, 3,
                                  3, 6, 10>(
                q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h, mid_w,
                grid_t, grid_h, grid_w, text_length, stream);
        }
        return;
    }
    if (dense_t == 1 && dense_h == 1 && dense_w == 1 && mid_t == 2 &&
        mid_h == 2 && mid_w == 3 && grid_t == 5 && grid_h == 6 &&
        grid_w == 10) {
        if (text_length > 0) {
            if (fuse_text_query) {
                launch_mixed_vra_attn<D, true, true, true, 1, 1, 1, 2, 2, 3,
                                      5, 6, 10>(
                    q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h,
                    mid_w, grid_t, grid_h, grid_w, text_length, stream);
            } else {
                launch_mixed_vra_attn<D, true, false, true, 1, 1, 1, 2, 2,
                                      3, 5, 6, 10>(
                    q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h,
                    mid_w, grid_t, grid_h, grid_w, text_length, stream);
            }
        } else {
            launch_mixed_vra_attn<D, false, false, true, 1, 1, 1, 2, 2, 3,
                                  5, 6, 10>(
                q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h, mid_w,
                grid_t, grid_h, grid_w, text_length, stream);
        }
        return;
    }
    if (text_length > 0) {
        if (fuse_text_query) {
            launch_mixed_vra_attn<D, true, true>(
                q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h, mid_w,
                grid_t, grid_h, grid_w, text_length, stream);
        } else {
            launch_mixed_vra_attn<D, true, false>(
                q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h, mid_w,
                grid_t, grid_h, grid_w, text_length, stream);
        }
    } else {
        launch_mixed_vra_attn<D, false, false>(
            q, k, v, o, dense_t, dense_h, dense_w, mid_t, mid_h, mid_w,
            grid_t, grid_h, grid_w, text_length, stream);
    }
}

}  // namespace

torch::Tensor packed_attn_h100(torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
                "packed_attn_h100 currently expects bf16 q");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
                "packed_attn_h100 currently expects bf16 k");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16,
                "packed_attn_h100 currently expects bf16 v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v batch dimensions must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "packed_attn_h100 currently requires q/k/v heads to match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, and v head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k and v sequence lengths must match");
    TORCH_CHECK(q.size(2) % 192 == 0,
                "q sequence length must be divisible by 192");
    TORCH_CHECK(k.size(2) % 128 == 0,
                "packed kv sequence length must be divisible by 128");

    auto o = torch::empty_like(q);
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q.size(3) == 64) {
        launch_attn<64, false>(q, k, v, o, static_cast<int>(k.size(2)),
                               stream);
    } else if (q.size(3) == 128) {
        launch_attn<128, false>(q, k, v, o, static_cast<int>(k.size(2)),
                                stream);
    } else {
        TORCH_CHECK(false,
                    "packed_attn_h100 only supports head_dim 64 or 128");
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return o;
}

torch::Tensor dense_attn_h100_valid_kv(torch::Tensor q,
                                       torch::Tensor k,
                                       torch::Tensor v,
                                       int valid_kv_rows) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
                "dense_attn_h100_valid_kv currently expects bf16 q");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
                "dense_attn_h100_valid_kv currently expects bf16 k");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16,
                "dense_attn_h100_valid_kv currently expects bf16 v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v batch dimensions must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "dense_attn_h100_valid_kv currently requires q/k/v heads to match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, and v head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k and v sequence lengths must match");
    TORCH_CHECK(q.size(2) % 64 == 0,
                "q sequence length must be divisible by 64");
    TORCH_CHECK(k.size(2) % 128 == 0,
                "padded kv sequence length must be divisible by 128");
    TORCH_CHECK(valid_kv_rows > 0 && valid_kv_rows <= k.size(2),
                "valid_kv_rows must be in (0, padded kv sequence length]");

    auto o = torch::empty_like(q);
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q.size(3) == 64) {
        launch_attn<64, false>(q, k, v, o, valid_kv_rows, stream);
    } else if (q.size(3) == 128) {
        launch_attn<128, false>(q, k, v, o, valid_kv_rows, stream);
    } else {
        TORCH_CHECK(false,
                    "dense_attn_h100_valid_kv only supports head_dim 64 or 128");
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return o;
}

torch::Tensor stride3_attn_h100(torch::Tensor q,
                                torch::Tensor k,
                                torch::Tensor v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
                "stride3_attn_h100 currently expects bf16 q");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
                "stride3_attn_h100 currently expects bf16 k");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16,
                "stride3_attn_h100 currently expects bf16 v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v batch dimensions must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "stride3_attn_h100 currently requires q/k/v heads to match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, and v head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k and v sequence lengths must match");
    TORCH_CHECK(q.size(2) % 192 == 0,
                "q sequence length must be divisible by 192");
    TORCH_CHECK(k.size(2) % 384 == 0,
                "dense kv sequence length must be divisible by 384");

    auto o = torch::empty_like(q);
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q.size(3) == 64) {
        launch_attn<64, true>(q, k, v, o, static_cast<int>(k.size(2) / 3),
                              stream);
    } else if (q.size(3) == 128) {
        launch_attn<128, true>(q, k, v, o, static_cast<int>(k.size(2) / 3),
                               stream);
    } else {
        TORCH_CHECK(false,
                    "stride3_attn_h100 only supports head_dim 64 or 128");
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return o;
}

torch::Tensor mixed_vra_attn_h100(torch::Tensor q,
                                  torch::Tensor k,
                                  torch::Tensor v,
                                  int dense_t,
                                  int dense_h,
                                  int dense_w,
                                  int mid_t,
                                  int mid_h,
                                  int mid_w,
                                  int grid_t,
                                  int grid_h,
                                  int grid_w,
                                  int text_length) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16,
                "mixed_vra_attn_h100 currently expects bf16 q");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
                "mixed_vra_attn_h100 currently expects bf16 k");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16,
                "mixed_vra_attn_h100 currently expects bf16 v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v batch dimensions must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "mixed_vra_attn_h100 currently requires q/k/v heads to match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, and v head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k and v sequence lengths must match");
    TORCH_CHECK(text_length >= 0,
                "mixed_vra_attn_h100 requires non-negative text_length");
    int64_t image_seq_len = static_cast<int64_t>(grid_t) * grid_h * grid_w * 384;
    TORCH_CHECK(q.size(2) >= image_seq_len,
                "q must contain at least the image sequence length");
    if (text_length == 0) {
        TORCH_CHECK(q.size(2) % 192 == 0,
                    "image-only q sequence length must be divisible by 192");
        TORCH_CHECK(image_seq_len == q.size(2),
                    "tile grid must match q sequence length when text_length is zero");
    } else {
        TORCH_CHECK(q.size(2) % 64 == 0,
                    "text q sequence length must be divisible by 64");
        int64_t padded_text_query_rows = q.size(2) - image_seq_len;
        TORCH_CHECK(padded_text_query_rows == 0 ||
                        padded_text_query_rows >= text_length,
                    "q must contain image rows and optionally padded text query rows");
    }
    int64_t padded_text_rows = k.size(2) - image_seq_len;
    int64_t required_text_rows = ((static_cast<int64_t>(text_length) + 127) /
                                  128) *
                                 128;
    TORCH_CHECK(padded_text_rows >= required_text_rows,
                "mixed_vra_attn_h100 requires k/v to include padded text rows");

    auto o = torch::empty_like(q);
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q.size(3) == 64) {
        dispatch_mixed_vra_attn<64>(q, k, v, o, dense_t, dense_h, dense_w,
                                    mid_t, mid_h, mid_w, grid_t, grid_h,
                                    grid_w, text_length, stream);
    } else if (q.size(3) == 128) {
        dispatch_mixed_vra_attn<128>(q, k, v, o, dense_t, dense_h, dense_w,
                                     mid_t, mid_h, mid_w, grid_t, grid_h,
                                     grid_w, text_length, stream);
    } else {
        TORCH_CHECK(false,
                    "mixed_vra_attn_h100 only supports head_dim 64 or 128");
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return o;
}

void register_packed_attn_h100(pybind11::module_& m) {
    m.def("packed_attn_h100", torch::wrap_pybind_function(packed_attn_h100),
          "Dense H100 attention over packed K/V");
    m.def("dense_attn_h100_valid_kv",
          torch::wrap_pybind_function(dense_attn_h100_valid_kv),
          "Dense H100 attention over padded K/V with an explicit valid row count");
    m.def("stride3_attn_h100", torch::wrap_pybind_function(stride3_attn_h100),
          "Experimental H100 attention with in-kernel stride-3 K/V gather");
    m.def("mixed_vra_attn_h100",
          torch::wrap_pybind_function(mixed_vra_attn_h100),
          "Experimental H100 attention with dense/stride-2/stride-3 VRA K/V schedule");
}
