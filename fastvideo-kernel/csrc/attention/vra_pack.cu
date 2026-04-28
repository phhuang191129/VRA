#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

template <typename scalar_t>
__global__ void vra_pack_kv_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   int64_t total_elements,
                                   int64_t heads,
                                   int64_t seq_len,
                                   int64_t dim,
                                   int64_t img_seq_len,
                                   int64_t packed_seq_len,
                                   int64_t packed_img_len,
                                   int64_t stride) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (; linear < total_elements; linear += step) {
        int64_t d = linear % dim;
        int64_t tmp = linear / dim;
        int64_t packed_s = tmp % packed_seq_len;
        tmp /= packed_seq_len;
        int64_t h = tmp % heads;
        int64_t b = tmp / heads;

        int64_t src_s = packed_s < packed_img_len
                            ? packed_s * stride
                            : img_seq_len + (packed_s - packed_img_len);
        output[linear] = input[((b * heads + h) * seq_len + src_s) * dim + d];
    }
}

__global__ void vra_pack_kv_vec16_kernel(const uint4* __restrict__ input,
                                         uint4* __restrict__ output,
                                         int64_t total_vecs,
                                         int64_t heads,
                                         int64_t seq_len,
                                         int64_t row_vecs,
                                         int64_t img_seq_len,
                                         int64_t packed_seq_len,
                                         int64_t packed_img_len,
                                         int64_t stride) {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (; linear < total_vecs; linear += step) {
        int64_t vec_in_row = linear % row_vecs;
        int64_t tmp = linear / row_vecs;
        int64_t packed_s = tmp % packed_seq_len;
        tmp /= packed_seq_len;
        int64_t h = tmp % heads;
        int64_t b = tmp / heads;

        int64_t src_s = packed_s < packed_img_len
                            ? packed_s * stride
                            : img_seq_len + (packed_s - packed_img_len);
        output[linear] =
            input[((b * heads + h) * seq_len + src_s) * row_vecs + vec_in_row];
    }
}

}  // namespace

torch::Tensor vra_pack_kv(torch::Tensor input,
                          int64_t img_seq_len,
                          int64_t text_length,
                          int64_t stride) {
    TORCH_CHECK(input.is_cuda(), "vra_pack_kv expects a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must have shape [B,H,S,D]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(img_seq_len >= 0, "img_seq_len must be non-negative");
    TORCH_CHECK(text_length >= 0, "text_length must be non-negative");
    TORCH_CHECK(input.size(2) == img_seq_len + text_length,
                "input seq_len does not match img_seq_len + text_length");

    const auto batch = input.size(0);
    const auto heads = input.size(1);
    const auto seq_len = input.size(2);
    const auto dim = input.size(3);
    const auto row_bytes = dim * input.element_size();
    const auto packed_img_len = (img_seq_len + stride - 1) / stride;
    const auto packed_seq_len = packed_img_len + text_length;

    auto output = torch::empty({batch, heads, packed_seq_len, dim},
                               input.options());
    const int64_t total_elements = output.numel();
    if (total_elements == 0) {
        return output;
    }

    const c10::cuda::OptionalCUDAGuard device_guard(input.device());
    const int threads = 256;
    const int64_t max_blocks = 65535;
    const int blocks =
        static_cast<int>(std::min<int64_t>((total_elements + threads - 1) /
                                               threads,
                                           max_blocks));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto input_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
    const auto output_addr = reinterpret_cast<uintptr_t>(output.data_ptr());
    if (row_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
        input_addr % alignof(uint4) == 0 &&
        output_addr % alignof(uint4) == 0) {
        const auto row_vecs = row_bytes / static_cast<int64_t>(sizeof(uint4));
        const auto total_vecs = batch * heads * packed_seq_len * row_vecs;
        const int vec_blocks =
            static_cast<int>(std::min<int64_t>((total_vecs + threads - 1) /
                                                   threads,
                                               max_blocks));
        vra_pack_kv_vec16_kernel<<<vec_blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(input.data_ptr()),
            reinterpret_cast<uint4*>(output.data_ptr()), total_vecs, heads,
            seq_len, row_vecs, img_seq_len, packed_seq_len, packed_img_len,
            stride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "vra_pack_kv", [&] {
            vra_pack_kv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                total_elements, heads, seq_len, dim, img_seq_len,
                packed_seq_len, packed_img_len, stride);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void register_vra_pack(pybind11::module_& m) {
    m.def("vra_pack_kv", torch::wrap_pybind_function(vra_pack_kv),
          "Pack image K/V rows by stride while preserving text rows");
}
