# Attention Backend Development Guide

This document describes the structure of the attention backend modules in `fastvideo/attention/backends/` and serves as a guide for implementing new attention backend algorithms.

## Overview
The attention backends abstract the complexity of integrating custom attention kernels (e.g., Sparse Attention, Sliding Tile Attention) into the diffusion transformer models. By separating metadata generation, tensor preprocessing/postprocessing, and kernel dispatching, the codebase supports modular and easily extensible attention mechanisms.

### Key Components

To implement a new attention backend, you must typically define four components by inheriting from the abstract base classes in `fastvideo.attention.backends.abstract` (or reusing existing ones if applicable):

1. **`AttentionBackend`**: The entry point/factory class.
2. **`AttentionMetadata`**: A dataclass to hold forward-pass specific data (like current timestep or partition indices).
3. **`AttentionMetadataBuilder`**: A builder class that constructs the `AttentionMetadata` on the fly.
4. **`AttentionImpl`**: The actual implementation layer that pre/post-processes inputs and executes the attention kernel.

---

## Detailed File Breakdown

### 1. `sliding_tile_attn.py` (Sliding Tile Attention)
This module implements the Sliding Tile Attention (STA), where different heads or layers can use varying local 3D window structures.
*   **`SlidingTileAttentionBackend`**: Registers the name `SLIDING_TILE_ATTN` and points to its specific Impl and Metadata classes.
*   **`SlidingTileAttentionMetadata` & `SlidingTileAttentionMetadataBuilder`**: Keeps track of `current_timestep` and the corresponding `STA_param` to select the right mask strategy window size for each layer.
*   **`SlidingTileAttentionImpl`**: Handles tiling (`tile()`) and untiling (`untile()`) to remap video patches into sequential tiles. During `forward()`, it unpacks keys/values and passes them to the `fastvideo_kernel.sliding_tile_attention`. It also features logic for a "searching" phase, computing L1/L2 differences against full attention for hyperparameter tuning.

### 2. `STA_configuration.py` (STA Configuration & Search Logic)
Provides hyperparameter loading, searching, and caching routines exclusively for STA. 
*   **`configure_sta`**: Configures the attention behavior based on four modes:
    *   `STA_searching`: Generates 3D masks to test several block configurations.
    *   `STA_tuning` / `STA_tuning_cfg`: Computes average L2 loss variations across head permutations, returning an optimized sparse mask strategy maximizing sparsity while maintaining minimal loss.
    *   `STA_inference`: Loads the best saved mask configurations for production.

### 3. `video_sparse_attn.py` (Video Sparse Attention)
This file implements a dynamic sparse attention relying on variable block lengths and top-k block retrievals.
*   **Helper Functions**: Contains heavily `lru_cache`-optimized math formulas to precompute non-padded tensor representations (`get_tile_partition_indices`, `construct_variable_block_sizes`).
*   **`VideoSparseAttentionMetadata` & `Builder`**: Builds the necessary shapes, precomputed indexing matrices, block lengths, and top-k sparsity thresholds relying heavily on the native `patch_size`.
*   **`VideoSparseAttentionImpl`**: 
    *   `preprocess_qkv()` and `postprocess_output()` reshape and pad the `qkv` tokens depending on `variable_block_sizes` using the precomputed indices.
    *   `forward()` invokes `fastvideo_kernel.video_sparse_attn`, which supports custom gate score projections (`gate_compress`) and selectively routes block subsets.

---

## Guide: How to Implement a New Attention Backend

Suppose you want to add a new algorithm named "MyCustomAttn". You should create a file `my_custom_attn.py` executing the following steps:

### Step 1: Define the `AttentionMetadata`
Create a data class grouping any auxiliary information the custom kernel needs per forward pass.

```python
from fastvideo.attention.backends.abstract import AttentionMetadata
from dataclasses import dataclass

@dataclass
class MyCustomAttentionMetadata(AttentionMetadata):
    current_timestep: int
    custom_indices: torch.Tensor
    sparsity_ratio: float
```

### Step 2: Define the `AttentionMetadataBuilder`
This constructs your metadata. Use this to do expensive index pre-calculations that shouldn't pollute the forward thread.

```python
from fastvideo.attention.backends.abstract import AttentionMetadataBuilder

class MyCustomAttentionMetadataBuilder(AttentionMetadataBuilder):
    def prepare(self): ...
        
    def build(self, current_timestep: int, sparsity_ratio: float, **kwargs) -> MyCustomAttentionMetadata:
        # Perform layout calculations
        indices = calculate_my_indices(...)
        return MyCustomAttentionMetadata(current_timestep, indices, sparsity_ratio)
```

### Step 3: Define the `AttentionImpl`
Write the execution block. It should initialize runtime variables once, do memory tensor layout conversions, and trigger the kernel.

```python
from fastvideo.attention.backends.abstract import AttentionImpl
import torch

class MyCustomAttentionImpl(AttentionImpl):
    def __init__(self, num_heads: int, head_size: int, causal: bool, **kwargs):
        self.num_heads = num_heads
        # ... setup layout mapping

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: MyCustomAttentionMetadata) -> torch.Tensor:
        # e.g., re-stride qkv from [B, S, H, D] into your desired format
        return tile_my_qkv(qkv, attn_metadata.custom_indices)

    def postprocess_output(self, output: torch.Tensor, attn_metadata: MyCustomAttentionMetadata) -> torch.Tensor:
        # Convert back to standard diffusion shapes
        return untile_my_output(output)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_metadata: MyCustomAttentionMetadata) -> torch.Tensor:
        # Call the fused kernel representation
        out = my_custom_cuda_kernel(q, k, v, attn_metadata.custom_indices, attn_metadata.sparsity_ratio)
        return out
```

### Step 4: Expose with `AttentionBackend`
Wrap everything inside the factory.

```python
from fastvideo.attention.backends.abstract import AttentionBackend

class MyCustomAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "MY_CUSTOM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MyCustomAttentionImpl"]:
        return MyCustomAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["MyCustomAttentionMetadata"]:
        return MyCustomAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MyCustomAttentionMetadataBuilder"]:
        return MyCustomAttentionMetadataBuilder
```

### Step 5: (Optional) Extract Complex Configuration
Similar to `STA_configuration.py`, if your attention mechanism depends heavily on dynamic config loading or caching states, keep those out of the forward loop. Expose a helper script to resolve config maps into simpler tensors/arrays before sending them into `build()`.
