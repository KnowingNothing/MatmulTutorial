# Level 5: Persistent Kernel + 2D Swizzle Tile Scheduler

## Overview

Level 5 converts the kernel to a **persistent kernel** with a **2D swizzle tile scheduler**, achieving better L2 cache reuse and eliminating per-tile launch overhead.

## Key Optimizations

### 1. Persistent Kernel

In Levels 1–4, we launched one CTA cluster per output tile:

```
gridDim = num_tiles × cluster_size
```

Each cluster computed exactly one 128×128 output tile, then exited. For a 8192×8192 output with 128×128 tiles, that's 4096 tiles = 4096 clusters (8192 CTAs). Most CTAs are queued and launched in waves.

In Level 5, we launch exactly **one CTA per SM** (192 CTAs = 96 clusters on GB200), and each cluster processes **multiple tiles** in a `while` loop:

```cpp
while (scheduler.get_next_block(m_block, n_block)) {
    // K-loop (TMA + MMA via warp specialization)
    // Epilogue (SMEM-staged coalesced write)
    // Barrier re-initialization for next tile
}
```

Benefits:
- **TMEM persists** across tiles — allocated once, deallocated at end
- **No kernel launch overhead** between tiles
- Enables the 2D swizzle scheduler for L2 cache reuse

### 2. 2D Swizzle Tile Scheduler

Tiles are assigned to clusters in round-robin order with 2D swizzling for L2 cache locality:

```
tile_idx = (++iter) * num_clusters + my_cluster_id
```

The `tile_idx` is then converted to `(m_block, n_block)` using 2D swizzle:

```
Groups of 8 consecutive M blocks share the same N block:
  Group 0: M=[0..7], iterate N=0, N=1, N=2, ...
  Group 1: M=[8..15], iterate N=0, N=1, N=2, ...
```

This means 8 consecutive tiles processed by a cluster share the same N block, keeping **B-tile data hot in L2 cache**. For K=8192 and LOAD_N_PER_CTA=64:

```
B-tile data per CTA = 64 × 8192 × 2B = 1 MB
With 8 tiles sharing the same N block → 8× B-data reuse in L2
```

This is the same strategy used by DeepGEMM's `Scheduler::get_swizzled_block_idx()`.

### 3. Barrier Re-initialization Between Tiles

Since barriers are reused across tiles, they must be re-initialized between tiles:

```cpp
cluster_sync();   // Ensure epilogue is complete
// Re-init all barriers (warp 1, elected thread)
for (s = 0..NUM_STAGES) {
    barrier_init(full_bar[s], CLUSTER_SIZE);
    barrier_init(empty_bar[s], 1);
}
barrier_init(tmem_bar, 1);
fence_barrier_init();
cluster_sync();   // Ensure barriers are visible
```

The two `cluster_sync()` calls ensure safe transitions between tiles:
1. First sync: all threads finish epilogue before touching barriers
2. Second sync: barrier re-init is visible before next tile's K-loop starts

## Inherited Optimizations

- **True Warp Specialization** (Level 3): TMA warp + MMA warp with independent loops
- **8-stage Deep Pipeline** (Level 3): overlap TMA loads with UMMA computation
- **cta_group::2 TMA** (Level 2): cross-CTA data loading with PEER_BIT_MASK
- **SWIZZLE_128B** (Level 2): bank-conflict-free SMEM layout
- **SMEM-staged Coalesced Epilogue** (Level 4): vectorized 8-byte writes to global memory

## Performance Results (GB200)

| Size | Level 4 | Level 5 | Improvement |
|------|---------|---------|-------------|
| 4096³ | 529 TFLOPS | 525 TFLOPS | ~same |
| 8192³ | 493 TFLOPS | 558 TFLOPS | +13.2% |
| 10240³ | — | 565 TFLOPS | — |
| 12288³ | — | 577 TFLOPS | — |

The persistent kernel + 2D swizzle scheduler shows clear benefits for larger matrices:
- **8192³**: 493 → 558 TFLOPS (+13.2%) — L2 cache reuse kicks in
- **12288³**: 577 TFLOPS — sustained high performance at scale
- **4096³**: similar (the matrix fits better in L2 regardless of swizzle)

Correctness: `cos_sim = 0.999999` ✓

## What's Next (Level 6)

The next optimization targets are:
- **TMA Store Epilogue**: replace the manual SMEM→Global writes with hardware TMA store,
  further reducing epilogue overhead
- **Dedicated Epilogue Warps**: overlap epilogue of tile N with MMA of tile N+1,
  enabling true pipelined tile processing
