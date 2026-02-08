# Level 7: Dual-Tile 2SM Architecture

## Core Insight — The 2x Performance Bug Fix

This level addresses the **single most impactful optimization** in the entire series: making both SMs in the cluster compute **different** tiles instead of the same tile.

### The Problem in Level 6

In Level 6, both CTAs in the 2SM cluster were assigned the **same tile** (same `m_block`, same `n_block`). The scheduler used `cluster_id = blockIdx.x / 2` — both CTAs got identical tile assignments.

In `cta_group::2` UMMA mode, each SM reads A from its **own CTA's shared memory**. Since both CTAs loaded the same A data, both SMs computed identical results. SM1's compute was entirely wasted — **50% of tensor core capacity was thrown away**.

### The Fix

The key insight comes from understanding how `cta_group::2` UMMA really works:

1. **A operand**: Each SM reads from its **own CTA's SMEM** (not the leader's)
2. **B operand**: The hardware combines B from **both CTAs' SMEM** (64 cols each → 128 total)
3. **Result**: Each SM's TMEM holds a **different** result, based on that CTA's A data

By assigning **different tiles** to the two CTAs (different `m_block`, same `n_block`), a single UMMA instruction processes two tiles simultaneously:

```
CTA0 loads: A[m0, :], B[n, 0:64]    CTA1 loads: A[m1, :], B[n, 64:128]
                    ↓ UMMA cta_group::2 ↓
SM0 TMEM = A[m0, :] × B[n, :]       SM1 TMEM = A[m1, :] × B[n, :]
       (tile 0 result)                       (tile 1 result)
```

## Key Changes from Level 6

### 1. Per-CTA Scheduler (not per-cluster)

```c
// Level 6: both CTAs in cluster get the SAME tile
cta_id = blockIdx.x / CLUSTER_SIZE;  // cluster_id

// Level 7: each CTA gets its OWN tile
cta_id = blockIdx.x;  // per-CTA
```

The 2D swizzle ensures that CTAs `(2k, 2k+1)` in the same cluster always get the **same `n_block`** but **different `m_block`** values, so B data is compatible.

### 2. Both CTAs Run Epilogue

Since each SM's TMEM now holds a **different** result, both CTAs must run the epilogue independently:

```c
// Level 6: only leader CTA
} else if (warp_idx >= 4 && is_leader) {

// Level 7: both CTAs
} else if (warp_idx >= 4) {
```

Each CTA reads its own SM's TMEM and TMA-stores to its own tile's output coordinates.

### 3. tmem_empty Barrier: Cluster-Addressed Arrive

The MMA warp must wait for **both** CTAs' epilogues to finish reading TMEM before overwriting it. The `tmem_empty` barrier init count is changed from 1 to `CLUSTER_SIZE=2`, and CTA1 signals the leader's barrier via cluster addressing:

```c
barrier_init(tmem_empty_bar[e], CLUSTER_SIZE);  // was 1

// In epilogue:
if (is_leader)
    barrier_arrive(tmem_empty_bar[accum_idx]);         // shared::cta
else
    barrier_arrive_cluster(tmem_empty_bar[accum_idx], 0); // shared::cluster → leader
```

### 4. Grid Launch

```c
// Level 6: grid = num_clusters * 2
num_ctas = num_clusters * CLUSTER_SIZE;

// Level 7: grid = num_sms (each CTA processes its own tile)
num_ctas = min(num_sms, num_tiles);
num_ctas = (num_ctas / CLUSTER_SIZE) * CLUSTER_SIZE;  // ensure even
```

## Why This Works Correctly

The 2D swizzle with `SWIZZLE_GROUP_SIZE=8` groups 8 M-blocks together before advancing N. Adjacent CTAs `(2k, 2k+1)` always fall in the same M-group:

```
Block 0: (m=0, n=0)   Block 1: (m=1, n=0)   ← same n, cluster 0
Block 2: (m=2, n=0)   Block 3: (m=3, n=0)   ← same n, cluster 1
...
Block 6: (m=6, n=0)   Block 7: (m=7, n=0)   ← same n, cluster 3
Block 8: (m=0, n=1)   Block 9: (m=1, n=1)   ← same n, cluster 4
```

DeepGEMM's multicast legality check ensures `num_m_blocks % 2 == 0`, guaranteeing no tile imbalance between paired CTAs (total tiles is always even).

## Performance

| Size | Level 6 (TFLOPS) | Level 7 (TFLOPS) | Speedup | cuBLAS (TFLOPS) | vs cuBLAS |
|------|-------------------|-------------------|---------|-----------------|-----------|
| 4096 | ~624 | 1242 | 2.0x | 1659 | 74.9% |
| 6144 | ~700 | 1414 | ~2.0x | 1632 | 86.6% |
| 8192 | ~700 | 1277 | ~1.8x | 1708 | 74.8% |
| 10240 | — | 1267 | — | 1458 | 86.9% |
| 12288 | — | 1215 | — | 1518 | 80.0% |

The dual-tile architecture provides a **near-perfect 2x speedup**, confirming that Level 6 was indeed wasting 50% of compute. We are now at **75–87% of cuBLAS**.

## Remaining Gap to cuBLAS (~15-25%)

Potential further optimizations to close the gap:
- **Swizzled CD output**: Use `SWIZZLE_128B` for TMA Store to avoid bank conflicts
- **Stage merging**: Combine pipeline stages to reduce UMMA commit overhead
- **Tensor core utilization control**: Throttle TC usage to prevent GPU clock drops
- **TMA multicast for B**: Use hardware multicast to halve B load bandwidth
- **Epilogue pipelining**: Multi-stage TMA store to overlap TMEM reads with stores

## Inherited Optimizations

All optimizations from Level 6 are preserved:
- BLOCK_M=256, BLOCK_N=128, BLOCK_K=64
- 3-role warp specialization (TMA, MMA, Epilogue)
- 4-stage asynchronous pipeline
- TMEM double buffering (2 epilogue stages)
- TMA Store epilogue with 8x TMEM loads, BF16 packing, 128-bit SMEM stores
- 2D swizzle tile scheduler
- Persistent kernel
