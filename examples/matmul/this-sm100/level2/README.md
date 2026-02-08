# Level 2: cta_group::2 TMA + 2-Stage Pipeline

## Optimization Points

### Inherited from Level 1
- 2SM UMMA (`tcgen05.mma.cta_group::2.kind::f16`) for BF16 GEMM
- TMA (`cp.async.bulk.tensor.2d`) for asynchronous global→shared memory loads
- TMA swizzle 128B (`CU_TENSOR_MAP_SWIZZLE_128B`), consistent with DeepGEMM
- TMEM for accumulator storage
- Warp specialization (warp 0 = TMA, warp 1 = MMA, warp 2 = TMEM alloc)
- Only leader CTA writes output

### New in Level 2

1. **2-Stage SMEM Double Buffering**
   - Two shared memory stages with per-stage `full_bar` and `empty_bar` barriers
   - Enables overlapping TMA loads with UMMA computation

2. **`cta_group::2` TMA (key optimization)**
   - Uses `cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile`
   - `PEER_BIT_MASK` (`0xFEFFFFFF`) applied to the **barrier address only** to route TMA completion signals to CTA 0's (leader's) barrier
   - Data destination address is NOT masked — each CTA writes to its own shared memory
   - Eliminates the need for `cluster_sync()` in the inner K loop

3. **Cluster-Scoped Barrier Signaling**
   - Leader CTA uses local `barrier_arrive_expect_tx` (shared::cta)
   - Non-leader CTA uses `mapa.shared::cluster` + `mbarrier.arrive.expect_tx.shared::cluster` to signal the leader's barrier remotely
   - `full_bar` initialized with `count = CLUSTER_SIZE` (one arrival per CTA)
   - `umma_commit_2sm` with `multicast::cluster` (mask `0x3`) signals `empty_bar` on **both** CTAs simultaneously

4. **True TMA/UMMA Pipeline Overlap**
   - Non-leader CTA only waits on `empty_bar` (not `full_bar`), allowing it to race ahead and issue TMA for the next stage while the leader CTA executes UMMA on the current stage
   - This achieves genuine overlap of memory transfers and tensor core computation

## Barrier Protocol

```
full_bar[s]:  init_count = 2 (CLUSTER_SIZE)
  - Both CTAs' TMA threads arrive with arrive_expect_tx(TMA_BYTES each)
  - cta_group::2 TMA completion signals leader's barrier via PEER_BIT_MASK
  - Only leader CTA waits on full_bar

empty_bar[s]: init_count = 1
  - umma_commit_2sm multicasts arrive to BOTH CTAs' empty_bar
  - Both CTAs wait on their own empty_bar before next TMA

tmem_bar:     init_count = 1
  - umma_commit_2sm on last K block signals both CTAs
  - Both CTAs wait before epilogue
```

## Performance

Compiled with `nvcc -O3 -gencode arch=compute_100a,code=sm_100a`.

| M=N=K | Level 1 | Level 2 | Improvement |
|-------|---------|---------|-------------|
| 4096  | ~100 TFLOPS | **193.2 TFLOPS** | ~+93% |
| 6144  | — | **166.4 TFLOPS** | — |
| 8192  | ~92 TFLOPS | **147.0 TFLOPS** | ~+60% |
| 10240 | — | **139.5 TFLOPS** | — |
| 12288 | — | **144.0 TFLOPS** | — |

Correctness: max_err=1.40, mean_err=0.10, cos_sim=0.999999 (8192x8192x8192).

## Key Lessons

- `PEER_BIT_MASK` must be applied to the **barrier address only**, not the data destination
  (per CUTLASS `SM100_TMA_2SM_LOAD_2D` implementation). Applying it to the destination
  corrupts data by routing it to the wrong CTA's shared memory.
- `cluster_sync()` in the inner loop was a major bottleneck: it serialized TMA and UMMA
  across CTAs, preventing pipeline overlap. Replacing it with fine-grained barrier signaling
  via `cta_group::2` TMA yielded 45-80% speedup.
- Non-leader CTA acting as a "TMA prefetch engine" enables true compute/memory overlap.
