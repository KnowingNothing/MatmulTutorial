# Level 3: True Warp Specialization + 8-Stage Deep Pipeline

## Optimization Points

### Inherited from Level 2
- cta_group::2 TMA with PEER_BIT_MASK (barrier-only, no cluster_sync in K loop)
- Cluster-scoped barrier signaling (leader: local, non-leader: mapa + cluster arrive)
- 2SM UMMA, TMA swizzle 128B, TMEM accumulator
- Only leader CTA writes output

### New in Level 3

1. **True Warp Specialization (key optimization)**
   - TMA warp (warp 0) and MMA warp (warp 1) run **completely independent loops**
   - In Level 2, all warps executed the same `for` loop with `if/else` branching
   - Now each warp has its own loop body, phase tracking (`tma_stage/tma_phase` vs `mma_stage/mma_phase`), and advancement logic
   - They communicate **only** through `full_bar` and `empty_bar` barriers

2. **8-Stage Deep Pipeline (was 2)**
   - 8 SMEM buffer stages: TMA warp can prefetch up to **8 K blocks ahead** of MMA warp
   - SMEM usage: 1024 (barriers) + 8 × 24,576 (data) = 197,632 bytes (< 232,448 SM100 capacity)
   - On the first pass, all 8 `empty_bar` waits return instantly (fresh barriers), allowing TMA to fill all 8 stages before MMA even starts

3. **Warp Roles**

   | Warp | Role | CTA | Description |
   |------|------|-----|-------------|
   | 0 | TMA | Both | Independent K-loop: wait empty → arrive_expect_tx → TMA load |
   | 1 | MMA | Leader only | Independent K-loop: wait full → UMMA × 4 → commit empty |
   | 2 | TMEM alloc | Both | Allocates TMEM, then idles |
   | 3 | Idle | — | Falls through to epilogue |

## Pipeline Timing

```
TMA warp:  [TMA0][TMA1][TMA2][TMA3][TMA4][TMA5][TMA6][TMA7][wait empty0]...
MMA warp:                          [wait full0][MMA0][MMA1][MMA2]...

                                    ↑ TMA is 4-8 stages ahead
                                    ↑ Memory latency fully hidden
```

The TMA warp races ahead, filling SMEM stages 0-7 while the MMA warp processes
them sequentially. With 8 stages of buffering, TMA latency is fully absorbed
by the compute pipeline.

## Performance

Compiled with `nvcc -O3 -gencode arch=compute_100a,code=sm_100a`.

| M=N=K | Level 2 | Level 3 | Improvement |
|-------|---------|---------|-------------|
| 4096  | 193 TFLOPS | **296 TFLOPS** | +53% |
| 6144  | 166 TFLOPS | **326 TFLOPS** | +96% |
| 8192  | 147 TFLOPS | **403 TFLOPS** | +174% |
| 10240 | 140 TFLOPS | **399 TFLOPS** | +187% |
| 12288 | 144 TFLOPS | **417 TFLOPS** | +189% |

Correctness: max_err=1.60, mean_err=0.10, cos_sim=0.999999 (8192x8192x8192).

## Why This Works

The ~3x improvement for large matrices comes from two effects:

1. **True asynchronous pipeline**: In Level 2, even though TMA and MMA were in separate
   branches, they shared the same loop body — one iteration had to complete (TMA + wait + UMMA)
   before the next could start. Now TMA and MMA run independently, overlapping in time.

2. **Deep buffering**: With only 2 stages, the TMA warp could be at most 1 stage ahead.
   If a single TMA load took longer than expected (e.g., L2 cache miss), the MMA warp
   would stall. With 8 stages, occasional TMA latency spikes are absorbed by the buffer.

## Key Lessons

- True warp specialization is the single most impactful optimization in this series.
  The key insight: each warp should have its own loop, not share one loop with branches.
- 8 pipeline stages fit within SM100's 232KB shared memory budget (using 197KB).
- The barrier protocol (full_bar/empty_bar) naturally supports independent warp loops —
  no code changes to the barrier helpers were needed, only restructuring the control flow.
