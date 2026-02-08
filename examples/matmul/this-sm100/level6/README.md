# Level 6: 3-Role Warp Specialization + TMA Store Epilogue

## Overview

Level 6 implements **full 3-role warp specialization**, **TMEM double buffering**, and a **TMA Store epilogue** with optimized data staging. This is the closest architecture to DeepGEMM's approach.

## Key Optimizations

### 1. Three Independent Persistent Loops

Each role has its own `while(get_next_block)` loop, enabling full overlap:

```
TMA warp:       [Tile0 TMA][Tile1 TMA][Tile2 TMA]...
MMA warp:       [Tile0 MMA][Tile1 MMA][Tile2 MMA]...
Epilogue warps:       [Tile0 epi][Tile1 epi][Tile2 epi]...
```

No barrier re-initialization between tiles — phase tracking persists naturally.

### 2. TMEM Double Buffering (2 Accumulator Stages)

256 TMEM columns: stage 0 (cols 0-127) and stage 1 (cols 128-255). MMA writes one stage while epilogue reads the other:

```
MMA:      write stage 0 | write stage 1 | write stage 0 | ...
Epilogue:               | read stage 0  | read stage 1  | ...
```

### 3. TMA Store Epilogue (Hardware DMA)

Replaces Level 4/5's manual `uint2` vectorized global writes with **hardware TMA Store** (`cp.async.bulk.tensor.2d.global.shared::cta.tile`):

```
Before (Level 5): TMEM → SMEM → manual store loop → Global Memory
After  (Level 6): TMEM → SMEM → TMA Store (hardware DMA) → Global Memory
```

Advantages:
- **No explicit store instructions** — TMA DMA engine handles the transfer
- **Automatic OOB handling** — no boundary checks needed in kernel code
- **Potential L2 cache bypass** — avoids polluting cache with output data
- **Higher bandwidth** — hardware-optimized memory controller path

### 4. Optimized TMEM→SMEM Data Staging

Three micro-optimizations for the TMEM-to-SMEM transfer:

| Optimization | Level 5 | Level 6 |
|---|---|---|
| TMEM loads | `tmem_load_4x` (4 cols/fence) | `tmem_load_8x` (8 cols/fence) |
| BF16 conversion | scalar `__float2bfloat16` + store | `pack_bf16()` → packed uint32_t |
| SMEM writes | 16-bit scalar stores | `st_shared_128` (128-bit = 16 bytes/store) |

The 8x TMEM load halves the fence overhead (16 → 16 iterations but only 16 fences instead of 32). The packed BF16 conversion writes 2 values per register. The 128-bit SMEM store writes an entire bank group (8 BF16 values) in one instruction.

### 5. 256 Threads (8 Warps) per CTA

```
Warp 0:    TMA load  (1 elected thread)
Warp 1:    MMA issue (1 elected thread, leader CTA only)
Warp 2:    TMEM alloc/dealloc
Warp 3:    idle
Warps 4-7: Epilogue  (128 threads, leader CTA only)
```

### 6. Dedicated Epilogue SMEM Buffer (32 KB)

Separate from the 192 KB pipeline stages, so TMA loads and epilogue writes never conflict.

## Barrier Protocol

| Barrier | Direction | Count | Purpose |
|---------|-----------|-------|---------|
| `full_bar[8]` | TMA → MMA | 2 | SMEM data ready |
| `empty_bar[8]` | MMA → TMA | 1 | SMEM stage free |
| `tmem_full_bar[2]` | MMA → Epilogue | 1 | TMEM accumulation done |
| `tmem_empty_bar[2]` | Epilogue → MMA | 1 | TMEM stage free |

## Performance Results (GB200)

| Size | Level 5 | Level 6 | Improvement |
|------|---------|---------|-------------|
| 4096³ | 525 TFLOPS | **754 TFLOPS** | **+43.6%** |
| 6144³ | 582 TFLOPS | **716 TFLOPS** | **+23.0%** |
| 8192³ | 558 TFLOPS | **650 TFLOPS** | **+16.5%** |
| 10240³ | 565 TFLOPS | **628 TFLOPS** | **+11.2%** |
| 12288³ | 577 TFLOPS | **634 TFLOPS** | **+9.9%** |

The combined optimizations (3-role warp specialization + TMEM double buffering + TMA Store + optimized data staging) yield 10-44% improvement over Level 5, with the largest gains at smaller matrix sizes where epilogue overhead is most significant.

Correctness: `cos_sim = 0.999999` ✓

## Inherited Optimizations

- Persistent kernel + 2D swizzle scheduler (Level 5)
- 8-stage deep pipeline (Level 3)
- cta_group::2 TMA with PEER_BIT_MASK (Level 2)
- SWIZZLE_128B SMEM layout for inputs (Level 2)
