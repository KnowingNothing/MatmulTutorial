# Level 1 — Baseline: Correct 2SM UMMA Kernel

从零开始实现的第一个能正确运行的 SM100 (GB200) BF16 GEMM kernel，使用 2SM UMMA 指令。
目标是跑通全流程、验证正确性，不做任何性能优化。

## 计算任务

```
D(M, N) = A(M, K) @ B(N, K)^T
```

- 输入: A、B 均为 BF16 row-major
- 累加器: FP32 (TMEM)
- 输出: D 为 BF16 row-major

## 核心概念

### SM100 新硬件特性

| 特性 | 说明 |
|------|------|
| **UMMA** (tcgen05.mma) | SM100 的 Tensor Core 矩阵乘指令，替代 SM90 的 WGMMA |
| **TMEM** (Tensor Memory) | Tensor Core 专用的片上存储，用于存放累加器，替代 SM90 的寄存器累加 |
| **TMA** (Tensor Memory Access) | 异步 Global → Shared Memory 拷贝（SM90 已有） |
| **2SM / cta_group::2** | 两个 SM 协作执行一条 UMMA，计算吞吐翻倍 |
| **Cluster** | 2 个 CTA 组成 cluster，共享 barrier 和 SMEM 访问 |

### Kernel 架构

```
┌──────────────── Cluster (2 CTAs) ─────────────────┐
│                                                     │
│  CTA 0 (SM 0)              CTA 1 (SM 1)           │
│  ┌──────────┐              ┌──────────┐            │
│  │ SMEM:    │              │ SMEM:    │            │
│  │  A[128×64]│              │  A[128×64]│  ← 同样的 A│
│  │  B[64×64] │              │  B[64×64] │  ← 不同的 B│
│  └────┬─────┘              └────┬─────┘            │
│       │                         │                   │
│       └────────┬────────────────┘                   │
│                ▼                                    │
│     tcgen05.mma.cta_group::2                       │
│     (UMMA_M=256, UMMA_N=128, UMMA_K=16)           │
│                │                                    │
│       ┌────────┴────────┐                           │
│       ▼                 ▼                           │
│   TMEM CTA 0        TMEM CTA 1                    │
│   (128 rows)         (128 rows)                    │
│       │                 │                           │
│       ▼                 ▼                           │
│   Store to D         Store to D (redundant)        │
└─────────────────────────────────────────────────────┘
```

### 执行流程 (单级流水线)

```
for each K block:
    1. Wait empty_bar  → 等 UMMA 释放 SMEM（首次通过 parity trick 跳过）
    2. TMA load A, B   → Global → Shared Memory
    3. Wait full_bar   → 等 TMA 完成
    4. cluster_sync    → 确保两个 CTA 都 load 完毕
    5. UMMA compute    → tcgen05.mma.cta_group::2 (BLOCK_K/UMMA_K = 4 次)
    6. commit          → 通知 barrier: SMEM 可以被重用 / TMEM 就绪

Epilogue:
    7. Wait tmem_bar   → 等 UMMA 结果写入 TMEM
    8. tcgen05.ld      → 从 TMEM 读取 FP32 结果
    9. Convert to BF16 → 写到 Global Memory D
```

## 关键调试经验

### Bug 1: BF16 格式码错误

InstrDescriptor 中 `a_format` / `b_format` 的 BF16 编码是 **1**，不是 3。

```
// 错误: d |= (3u << 7);   // a_format = 3
// 正确: d |= (1u << 7);   // a_format = 1 (BF16)
```

参考 CUTLASS 的 `F16F32Format` 枚举:
- F16 = 0, BF16 = 1, TF32 = 2

### Bug 2: InstrDescriptor 必须用 32-bit 传递

`tcgen05.mma` 的 `idesc` 操作数是 **32-bit** 寄存器 (`"r"` constraint)，
不是 64-bit (`"l"`)。CUTLASS 的 `make_runtime_instr_desc` 返回 `uint64_t`
（descriptor 放在高 32 位），但传入 inline ASM 时需要手动提取高 32 位:

```cpp
// CUTLASS 的做法:
"r"(static_cast<uint32_t>(desc >> 32))

// 而非:
"l"(idescE)  // ❌ ptxas: Arguments mismatch
```

### Bug 3: tcgen05.mma 需要 scaleD predicate

UMMA 指令必须带 `scaleD` predicate 参数控制是否累加:
```asm
// ❌ tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3;
// ✅ tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;
```

### SmemDescriptor 位布局

```
Bits [0:14)   start_address (addr >> 4)
Bits [16:30)  leading_byte_offset (LBO >> 4), K-major 时为 0
Bits [32:46)  stride_byte_offset (SBO >> 4)
Bits [46:48)  version = 1 (SM100)
Bits [49:52)  base_offset = 0
Bits [52:53)  lbo_mode = 0
Bits [61:64)  layout_type = 2 (SWIZZLE_128B)
```

## 性能

| Shape | Ours | Torch | Ratio |
|-------|------|-------|-------|
| 8192×8192×8192 | 91.5 TFLOPS (12.015 ms) | 1705.7 TFLOPS (0.645 ms) | 0.054x |

仅达到 torch.matmul 性能的 **5.4%**。

## 性能瓶颈分析

1. **冗余计算 (50%)**: 两个 CTA 加载相同的 A 数据，UMMA_M=256 但只有 128 行有效输出，浪费一半算力
2. **无流水线**: 单级 pipeline，TMA 和 UMMA 完全串行，无法隐藏访存延迟
3. **低效 epilogue**: 逐元素从 TMEM 读取 + 转换 + 写回，无合并访存优化
4. **重量级同步**: 每个 K 步都做 `cluster_sync()`，开销大
5. **无 Tile 调度**: 简单行主序遍历，L2 cache 利用率差

## 下一步优化方向 (Level 1)

- 让两个 CTA 加载不同的 A 行 (消除冗余计算)
- 多级流水线 (overlap TMA 和 UMMA)
- 优化 epilogue 写回
