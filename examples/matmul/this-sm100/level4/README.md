# Level 4: SMEM-Staged Coalesced Epilogue

## 优化内容

在 Level 3（真正 warp 特化 + 8 级深流水线）的基础上，优化 epilogue（结果写回）的内存访问模式。

### 问题分析：Level 3 的 epilogue 瓶颈

Level 3 的 epilogue 中，每个线程直接从 TMEM 加载自己行的数据并写入 global memory：

```
Thread 0 → D[row_0, col:col+3]  // 地址: base + 0 * N * 2
Thread 1 → D[row_1, col:col+3]  // 地址: base + 1 * N * 2
...
Thread 31 → D[row_31, col:col+3] // 地址: base + 31 * N * 2
```

同一个 warp 内 32 个线程写入 **32 个不同的行**，地址间隔 `N × 2` 字节（例如 N=8192 时间隔 16384 字节），导致每次写操作产生 **32 个独立的 cache line 事务**——完全不合并（uncoalesced）。

### 解决方案：SMEM 转置 + 合并写入

**Phase 1: TMEM → SMEM（保持原有 thread-per-row 映射）**
- 每个线程从 TMEM 读取其行的 FP32 数据，转换为 BF16 后写入 shared memory
- SMEM 中存储完整的 128×128 BF16 output tile（32 KB）

**Phase 2: SMEM → Global（重新映射线程实现合并访问）**
- 4 个 warp 处理 4 个不同的行（每步 4 行）
- 每个 warp 的 32 个线程写同一行的连续 4×BF16（8 字节 `uint2`）
- 同一 warp 内的写入地址完全连续 → **全合并（fully coalesced）**

```
Step i:
  Warp 0: 32 threads → D[row_0, col_0..col_127]  // 256 bytes = 2 cache lines
  Warp 1: 32 threads → D[row_1, col_0..col_127]  // 256 bytes = 2 cache lines
  Warp 2: 32 threads → D[row_2, col_0..col_127]  // 256 bytes = 2 cache lines
  Warp 3: 32 threads → D[row_3, col_0..col_127]  // 256 bytes = 2 cache lines
```

### 关键数据

| 指标 | Level 3 | Level 4 |
|------|---------|---------|
| 每 warp 每步 cache line 事务数 | 32（不合并） | 2（全合并） |
| 每 tile 总 cache line 事务数 | ~4096 | 256 |
| 全局写带宽利用率 | ~3% | ~100% |

## 继承的优化

- 真正 warp 特化（TMA warp + MMA warp 独立循环）
- 8 级深流水线（NUM_STAGES=8）
- cta_group::2 TMA + PEER_BIT_MASK
- SWIZZLE_128B shared memory 布局
- 2SM UMMA (tcgen05.mma.cta_group::2)

## 性能结果

| Matrix Size | Level 3 (TFLOPS) | Level 4 (TFLOPS) | 提升 | cuBLAS (TFLOPS) | vs cuBLAS |
|-------------|-------------------|-------------------|------|-----------------|-----------|
| 4096³       | 296               | 529               | +79% | 1713            | 30.9%     |
| 6144³       | 417               | 560               | +34% | 1764            | 31.7%     |
| 8192³       | 403               | 493               | +22% | 1620            | 30.4%     |
| 10240³      | 364               | 467               | +28% | 1482            | 31.5%     |
| 12288³      | 384               | 483               | +26% | 1549            | 31.2%     |

- 所有尺寸均超过 100 TFLOPS 目标
- 平均性能提升 ~38%
- 正确性验证：cos_sim = 0.999999，allclose PASSED
