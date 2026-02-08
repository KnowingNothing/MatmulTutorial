# Level 8: BLOCK_M=128 + Swizzled CD + Deep Pipeline + ASAP Signal

## 核心优化

在 Level 7 的双tile 2SM架构基础上，Level 8 做了四项关键优化：

### 1. BLOCK_M=128 + 8级流水线 (原先 BLOCK_M=256 + 4级)

在双tile模式下，每个SM只计算128行输出。将BLOCK_M从256降到128：
- A的SMEM每级从32KB减半到16KB → 可容纳8级流水线（原来只有4级）
- 消除了M-wave循环 → 每个k-block只需4次UMMA（原来8次）
- 更简单的epilogue：无需wave循环

SMEM预算（总量 232,448 bytes）:
```
barriers + tmem_ptr:  1,024 bytes
epilogue CD (2期):   32,768 bytes  (2 × 128 × 128 bytes)
8 pipeline stages:  196,608 bytes  (8 × 24,576)
总计:              230,400 bytes ✓
```

### 2. Swizzled CD输出 (SWIZZLE_128B)

TMA Store使用128B swizzle模式，避免epilogue中的SMEM bank conflict：

- `STORE_BLOCK_N = 64` (半个BLOCK_N)，每tile需2次TMA Store
- 2个TMA Store stage实现流水线重叠
- Swizzle公式：`col = i ^ (thread_id % 8)`
  - 128线程各处理1个TMEM row
  - 每线程写8个bank group（每个16 bytes = 8个BF16）
  - XOR保证同warp内线程访问不同bank

```
无swizzle: 所有线程访问同一bank → 32路冲突
128B swizzle: 冲突降低到4路（~8x改善）
```

### 3. ASAP tmem_empty信号

Level 7在TMA Store sync之后才signal tmem_empty。Level 8将信号提前：

```
Level 7:  TMEM读取 → SMEM写入 → named_barrier_sync → tmem_empty信号 → TMA Store
Level 8:  TMEM读取 → SMEM写入 → tcgen05_fence → tmem_empty信号 → sync → TMA Store
```

MMA warp可以在epilogue还在做TMA Store时就开始下一个tile的累加。
（注：由于TMEM双缓冲提供了2-tile的slack，此优化的实际影响较小。）

### 4. TC利用率控制（可选）

编译时设置 `TC_UTIL_PERCENT` (0-100) 可在UMMA commit之间插入空闲周期，
防止持续tensor core使用导致GPU降频。

```bash
# 默认：不启用（100%利用率）
nvcc -DTC_UTIL_PERCENT=100 ...

# 启用：85%利用率（每k-block约45个空闲周期）
nvcc -DTC_UTIL_PERCENT=85 ...
```

**实验结论**：在GB200上，TC_UTIL < 100反而严重降低性能（~50%），
说明此GPU不存在明显的频率降档问题，或降档的影响小于流水线序列化的代价。

## 性能结果

| Size | Level 8 (TFLOPS) | cuBLAS (TFLOPS) | Ratio |
|------|----------------:|----------------:|------:|
| 4096 | 1,428 | 1,667 | 85.7% |
| 6144 | 1,449 | 1,665 | 87.0% |
| 8192 | 1,290 | 1,486 | 86.8% |
| 10240 | 1,240 | 1,454 | 85.3% |
| 12288 | 1,157 | 1,566 | 73.9% |

Level 7对比（BLOCK_M=256, 4级, SWIZZLE_NONE）：

| Size | Level 7 | Level 8 | 比较 |
|------|--------:|--------:|------|
| 4096 | 1,245 | 1,428 | L8 +14.7% |
| 6144 | 1,463 | 1,449 | L7 +1.0% |
| 8192 | 1,321 | 1,290 | L7 +2.4% |
| 12288 | 1,221 | 1,157 | L7 +5.5% |

**结论**：Level 8在4096等小矩阵上明显优于Level 7（+15%），但在大矩阵上Level 7略胜。
最优BLOCK_M取决于矩阵大小。

## 剩余性能差距分析

我们与cuBLAS的差距约为15-25%。通过实验排除了以下假设：

### 已验证无效的优化

| 优化 | 预期 | 实际 | 原因 |
|------|------|------|------|
| shfl描述符缓存 | 减少寄存器压力 | 性能下降 | 额外shfl同步开销 > 寄存器节省 |
| MMA warp全线程参与barrier_wait | 更快轮询 | 性能下降 | 31个额外线程消耗调度资源 |
| TC利用率控制 (85-95%) | 防止降频 | 性能腰斩 | 流水线序列化 > 频率收益 |
| ASAP tmem_empty | 提前MMA重启 | 无明显变化 | TMEM双缓冲已提供足够slack |

### 可能的根本原因

1. **SASS级指令调度**：cuBLAS的机器码经过手工调优，能实现更紧凑的UMMA流水线。
   编译器从PTX生成的代码难以达到相同的指令级并行度。

2. **UMMA流水线气泡**：k-block之间的barrier_wait + commit造成流水线气泡。
   cuBLAS可能使用了不同的流水线结构来减少气泡。

3. **不同的UMMA模式**：cuBLAS可能对计算密集型workload使用 `cta_group::1`，
   消除跨CTA同步开销，每个SM独立处理tile。

## 继承的优化

- Level 7: 双tile 2SM架构（每CTA独立tile）
- Level 5: 持久化kernel + 2D swizzle tile调度
- Level 6: 3角色warp特化 + TMEM双缓冲
- Level 4: TMA Store epilogue
- Level 2: TMA + SWIZZLE_128B加载
