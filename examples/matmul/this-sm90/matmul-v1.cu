/*
 * Golden Sample: WGMMA BF16 Basic
 * Level 1 - Basic WGMMA with B128 swizzle layout
 * 
 * Key features:
 * - Uses wgmma.mma_async.m64n64k16.f32.bf16.bf16
 * - B128 swizzle layout (layout_type=1)
 * - BLOCK_K = 64 (128 bytes / 2 bytes per bf16)
 * - stride = 1024 bytes (8 rows * 64 elements * 2 bytes)
 * - Swizzle formula: k' = k XOR ((m & 7) * 8)
 *   This is IDENTICAL to TMA 128B swizzle!
 * 
 * CRITICAL SYNCHRONIZATION:
 * - fence.proxy.async required after shared memory loads
 * - This ensures generic proxy writes are visible to async proxy (tensor core)
 * - Without TMA, this fence is mandatory for correctness
 * 
 * LayoutType enum (from cute):
 *   INTERLEAVE = 0, B128 = 1, B64 = 2, B32 = 3
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cstdint>
 
 struct GmmaDescriptor {
     uint64_t desc;
     __device__ static GmmaDescriptor make(const void* ptr, int leading, int stride, int layout) {
         GmmaDescriptor d;
         uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
         d.desc = ((uint64_t)((addr >> 4) & 0x3FFF)) |
                  ((uint64_t)((leading >> 4) & 0x3FFF) << 16) |
                  ((uint64_t)((stride >> 4) & 0x3FFF) << 32) |
                  ((uint64_t)(layout & 0x3) << 62);
         return d;
     }
 };
 
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 
 __device__ __forceinline__ void wgmma_fence_operand(float& reg) {
     asm volatile("" : "+f"(reg) :: "memory");
 }
 
 __device__ void wgmma_m64n64k16_bf16(float* acc, uint64_t da, uint64_t db, int scale_d) {
     asm volatile(
         "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
         "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
         "%32,%33,p,1,1,0,0;\n}\n"
         : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
           "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
           "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
           "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
         : "l"(da), "l"(db), "r"(scale_d));
 }
 
 constexpr int BLOCK_M = 64;
 constexpr int BLOCK_N = 64;
 constexpr int BLOCK_K = 64;  // Must be 64 for B128 swizzle
 constexpr int WGMMA_K = 16;
 constexpr int WGMMA_STRIDE = 8 * BLOCK_K * sizeof(__nv_bfloat16);  // 1024 bytes
 
 // B128 swizzle formula: k' = k XOR ((m & 7) * 8)
 __device__ __forceinline__ int idx_swizzle(int row, int k) {
     int k_swizzled = k ^ ((row & 7) * 8);
     return row * BLOCK_K + k_swizzled;
 }
 
 // Output register to matrix coordinate mapping for m64n64
 __device__ void get_coord(int tid, int reg, int& row, int& col) {
     int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
     int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
     int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
     row = lin % 64; 
     col = lin / 64;
 }
 
 __global__ __launch_bounds__(128)
 void wgmma_bf16_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
     __shared__ __align__(1024) __nv_bfloat16 sA[BLOCK_M * BLOCK_K];
     __shared__ __align__(1024) __nv_bfloat16 sB[BLOCK_N * BLOCK_K];
     
     int tid = threadIdx.x;
     int bm = blockIdx.y * BLOCK_M, bn = blockIdx.x * BLOCK_N;
     if (bm >= M || bn >= N) return;
     
     float acc[32];
     #pragma unroll
     for (int i = 0; i < 32; i++) acc[i] = 0.0f;
     
     for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
         // Load A[BLOCK_M][BLOCK_K] with B128 swizzle
         for (int i = tid; i < BLOCK_M * BLOCK_K; i += 128) {
             int m = i / BLOCK_K, k = i % BLOCK_K;
             int gm = bm + m, gk = k_base + k;
             __nv_bfloat16 val = (gm < M && gk < K) ? A[gm * K + gk] : __float2bfloat16(0.0f);
             sA[idx_swizzle(m, k)] = val;
         }
         
         // Load B[BLOCK_K][BLOCK_N] with B128 swizzle (treat as B_t[BLOCK_N][BLOCK_K])
         for (int i = tid; i < BLOCK_K * BLOCK_N; i += 128) {
             int k = i / BLOCK_N, n = i % BLOCK_N;
             int gk = k_base + k, gn = bn + n;
             __nv_bfloat16 val = (gk < K && gn < N) ? B[gk * N + gn] : __float2bfloat16(0.0f);
             sB[idx_swizzle(n, k)] = val;
         }
         
         __syncthreads();
         asm volatile("fence.proxy.async;\n" ::: "memory");
         __syncwarp();
 
         #pragma unroll
         for (int ki = 0; ki < BLOCK_K; ki += WGMMA_K) {
             #pragma unroll
             for (int i = 0; i < 32; i++) wgmma_fence_operand(acc[i]);
             
             wgmma_fence();
             uint64_t da = GmmaDescriptor::make(sA + ki, 0, WGMMA_STRIDE, 1).desc;
             uint64_t db = GmmaDescriptor::make(sB + ki, 0, WGMMA_STRIDE, 1).desc;
             wgmma_m64n64k16_bf16(acc, da, db, 1);
             wgmma_commit();
             
             #pragma unroll
             for (int i = 0; i < 32; i++) wgmma_fence_operand(acc[i]);
             wgmma_wait();
         }
     }
     
     #pragma unroll
     for (int r = 0; r < 32; r++) {
         int lm, ln;
         get_coord(tid, r, lm, ln);
         int gm = bm + lm, gn = bn + ln;
         if (gm < M && gn < N) {
             C[gm * N + gn] = acc[r];
         }
     }
 }
 
 extern "C" void gemm_bf16_launch(const void* A, const void* B, void* C, int M, int N, int K) {
     dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
     wgmma_bf16_gemm<<<grid, 128>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B, (float*)C, M, N, K);
 }
 
 extern "C" float benchmark_kernel(const void* A, const void* B, void* C,
                                    int M, int N, int K, int warmup, int iters) {
     dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     for (int i = 0; i < warmup; i++)
         wgmma_bf16_gemm<<<grid, 128>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B, (float*)C, M, N, K);
     cudaDeviceSynchronize();
     cudaEventRecord(start);
     for (int i = 0; i < iters; i++)
         wgmma_bf16_gemm<<<grid, 128>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B, (float*)C, M, N, K);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     float ms;
     cudaEventElapsedTime(&ms, start, stop);
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
     return ms / iters;
 }
 