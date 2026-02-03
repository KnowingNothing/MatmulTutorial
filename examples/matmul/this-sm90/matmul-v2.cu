/*
 * Level 2: TMA + WGMMA GEMM
 * 
 * Key insight: TMA 128B swizzle matches WGMMA B128 layout exactly!
 * - TMA 128B swizzle: k' = k XOR ((m & 7) * 8)
 * - WGMMA B128 (layout_type=1): same formula!
 * 
 * So TMA loaded data can be directly used by WGMMA without rearrangement.
 * 
 * Layout (K-major for both A and B):
 * - A[M][K]: K is contiguous
 * - B stored as B^T[N][K]: K is contiguous
 * 
 * BLOCK_K = 64 to match TMA 128B swizzle atom (128 bytes / 2 bytes = 64 bf16)
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 
 using TmaDescriptor = CUtensorMap;
 
 __device__ void mbarrier_init(uint64_t* m, uint32_t c) {
     asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(c));
 }
 __device__ void mbarrier_arrive_expect_tx(uint64_t* m, uint32_t tx) {
     asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(tx));
 }
 __device__ void mbarrier_wait_parity(uint64_t* m, uint32_t p) {
     uint32_t a = __cvta_generic_to_shared(m);
     asm volatile("{\n.reg .pred P;\nL:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra L;\n}\n" :: "r"(a), "r"(p));
 }
 __device__ void mbarrier_invalidate(uint64_t* m) {
     asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(m)));
 }
 __device__ void tma_load_2d(const TmaDescriptor* d, uint64_t* m, void* s, int32_t x, int32_t y) {
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
         :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(x), "r"(y) : "memory");
 }
 
 struct GmmaDescriptor {
     uint64_t desc;
     __device__ static GmmaDescriptor make(const void* p, int leading, int stride, int layout) {
         GmmaDescriptor d; uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
         d.desc = ((uint64_t)((a >> 4) & 0x3FFF)) | ((uint64_t)((leading >> 4) & 0x3FFF) << 16) |
                  ((uint64_t)((stride >> 4) & 0x3FFF) << 32) | ((uint64_t)(layout & 0x3) << 62);
         return d;
     }
 };
 
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 __device__ __forceinline__ void wgmma_fence_operand(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
 
 __device__ void wgmma_m64n64k16_bf16(float* c, uint64_t a, uint64_t b, int s) {
     asm volatile(
         "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
         "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;\n}\n"
         : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
           "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
           "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
           "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
         : "l"(a), "l"(b), "r"(s));
 }
 
 __device__ void get_coord(int t, int r, int& row, int& col) {
     int t0 = t % 4, t1 = (t / 4) % 8, t2 = t / 32;
     int r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
     int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
     row = lin % 64; col = lin / 64;
 }
 
 // Block sizes - BLOCK_K=64 matches TMA 128B atom (128 bytes / 2 = 64 bf16)
 constexpr int BM = 64, BN = 64, BK = 64;
 constexpr int WGMMA_K = 16;
 // WGMMA B128 stride: 8 rows * BK elements * 2 bytes
 constexpr int WGMMA_STRIDE = 8 * BK * sizeof(__nv_bfloat16);  // = 1024 bytes
 
 __global__ __launch_bounds__(128)
 void tma_wgmma_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB,
                       float* C, int M, int N, int K) {
     extern __shared__ char smem[];
     // TMA 128B swizzle directly creates WGMMA B128 layout - no rearrangement needed!
     __nv_bfloat16* sA = (__nv_bfloat16*)smem;
     __nv_bfloat16* sB = sA + BM * BK;
     size_t moff = (BM * BK + BN * BK) * 2;
     moff = (moff + 7) & ~7;
     uint64_t* mbar = (uint64_t*)(smem + moff);
     
     int tid = threadIdx.x, bm = blockIdx.y * BM, bn = blockIdx.x * BN;
     if (bm >= M || bn >= N) return;
     
     float acc[32];
     for (int i = 0; i < 32; i++) acc[i] = 0.0f;
     
     if (tid == 0) mbarrier_init(mbar, 1);
     __syncthreads();
     
     for (int kb = 0; kb < K; kb += BK) {
         if (tid == 0) {
             mbarrier_arrive_expect_tx(mbar, (BM * BK + BN * BK) * 2);
             // TMA coords: (k_offset, m_offset) for K-contiguous A[M][K]
             tma_load_2d(&dA, mbar, sA, kb, bm);
             // TMA coords: (k_offset, n_offset) for K-contiguous B_t[N][K]
             tma_load_2d(&dB, mbar, sB, kb, bn);
         }
         mbarrier_wait_parity(mbar, (kb / BK) & 1);
         __syncwarp();
         
         for (int i = 0; i < 32; i++) wgmma_fence_operand(acc[i]);
         wgmma_fence();
         
         // Process K in chunks of 16 (WGMMA K-dimension)
         // TMA 128B swizzle layout = WGMMA B128 layout, direct use!
         #pragma unroll
         for (int ki = 0; ki < BK; ki += WGMMA_K) {
             // Point to the start of this K-slice
             const void* pA = sA + ki;
             const void* pB = sB + ki;
             
             // WGMMA B128: layout_type=1, stride=1024 bytes
             uint64_t da = GmmaDescriptor::make(pA, 0, WGMMA_STRIDE, 1).desc;
             uint64_t db = GmmaDescriptor::make(pB, 0, WGMMA_STRIDE, 1).desc;
             wgmma_m64n64k16_bf16(acc, da, db, 1);
         }
         
         wgmma_commit();
         for (int i = 0; i < 32; i++) wgmma_fence_operand(acc[i]);
         wgmma_wait();
     }
     
     if (tid == 0) mbarrier_invalidate(mbar);
     
     for (int r = 0; r < 32; r++) {
         int lm, ln; get_coord(tid, r, lm, ln);
         int gm = bm + lm, gn = bn + ln;
         if (gm < M && gn < N) C[gm * N + gn] = acc[r];
     }
 }
 
 extern "C" void gemm_bf16_launch(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K) {
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     size_t smem = (BM * BK + BN * BK) * 2 + 16;
     cudaFuncSetAttribute(tma_wgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
     cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = smem;
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     cudaLaunchKernelExC(&cfg, (const void*)tma_wgmma_kernel, args);
 }
 
 extern "C" float benchmark_kernel(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K, int warmup, int iters) {
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     size_t smem = (BM * BK + BN * BK) * 2 + 16;
     cudaFuncSetAttribute(tma_wgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
     cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = smem;
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     for (int i = 0; i < warmup; i++) cudaLaunchKernelExC(&cfg, (const void*)tma_wgmma_kernel, args);
     cudaDeviceSynchronize();
     cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
     for (int i = 0; i < iters; i++) cudaLaunchKernelExC(&cfg, (const void*)tma_wgmma_kernel, args);
     cudaEventRecord(t1); cudaEventSynchronize(t1);
     float ms; cudaEventElapsedTime(&ms, t0, t1); cudaEventDestroy(t0); cudaEventDestroy(t1);
     return ms / iters;
 }
 