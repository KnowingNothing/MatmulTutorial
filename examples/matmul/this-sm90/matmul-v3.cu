/*
 * Level 3: Warp Specialization GEMM with Multi-Stage Pipelining
 * 
 * Architecture:
 * - 256 threads = 2 warpgroups
 * - Warpgroup 0 (threads 0-127): Producer - TMA loads
 * - Warpgroup 1 (threads 128-255): Consumer - WGMMA compute
 * - Double buffering for TMA-WGMMA overlap
 * 
 * Note: This architecture shows lower performance than Level 3 (128 threads)
 * because only half the threads are doing compute. The benefit of warp 
 * specialization comes with persistent kernels and TMA multicast (not 
 * implemented here).
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 
 using TmaDescriptor = CUtensorMap;
 
 constexpr int BM = 64, BN = 64, BK = 64;
 constexpr int WGMMA_K = 16;
 constexpr int WGMMA_STRIDE = 8 * BK * sizeof(__nv_bfloat16);
 constexpr int NUM_STAGES = 3;
 constexpr int TOTAL_THREADS = 256;
 
 __device__ void mbar_init(uint64_t* m, uint32_t c) {
     asm volatile("mbarrier.init.shared.b64 [%0], %1;" 
         :: "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(c));
 }
 __device__ void mbar_arrive(uint64_t* m) {
     asm volatile("mbarrier.arrive.shared.b64 _, [%0];" 
         :: "r"((uint32_t)__cvta_generic_to_shared(m)));
 }
 __device__ void mbar_arrive_tx(uint64_t* m, uint32_t tx) {
     asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" 
         :: "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(tx));
 }
 __device__ void mbar_wait(uint64_t* m, uint32_t phase) {
     asm volatile(
         "{\n.reg .pred P;\nWAIT:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT;\n}\n"
         :: "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(phase));
 }
 __device__ void tma_load(const TmaDescriptor* d, uint64_t* m, void* s, int32_t x, int32_t y) {
     asm volatile(
         "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
         :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"((uint64_t)d), 
            "r"((uint32_t)__cvta_generic_to_shared(m)), "r"(x), "r"(y) : "memory");
 }
 
 __device__ uint64_t make_desc(const void* p, int stride) {
     uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
     return ((uint64_t)((a >> 4) & 0x3FFF)) | 
            ((uint64_t)((stride >> 4) & 0x3FFF) << 32) | 
            ((uint64_t)1 << 62);
 }
 
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 __device__ void wgmma_fence_op(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
 
 __device__ void wgmma_m64n64k16(float* c, uint64_t a, uint64_t b) {
     asm volatile(
         "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
         "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;\n}\n"
         : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
           "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
           "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
           "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
         : "l"(a), "l"(b));
 }
 
 __device__ void get_coord(int t, int r, int& row, int& col) {
     int t0 = t % 4, t1 = (t / 4) % 8, t2 = t / 32;
     int r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
     int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
     row = lin % 64; col = lin / 64;
 }
 
 constexpr int SMEM_AB = (BM * BK + BN * BK) * sizeof(__nv_bfloat16);
 constexpr int TX_BYTES = SMEM_AB;
 
 __global__ __launch_bounds__(256, 1)
 void kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB,
             float* C, int M, int N, int K) {
     extern __shared__ __align__(1024) char smem[];
     
     __nv_bfloat16* sA[NUM_STAGES];
     __nv_bfloat16* sB[NUM_STAGES];
     uint64_t* full[NUM_STAGES];
     uint64_t* empty[NUM_STAGES];
     
     for (int s = 0; s < NUM_STAGES; s++) {
         sA[s] = (__nv_bfloat16*)(smem + s * SMEM_AB);
         sB[s] = sA[s] + BM * BK;
     }
     uint64_t* barriers = (uint64_t*)(smem + NUM_STAGES * SMEM_AB);
     for (int s = 0; s < NUM_STAGES; s++) {
         full[s] = barriers + s;
         empty[s] = barriers + NUM_STAGES + s;
     }
     
     int tid = threadIdx.x;
     int wg = tid / 128;
     int lane = tid % 32;
     int ltid = tid % 128;
     
     int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
     
     if (tid == 0) {
         for (int s = 0; s < NUM_STAGES; s++) {
             mbar_init(full[s], 1);
             mbar_init(empty[s], 4);
         }
     }
     __syncthreads();
     asm volatile("fence.proxy.async;\n" ::: "memory");
     __syncthreads();
     
     if (bm >= M || bn >= N) return;
     
     int nk = (K + BK - 1) / BK;
     
     if (wg == 0) {
         if (tid == 0) {
             for (int k = 0; k < nk; k++) {
                 int s = k % NUM_STAGES;
                 int p = (k / NUM_STAGES) & 1;
                 
                 if (k >= NUM_STAGES) mbar_wait(empty[s], p ^ 1);
                 
                 mbar_arrive_tx(full[s], TX_BYTES);
                 tma_load(&dA, full[s], sA[s], k * BK, bm);
                 tma_load(&dB, full[s], sB[s], k * BK, bn);
             }
         }
     } else {
         float acc[32] = {0};
         
         for (int k = 0; k < nk; k++) {
             int s = k % NUM_STAGES;
             int p = (k / NUM_STAGES) & 1;
             
             mbar_wait(full[s], p);
             __syncwarp();
             
             for (int i = 0; i < 32; i++) wgmma_fence_op(acc[i]);
             wgmma_fence();
             
             for (int ki = 0; ki < BK; ki += WGMMA_K) {
                 uint64_t da = make_desc(sA[s] + ki, WGMMA_STRIDE);
                 uint64_t db = make_desc(sB[s] + ki, WGMMA_STRIDE);
                 wgmma_m64n64k16(acc, da, db);
             }
             
             wgmma_commit();
             for (int i = 0; i < 32; i++) wgmma_fence_op(acc[i]);
             wgmma_wait();
             
             if (lane == 0) mbar_arrive(empty[s]);
             __syncwarp();
         }
         
         for (int r = 0; r < 32; r++) {
             int lm, ln; get_coord(ltid, r, lm, ln);
             int gm = bm + lm, gn = bn + ln;
             if (gm < M && gn < N) C[gm * N + gn] = acc[r];
         }
     }
 }
 
 constexpr size_t SMEM = NUM_STAGES * SMEM_AB + 2 * NUM_STAGES * sizeof(uint64_t);
 
 extern "C" void gemm_bf16_launch(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K) {
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
     cudaLaunchConfig_t cfg = {}; 
     cfg.gridDim = grid; cfg.blockDim = 256; cfg.dynamicSmemBytes = SMEM;
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     cudaLaunchKernelExC(&cfg, (const void*)kernel, args);
 }
 
 extern "C" void gemm_bf16_launch_cooperative(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K) {
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     cudaLaunchCooperativeKernel((const void*)kernel, grid, 256, args, SMEM);
 }
 
 extern "C" float benchmark_kernel(const CUtensorMap* dA, const CUtensorMap* dB, void* C, 
                                    int M, int N, int K, int warmup, int iters) {
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
     cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = 256; cfg.dynamicSmemBytes = SMEM;
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     for (int i = 0; i < warmup; i++) cudaLaunchKernelExC(&cfg, (const void*)kernel, args);
     cudaDeviceSynchronize();
     cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
     for (int i = 0; i < iters; i++) cudaLaunchKernelExC(&cfg, (const void*)kernel, args);
     cudaEventRecord(t1); cudaEventSynchronize(t1);
     float ms; cudaEventElapsedTime(&ms, t0, t1); cudaEventDestroy(t0); cudaEventDestroy(t1);
     return ms / iters;
 }
 