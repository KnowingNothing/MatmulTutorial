/*
 * Level 4: Large Tile GEMM (BM=128, BN=128, BK=64)
 * Uses 4x m64n64k16 WGMMAs to compute 128x128 output
 * 4x compute per block vs Level 4
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 using TmaDescriptor = CUtensorMap;
 constexpr int BM = 128, BN = 128, BK = 64, WGMMA_K = 16;
 constexpr int NUM_STAGES = 2;
 constexpr int WGMMA_STRIDE = 8 * BK * sizeof(__nv_bfloat16);
 struct TxBarrier {
     uint64_t barrier;
     __device__ void init(uint32_t c) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(c)); }
     __device__ void arrive() { asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier))); }
     __device__ void arrive_tx(uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(tx)); }
     __device__ void wait(uint32_t phase) { asm volatile("{\n.reg .pred P;\nWAIT:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(phase)); }
 };
 __device__ void tma_load_2d(const TmaDescriptor* d, TxBarrier* m, void* s, int32_t x, int32_t y) {
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&m->barrier)), "r"(x), "r"(y) : "memory");
 }
 __device__ uint64_t make_desc(const void* p, int stride) {
     uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
     return ((uint64_t)((a >> 4) & 0x3FFF)) | ((uint64_t)((stride >> 4) & 0x3FFF) << 32) | ((uint64_t)1 << 62);
 }
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 __device__ void wgmma_fence_op(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
 __device__ void wgmma_m64n64k16(float* c, uint64_t a, uint64_t b) {
     asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;\n}\n" : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]) : "l"(a), "l"(b));
 }
 __device__ void get_coord(int t, int r, int& row, int& col) {
     int t0 = t % 4, t1 = (t / 4) % 8, t2 = t / 32, r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
     int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512; row = lin % 64; col = lin / 64;
 }
 constexpr int SMEM_A = BM * BK * 2, SMEM_B = BN * BK * 2, SMEM_AB = SMEM_A + SMEM_B, TX = SMEM_AB;
 __global__ __launch_bounds__(256, 1) void kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int M, int N, int K) {
     extern __shared__ __align__(1024) char smem[];
     __nv_bfloat16 *sA[NUM_STAGES], *sB[NUM_STAGES]; TxBarrier *full[NUM_STAGES], *empty[NUM_STAGES];
     for (int s = 0; s < NUM_STAGES; s++) { sA[s] = (__nv_bfloat16*)(smem + s * SMEM_AB); sB[s] = (__nv_bfloat16*)(smem + s * SMEM_AB + SMEM_A); }
     TxBarrier* b = (TxBarrier*)(smem + NUM_STAGES * SMEM_AB); for (int s = 0; s < NUM_STAGES; s++) { full[s] = b + s; empty[s] = b + NUM_STAGES + s; }
     int tid = threadIdx.x, wg = tid / 128, lane = tid % 32, ltid = tid % 128, bm = blockIdx.y * BM, bn = blockIdx.x * BN;
     if (tid == 0) for (int s = 0; s < NUM_STAGES; s++) { full[s]->init(1); empty[s]->init(4); }
     __syncthreads(); asm volatile("fence.proxy.async;\n" ::: "memory"); __syncthreads();
     if (bm >= M || bn >= N) return; int nk = (K + BK - 1) / BK;
     if (wg == 0 && tid == 0) { for (int k = 0; k < nk; k++) { int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1; if (k >= NUM_STAGES) empty[s]->wait(p ^ 1); full[s]->arrive_tx(TX);
         for (int ma = 0; ma < 2; ma++) tma_load_2d(&dA, full[s], sA[s] + ma * 64 * BK, k * BK, bm + ma * 64);
         for (int nb = 0; nb < 2; nb++) tma_load_2d(&dB, full[s], sB[s] + nb * 64 * BK, k * BK, bn + nb * 64); } }
     else if (wg == 1) {
         float acc00[32]={0}, acc01[32]={0}, acc10[32]={0}, acc11[32]={0};
         for (int k = 0; k < nk; k++) { int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1; full[s]->wait(p); __syncwarp();
             for (int i = 0; i < 32; i++) { wgmma_fence_op(acc00[i]); wgmma_fence_op(acc01[i]); wgmma_fence_op(acc10[i]); wgmma_fence_op(acc11[i]); } wgmma_fence();
             for (int ki = 0; ki < BK; ki += WGMMA_K) {
                 wgmma_m64n64k16(acc00, make_desc(sA[s] + ki, WGMMA_STRIDE), make_desc(sB[s] + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc01, make_desc(sA[s] + ki, WGMMA_STRIDE), make_desc(sB[s] + 64 * BK + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc10, make_desc(sA[s] + 64 * BK + ki, WGMMA_STRIDE), make_desc(sB[s] + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc11, make_desc(sA[s] + 64 * BK + ki, WGMMA_STRIDE), make_desc(sB[s] + 64 * BK + ki, WGMMA_STRIDE)); }
             wgmma_commit(); for (int i = 0; i < 32; i++) { wgmma_fence_op(acc00[i]); wgmma_fence_op(acc01[i]); wgmma_fence_op(acc10[i]); wgmma_fence_op(acc11[i]); } wgmma_wait();
             if (lane == 0) empty[s]->arrive(); __syncwarp(); }
         for (int r = 0; r < 32; r++) { int lm, ln; get_coord(ltid, r, lm, ln);
             if (bm + lm < M && bn + ln < N) C[(bm + lm) * N + bn + ln] = acc00[r];
             if (bm + lm < M && bn + 64 + ln < N) C[(bm + lm) * N + bn + 64 + ln] = acc01[r];
             if (bm + 64 + lm < M && bn + ln < N) C[(bm + 64 + lm) * N + bn + ln] = acc10[r];
             if (bm + 64 + lm < M && bn + 64 + ln < N) C[(bm + 64 + lm) * N + bn + 64 + ln] = acc11[r]; } } }
 constexpr size_t SMEM = NUM_STAGES * SMEM_AB + 2 * NUM_STAGES * sizeof(TxBarrier);
 extern "C" void gemm_bf16_launch(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K) { dim3 g((N+BN-1)/BN,(M+BM-1)/BM); cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); cudaLaunchConfig_t c={}; c.gridDim=g;c.blockDim=256;c.dynamicSmemBytes=SMEM; void* a[]={(void*)dA,(void*)dB,&C,&M,&N,&K}; cudaLaunchKernelExC(&c,(const void*)kernel,a); }
 extern "C" float benchmark_kernel(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K, int w, int it) { dim3 g((N+BN-1)/BN,(M+BM-1)/BM); cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); cudaLaunchConfig_t c={}; c.gridDim=g;c.blockDim=256;c.dynamicSmemBytes=SMEM; void* a[]={(void*)dA,(void*)dB,&C,&M,&N,&K}; for(int i=0;i<w;i++)cudaLaunchKernelExC(&c,(const void*)kernel,a); cudaDeviceSynchronize(); cudaEvent_t t0,t1; cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventRecord(t0); for(int i=0;i<it;i++)cudaLaunchKernelExC(&c,(const void*)kernel,a); cudaEventRecord(t1);cudaEventSynchronize(t1); float ms; cudaEventElapsedTime(&ms,t0,t1);cudaEventDestroy(t0);cudaEventDestroy(t1); return ms/it; }
 