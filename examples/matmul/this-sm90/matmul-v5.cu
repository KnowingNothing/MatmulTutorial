/* Level 5: BM=128, BN=128, BK=64, Cluster=2x1x1, TMA Multicast B
 * 
 * Features:
 * - DeepGEMM-style register reconfiguration (TMA 48 regs, Math 232 regs)
 * - Cluster of 2 CTAs sharing B matrix via TMA multicast
 * - Each CTA loads its own A, CTA 0 multicasts B to both
 * - Performance: ~317-346 TFLOPS on H800
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 
 using TmaDescriptor = CUtensorMap;
 
 constexpr int BM = 128, BN = 128, BK = 64, WGMMA_K = 16;
 constexpr int NUM_STAGES = 2, CLUSTER_SIZE = 2;
 constexpr int TOTAL_THREADS = 256;
 constexpr int NUM_TMA_REGS = 48, NUM_MATH_REGS = 232;
 constexpr int WGMMA_STRIDE = 8 * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_A = BM * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_B = BN * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_AB = SMEM_A + SMEM_B;
 constexpr int TX = SMEM_AB;
 
 struct ClusterBarrier {
     uint64_t barrier;
     __device__ void init(uint32_t c) { 
         asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(c)); 
     }
     __device__ void arrive() { 
         asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier))); 
     }
     __device__ void arrive_remote(uint32_t target_cta) {
         uint32_t smem_addr = __cvta_generic_to_shared(&barrier);
         uint32_t remote_addr;
         asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
         asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_addr) : "memory");
     }
     __device__ void arrive_tx(uint32_t tx) { 
         asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(tx) : "memory"); 
     }
     __device__ void wait(uint32_t p) { 
         asm volatile("{\n.reg .pred P;\nWAIT:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT;\n}\n" 
             :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(p)); 
     }
 };
 
 __device__ void tma_load_2d(const TmaDescriptor* d, ClusterBarrier* m, void* s, int32_t x, int32_t y) {
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" 
         :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"((uint64_t)d), 
            "r"((uint32_t)__cvta_generic_to_shared(&m->barrier)), "r"(x), "r"(y) : "memory");
 }
 
 __device__ void tma_multicast_2d(const TmaDescriptor* d, ClusterBarrier* m, void* s, int32_t x, int32_t y, uint16_t mask) {
     uint64_t cache_hint = 0;
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;" 
         :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"((uint64_t)d), 
            "r"((uint32_t)__cvta_generic_to_shared(&m->barrier)), "h"(mask), "r"(x), "r"(y), "l"(cache_hint) : "memory");
 }
 
 __device__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r)); return r; }
 __device__ void cluster_sync() { asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory"); }
 __device__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
 
 template<int N> __device__ void warpgroup_reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
 template<int N> __device__ void warpgroup_reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
 
 __device__ uint64_t make_desc(const void* p, int stride) {
     uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
     return ((uint64_t)((a >> 4) & 0x3FFF)) | ((uint64_t)((stride >> 4) & 0x3FFF) << 32) | ((uint64_t)1 << 62);
 }
 
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 __device__ void wgmma_fence_op(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
 
 __device__ void wgmma_m64n64k16(float* c, uint64_t a, uint64_t b) {
     asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;\n}\n" 
         : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
           "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
           "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
           "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]) 
         : "l"(a), "l"(b));
 }
 
 __device__ void get_coord(int t, int r, int& row, int& col) {
     int t0 = t % 4, t1 = (t / 4) % 8, t2 = t / 32, r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
     int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512; row = lin % 64; col = lin / 64;
 }
 
 constexpr size_t SMEM_SIZE = NUM_STAGES * SMEM_AB + 2 * NUM_STAGES * sizeof(ClusterBarrier);
 
 __global__ __launch_bounds__(TOTAL_THREADS, 1)
 void kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int M, int N, int K) {
     extern __shared__ __align__(1024) char smem[];
     __nv_bfloat16 *sA[NUM_STAGES], *sB[NUM_STAGES]; 
     ClusterBarrier *full[NUM_STAGES], *empty[NUM_STAGES];
     for (int s = 0; s < NUM_STAGES; s++) { 
         sA[s] = (__nv_bfloat16*)(smem + s * SMEM_AB); 
         sB[s] = (__nv_bfloat16*)(smem + s * SMEM_AB + SMEM_A); 
     }
     ClusterBarrier* b = (ClusterBarrier*)(smem + NUM_STAGES * SMEM_AB);
     for (int s = 0; s < NUM_STAGES; s++) { full[s] = b + s; empty[s] = b + NUM_STAGES + s; }
 
     int tid = threadIdx.x, wg = tid / 128, lane = tid % 32, ltid = tid % 128;
     uint32_t cta = cluster_rank();
     
     int n_tile = blockIdx.x / CLUSTER_SIZE;
     int bn = n_tile * BN;
     int cluster_m = blockIdx.y * (BM * CLUSTER_SIZE);
     int bm = cluster_m + cta * BM;
 
     if (tid == 0) {
         for (int s = 0; s < NUM_STAGES; s++) { 
             full[s]->init(1); 
             empty[s]->init(CLUSTER_SIZE * 4);
         }
         fence_barrier_init();
     }
     cluster_sync();
     
     if (bm >= M || bn >= N) return;
     int nk = (K + BK - 1) / BK;
 
     if (wg == 0) {
         warpgroup_reg_dealloc<NUM_TMA_REGS>();
         if (tid == 0) {
             for (int k = 0; k < nk; k++) {
                 int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1;
                 if (k >= NUM_STAGES) empty[s]->wait(p ^ 1);
                 for (int ma = 0; ma < 2; ma++) 
                     tma_load_2d(&dA, full[s], sA[s] + ma * 64 * BK, k * BK, bm + ma * 64);
                 if (cta == 0) {
                     for (int nb = 0; nb < 2; nb++) 
                         tma_multicast_2d(&dB, full[s], sB[s] + nb * 64 * BK, k * BK, bn + nb * 64, 0x3);
                 }
                 full[s]->arrive_tx(TX);
             }
         }
     }
     else if (wg == 1) {
         warpgroup_reg_alloc<NUM_MATH_REGS>();
         float acc00[32]={0}, acc01[32]={0}, acc10[32]={0}, acc11[32]={0};
         
         for (int k = 0; k < nk; k++) {
             int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1;
             full[s]->wait(p); __syncwarp();
             for (int i = 0; i < 32; i++) { wgmma_fence_op(acc00[i]); wgmma_fence_op(acc01[i]); wgmma_fence_op(acc10[i]); wgmma_fence_op(acc11[i]); } 
             wgmma_fence();
             for (int ki = 0; ki < BK; ki += WGMMA_K) {
                 wgmma_m64n64k16(acc00, make_desc(sA[s] + ki, WGMMA_STRIDE), make_desc(sB[s] + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc01, make_desc(sA[s] + ki, WGMMA_STRIDE), make_desc(sB[s] + 64 * BK + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc10, make_desc(sA[s] + 64 * BK + ki, WGMMA_STRIDE), make_desc(sB[s] + ki, WGMMA_STRIDE));
                 wgmma_m64n64k16(acc11, make_desc(sA[s] + 64 * BK + ki, WGMMA_STRIDE), make_desc(sB[s] + 64 * BK + ki, WGMMA_STRIDE));
             }
             wgmma_commit(); 
             for (int i = 0; i < 32; i++) { wgmma_fence_op(acc00[i]); wgmma_fence_op(acc01[i]); wgmma_fence_op(acc10[i]); wgmma_fence_op(acc11[i]); } 
             wgmma_wait();
             if (lane < CLUSTER_SIZE) empty[s]->arrive_remote(lane);
             __syncwarp();
         }
         
         for (int r = 0; r < 32; r++) { 
             int lm, ln; get_coord(ltid, r, lm, ln);
             if (bm + lm < M && bn + ln < N) C[(bm + lm) * N + bn + ln] = acc00[r];
             if (bm + lm < M && bn + 64 + ln < N) C[(bm + lm) * N + bn + 64 + ln] = acc01[r];
             if (bm + 64 + lm < M && bn + ln < N) C[(bm + 64 + lm) * N + bn + ln] = acc10[r];
             if (bm + 64 + lm < M && bn + 64 + ln < N) C[(bm + 64 + lm) * N + bn + 64 + ln] = acc11[r]; 
         }
     }
 }
 
 extern "C" void gemm_bf16_launch(const CUtensorMap* dA, const CUtensorMap* dB, void* C, int M, int N, int K) {
     int num_n_tiles = (N + BN - 1) / BN;
     int num_m_clusters = (M + BM * CLUSTER_SIZE - 1) / (BM * CLUSTER_SIZE);
     dim3 grid(num_n_tiles * CLUSTER_SIZE, num_m_clusters);
     
     cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
     
     cudaLaunchConfig_t cfg = {}; 
     cfg.gridDim = grid; cfg.blockDim = TOTAL_THREADS; cfg.dynamicSmemBytes = SMEM_SIZE;
     cudaLaunchAttribute attrs[1]; 
     attrs[0].id = cudaLaunchAttributeClusterDimension;
     attrs[0].val.clusterDim.x = CLUSTER_SIZE; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
     cfg.attrs = attrs; cfg.numAttrs = 1;
     
     void* args[] = {(void*)dA, (void*)dB, &C, &M, &N, &K};
     cudaLaunchKernelExC(&cfg, (const void*)kernel, args);
 }
 