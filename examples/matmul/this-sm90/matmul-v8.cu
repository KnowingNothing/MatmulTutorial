/**
 * Level 8: TMA Store with BF16 Output
 * 
 * Key improvements over Level 8:
 * - Output in BF16 format (instead of FP32)
 * - TMA store from shared memory to global memory
 * - Math warpgroups write BF16 to shared memory
 * - TMA store after all math complete per tile
 * 
 * BM=128, BN=256, BK=64 per CTA
 * Cluster 2x1x1: 2 CTAs share B via bidirectional TMA multicast
 * 3 warpgroups: 1 TMA + 2 math (each 64x256)
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 
 using TmaDescriptor = CUtensorMap;
 
 constexpr int BM = 128, BN = 256, BK = 64, WGMMA_K = 16;
 constexpr int NUM_STAGES = 3, CLUSTER_SIZE = 2;
 constexpr int TOTAL_THREADS = 384;
 constexpr int GROUP_M = 8;
 
 constexpr int SMEM_A = BM * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_B = BN * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_AB = SMEM_A + SMEM_B;
 constexpr int SMEM_C = BM * BN * sizeof(__nv_bfloat16);
 constexpr int TX = SMEM_A + SMEM_B;
 constexpr int WGMMA_SBO = 8 * BK * sizeof(__nv_bfloat16);
 // Total: 3*48KB + padding + 64KB = ~210KB
 constexpr size_t SMEM_SIZE = NUM_STAGES * SMEM_AB + 128 + SMEM_C;
 
 struct ClusterBarrier {
     uint64_t barrier;
     __device__ void init(uint32_t count) {
         asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(count));
     }
     __device__ void arrive_tx(uint32_t tx_bytes) {
         asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(tx_bytes) : "memory");
     }
     __device__ void wait(uint32_t phase) {
         asm volatile("{\n.reg .pred P;\nWAIT:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)), "r"(phase));
     }
     __device__ void arrive_remote(uint32_t target_cta) {
         uint32_t smem_addr = __cvta_generic_to_shared(&barrier);
         uint32_t remote_addr;
         asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
         asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_addr) : "memory");
     }
 };
 
 __device__ void tma_load_2d(const TmaDescriptor* d, ClusterBarrier* bar, void* smem, int32_t c0, int32_t c1) {
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar->barrier)), "r"(c0), "r"(c1) : "memory");
 }
 
 __device__ void tma_multicast_2d(const TmaDescriptor* d, ClusterBarrier* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
     uint64_t cache_hint = 0;
     asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;" :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar->barrier)), "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
 }
 
 __device__ void tma_store_2d(const TmaDescriptor* d, void* smem, int32_t c0, int32_t c1) {
     asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)), "r"(c0), "r"(c1) : "memory");
 }
 
 __device__ void tma_store_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
 // Sync threads within a warpgroup (128 threads, using named barrier)
 __device__ void warpgroup_sync(int wg) { 
     // bar.sync with barrier_id = wg, count = 128
     asm volatile("bar.sync %0, 128;" :: "r"(wg)); 
 }
 __device__ void tma_store_arrive() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
 __device__ void tma_store_wait() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
 
 __device__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r)); return r; }
 __device__ void cluster_sync() { asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory"); }
 __device__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
 
 __device__ uint64_t make_desc_b128(const void* smem_ptr, int sbo) {
     uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
     uint64_t desc = 0;
     desc |= (uint64_t)((addr >> 4) & 0x3FFF);
     desc |= (uint64_t)(3) << 14;
     desc |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
     desc |= (uint64_t)(1) << 62;
     return desc;
 }
 
 __device__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 __device__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
 
 __device__ __forceinline__ void wgmma_m64n256k16(float* c, uint64_t desc_a, uint64_t desc_b) {
     asm volatile(
         "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
         "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
         "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
         "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
         "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
         "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
         "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
         "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
         "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
         "%128,%129,p,1,1,0,0;\n}\n"
         : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
           "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
           "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
           "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),
           "+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),
           "+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),
           "+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),
           "+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63]),
           "+f"(c[64]),"+f"(c[65]),"+f"(c[66]),"+f"(c[67]),"+f"(c[68]),"+f"(c[69]),"+f"(c[70]),"+f"(c[71]),
           "+f"(c[72]),"+f"(c[73]),"+f"(c[74]),"+f"(c[75]),"+f"(c[76]),"+f"(c[77]),"+f"(c[78]),"+f"(c[79]),
           "+f"(c[80]),"+f"(c[81]),"+f"(c[82]),"+f"(c[83]),"+f"(c[84]),"+f"(c[85]),"+f"(c[86]),"+f"(c[87]),
           "+f"(c[88]),"+f"(c[89]),"+f"(c[90]),"+f"(c[91]),"+f"(c[92]),"+f"(c[93]),"+f"(c[94]),"+f"(c[95]),
           "+f"(c[96]),"+f"(c[97]),"+f"(c[98]),"+f"(c[99]),"+f"(c[100]),"+f"(c[101]),"+f"(c[102]),"+f"(c[103]),
           "+f"(c[104]),"+f"(c[105]),"+f"(c[106]),"+f"(c[107]),"+f"(c[108]),"+f"(c[109]),"+f"(c[110]),"+f"(c[111]),
           "+f"(c[112]),"+f"(c[113]),"+f"(c[114]),"+f"(c[115]),"+f"(c[116]),"+f"(c[117]),"+f"(c[118]),"+f"(c[119]),
           "+f"(c[120]),"+f"(c[121]),"+f"(c[122]),"+f"(c[123]),"+f"(c[124]),"+f"(c[125]),"+f"(c[126]),"+f"(c[127])
         : "l"(desc_a), "l"(desc_b));
 }
 
 // Store accumulator to shared memory using DeepGEMM-style layout
 // WGMMA m64n256k16 output: each lane owns 128 regs covering 2 rows x 256 cols
 // Layout: acc[i*4+0..3] covers (row0,col), (row0,col+1), (row1,col), (row1,col+1)
 // where row0 = warp*16 + lane/4, row1 = row0+8, col = (lane%4)*2 + i*8
 __device__ __forceinline__ void store_accum_to_smem(__nv_bfloat16* sC, float* acc, int ltid, int row_offset) {
     int warp = ltid >> 5;
     int lane = ltid & 31;
     int row0 = row_offset + warp * 16 + (lane >> 2);
     int row1 = row0 + 8;
     int col_base = (lane & 3) * 2;
     
     #pragma unroll
     for (int i = 0; i < 32; i++) {
         int col = col_base + i * 8;
         sC[row0 * BN + col + 0] = __float2bfloat16(acc[i * 4 + 0]);
         sC[row0 * BN + col + 1] = __float2bfloat16(acc[i * 4 + 1]);
         sC[row1 * BN + col + 0] = __float2bfloat16(acc[i * 4 + 2]);
         sC[row1 * BN + col + 1] = __float2bfloat16(acc[i * 4 + 3]);
     }
 }
 
 __global__ __launch_bounds__(TOTAL_THREADS, 1)
 void kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB,
             const __grid_constant__ TmaDescriptor dC,
             __nv_bfloat16* C, int M, int N, int K, int num_tiles, int num_clusters) {
     extern __shared__ __align__(1024) char smem[];
     
     __nv_bfloat16 *sA[NUM_STAGES], *sB[NUM_STAGES];
     ClusterBarrier *full[NUM_STAGES], *empty[NUM_STAGES];
     for (int s = 0; s < NUM_STAGES; s++) {
         sA[s] = (__nv_bfloat16*)(smem + s * SMEM_AB);
         sB[s] = (__nv_bfloat16*)(smem + s * SMEM_AB + SMEM_A);
     }
     ClusterBarrier* bars = (ClusterBarrier*)(smem + NUM_STAGES * SMEM_AB);
     for (int s = 0; s < NUM_STAGES; s++) { full[s] = bars + s; empty[s] = bars + NUM_STAGES + s; }
     
     // Output buffer - 128-byte aligned after barriers
     __nv_bfloat16* sC = (__nv_bfloat16*)(smem + NUM_STAGES * SMEM_AB + 128);
     
     int tid = threadIdx.x, wg = tid / 128, ltid = tid % 128, lane = tid % 32;
     uint32_t cta = cluster_rank();
     
     int num_n_tiles = (N + BN - 1) / BN;
     int num_m_clusters = (M + BM * CLUSTER_SIZE - 1) / (BM * CLUSTER_SIZE);
     int nk = (K + BK - 1) / BK;
     int cluster_id = blockIdx.x / CLUSTER_SIZE;
     
     // Persistent loop with L2 swizzle (same as Level 8)
     for (int tile_idx = cluster_id; tile_idx < num_tiles; tile_idx += num_clusters) {
         // Init barriers per tile (reliable approach from Level 8)
         if (tid == 0) {
             for (int s = 0; s < NUM_STAGES; s++) {
                 full[s]->init(1);
                 empty[s]->init(CLUSTER_SIZE * 8);
             }
             fence_barrier_init();
         }
         cluster_sync();
         
         // L2 swizzle
         int num_pid_in_group = GROUP_M * num_n_tiles;
         int group_id = tile_idx / num_pid_in_group;
         int first_pid_m = group_id * GROUP_M;
         int group_size_m = min(GROUP_M, num_m_clusters - first_pid_m);
         int tile_m = first_pid_m + (tile_idx % group_size_m);
         int tile_n = (tile_idx % num_pid_in_group) / group_size_m;
         
         int bn = tile_n * BN;
         int bm = tile_m * (BM * CLUSTER_SIZE) + cta * BM;
         
         if (bm >= M || bn >= N) continue;
         
         float acc[128];
         #pragma unroll
         for (int i = 0; i < 128; i++) acc[i] = 0.0f;
         
         // TMA warpgroup (only tid 0 does TMA, others wait)
         if (wg == 0) {
             if (tid == 0) {
                 for (int k = 0; k < nk; k++) {
                     int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1;
                     if (k >= NUM_STAGES) empty[s]->wait(p ^ 1);
                     tma_load_2d(&dA, full[s], sA[s], k * BK, bm);
                     int b_offset = cta * (BN / 2);
                     tma_multicast_2d(&dB, full[s], sB[s] + b_offset * BK, k * BK, bn + b_offset, 0x3);
                     full[s]->arrive_tx(TX);
                 }
             }
             // wg0 threads wait here for sync with other warpgroups at end of tile
         }
         // Warpgroup 1: math for rows [0, 64)
         else if (wg == 1) {
             for (int k = 0; k < nk; k++) {
                 int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1;
                 full[s]->wait(p);
                 __syncwarp();
                 wgmma_fence();
                 for (int ki = 0; ki < BK; ki += WGMMA_K) {
                     uint64_t da = make_desc_b128(sA[s] + ki, WGMMA_SBO);
                     uint64_t db = make_desc_b128(sB[s] + ki, WGMMA_SBO);
                     wgmma_m64n256k16(acc, da, db);
                 }
                 wgmma_commit();
                 wgmma_wait();
                 if (lane < CLUSTER_SIZE) empty[s]->arrive_remote(lane);
                 __syncwarp();
             }
             // Write to sC (rows 0-63) using optimized store
             store_accum_to_smem(sC, acc, ltid, 0);
         }
         // Warpgroup 2: math for rows [64, 128)
         else if (wg == 2) {
             for (int k = 0; k < nk; k++) {
                 int s = k % NUM_STAGES, p = (k / NUM_STAGES) & 1;
                 full[s]->wait(p);
                 __syncwarp();
                 wgmma_fence();
                 for (int ki = 0; ki < BK; ki += WGMMA_K) {
                     uint64_t da = make_desc_b128(sA[s] + 64 * BK + ki, WGMMA_SBO);
                     uint64_t db = make_desc_b128(sB[s] + ki, WGMMA_SBO);
                     wgmma_m64n256k16(acc, da, db);
                 }
                 wgmma_commit();
                 wgmma_wait();
                 if (lane < CLUSTER_SIZE) empty[s]->arrive_remote(lane);
                 __syncwarp();
             }
             // Write to sC (rows 64-127) using optimized store
             store_accum_to_smem(sC, acc, ltid, 64);
         }
         
         // Sync all warpgroups, fence, then TMA store
         __syncthreads();
         tma_store_fence();
         if (tid == 0) {
             tma_store_2d(&dC, sC, bn, bm);
             tma_store_2d(&dC, sC + 64 * BN, bn, bm + 64);
             tma_store_arrive();
             tma_store_wait();
         }
         __syncthreads();
     }
 }
 
 extern "C" void gemm_bf16_launch(const CUtensorMap* dA, const CUtensorMap* dB,
                                   const CUtensorMap* dC, void* C, int M, int N, int K) {
     int num_n_tiles = (N + BN - 1) / BN;
     int num_m_clusters = (M + BM * CLUSTER_SIZE - 1) / (BM * CLUSTER_SIZE);
     int num_tiles = num_m_clusters * num_n_tiles;
     int num_clusters = min(132, num_tiles);
     dim3 grid(num_clusters * CLUSTER_SIZE, 1);
     
     cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
     cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
     
     cudaLaunchConfig_t cfg = {};
     cfg.gridDim = grid;
     cfg.blockDim = TOTAL_THREADS;
     cfg.dynamicSmemBytes = SMEM_SIZE;
     
     cudaLaunchAttribute attrs[1];
     attrs[0].id = cudaLaunchAttributeClusterDimension;
     attrs[0].val.clusterDim.x = CLUSTER_SIZE;
     attrs[0].val.clusterDim.y = 1;
     attrs[0].val.clusterDim.z = 1;
     cfg.attrs = attrs;
     cfg.numAttrs = 1;
     
     void* args[] = {(void*)dA, (void*)dB, (void*)dC, &C, &M, &N, &K, &num_tiles, &num_clusters};
     cudaLaunchKernelExC(&cfg, (const void*)kernel, args);
 }
 