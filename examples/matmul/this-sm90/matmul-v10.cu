/**
 * Level 10: DeepGEMM Style with Full Optimizations
 * 
 * BM=128, BN=256, BK=64 per CTA with all DeepGEMM optimizations:
 * - 128B swizzle for output with STSM instructions
 * - 4 parallel TMA stores (64 columns each)
 * - Optimized WGMMA descriptor computation (pre-computed base + offsets)
 * - 3 warpgroups: 1 TMA + 2 Math
 * - Supports arbitrary M, N dimensions (K aligned to BK)
 * 
 * Key features:
 * - 2 math warpgroups, each processing 64 rows (m64n256k16 WGMMA)
 * - Dynamic register allocation (48 for TMA, 224 for Math)
 * - Independent tile scheduling between TMA and Math
 * - NUM_STAGES=3 for BM=128
 * - Named barrier sync for math warpgroup coordination
 * - TMA oobFill=1 for zero-padding out-of-bounds loads
 * 
 * Performance: ~800 TFLOPS on H800 (8192x8192)
 * 
 * Cluster 2x1x1: 2 CTAs share B via bidirectional TMA multicast
 * 
 * Arbitrary Shape Support:
 * - K must be aligned to BK (64) for TMA 128B swizzle
 * - N stride must be aligned to 64 for TMA store swizzle (C tensor)
 * - TMA oobFill=1 handles out-of-bounds loads (zero-fill)
 * - TMA stores are skipped for out-of-bounds output regions
 * - All tiles are processed to maintain barrier synchronization
 */
 #include <cuda_runtime.h>
 #include <cuda_bf16.h>
 #include <cuda.h>
 #include <cstdint>
 
 using TmaDescriptor = CUtensorMap;
 
 // BM=128, BN=256 with 2 math warpgroups
 constexpr int BM = 128, BN = 256, BK = 64, WGMMA_K = 16;
 constexpr int NUM_STAGES = 3, CLUSTER_SIZE = 2;
 constexpr int TOTAL_THREADS = 384;  // 1 TMA + 2 Math warpgroups
 constexpr int GROUP_M = 8;
 
 // 128B swizzle for output (DeepGEMM style)
 constexpr int SWIZZLE_D_MODE = 128;  // 128 bytes
 constexpr int TMA_D_BLOCK_N = SWIZZLE_D_MODE / sizeof(__nv_bfloat16);  // 64 columns per TMA store
 constexpr int NUM_TMA_D_BLOCKS = BN / TMA_D_BLOCK_N;  // 4 parallel TMA stores
 constexpr int WGMMA_M_PER_WARP = 64 / 4;  // 16 rows per warp in each math WG
 
 constexpr int SMEM_A = BM * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_B = BN * BK * sizeof(__nv_bfloat16);
 constexpr int SMEM_AB = SMEM_A + SMEM_B;
 // SMEM_D with swizzle: BM=128 requires 64KB output buffer, aligned to 1024
 constexpr int SMEM_D_SIZE = ((BM * BN * sizeof(__nv_bfloat16) + 1023) / 1024) * 1024;  // 64KB aligned
 constexpr int TX = SMEM_A + SMEM_B;
 constexpr int WGMMA_SBO = 8 * BK * sizeof(__nv_bfloat16);
 constexpr size_t SMEM_SIZE = SMEM_D_SIZE + NUM_STAGES * SMEM_AB + 128;  // D first, then A/B
 
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
     __device__ void arrive() {
         asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&barrier)) : "memory");
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
 __device__ void tma_store_arrive() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
 __device__ void tma_store_wait() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
 
 __device__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r)); return r; }
 __device__ void cluster_sync() { asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory"); }
 __device__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
 
 // Named barrier for math warpgroup sync (before TMA store)
 __device__ void named_barrier_sync(int bar_id, int count) {
     asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count));
 }
 
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
 
 // Fence operand to prevent compiler from reordering accumulator access
 __device__ __forceinline__ void wgmma_fence_operand(float& reg) {
     asm volatile("" : "+f"(reg) :: "memory");
 }
 
 // DeepGEMM style warpgroup functions
 __device__ void warpgroup_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
 __device__ void warpgroup_commit_batch() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
 
 template<int N> __device__ void warpgroup_reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
 template<int N> __device__ void warpgroup_reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
 
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
 
 // STSM instruction: stmatrix.sync.aligned.x2.m8n8.shared.b16
 // Stores 2 bf162 values (8 bytes) using matrix store instruction
 __device__ __forceinline__ void stsm_x2(__nv_bfloat162 v0, __nv_bfloat162 v1, void* smem_ptr) {
     uint32_t src0 = *reinterpret_cast<uint32_t*>(&v0);
     uint32_t src1 = *reinterpret_cast<uint32_t*>(&v1);
     asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                  :: "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)), "r"(src0), "r"(src1));
 }
 
 // Store accumulator to SMEM with 128B swizzle layout (DeepGEMM style)
 // For BM=128: each math warpgroup handles 64 rows, with m_offset indicating row group
 // SMEM layout: column-major with 128B swizzle for TMA store
 __device__ __forceinline__ void store_accum_to_smem_swizzle(__nv_bfloat16* sD, float* acc, int warp_idx, int lane_idx, int m_offset) {
     constexpr int NUM_BANK_GROUP_BYTES = 16;  // 16 bytes = 8 bf16 per bank group
     constexpr int WAVE_BLOCK_M = 64;  // Each WGMMA handles 64 rows
     
     // 128 accumulators / 4 = 32 iterations, each stores 4 bf16 values
     #pragma unroll
     for (int i = 0; i < 32; i++) {
         // Which 64-column atom does this belong to? (0-3 for BN=256)
         int atom_offset = i / (TMA_D_BLOCK_N / 8);  // i/8, so 0-3
         int in_atom_offset = i % (TMA_D_BLOCK_N / 8);  // i%8, so 0-7
         
         // Reshape and swizzle: kHasShortcut = true since 128/16 == 8
         int row = in_atom_offset / 8 + lane_idx;  // = lane_idx
         int col = in_atom_offset;
         col ^= row % (SWIZZLE_D_MODE / 16);  // XOR swizzle: row % 8
         
         // Calculate SMEM address with m_offset for row group (0 or 64)
         uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(sD) +
             warp_idx * (WGMMA_M_PER_WARP * SWIZZLE_D_MODE) +  // Warp offset in this group
             m_offset * SWIZZLE_D_MODE +                        // Row group offset (0 or 64*128)
             atom_offset * BM * SWIZZLE_D_MODE +               // Atom offset: atom * 128 * 128
             row * (NUM_BANK_GROUP_BYTES * 8) + col * NUM_BANK_GROUP_BYTES;
         
         __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i * 4 + 0], acc[i * 4 + 1]);
         __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i * 4 + 2], acc[i * 4 + 3]);
         
         stsm_x2(v0, v1, smem_ptr);
     }
 }
 
 // Tile scheduler: compute tile coordinates from linear tile index
 __device__ __forceinline__ void get_tile_coords(int tile_idx, int num_n_tiles, int num_m_clusters,
                                                   int& tile_m, int& tile_n) {
     int num_pid_in_group = GROUP_M * num_n_tiles;
     int group_id = tile_idx / num_pid_in_group;
     int first_pid_m = group_id * GROUP_M;
     int group_size_m = min(GROUP_M, num_m_clusters - first_pid_m);
     tile_m = first_pid_m + (tile_idx % group_size_m);
     tile_n = (tile_idx % num_pid_in_group) / group_size_m;
 }
 
 __global__ __launch_bounds__(TOTAL_THREADS, 1)
 void kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB,
             const __grid_constant__ TmaDescriptor dC,
             __nv_bfloat16* C, int M, int N, int K, int num_tiles, int num_clusters) {
     // Align to 1024 bytes for swizzle-128B (DeepGEMM style)
     extern __shared__ __align__(1024) char smem[];
     
     // DeepGEMM layout: D first, then A/B stages, then barriers
     __nv_bfloat16* sD = (__nv_bfloat16*)smem;  // Output buffer with swizzle
     
     __nv_bfloat16 *sA[NUM_STAGES], *sB[NUM_STAGES];
     ClusterBarrier *full[NUM_STAGES], *empty[NUM_STAGES];
     for (int s = 0; s < NUM_STAGES; s++) {
         sA[s] = (__nv_bfloat16*)(smem + SMEM_D_SIZE + s * SMEM_AB);
         sB[s] = (__nv_bfloat16*)(smem + SMEM_D_SIZE + s * SMEM_AB + SMEM_A);
     }
     ClusterBarrier* bars = (ClusterBarrier*)(smem + SMEM_D_SIZE + NUM_STAGES * SMEM_AB);
     for (int s = 0; s < NUM_STAGES; s++) { full[s] = bars + s; empty[s] = bars + NUM_STAGES + s; }
     
     int tid = threadIdx.x, wg = tid / 128, ltid = tid % 128, lane = tid % 32;
     uint32_t cta = cluster_rank();
     
     int num_n_tiles = (N + BN - 1) / BN;
     int num_m_clusters = (M + BM * CLUSTER_SIZE - 1) / (BM * CLUSTER_SIZE);
     int nk = (K + BK - 1) / BK;
     int cluster_id = blockIdx.x / CLUSTER_SIZE;
     
     // Total k iterations across all tiles for this cluster
     int total_tiles_for_cluster = (num_tiles - cluster_id + num_clusters - 1) / num_clusters;
     int total_k_iters = total_tiles_for_cluster * nk;
     
     // =========================================================
     // BARRIER INITIALIZATION - ONCE AT KERNEL START
     // =========================================================
     if (tid == 0) {
         #pragma unroll
         for (int s = 0; s < NUM_STAGES; s++) {
             full[s]->init(1);  // TMA producer arrives once
             empty[s]->init(CLUSTER_SIZE * 8);  // 2 math WG * 4 warps * 2 CTAs = 16
         }
         fence_barrier_init();
     }
     cluster_sync();
     
     // =========================================================
     // TMA WARPGROUP - Independent tile scheduling
     // =========================================================
     if (wg == 0) {
         // TMA warpgroup releases registers for math warpgroup (keep 48)
         warpgroup_reg_dealloc<40>();
         
         if (tid == 0) {
             int stage = 0;
             int phase = 0;
             
             for (int tile_idx = cluster_id; tile_idx < num_tiles; tile_idx += num_clusters) {
                 int tile_m, tile_n;
                 get_tile_coords(tile_idx, num_n_tiles, num_m_clusters, tile_m, tile_n);
                 int bn = tile_n * BN;
                 int bm = tile_m * (BM * CLUSTER_SIZE) + cta * BM;
                 
                 // Don't skip tiles - TMA oobFill will handle out-of-bounds
                 // Must process all tiles to maintain barrier sync with math warpgroups
                 
                 for (int k = 0; k < nk; k++) {
                     // Wait for empty barrier (consumers released this stage)
                     if (stage == 0 && phase > 0) {
                         empty[0]->wait(phase ^ 1);
                     } else if (tile_idx > cluster_id || k >= NUM_STAGES) {
                         empty[stage]->wait(phase ^ 1);
                     }
                     
                     // Issue TMA loads - TMA oobFill=1 will zero-pad out-of-bounds accesses
                     tma_load_2d(&dA, full[stage], sA[stage], k * BK, bm);
                     int b_offset = cta * (BN / 2);
                     tma_multicast_2d(&dB, full[stage], sB[stage] + b_offset * BK, k * BK, bn + b_offset, 0x3);
                     full[stage]->arrive_tx(TX);
                     
                     // Advance pipeline
                     stage++;
                     if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
                 }
             }
         }
         // Other threads in wg0 do nothing
     }
     // =========================================================
     // MATH WARPGROUPS - wg1 handles rows 0-63, wg2 handles rows 64-127
     // =========================================================
     else {
         // Math warpgroups claim more registers (224 for 2 WGs)
         warpgroup_reg_alloc<224>();
         
         int math_wg = wg - 1;  // 0 or 1
         int m_offset = math_wg * 64;  // Row offset: 0 or 64
         int warp_in_wg = ltid / 32;  // 0-3 within this warpgroup
         int lane_idx = ltid % 32;
         
         int stage = 0, phase = 0;
         bool first_tile = true;
         float acc[128];
         
         // Pre-compute base descriptor (DeepGEMM optimization)
         uint32_t base_desc_a = ((uint32_t)__cvta_generic_to_shared(sA[0])) >> 4;
         uint32_t base_desc_b = ((uint32_t)__cvta_generic_to_shared(sB[0])) >> 4;
         constexpr uint32_t desc_k_stride = WGMMA_K * sizeof(__nv_bfloat16) / 16;
         constexpr uint32_t stage_stride = SMEM_AB / 16;
         constexpr uint32_t m_stride = 64 * BK * sizeof(__nv_bfloat16) / 16;  // Offset for rows 64-127 in A
         constexpr uint64_t desc_hi = ((uint64_t)3 << 14) |
                                      ((uint64_t)((WGMMA_SBO >> 4) & 0x3FFF) << 32) |
                                      ((uint64_t)1 << 62);
         
         for (int tile_idx = cluster_id; tile_idx < num_tiles; tile_idx += num_clusters) {
             int tile_m, tile_n;
             get_tile_coords(tile_idx, num_n_tiles, num_m_clusters, tile_m, tile_n);
             int bn = tile_n * BN;
             int bm = tile_m * (BM * CLUSTER_SIZE) + cta * BM;
             
             // Don't skip tiles - always process to maintain barrier sync with TMA warpgroup
             // TMA oobFill=1 will zero-pad out-of-bounds loads
             
             #pragma unroll
             for (int i = 0; i < 128; i++) acc[i] = 0.0f;
             
             for (int k = 0; k < nk; k++) {
                 full[stage]->wait(phase);
                 
                 warpgroup_arrive();
                 uint32_t s_off = stage * stage_stride;
                 uint32_t a_m_off = math_wg * m_stride;  // Offset for this warpgroup's rows in A
                 #pragma unroll
                 for (int ki = 0; ki < BK; ki += WGMMA_K) {
                     uint32_t k_off = (ki / WGMMA_K) * desc_k_stride;
                     uint64_t da = ((base_desc_a + s_off + a_m_off + k_off) & 0x3FFF) | desc_hi;
                     uint64_t db = ((base_desc_b + s_off + k_off) & 0x3FFF) | desc_hi;
                     wgmma_m64n256k16(acc, da, db);
                 }
                 warpgroup_commit_batch();
                 wgmma_wait();
                 
                 if (lane < CLUSTER_SIZE) empty[stage]->arrive_remote(lane);
                 
                 stage++;
                 if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
             }
             
             // Wait for previous TMA stores to complete
             if (!first_tile) {
                 if (math_wg == 0 && ltid == 0) tma_store_wait();
                 named_barrier_sync(0, 256);
             }
             first_tile = false;
             
             // Store to SMEM with 128B swizzle layout using STSM
             store_accum_to_smem_swizzle(sD, acc, warp_in_wg, lane_idx, m_offset);
             
             // Fence for TMA
             tma_store_fence();
             named_barrier_sync(0, 256);
             
             // Issue TMA stores - only for valid output region
             // Thread 0 issues all stores, checking bounds for each
             if (math_wg == 0 && ltid == 0) {
                 // Only store if this tile is within the valid output region
                 if (bm < M && bn < N) {
                     for (int t = 0; t < NUM_TMA_D_BLOCKS; t++) {
                         int col_offset = t * TMA_D_BLOCK_N;
                         if (bn + col_offset < N) {
                             __nv_bfloat16* atom_ptr = sD + t * BM * (SWIZZLE_D_MODE / sizeof(__nv_bfloat16));
                             tma_store_2d(&dC, atom_ptr, bn + col_offset, bm);
                         }
                     }
                 }
                 tma_store_arrive();
             }
         }
         
         // Final wait for all TMA stores
         if (math_wg == 0 && ltid == 0) tma_store_wait();
     }
     
     // Sync all warpgroups before kernel ends to ensure barriers are in clean state
     cluster_sync();
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
 