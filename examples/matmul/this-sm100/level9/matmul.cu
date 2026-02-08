// level9/matmul.cu
// SM100 (GB200) BF16 GEMM — Matching DeepGEMM Architecture
//
// Key changes from Level 8 (derived from DeepGEMM analysis):
//   1. BLOCK_M=256, BLOCK_N=256, NUM_STAGES=4.  Larger tiles = fewer
//      scheduler iterations + better L2 reuse.  Each tile computes 256×256
//      output with 2 M-waves of 128 rows each.
//   2. Separated A/B SMEM layout (A[0..3] contiguous, then B[0..3]).
//      Enables stride-based __shfl_sync descriptor caching.
//   3. MMA warp: all 32 threads participate in barrier_wait + __shfl_sync.
//      elect_one() guards only UMMA issuance + commit (matching CUTLASS).
//   4. tmem_empty: ALL 128 epilogue threads on BOTH CTAs arrive at the
//      leader's barrier via cluster addressing (mapa + shared::cluster).
//      Init count = CLUSTER_SIZE * 128 = 256.
//   5. L2 promotion L2_256B in TMA descriptors.
//   6. Scheduler swizzle group = 16 (was 8).
//   7. NUM_EPILOGUE_STAGES = 1 (no TMEM double buffering; TMEM_COLS=512).
//
// Computes D = A @ B^T
//   A: (M,K) row-major BF16,  B: (N,K) row-major BF16,  D: (M,N) row-major BF16

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ============================================================
//  Configuration
// ============================================================
static constexpr uint32_t BLOCK_M          = 256;
static constexpr uint32_t BLOCK_N          = 256;
static constexpr uint32_t BLOCK_K          = 64;
static constexpr uint32_t CLUSTER_SIZE     = 2;
static constexpr uint32_t NUM_STAGES       = 4;
static constexpr uint32_t NUM_EPILOGUE_STAGES = 1;   // no TMEM double buffer

static constexpr uint32_t WAVE_BLOCK_M     = 128;    // LAYOUT_AD_M
static constexpr uint32_t NUM_M_WAVES      = BLOCK_M / WAVE_BLOCK_M;  // 2
static constexpr uint32_t LOAD_N_PER_CTA   = BLOCK_N / CLUSTER_SIZE;  // 128

// UMMA config for cta_group::2
static constexpr uint32_t UMMA_M           = WAVE_BLOCK_M * CLUSTER_SIZE; // 256
static constexpr uint32_t UMMA_N           = BLOCK_N;                     // 256
static constexpr uint32_t UMMA_K           = 16;

// SMEM sizes per stage
static constexpr uint32_t SMEM_A_SIZE      = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);        // 32768
static constexpr uint32_t SMEM_B_SIZE      = LOAD_N_PER_CTA * BLOCK_K * sizeof(__nv_bfloat16); // 16384
static constexpr uint32_t TMA_BYTES        = SMEM_A_SIZE + SMEM_B_SIZE;                         // 49152
static constexpr uint32_t UMMA_SBO         = 8u * BLOCK_K * sizeof(__nv_bfloat16);  // 1024

// TMEM: 1 epilogue stage × 2 M-waves × BLOCK_N = 512 columns
static constexpr uint32_t TMEM_COLS        = NUM_EPILOGUE_STAGES * NUM_M_WAVES * BLOCK_N; // 512

// Thread config: 128 non-epilogue + 128 epilogue
static constexpr uint32_t NUM_THREADS         = 256;
static constexpr uint32_t NUM_EPILOGUE_THREADS = 128;
static constexpr uint32_t SWIZZLE_GROUP_SIZE   = 16;

// Swizzled CD output (SWIZZLE_128B for BF16)
static constexpr uint32_t SWIZZLE_CD_BYTES     = 128;
static constexpr uint32_t STORE_BLOCK_M        = WAVE_BLOCK_M;  // 128
static constexpr uint32_t STORE_BLOCK_N        = SWIZZLE_CD_BYTES / sizeof(__nv_bfloat16); // 64
static constexpr uint32_t NUM_STORES           = BLOCK_N / STORE_BLOCK_N; // 4
static constexpr uint32_t NUM_TMA_STORE_STAGES = 2;
static constexpr uint32_t SMEM_CD_PER_STAGE    = STORE_BLOCK_M * SWIZZLE_CD_BYTES; // 16384
static constexpr uint32_t SMEM_CD_SIZE         = NUM_TMA_STORE_STAGES * SMEM_CD_PER_STAGE;

// Separated SMEM layout: [CD][A0..A3][B0..B3][barriers][tmem_ptr]
static constexpr uint32_t OFF_CD    = 0;
static constexpr uint32_t OFF_A     = SMEM_CD_SIZE;                                  // 32768
static constexpr uint32_t OFF_B     = OFF_A + NUM_STAGES * SMEM_A_SIZE;              // 32768 + 131072 = 163840
static constexpr uint32_t OFF_BAR   = OFF_B + NUM_STAGES * SMEM_B_SIZE;              // 163840 + 65536 = 229376
// Barriers: full[4] + empty[4] + tmem_full[1] + tmem_empty[1] + tc_full[1] = 11 × 8
static constexpr uint32_t NUM_BARS  = NUM_STAGES * 2 + NUM_EPILOGUE_STAGES * 2 + 1;
static constexpr uint32_t OFF_TMEM  = OFF_BAR + NUM_BARS * 8;
static constexpr uint32_t SMEM_SIZE = OFF_TMEM + 4;

static constexpr uint32_t PEER_BIT_MASK     = 0xFEFFFFFF;
static constexpr uint32_t EPILOGUE_BAR_ID   = 1;
static constexpr uint32_t BANK_GROUP_BYTES  = 16;
static constexpr uint32_t ELEMS_PER_BANK_GROUP = BANK_GROUP_BYTES / sizeof(__nv_bfloat16); // 8
static constexpr uint32_t BANK_GROUPS_PER_SWIZZLE = SWIZZLE_CD_BYTES / BANK_GROUP_BYTES;   // 8

// ============================================================
//  PTX helpers
// ============================================================
__device__ __forceinline__ uint32_t get_warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ uint32_t get_lane_id() {
    uint32_t r; asm volatile("mov.u32 %0, %laneid;" : "=r"(r)); return r;
}
__device__ __forceinline__ uint32_t get_cluster_rank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;\n"
                 "barrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xFFFFFFFF;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(pred));
    return pred != 0;
}

// --- Barrier helpers ---
__device__ __forceinline__ void barrier_init(void* bar, uint32_t count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void barrier_arrive_local(void* bar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}
// Arrive at CTA target_cta's copy of bar (cluster addressing)
__device__ __forceinline__ void barrier_arrive_cluster(void* bar, uint32_t target_cta) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_a));
}
__device__ __forceinline__ void barrier_arrive_expect_tx(void* bar, uint32_t tx) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx));
}
__device__ __forceinline__ void barrier_arrive_expect_tx_cluster(
    void* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}
__device__ __forceinline__ void barrier_wait(void* bar, uint32_t phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    int rdy = 0;
    while (!rdy)
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(rdy) : "r"(a), "r"(phase));
}
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void named_barrier_sync(uint32_t num_threads, uint32_t bar_id) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads) : "memory");
}

// --- TMA helpers ---
__device__ __forceinline__ void prefetch_tma(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}
__device__ __forceinline__ void tma_load_2d_cg2(
    void* smem, const void* desc, void* bar, int32_t c0, int32_t c1) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t ba = static_cast<uint32_t>(__cvta_generic_to_shared(bar)) & PEER_BIT_MASK;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"(desc), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
__device__ __forceinline__ void tma_store_2d(
    void* smem, const void* desc, int32_t c0, int32_t c1) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(desc), "r"(c0), "r"(c1), "r"(sa) : "memory");
}
__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

// --- TMEM / tcgen05 helpers ---
__device__ __forceinline__ void tmem_alloc_2sm(void* dst_smem, int ncols) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_2sm(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void tmem_load_8x(
    uint32_t col, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t& r4, uint32_t& r5, uint32_t& r6, uint32_t& r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                   "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}
__device__ __forceinline__ void st_shared_128(
    uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
// tcgen05.commit with arrive::one.  Must be called by exactly 1 thread.
__device__ __forceinline__ void umma_commit_2sm(void* bar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
// UMMA instruction: cta_group::2, kind::f16
__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

// ============================================================
//  Descriptors
// ============================================================
//
// SM100 UMMA SMEM descriptor layout (64-bit, version=1):
//   [0:14)   start_address  = smem_addr >> 4
//   [16:30)  leading_byte_offset >> 4  (0 for K-major)
//   [32:46)  stride_byte_offset >> 4   (SBO)
//   [46:48)  version = 1 (SM100)
//   [49:52)  base_offset = 0
//   [52]     lbo_mode = 0
//   [61:64)  layout_type = 2 (SWIZZLE_128B)
//
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);                  // start_address [0:14)
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;      // stride_byte_offset [32:46)
    d |= (uint64_t)1 << 46;                           // version = 1 [46:48)
    d |= (uint64_t)2 << 61;                           // layout_type = SWIZZLE_128B [61:64)
    return d;
}

// Instruction descriptor for tcgen05.mma.cta_group::2.kind::f16
//   BF16 A (K-major) × BF16 B (K-major) → FP32 accumulator
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = F32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

// ============================================================
//  2D Swizzle Tile Scheduler (per-CTA, group=16)
// ============================================================
struct TileScheduler {
    uint32_t num_m_blocks, num_n_blocks, num_tiles, num_ctas, cta_id;
    int current_iter;
    __device__ TileScheduler(uint32_t M, uint32_t N, uint32_t nc)
        : num_m_blocks((M + BLOCK_M - 1) / BLOCK_M),
          num_n_blocks((N + BLOCK_N - 1) / BLOCK_N),
          num_tiles(num_m_blocks * num_n_blocks),
          num_ctas(nc), cta_id(blockIdx.x), current_iter(-1) {}
    __device__ bool get_next_block(uint32_t& m_block, uint32_t& n_block) {
        uint32_t tile_idx = static_cast<uint32_t>(++current_iter) * num_ctas + cta_id;
        if (tile_idx >= num_tiles) return false;
        // Group M-blocks in chunks of SWIZZLE_GROUP_SIZE, sweep N within each group
        uint32_t tpg = num_n_blocks * SWIZZLE_GROUP_SIZE;
        uint32_t gi  = tile_idx / tpg;
        uint32_t fm  = gi * SWIZZLE_GROUP_SIZE;
        uint32_t ig  = tile_idx % tpg;
        uint32_t mg  = min(SWIZZLE_GROUP_SIZE, num_m_blocks - fm);
        m_block = fm + ig % mg;
        n_block = ig / mg;
        return true;
    }
};

// ============================================================
//  Kernel
// ============================================================
__global__ void
__cluster_dims__(2, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
bf16_gemm_2sm_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_d,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t num_ctas)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    const uint32_t warp_idx  = get_warp_id();
    const uint32_t lane_idx  = get_lane_id();
    const uint32_t cta_rank  = get_cluster_rank();
    const bool     is_leader = (cta_rank == 0);

    extern __shared__ __align__(1024) uint8_t smem_buf[];

    // --- Barrier setup ---
    uint64_t* full_bar[NUM_STAGES];
    uint64_t* empty_bar[NUM_STAGES];
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        full_bar[s]  = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + s * 8);
        empty_bar[s] = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 8 + s * 8);
    }
    uint64_t* tmem_full_bar[NUM_EPILOGUE_STAGES];
    uint64_t* tmem_empty_bar[NUM_EPILOGUE_STAGES];
    for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
        tmem_full_bar[e]  = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 16 + e * 8);
        tmem_empty_bar[e] = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 16 + NUM_EPILOGUE_STAGES * 8 + e * 8);
    }
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(smem_buf + OFF_TMEM);

    // SMEM pointers (separated layout)
    uint8_t* smem_cd_base = smem_buf + OFF_CD;
    auto get_smem_a = [&](uint32_t s) -> void* { return smem_buf + OFF_A + s * SMEM_A_SIZE; };
    auto get_smem_b = [&](uint32_t s) -> void* { return smem_buf + OFF_B + s * SMEM_B_SIZE; };

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one()) {
        prefetch_tma(&tma_a);
        prefetch_tma(&tma_b);
        prefetch_tma(&tma_d);
    }

    // Initialize barriers (warp 1)
    if (warp_idx == 1 && elect_one()) {
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            barrier_init(full_bar[s],  CLUSTER_SIZE);  // Both CTAs' TMAs signal
            barrier_init(empty_bar[s], 1);             // 1 commit per k-block
        }
        for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
            barrier_init(tmem_full_bar[e], 1);  // 1 commit at end of K
            // All 128 epilogue threads × 2 CTAs = 256 arrivals
            barrier_init(tmem_empty_bar[e], CLUSTER_SIZE * NUM_EPILOGUE_THREADS);
        }
        fence_barrier_init();
    }

    // Allocate TMEM (warp 2)
    if (warp_idx == 2) tmem_alloc_2sm(tmem_addr, TMEM_COLS);
    cluster_sync();

    const uint32_t num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);

    // ================================================================
    //  Three-role warp dispatch
    // ================================================================

    if (warp_idx == 0 && elect_one()) {
        // ======================== TMA WARP ========================
        // 1 elected thread per CTA. Loads A (full) and B (half per CTA).
        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;
        uint32_t stage = 0, phase = 0;
        while (scheduler.get_next_block(m_block, n_block)) {
            const int32_t m_coord = m_block * BLOCK_M;
            const int32_t n_coord = n_block * BLOCK_N + cta_rank * LOAD_N_PER_CTA;
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                // Wait for consumer (MMA) to release this stage
                barrier_wait(empty_bar[stage], phase ^ 1);

                // Signal expected TX bytes on leader's full barrier
                if (is_leader)
                    barrier_arrive_expect_tx(full_bar[stage], TMA_BYTES * CLUSTER_SIZE);
                else
                    barrier_arrive_cluster(full_bar[stage], 0);  // arrive(0u) on leader

                int32_t kc = kb * BLOCK_K;
                tma_load_2d_cg2(get_smem_a(stage), &tma_a, full_bar[stage], kc, m_coord);
                tma_load_2d_cg2(get_smem_b(stage), &tma_b, full_bar[stage], kc, n_coord);

                stage = (stage + 1) % NUM_STAGES;
                phase ^= (stage == 0);
            }
        }

    } else if (warp_idx == 1 && is_leader) {
        // ======================== MMA WARP ========================
        // All 32 threads of warp 1 on leader CTA participate.
        // elect_one() used only for UMMA + commit instructions.

        // Build base descriptors from stage 0
        uint64_t desc0_a = make_smem_desc(get_smem_a(0), UMMA_SBO);
        uint64_t desc0_b = make_smem_desc(get_smem_b(0), UMMA_SBO);
        uint32_t ahi = static_cast<uint32_t>(desc0_a >> 32);
        uint32_t bhi = static_cast<uint32_t>(desc0_b >> 32);

        // __shfl_sync descriptor caching: lane i stores stage i's descriptor lo
        // Separated SMEM means constant stride between stages
        uint32_t a_lo_base = static_cast<uint32_t>(desc0_a);
        uint32_t b_lo_base = static_cast<uint32_t>(desc0_b);
        uint32_t my_a_lo = a_lo_base + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_A_SIZE / 16) : 0u);
        uint32_t my_b_lo = b_lo_base + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_B_SIZE / 16) : 0u);

        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;
        uint32_t stage = 0, phase = 0;

        while (scheduler.get_next_block(m_block, n_block)) {
            uint32_t accum_idx   = scheduler.current_iter % NUM_EPILOGUE_STAGES; // always 0
            uint32_t accum_phase = (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1;

            // Wait for epilogue to release TMEM
            barrier_wait(tmem_empty_bar[accum_idx], accum_phase ^ 1);
            tcgen05_fence_after();

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                // All 32 threads wait for TMA data
                barrier_wait(full_bar[stage], phase);
                tcgen05_fence_after();

                // Shuffle to get current stage's descriptor lo
                uint32_t cur_a_lo = __shfl_sync(0xFFFFFFFF, my_a_lo, stage);
                uint32_t cur_b_lo = __shfl_sync(0xFFFFFFFF, my_b_lo, stage);

                // Only elected thread issues UMMA
                if (elect_one()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                        // B descriptor: only varies with k-step
                        uint32_t b_lo = cur_b_lo + k * 2;
                        uint64_t bd = ((uint64_t)bhi << 32) | b_lo;

                        #pragma unroll
                        for (uint32_t w = 0; w < NUM_M_WAVES; ++w) {
                            // A descriptor: varies with k-step and M-wave
                            // M-wave offset = w * WAVE_BLOCK_M * BLOCK_K * sizeof(bf16) / 16
                            //               = w * 128 * 64 * 2 / 16 = w * 1024
                            uint32_t a_lo = cur_a_lo + w * (WAVE_BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16) / 16) + k * 2;
                            uint64_t ad = ((uint64_t)ahi << 32) | a_lo;

                            uint32_t tmem_offset = accum_idx * NUM_M_WAVES * BLOCK_N + w * BLOCK_N;
                            // Each M-wave writes to separate TMEM columns, so
                            // ALL waves clear on first k-step (not just wave 0)
                            uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;
                            umma_f16_cg2(tmem_offset, ad, bd, idesc, accum);
                        }
                    }
                }

                // Commit: elect_one() for arrive::one commit
                if (elect_one()) {
                    umma_commit_2sm(empty_bar[stage]);
                    if (kb == num_k_blocks - 1)
                        umma_commit_2sm(tmem_full_bar[accum_idx]);
                }

                stage = (stage + 1) % NUM_STAGES;
                phase ^= (stage == 0);
            }
        }

        // Extra wait for safe barrier destruction
        int last_iter = scheduler.current_iter - 1;
        if (last_iter >= 0) {
            uint32_t li = last_iter % NUM_EPILOGUE_STAGES;
            uint32_t lp = (last_iter / NUM_EPILOGUE_STAGES) & 1;
            barrier_wait(tmem_empty_bar[li], lp);
        }

    } else if (warp_idx >= 4) {
        // ======================== EPILOGUE WARPS ========================
        // 128 threads (warps 4-7) on BOTH CTAs.
        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;
        const uint32_t local_tid = threadIdx.x - 128;
        const uint32_t epi_warp  = local_tid / 32;
        uint32_t tma_store_stage = 0;

        while (scheduler.get_next_block(m_block, n_block)) {
            uint32_t accum_idx   = scheduler.current_iter % NUM_EPILOGUE_STAGES;
            uint32_t accum_phase = (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1;

            // Wait for UMMA to finish writing TMEM
            barrier_wait(tmem_full_bar[accum_idx], accum_phase);
            tcgen05_fence_after();

            // Iterate over M-waves and stores
            #pragma unroll
            for (uint32_t w = 0; w < NUM_M_WAVES; ++w) {
                #pragma unroll
                for (uint32_t s = 0; s < NUM_STORES; ++s) {
                    // Wait for TMA store stage to be free
                    if (epi_warp == 0 && elect_one())
                        tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                    named_barrier_sync(NUM_EPILOGUE_THREADS, EPILOGUE_BAR_ID);

                    // TMEM → SMEM with 128B swizzle
                    uint32_t smem_stage_base = static_cast<uint32_t>(
                        __cvta_generic_to_shared(smem_cd_base + tma_store_stage * SMEM_CD_PER_STAGE));

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / ELEMS_PER_BANK_GROUP; ++i) {
                        uint32_t tmem_col = accum_idx * NUM_M_WAVES * BLOCK_N
                                          + w * BLOCK_N
                                          + s * STORE_BLOCK_N
                                          + i * ELEMS_PER_BANK_GROUP;
                        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                        tmem_load_8x(tmem_col, r0, r1, r2, r3, r4, r5, r6, r7);
                        tmem_load_fence();

                        uint32_t p0 = pack_bf16(r0, r1);
                        uint32_t p1 = pack_bf16(r2, r3);
                        uint32_t p2 = pack_bf16(r4, r5);
                        uint32_t p3 = pack_bf16(r6, r7);

                        uint32_t swizzled_col = i ^ (local_tid % BANK_GROUPS_PER_SWIZZLE);
                        uint32_t smem_addr = smem_stage_base
                                           + local_tid * SWIZZLE_CD_BYTES
                                           + swizzled_col * BANK_GROUP_BYTES;
                        st_shared_128(smem_addr, p0, p1, p2, p3);
                    }

                    // Signal tmem_empty ASAP on the LAST M-wave, LAST store
                    // All 128 threads arrive at leader's (CTA 0) barrier via cluster addr
                    if (w == NUM_M_WAVES - 1 && s == NUM_STORES - 1) {
                        tcgen05_fence_before();
                        barrier_arrive_cluster(tmem_empty_bar[accum_idx], 0);
                    }
                    __syncwarp();

                    // Make SMEM writes visible and issue TMA store
                    tma_store_fence();
                    named_barrier_sync(NUM_EPILOGUE_THREADS, EPILOGUE_BAR_ID);
                    if (epi_warp == 0 && elect_one()) {
                        int32_t n_idx = n_block * BLOCK_N + s * STORE_BLOCK_N;
                        int32_t m_idx = m_block * BLOCK_M + w * WAVE_BLOCK_M;
                        tma_store_2d(smem_cd_base + tma_store_stage * SMEM_CD_PER_STAGE,
                                     &tma_d, n_idx, m_idx);
                        tma_store_commit();
                    }

                    tma_store_stage = (tma_store_stage + 1) % NUM_TMA_STORE_STAGES;
                }
            }
        }
        if (epi_warp == 0 && elect_one()) tma_store_wait<0>();
    }

    __syncthreads();
    cluster_sync();
    if (warp_idx == 2) tmem_dealloc_2sm(0, TMEM_COLS);
#endif
}

// ============================================================
//  Host
// ============================================================
static void create_tma_desc(
    CUtensorMap* map, const void* ptr,
    uint64_t dim0, uint64_t dim1,
    uint32_t box0, uint32_t box1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {
    uint64_t dims[2]    = {dim0, dim1};
    uint64_t strides[1] = {dim0 * sizeof(__nv_bfloat16)};
    uint32_t box[2]     = {box0, box1};
    uint32_t estrides[2]= {1, 1};
    cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(ptr), dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // Match DeepGEMM
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

static void launch_kernel(
    __nv_bfloat16* D, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K) {
    CUtensorMap tma_a, tma_b, tma_d;
    // A: (K, M) box = (BLOCK_K=64, BLOCK_M=256)
    create_tma_desc(&tma_a, A, (uint64_t)K, (uint64_t)M, BLOCK_K, BLOCK_M);
    // B: (K, N) box = (BLOCK_K=64, LOAD_N_PER_CTA=128), each CTA loads half
    create_tma_desc(&tma_b, B, (uint64_t)K, (uint64_t)N, BLOCK_K, LOAD_N_PER_CTA);
    // D: (N, M) box = (STORE_BLOCK_N=64, STORE_BLOCK_M=128), swizzle 128B
    create_tma_desc(&tma_d, D, (uint64_t)N, (uint64_t)M,
                    STORE_BLOCK_N, STORE_BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);

    uint32_t num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t num_tiles    = num_m_blocks * num_n_blocks;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_ctas = min((uint32_t)num_sms, num_tiles);
    num_ctas = (num_ctas / CLUSTER_SIZE) * CLUSTER_SIZE;

    cudaFuncSetAttribute(
        (const void*)bf16_gemm_2sm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3(num_ctas, 1, 1);
    config.blockDim = dim3(NUM_THREADS, 1, 1);
    config.dynamicSmemBytes = SMEM_SIZE;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
    config.attrs    = attrs;
    config.numAttrs = 1;

    uint32_t Mu = M, Nu = N, Ku = K;
    cudaLaunchKernelEx(&config, bf16_gemm_2sm_kernel,
                       tma_a, tma_b, tma_d, Mu, Nu, Ku, num_ctas);
}

extern "C" void gemm_bf16_launch(
    const void* A, const void* B, void* D, int M, int N, int K) {
    launch_kernel(
        reinterpret_cast<__nv_bfloat16*>(D),
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B), M, N, K);
}

extern "C" float benchmark_kernel(
    const void* A, const void* B, void* D,
    int M, int N, int K, int warmup, int iters) {
    auto* Dp = reinterpret_cast<__nv_bfloat16*>(D);
    auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A);
    auto* Bp = reinterpret_cast<const __nv_bfloat16*>(B);
    for (int i = 0; i < warmup; i++) launch_kernel(Dp, Ap, Bp, M, N, K);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) launch_kernel(Dp, Ap, Bp, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}
