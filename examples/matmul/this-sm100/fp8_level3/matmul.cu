// =============================================================================
// FP8 GEMM Level 3 — SM100 (GB200) — Large Tile (BLOCK_N=224)
//
// D (BF16, M×N) = A (FP8 E4M3, M×K) × B^T (FP8 E4M3, N×K)
//
// Level 3 improvements over Level 2:
//   1. BLOCK_N=224 (vs 128): higher compute-to-memory ratio, matches DeepGEMM
//   2. Fixed epilogue swizzle for SWIZZLE_CD != 128B (the bug that hung Level 2
//      when BLOCK_N > 128: lt%8 must be lt%(SWIZZLE_CD/16))
//   3. Generalized CD swizzle logic for any BLOCK_N
//
// Architecture: cta_group::1 (single SM, no cluster)
// Warp assignment (256 threads = 8 warps):
//   Warp 0:   TMA Load (A, B, SFA, SFB)
//   Warp 1:   MMA Issue (UTCCP SF copy + block-scaled UMMA)
//   Warp 2:   UTCCP Transposer (SF SMEM transpose)
//   Warp 3:   TMEM Allocator (then idle)
//   Warp 4-7: Epilogue (TMEM → SMEM → TMA Store)
// =============================================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// Configuration
// =============================================================================

#ifndef TILE_N
#define TILE_N 224
#endif

static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = TILE_N;
static constexpr uint32_t BLOCK_K = 128;
static constexpr uint32_t UMMA_K  = 32;

// Scale factor config
static constexpr uint32_t GRAN_K = 128;
static constexpr uint32_t NUM_SF_PER_LOAD = (GRAN_K == 32) ? 1 : 4;

// SF block sizes (aligned to 128 for UTCCP)
static constexpr uint32_t SF_BLOCK_M = ((BLOCK_M + 127) / 128) * 128;
static constexpr uint32_t SF_BLOCK_N = ((BLOCK_N + 127) / 128) * 128;

// SMEM sizes per pipeline stage
static constexpr uint32_t SMEM_A   = BLOCK_M * BLOCK_K * 1;
static constexpr uint32_t SMEM_B   = BLOCK_N * BLOCK_K * 1;
static constexpr uint32_t SMEM_SFA = SF_BLOCK_M * sizeof(uint32_t);
static constexpr uint32_t SMEM_SFB = SF_BLOCK_N * sizeof(uint32_t);
static constexpr uint32_t SMEM_STAGE = SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB;

// TMA transfer sizes (may differ from SMEM alloc when SF_BLOCK_N > BLOCK_N)
static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(uint32_t);
static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(uint32_t);

// Swizzle mode for CD: pick the largest mode where (BLOCK_N * sizeof(BF16)) % mode == 0
// For BF16 (2B): BLOCK_N * 2 must be divisible by the swizzle mode
static constexpr uint32_t SWIZZLE_CD =
    ((BLOCK_N * 2) % 128 == 0) ? 128 :
    ((BLOCK_N * 2) % 64  == 0) ? 64  :
    ((BLOCK_N * 2) % 32  == 0) ? 32  : 16;

// CD staging (BF16 output)
static constexpr uint32_t STORE_M       = 128;
static constexpr uint32_t STORE_N       = SWIZZLE_CD / 2;   // BF16=2B
static constexpr uint32_t CD_STORES     = BLOCK_N / STORE_N;
static constexpr uint32_t TMA_ST_STAGES = 2;
static constexpr uint32_t SMEM_CD_STAGE = STORE_M * SWIZZLE_CD;
static constexpr uint32_t SMEM_CD       = SMEM_CD_STAGE * TMA_ST_STAGES;

// Number of bank groups per swizzle atom (each bank group = 16 bytes)
static constexpr uint32_t BANK_GROUPS = SWIZZLE_CD / 16;

// Number of output rows per 128-byte SMEM atom (bank line).
// TMA swizzle always operates on 128-byte atoms. For SWIZZLE_128B, each atom
// holds 1 row (128 bytes). For SWIZZLE_64B, each atom holds 2 rows (64 bytes
// each). The XOR pattern uses the atom row index, NOT the physical row index.
static constexpr uint32_t ROWS_PER_ATOM = 128 / SWIZZLE_CD;

// TMEM layout
static constexpr uint32_t NUM_M_WAVES    = BLOCK_M / 128;
static constexpr uint32_t NUM_EPI_STAGES = 2;
static constexpr uint32_t ACCUM_COLS     = NUM_EPI_STAGES * NUM_M_WAVES * BLOCK_N;
static constexpr uint32_t SFA_TMEM_COLS  = SF_BLOCK_M / 32;
static constexpr uint32_t SFB_TMEM_COLS  = SF_BLOCK_N / 32;
static constexpr uint32_t SFA_TMEM_OFF   = ACCUM_COLS;
static constexpr uint32_t SFB_TMEM_OFF   = SFA_TMEM_OFF + SFA_TMEM_COLS;

// Align TMEM columns to power-of-2
static constexpr uint32_t TMEM_COLS_RAW  = ACCUM_COLS + SFA_TMEM_COLS + SFB_TMEM_COLS;
static constexpr uint32_t TMEM_COLS =
    TMEM_COLS_RAW <=  32 ?  32 :
    TMEM_COLS_RAW <=  64 ?  64 :
    TMEM_COLS_RAW <= 128 ? 128 :
    TMEM_COLS_RAW <= 256 ? 256 : 512;

static_assert(TMEM_COLS <= 512, "TMEM columns exceed 512, reduce BLOCK_N");

// Maximize pipeline stages
static constexpr uint32_t SMEM_CAPACITY = 232448;
static constexpr uint32_t SMEM_FIXED = SMEM_CD + 4;
static constexpr uint32_t SMEM_EPI_BAR = NUM_EPI_STAGES * 2 * 8;
static constexpr uint32_t SMEM_PER_STAGE = SMEM_STAGE + 3 * 8;

static constexpr uint32_t NUM_STAGES =
    (SMEM_CAPACITY - SMEM_FIXED - SMEM_EPI_BAR) / SMEM_PER_STAGE;
static constexpr uint32_t CAPPED_STAGES = NUM_STAGES > 32 ? 32 : NUM_STAGES;

// Thread config
static constexpr uint32_t N_MMA_THREADS = 128;
static constexpr uint32_t N_EPI_THREADS = 128;
static constexpr uint32_t N_THREADS     = 256;

// Scheduler
static constexpr uint32_t SWIZZLE_GROUP = 16;

// =============================================================================
// PTX Helpers
// =============================================================================

__device__ __forceinline__ void bar_init(void* bar, uint32_t cnt) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt));
}

__device__ __forceinline__ void bar_wait(void* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done;
    do {
        asm volatile(
            "{\n\t.reg .pred P;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, P;\n\t}\n"
            : "=r"(done) : "r"(a), "r"(phase));
    } while (!done);
}

__device__ __forceinline__ void bar_arrive(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}

__device__ __forceinline__ void bar_expect_tx(void* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(bytes));
}

__device__ __forceinline__ void bar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "elect.sync _|P, 0xFFFFFFFF;\n\t"
                 "selp.b32 %0, 1, 0, P;\n\t}\n"
                 : "=r"(p));
    return p != 0;
}

// ----- TMA -----

__device__ __forceinline__ void tma_load(void* dst, const void* desc,
                                          void* bar, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"(desc), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store(void* src, const void* desc,
                                           int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(desc), "r"(c0), "r"(c1), "r"(sa) : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}
template <int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__ void prefetch_tma(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}

// ----- TMEM -----

__device__ __forceinline__ void tmem_alloc(uint32_t* smem, uint32_t ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tmem_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tmem_ld_8x(uint32_t col,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t& r4, uint32_t& r5, uint32_t& r6, uint32_t& r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                 " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
                   "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// ----- UMMA -----

__device__ __forceinline__ void umma_commit(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void umma_fp8(
    uint32_t tmem_c, uint64_t da, uint64_t db,
    uint32_t idesc, uint32_t accum,
    uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
        :: "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(accum),
           "r"(tmem_sfa), "r"(tmem_sfb));
}

// ----- UTCCP -----

__device__ __forceinline__ void utccp_copy(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(smem_desc));
}

__device__ __forceinline__ uint32_t ld_s32(const uint32_t* p) {
    uint32_t v, a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
    return v;
}

__device__ __forceinline__ void st_s32(uint32_t* p, uint32_t v) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(v));
}

__device__ __forceinline__ void utccp_transpose(uint32_t* smem, uint32_t lane) {
    uint32_t vals[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++)
        vals[i] = ld_s32(smem + (i ^ (lane >> 3)) * 32 + lane);
    __syncwarp();
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++)
        st_s32(smem + lane * 4 + (i ^ (lane >> 3)), vals[i]);
}

__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ----- Descriptors -----

__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo, uint32_t layout = 2) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_sf_desc(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((128u >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

__device__ __forceinline__ void replace_desc_addr(uint64_t& desc, void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    desc = (desc & ~(uint64_t)0x3FFF) | (uint64_t)(addr & 0x3FFF);
}

// ----- FP8 Instruction Descriptor -----

__device__ __forceinline__ uint32_t make_fp8_idesc() {
    uint32_t d = 0;
    d |= ((BLOCK_N / 8) << 17);   // n_dim
    d |= (1u << 23);              // scale_format = E8M0
    d |= ((BLOCK_M / 16) << 24);  // m_dim
    return d;
}

__device__ __forceinline__ uint32_t set_sf_ids(uint32_t d, uint32_t a_id, uint32_t b_id) {
    d &= ~(0x3u << 29);
    d &= ~(0x3u << 4);
    d |= ((a_id & 0x3) << 29);
    d |= ((b_id & 0x3) << 4);
    return d;
}

// ----- BF16 Pack -----

__device__ __forceinline__ uint32_t pack_bf16(uint32_t a, uint32_t b) {
    __nv_bfloat16 ha = __float2bfloat16(__uint_as_float(a));
    __nv_bfloat16 hb = __float2bfloat16(__uint_as_float(b));
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r)
        : "h"(*(uint16_t*)&ha), "h"(*(uint16_t*)&hb));
    return r;
}

__device__ __forceinline__ void st_s128(uint32_t addr,
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

// ----- Tile Scheduler -----

struct Scheduler {
    uint32_t mb, nb, ntiles, grid, cta;
    int32_t  iter;

    __device__ Scheduler(uint32_t M, uint32_t N, uint32_t g, uint32_t bid)
        : mb((M + BLOCK_M - 1) / BLOCK_M), nb((N + BLOCK_N - 1) / BLOCK_N),
          grid(g), cta(bid), iter(-1) { ntiles = mb * nb; }

    __device__ bool next(uint32_t& m, uint32_t& n) {
        uint32_t t = (uint32_t)(++iter) * grid + cta;
        if (t >= ntiles) return false;
        uint32_t grp_sz = nb * SWIZZLE_GROUP;
        uint32_t grp = t / grp_sz, rem = t % grp_sz;
        uint32_t mg = min(SWIZZLE_GROUP, mb - grp * SWIZZLE_GROUP);
        m = grp * SWIZZLE_GROUP + rem % mg;
        n = rem / mg;
        return true;
    }
};

// =============================================================================
// SMEM Layout
// =============================================================================

static constexpr uint32_t O_A   = SMEM_CD;
static constexpr uint32_t O_B   = O_A + CAPPED_STAGES * SMEM_A;
static constexpr uint32_t O_SFA = O_B + CAPPED_STAGES * SMEM_B;
static constexpr uint32_t O_SFB = O_SFA + CAPPED_STAGES * SMEM_SFA;
static constexpr uint32_t O_BAR = O_SFB + CAPPED_STAGES * SMEM_SFB;

static constexpr uint32_t N_BARS =
    CAPPED_STAGES * 3 + NUM_EPI_STAGES * 2;

static constexpr uint32_t TOTAL_SMEM = O_BAR + N_BARS * 8 + 4;

static_assert(TOTAL_SMEM <= SMEM_CAPACITY, "Exceeds shared memory capacity");

// =============================================================================
// Kernel
// =============================================================================

__global__ void __cluster_dims__(1, 1, 1)
__launch_bounds__(N_THREADS, 1)
fp8_gemm_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_sfa,
    const __grid_constant__ CUtensorMap tma_sfb,
    const __grid_constant__ CUtensorMap tma_d,
    uint32_t M, uint32_t N, uint32_t K)
{
    extern __shared__ __align__(1024) uint8_t smem[];

    // Barrier pointers
    uint64_t* full_bar    = (uint64_t*)(smem + O_BAR);
    uint64_t* empty_bar   = full_bar + CAPPED_STAGES;
    uint64_t* sf_full_bar = empty_bar + CAPPED_STAGES;
    uint64_t* tf_bar      = sf_full_bar + CAPPED_STAGES;
    uint64_t* te_bar      = tf_bar + NUM_EPI_STAGES;
    uint32_t* tmem_smem   = (uint32_t*)(te_bar + NUM_EPI_STAGES);

    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    // Init barriers + TMEM alloc
    if (wid == 1 && elect_one()) {
        for (uint32_t s = 0; s < CAPPED_STAGES; s++) {
            bar_init(&full_bar[s], 1);
            bar_init(&empty_bar[s], 1);
            bar_init(&sf_full_bar[s], 32);
        }
        for (uint32_t e = 0; e < NUM_EPI_STAGES; e++) {
            bar_init(&tf_bar[e], 1);
            bar_init(&te_bar[e], N_EPI_THREADS);
        }
        bar_fence_init();
    }
    if (wid == 3)
        tmem_alloc(tmem_smem, TMEM_COLS);
    __syncthreads();

    uint32_t tmem_base = *tmem_smem;

    if (wid == 0 && elect_one()) {
        prefetch_tma(&tma_a);
        prefetch_tma(&tma_b);
        prefetch_tma(&tma_sfa);
        prefetch_tma(&tma_sfb);
        prefetch_tma(&tma_d);
    }

    Scheduler sched(M, N, gridDim.x, blockIdx.x);
    uint32_t mb, nb;
    uint32_t num_kb = (K + BLOCK_K - 1) / BLOCK_K;

    uint32_t stg = 0, ph = 0;
    auto adv = [&]() { stg = (stg + 1) % CAPPED_STAGES; ph ^= (stg == 0); };

    // =====================================================================
    // Warp 0: TMA Load
    // =====================================================================
    if (wid == 0 && elect_one()) {
        while (sched.next(mb, nb)) {
            for (uint32_t kb = 0; kb < num_kb; kb++) {
                bar_wait(&empty_bar[stg], ph ^ 1);

                uint32_t k = kb * BLOCK_K;
                tma_load(smem + O_A + stg * SMEM_A, &tma_a, &full_bar[stg],
                         (int32_t)k, (int32_t)(mb * BLOCK_M));
                tma_load(smem + O_B + stg * SMEM_B, &tma_b, &full_bar[stg],
                         (int32_t)k, (int32_t)(nb * BLOCK_N));

                // Use actual TMA transfer sizes, NOT SMEM allocation sizes
                // (SF_BLOCK_N may be > BLOCK_N due to 128-alignment for UTCCP)
                uint32_t tx = SMEM_A + SMEM_B;

                if (kb % NUM_SF_PER_LOAD == 0) {
                    uint32_t sfk = kb / NUM_SF_PER_LOAD;
                    tma_load(smem + O_SFA + stg * SMEM_SFA, &tma_sfa,
                             &full_bar[stg],
                             (int32_t)(mb * BLOCK_M), (int32_t)sfk);
                    tma_load(smem + O_SFB + stg * SMEM_SFB, &tma_sfb,
                             &full_bar[stg],
                             (int32_t)(nb * BLOCK_N), (int32_t)sfk);
                    tx += TMA_SFA_BYTES + TMA_SFB_BYTES;
                }
                bar_expect_tx(&full_bar[stg], tx);
                adv();
            }
        }
    }

    // =====================================================================
    // Warp 2: UTCCP Transposer
    // =====================================================================
    else if (wid == 2) {
        while (sched.next(mb, nb)) {
            for (uint32_t kb = 0; kb < num_kb; kb++) {
                bar_wait(&full_bar[stg], ph);

                if (kb % NUM_SF_PER_LOAD == 0) {
                    for (uint32_t i = 0; i < SF_BLOCK_M / 128; i++)
                        utccp_transpose(
                            (uint32_t*)(smem + O_SFA + stg * SMEM_SFA) + i*128,
                            lane);
                    fence_async_smem();

                    for (uint32_t i = 0; i < SF_BLOCK_N / 128; i++)
                        utccp_transpose(
                            (uint32_t*)(smem + O_SFB + stg * SMEM_SFB) + i*128,
                            lane);
                    fence_async_smem();
                }

                bar_arrive(&sf_full_bar[stg]);
                adv();
            }
        }
    }

    // =====================================================================
    // Warp 1: MMA Issue
    // =====================================================================
    else if (wid == 1) {
        static constexpr uint32_t SBO = 8 * BLOCK_K * 1;

        uint32_t base_idesc = make_fp8_idesc();

        uint64_t a_base_desc = make_smem_desc(smem + O_A, SBO);
        uint64_t b_base_desc = make_smem_desc(smem + O_B, SBO);
        uint32_t a_hi = (uint32_t)(a_base_desc >> 32);
        uint32_t b_hi = (uint32_t)(b_base_desc >> 32);

        uint32_t my_a_lo = 0, my_b_lo = 0;
        if (lane < CAPPED_STAGES) {
            my_a_lo = (uint32_t)make_smem_desc(smem + O_A + lane * SMEM_A, SBO);
            my_b_lo = (uint32_t)make_smem_desc(smem + O_B + lane * SMEM_B, SBO);
        }

        while (sched.next(mb, nb)) {
            uint32_t ai = sched.iter % NUM_EPI_STAGES;
            uint32_t ap = (sched.iter / NUM_EPI_STAGES) & 1;

            bar_wait(&te_bar[ai], ap ^ 1);
            tmem_fence_after();

            for (uint32_t kb = 0; kb < num_kb; kb++) {
                bar_wait(&sf_full_bar[stg], ph);
                tmem_fence_after();

                uint32_t sf_idx = kb % NUM_SF_PER_LOAD;
                if (sf_idx == 0 && elect_one()) {
                    for (uint32_t i = 0; i < SF_BLOCK_M / 128; i++) {
                        uint64_t d = make_sf_desc(
                            smem + O_SFA + stg * SMEM_SFA + i * 128 * 4);
                        utccp_copy(d, tmem_base + SFA_TMEM_OFF + i * 4);
                    }
                    for (uint32_t i = 0; i < SF_BLOCK_N / 128; i++) {
                        uint64_t d = make_sf_desc(
                            smem + O_SFB + stg * SMEM_SFB + i * 128 * 4);
                        utccp_copy(d, tmem_base + SFB_TMEM_OFF + i * 4);
                    }
                }
                __syncwarp();

                uint32_t a_lo = __shfl_sync(0xFFFFFFFF, my_a_lo, stg);
                uint32_t b_lo = __shfl_sync(0xFFFFFFFF, my_b_lo, stg);

                uint32_t idesc = set_sf_ids(base_idesc, sf_idx, sf_idx);
                uint32_t accum = (kb > 0) ? 1u : 0u;

                if (elect_one()) {
                    #pragma unroll
                    for (uint32_t kk = 0; kk < BLOCK_K / UMMA_K; kk++) {
                        uint64_t da = ((uint64_t)a_hi << 32) | (uint64_t)(a_lo + kk * 2);
                        uint64_t db = ((uint64_t)b_hi << 32) | (uint64_t)(b_lo + kk * 2);

                        umma_fp8(
                            tmem_base + ai * NUM_M_WAVES * BLOCK_N,
                            da, db, idesc, accum,
                            tmem_base + SFA_TMEM_OFF,
                            tmem_base + SFB_TMEM_OFF);
                        accum = 1u;
                    }
                }

                if (elect_one()) {
                    if (kb == num_kb - 1)
                        umma_commit(&tf_bar[ai]);
                    umma_commit(&empty_bar[stg]);
                }

                adv();
            }
        }
    }

    // =====================================================================
    // Warp 4-7: Epilogue
    //
    // TMA swizzle always uses 128-byte atoms (bank lines). The XOR pattern
    // is based on the ATOM row index, not the physical row index:
    //   atom_row = lt / ROWS_PER_ATOM
    //   swizzled_bg = i ^ (atom_row % BANK_GROUPS)
    //
    // For SWIZZLE_128B: ROWS_PER_ATOM=1, atom_row=lt, bg = i ^ (lt % 8)
    // For SWIZZLE_64B:  ROWS_PER_ATOM=2, atom_row=lt/2, bg = i ^ ((lt/2) % 4)
    //
    // Two adjacent rows share one 128-byte atom for SWIZZLE_64B:
    //   Even rows use bank groups 0-3 (first 64 bytes of the atom)
    //   Odd rows use bank groups 4-7 (second 64 bytes of the atom)
    // =====================================================================
    else if (wid >= 4) {
        uint32_t ew = wid - 4;
        uint32_t lt = threadIdx.x - N_MMA_THREADS;
        uint32_t ts = 0;

        while (sched.next(mb, nb)) {
            uint32_t ai = sched.iter % NUM_EPI_STAGES;
            uint32_t ap = (sched.iter / NUM_EPI_STAGES) & 1;

            bar_wait(&tf_bar[ai], ap);
            tmem_fence_after();

            for (uint32_t s = 0; s < CD_STORES; s++) {
                if (ew == 0) tma_store_wait<TMA_ST_STAGES - 1>();
                asm volatile("bar.sync 0, %0;" :: "r"(N_EPI_THREADS));

                uint32_t row_base = ts * SMEM_CD_STAGE + lt * SWIZZLE_CD;
                uint32_t smem_base_addr = (uint32_t)__cvta_generic_to_shared(smem);

                #pragma unroll
                for (uint32_t i = 0; i < STORE_N / 8; i++) {
                    uint32_t col = tmem_base + ai * NUM_M_WAVES * BLOCK_N
                                 + s * STORE_N + i * 8;

                    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                    tmem_ld_8x(col, r0, r1, r2, r3, r4, r5, r6, r7);
                    tmem_wait_ld();

                    uint32_t p0 = pack_bf16(r0, r1);
                    uint32_t p1 = pack_bf16(r2, r3);
                    uint32_t p2 = pack_bf16(r4, r5);
                    uint32_t p3 = pack_bf16(r6, r7);

                    // TMA swizzle XOR uses the ATOM row, not the physical row.
                    // ROWS_PER_ATOM = 128 / SWIZZLE_CD (1 for 128B, 2 for 64B).
                    // For SWIZZLE_128B: atom_row = lt/1 = lt, bg = i ^ (lt % 8)
                    // For SWIZZLE_64B:  atom_row = lt/2,     bg = i ^ ((lt/2) % 4)
                    uint32_t atom_row = lt / ROWS_PER_ATOM;
                    uint32_t bg = i ^ (atom_row % BANK_GROUPS);
                    uint32_t addr = smem_base_addr + row_base + bg * 16;
                    st_s128(addr, p0, p1, p2, p3);
                }

                if (s == CD_STORES - 1) {
                    tmem_fence_before();
                    bar_arrive(&te_bar[ai]);
                }

                tma_store_fence();
                asm volatile("bar.sync 0, %0;" :: "r"(N_EPI_THREADS));

                if (ew == 0 && elect_one()) {
                    tma_store(smem + ts * SMEM_CD_STAGE, &tma_d,
                              (int32_t)(nb * BLOCK_N + s * STORE_N),
                              (int32_t)(mb * BLOCK_M));
                    tma_store_commit();
                }
                ts = (ts + 1) % TMA_ST_STAGES;
            }
        }

        tma_store_wait<0>();
        if (ew == 3)
            tmem_dealloc(tmem_base, TMEM_COLS);
    }
}

// =============================================================================
// Host Launch
// =============================================================================

extern "C" {

void launch_fp8_gemm(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    void* D,
    int M, int N, int K)
{
    uint32_t sf_k = (K + GRAN_K * 4 - 1) / (GRAN_K * 4);

    // A: FP8 K-major
    CUtensorMap tma_a;
    {
        uint64_t d[2] = {(uint64_t)K, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {BLOCK_K, BLOCK_M};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2, const_cast<void*>(A), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // B: FP8 K-major
    CUtensorMap tma_b;
    {
        uint64_t d[2] = {(uint64_t)K, (uint64_t)N};
        uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {BLOCK_K, BLOCK_N};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_b, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2, const_cast<void*>(B), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // SFA: int32
    CUtensorMap tma_sfa;
    {
        uint64_t d[2] = {(uint64_t)M, (uint64_t)sf_k};
        uint64_t s[1] = {(uint64_t)M * 4};
        uint32_t b[2] = {BLOCK_M, 1};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_sfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
            2, const_cast<void*>(SFA), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // SFB: int32
    CUtensorMap tma_sfb;
    {
        uint64_t d[2] = {(uint64_t)N, (uint64_t)sf_k};
        uint64_t s[1] = {(uint64_t)N * 4};
        uint32_t b[2] = {BLOCK_N, 1};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_sfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
            2, const_cast<void*>(SFB), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // D: BF16
    CUtensorMap tma_d;
    {
        uint64_t d[2] = {(uint64_t)N, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)N * 2};
        uint32_t b[2] = {STORE_N, STORE_M};
        uint32_t e[2] = {1, 1};
        CUtensorMapSwizzle sw =
            SWIZZLE_CD == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
            SWIZZLE_CD ==  64 ? CU_TENSOR_MAP_SWIZZLE_64B  :
            SWIZZLE_CD ==  32 ? CU_TENSOR_MAP_SWIZZLE_32B  :
                                CU_TENSOR_MAP_SWIZZLE_NONE;
        cuTensorMapEncodeTiled(&tma_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, D, d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // Launch
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t num_tiles = num_m_blocks * num_n_blocks;
    uint32_t num_ctas = (num_tiles < (uint32_t)num_sms)
                      ? num_tiles : (uint32_t)num_sms;

    cudaFuncSetAttribute(fp8_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3(num_ctas, 1, 1);
    config.blockDim = dim3(N_THREADS, 1, 1);
    config.dynamicSmemBytes = TOTAL_SMEM;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    config.attrs    = attrs;
    config.numAttrs = 1;

    uint32_t Mu = M, Nu = N, Ku = K;
    auto err = cudaLaunchKernelEx(&config, fp8_gemm_kernel,
        tma_a, tma_b, tma_sfa, tma_sfb, tma_d, Mu, Nu, Ku);
    if (err != cudaSuccess)
        printf("Launch error: %s\n", cudaGetErrorString(err));
}

} // extern "C"
