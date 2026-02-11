// =============================================================================
// FP8 GEMM Level 1 — SM100 (GB200)
//
// D (BF16, M×N) = A (FP8 E4M3, M×K) × B^T (FP8 E4M3, N×K)
//
// Architecture: cta_group::1 (single SM, no cluster)
// Key FP8-specific features:
//   1. Block-scaled UMMA: tcgen05.mma.kind::mxf8f6f4.block_scale
//   2. Scale factors (SFA/SFB) in UE8M0 format (4 E8M0 packed per uint32)
//   3. UTCCP transposer warp — in-SMEM transpose for UTCCP copy
//   4. UTCCP copy from SMEM to TMEM for scale factors
//   5. BLOCK_K=128 (vs 64 for BF16), UMMA_K=32 (vs 16 for BF16)
//
// Warp assignment (256 threads = 8 warps):
//   Warp 0:   TMA Load (A, B, SFA, SFB)
//   Warp 1:   MMA Issue (UTCCP SF copy + block-scaled UMMA)
//   Warp 2:   UTCCP Transposer (SF SMEM transpose)
//   Warp 3:   TMEM Allocator (then idle)
//   Warp 4-7: Epilogue (TMEM → SMEM → TMA Store)
//
// Pipeline: 5 stages with 3 barrier sets:
//   full_bar:         TMA → Warp2  (data + SF delivered)
//   with_sf_full_bar: Warp2 → MMA  (SF transposed, ready for UTCCP)
//   empty_bar:        MMA → TMA    (SMEM consumed, can refill)
// =============================================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// Configuration
// =============================================================================

static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = 128;
static constexpr uint32_t BLOCK_K = 128;   // FP8 uses 128 (vs BF16's 64)
static constexpr uint32_t UMMA_K  = 32;    // FP8 uses 32  (vs BF16's 16)

static constexpr uint32_t NUM_STAGES = 5;

// Scale factor config: gran_k=128 → one SF per 128 K-elements per BLOCK_K step
// 4 SFs packed into one uint32_t (UE8M0), loaded every 4 pipeline stages
static constexpr uint32_t GRAN_K = 128;
static constexpr uint32_t NUM_SF_PER_LOAD = (GRAN_K == 32) ? 1 : 4;

// SF block sizes (must be aligned to 128 for UTCCP)
static constexpr uint32_t SF_BLOCK_M = ((BLOCK_M + 127) / 128) * 128;
static constexpr uint32_t SF_BLOCK_N = ((BLOCK_N + 127) / 128) * 128;

// SMEM sizes per pipeline stage
static constexpr uint32_t SMEM_A   = BLOCK_M * BLOCK_K * 1;            // 16384 (FP8=1B)
static constexpr uint32_t SMEM_B   = BLOCK_N * BLOCK_K * 1;            // 16384
static constexpr uint32_t SMEM_SFA = SF_BLOCK_M * sizeof(uint32_t);    // 512
static constexpr uint32_t SMEM_SFB = SF_BLOCK_N * sizeof(uint32_t);    // 512

// CD staging (BF16 output)
static constexpr uint32_t SWIZZLE_CD    = 128;                         // 128B swizzle
static constexpr uint32_t STORE_M       = 128;
static constexpr uint32_t STORE_N       = SWIZZLE_CD / 2;              // 64 (BF16=2B)
static constexpr uint32_t CD_STORES     = BLOCK_N / STORE_N;           // 2
static constexpr uint32_t TMA_ST_STAGES = 2;
static constexpr uint32_t SMEM_CD_STAGE = STORE_M * SWIZZLE_CD;        // 16384
static constexpr uint32_t SMEM_CD       = SMEM_CD_STAGE * TMA_ST_STAGES;

// TMEM layout
static constexpr uint32_t NUM_M_WAVES    = BLOCK_M / 128;              // 1
static constexpr uint32_t NUM_EPI_STAGES = 2;                          // TMEM double buffer
static constexpr uint32_t ACCUM_COLS     = NUM_EPI_STAGES * NUM_M_WAVES * BLOCK_N; // 256
static constexpr uint32_t SFA_TMEM_COLS  = SF_BLOCK_M / 32;            // 4
static constexpr uint32_t SFB_TMEM_COLS  = SF_BLOCK_N / 32;            // 4
static constexpr uint32_t SFA_TMEM_OFF   = ACCUM_COLS;                 // 256
static constexpr uint32_t SFB_TMEM_OFF   = SFA_TMEM_OFF + SFA_TMEM_COLS; // 260
static constexpr uint32_t TMEM_COLS      = 512;                        // aligned

// Thread config
static constexpr uint32_t N_MMA_THREADS  = 128;                        // warps 0-3
static constexpr uint32_t N_EPI_THREADS  = 128;                        // warps 4-7
static constexpr uint32_t N_THREADS      = 256;

// Scheduler
static constexpr uint32_t SWIZZLE_GROUP = 16;

// =============================================================================
// PTX Helpers
// =============================================================================

// ----- Barriers -----

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

// ----- elect_one -----

__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "elect.sync _|P, 0xFFFFFFFF;\n\t"
                 "selp.b32 %0, 1, 0, P;\n\t}\n"
                 : "=r"(p));
    return p != 0;
}

// ----- TMA Load (cta_group::1, no cluster) -----

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

// ----- TMA Store -----

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

// ----- TMA Prefetch -----

__device__ __forceinline__ void prefetch_tma(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}

// ----- TMEM Operations -----

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

// ----- UMMA Commit (cta_group::1) -----

__device__ __forceinline__ void umma_commit(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

// ----- Block-Scaled FP8 UMMA (cta_group::1) -----
//
// PTX: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
//        [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p;
//
// Compared to BF16 UMMA:
//   - kind::mxf8f6f4.block_scale (not kind::f16)
//   - Extra TMEM operands [tmem_sfa] and [tmem_sfb] for scale factors
//   - Scale factors must be copied to TMEM first (via UTCCP)

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

// ----- UTCCP: Copy Scale Factors SMEM → TMEM -----
//
// tcgen05.cp.cta_group::1.32x128b.warpx4 [tmem_col], smem_desc;
//
// Copies 128 uint32_t values from SMEM to 4 TMEM columns (32 values each).
// Only 1 thread issues the instruction (via elect_one).
// SMEM data must be transposed first (via utccp_transpose).

__device__ __forceinline__ void utccp_copy(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(smem_desc));
}

// ----- UTCCP SMEM Transpose -----
//
// TMA loads scale factors linearly: SF[0], SF[1], ..., SF[127]
// UTCCP needs them in a specific interleaved layout.
// All 32 threads in the warp participate in this in-place transpose.

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
    // Read 4 values with XOR-based bank-conflict avoidance
    uint32_t vals[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++)
        vals[i] = ld_s32(smem + (i ^ (lane >> 3)) * 32 + lane);
    __syncwarp();
    // Write transposed
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++)
        st_s32(smem + lane * 4 + (i ^ (lane >> 3)), vals[i]);
}

__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ----- SMEM Descriptors -----

// For A/B data with SWIZZLE_128B:
//   SBO = 8 × BLOCK_K × sizeof(FP8) = 8 × 128 × 1 = 1024
//   layout_type = 2 (SWIZZLE_128B), version = 1 (SM100)
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo, uint32_t layout = 2) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);           // start_address [0:14)
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; // SBO [32:46)
    d |= (uint64_t)1 << 46;                   // version = 1 [46]
    d |= (uint64_t)layout << 61;              // layout [61:63)
    return d;
}

// For SF data with SWIZZLE_NONE:
//   SBO = 128 (= 8 × 16 bytes per TMEM bank group)
__device__ __forceinline__ uint64_t make_sf_desc(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((128u >> 4) & 0x3FFF) << 32; // SBO = 8
    d |= (uint64_t)1 << 46;                       // version = 1
    // layout_type = SWIZZLE_NONE = 0 (default)
    return d;
}

// ----- FP8 Instruction Descriptor (Block-Scaled) -----
//
// Bitfield layout (32 bits):
//   [4:6)   b_sf_id    (runtime, selects byte in uint32_t)
//   [7:10)  a_format   = 0 (E4M3)
//   [10:13) b_format   = 0 (E4M3)
//   [15]    a_major    = 0 (K-major)
//   [16]    b_major    = 0 (K-major)
//   [17:23) n_dim      = N/8
//   [23]    scale_fmt  = 1 (E8M0 = UE8M0)
//   [24:29) m_dim      = M/16
//   [29:31) a_sf_id    (runtime, selects byte in uint32_t)

__device__ __forceinline__ uint32_t make_fp8_idesc() {
    uint32_t d = 0;
    d |= ((BLOCK_N / 8) << 17);  // n_dim
    d |= (1u << 23);             // scale_format = E8M0
    d |= ((BLOCK_M / 16) << 24); // m_dim
    return d;
}

__device__ __forceinline__ uint32_t set_sf_ids(uint32_t d, uint32_t a_id, uint32_t b_id) {
    d &= ~(0x3u << 29);
    d &= ~(0x3u << 4);
    d |= ((a_id & 0x3) << 29);
    d |= ((b_id & 0x3) << 4);
    return d;
}

// ----- BF16 Helpers -----

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

// ----- Tile Scheduler (2D Swizzle) -----

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
//   [CD: SMEM_CD]                              32768 bytes
//   [A × NUM_STAGES]                           81920 bytes
//   [B × NUM_STAGES]                           81920 bytes
//   [SFA × NUM_STAGES]                          2560 bytes
//   [SFB × NUM_STAGES]                          2560 bytes
//   [Barriers]                                   152 bytes
//   [TMEM ptr]                                     4 bytes
// Total: ~201,984 bytes (< 232,448 limit)

static constexpr uint32_t O_A   = SMEM_CD;
static constexpr uint32_t O_B   = O_A + NUM_STAGES * SMEM_A;
static constexpr uint32_t O_SFA = O_B + NUM_STAGES * SMEM_B;
static constexpr uint32_t O_SFB = O_SFA + NUM_STAGES * SMEM_SFA;
static constexpr uint32_t O_BAR = O_SFB + NUM_STAGES * SMEM_SFB;

// Barrier layout: full[5], empty[5], sf_full[5], tmem_full[2], tmem_empty[2]
static constexpr uint32_t N_BARS =
    NUM_STAGES * 3 + NUM_EPI_STAGES * 2; // 19

static constexpr uint32_t TOTAL_SMEM = O_BAR + N_BARS * 8 + 4;

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
    uint64_t* empty_bar   = full_bar + NUM_STAGES;
    uint64_t* sf_full_bar = empty_bar + NUM_STAGES;
    uint64_t* tf_bar      = sf_full_bar + NUM_STAGES;   // tmem_full
    uint64_t* te_bar      = tf_bar + NUM_EPI_STAGES;    // tmem_empty
    uint32_t* tmem_smem   = (uint32_t*)(te_bar + NUM_EPI_STAGES);

    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    // =========================================================================
    // Init barriers + TMEM alloc
    // =========================================================================
    if (wid == 1 && elect_one()) {
        for (uint32_t s = 0; s < NUM_STAGES; s++) {
            bar_init(&full_bar[s], 1);      // TMA arrive
            bar_init(&empty_bar[s], 1);     // UMMA commit
            bar_init(&sf_full_bar[s], 32);  // Warp 2 (32 threads)
        }
        for (uint32_t e = 0; e < NUM_EPI_STAGES; e++) {
            bar_init(&tf_bar[e], 1);        // UMMA commit
            bar_init(&te_bar[e], N_EPI_THREADS); // 128 epilogue threads
        }
        bar_fence_init();
    }
    // tcgen05.alloc has .sync.aligned — ALL threads in the warp must call it
    if (wid == 3)
        tmem_alloc(tmem_smem, TMEM_COLS);
    __syncthreads();

    uint32_t tmem_base = *tmem_smem;

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("KERNEL: tmem_base=%u, num_kb=%u, TOTAL_SMEM=%u\n",
               tmem_base, (K + BLOCK_K - 1) / BLOCK_K, TOTAL_SMEM);

    // Prefetch TMA descriptors
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

    // Pipeline state (shared across all warps — each warp has its own copy)
    uint32_t stg = 0, ph = 0;
    auto adv = [&]() { stg = (stg + 1) % NUM_STAGES; ph ^= (stg == 0); };

    // =====================================================================
    // Warp 0: TMA Load
    // =====================================================================
    if (wid == 0 && elect_one()) {
        while (sched.next(mb, nb)) {
            for (uint32_t kb = 0; kb < num_kb; kb++) {
                bar_wait(&empty_bar[stg], ph ^ 1);

                uint32_t k = kb * BLOCK_K;
                uint8_t* a_ptr = smem + O_A + stg * SMEM_A;
                uint8_t* b_ptr = smem + O_B + stg * SMEM_B;

                // A: K-major, coord = {k_offset, m_offset}
                tma_load(a_ptr, &tma_a, &full_bar[stg],
                         (int32_t)k, (int32_t)(mb * BLOCK_M));
                // B: K-major, coord = {k_offset, n_offset}
                tma_load(b_ptr, &tma_b, &full_bar[stg],
                         (int32_t)k, (int32_t)(nb * BLOCK_N));

                uint32_t tx = SMEM_A + SMEM_B;

                // Load SFA/SFB every NUM_SF_PER_LOAD stages
                if (kb % NUM_SF_PER_LOAD == 0) {
                    uint32_t sfk = kb / NUM_SF_PER_LOAD;
                    // SFA: coord = {m_offset, sf_k_offset}
                    tma_load(smem + O_SFA + stg * SMEM_SFA, &tma_sfa,
                             &full_bar[stg],
                             (int32_t)(mb * BLOCK_M), (int32_t)sfk);
                    // SFB: coord = {n_offset, sf_k_offset}
                    tma_load(smem + O_SFB + stg * SMEM_SFB, &tma_sfb,
                             &full_bar[stg],
                             (int32_t)(nb * BLOCK_N), (int32_t)sfk);
                    tx += SMEM_SFA + SMEM_SFB;
                }
                bar_expect_tx(&full_bar[stg], tx);
                adv();
            }
        }
    }

    // =====================================================================
    // Warp 2: UTCCP Transposer
    //
    // Wait for TMA data → transpose SF in-place → signal MMA warp
    // =====================================================================
    else if (wid == 2) {
        while (sched.next(mb, nb)) {
            for (uint32_t kb = 0; kb < num_kb; kb++) {
                bar_wait(&full_bar[stg], ph);

                if (kb % NUM_SF_PER_LOAD == 0) {
                    // Transpose SFA
                    for (uint32_t i = 0; i < SF_BLOCK_M / 128; i++)
                        utccp_transpose(
                            (uint32_t*)(smem + O_SFA + stg * SMEM_SFA) + i*128,
                            lane);
                    fence_async_smem();

                    // Transpose SFB
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
    //
    // Wait for SF-ready data → UTCCP copy SF to TMEM → block-scaled UMMA
    // =====================================================================
    else if (wid == 1) {
        static constexpr uint32_t SBO = 8 * BLOCK_K * 1;  // 1024 for swizzle, same for no-swizzle

        uint32_t base_idesc = make_fp8_idesc();

        while (sched.next(mb, nb)) {
            uint32_t ai = sched.iter % NUM_EPI_STAGES;
            uint32_t ap = (sched.iter / NUM_EPI_STAGES) & 1;

            // Wait for TMEM to be available
            bar_wait(&te_bar[ai], ap ^ 1);
            tmem_fence_after();

            for (uint32_t kb = 0; kb < num_kb; kb++) {
                // Wait for SF transpose
                bar_wait(&sf_full_bar[stg], ph);
                tmem_fence_after();

                // UTCCP copy SF to TMEM (at SF-load stages)
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

                // Build descriptors for current stage
                uint64_t a_desc = make_smem_desc(
                    smem + O_A + stg * SMEM_A, SBO);
                uint64_t b_desc = make_smem_desc(
                    smem + O_B + stg * SMEM_B, SBO);
                // Split descriptor at bit 32 (lo=lower 32 bits, hi=upper 32 bits)
                // NOT at bit 14! Splitting at 14 truncates version+layout_type fields.
                uint32_t a_lo = (uint32_t)(a_desc);
                uint32_t a_hi = (uint32_t)(a_desc >> 32);
                uint32_t b_lo = (uint32_t)(b_desc);
                uint32_t b_hi = (uint32_t)(b_desc >> 32);

                uint32_t idesc = set_sf_ids(base_idesc, sf_idx, sf_idx);
                uint32_t accum = (kb > 0) ? 1u : 0u;

                if (elect_one()) {
                    // 4 UMMAs per BLOCK_K (128 / 32 = 4)
                    #pragma unroll
                    for (uint32_t kk = 0; kk < BLOCK_K / UMMA_K; kk++) {
                        // lo advances by kk * UMMA_K * sizeof(FP8) / 16 = kk * 2
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

                // Commit arrivals
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
    // Warp 4-7: Epilogue (TMEM → BF16 Pack → Swizzled SMEM → TMA Store)
    // =====================================================================
    else if (wid >= 4) {
        uint32_t ew = wid - 4;           // epilogue warp index (0-3)
        uint32_t lt = threadIdx.x - N_MMA_THREADS; // local thread id (0-127)
        uint32_t ts = 0;                  // tma store stage

        while (sched.next(mb, nb)) {
            uint32_t ai = sched.iter % NUM_EPI_STAGES;
            uint32_t ap = (sched.iter / NUM_EPI_STAGES) & 1;

            bar_wait(&tf_bar[ai], ap);
            tmem_fence_after();

            for (uint32_t s = 0; s < CD_STORES; s++) {
                // Wait for previous TMA store to finish
                if (ew == 0) tma_store_wait<TMA_ST_STAGES - 1>();
                asm volatile("bar.sync 0, %0;" :: "r"(N_EPI_THREADS));

                // TMEM → registers → BF16 pack → swizzled SMEM
                uint32_t row_base = ts * SMEM_CD_STAGE + lt * SWIZZLE_CD;
                uint32_t smem_base_addr = (uint32_t)__cvta_generic_to_shared(smem);

                #pragma unroll
                for (uint32_t i = 0; i < STORE_N / 8; i++) {  // 8 iterations
                    uint32_t col = tmem_base + ai * NUM_M_WAVES * BLOCK_N
                                 + s * STORE_N + i * 8;

                    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                    tmem_ld_8x(col, r0, r1, r2, r3, r4, r5, r6, r7);
                    tmem_wait_ld();

                    // FP32 → BF16 pack (8 FP32 → 4 × 2 BF16)
                    uint32_t p0 = pack_bf16(r0, r1);
                    uint32_t p1 = pack_bf16(r2, r3);
                    uint32_t p2 = pack_bf16(r4, r5);
                    uint32_t p3 = pack_bf16(r6, r7);

                    // Swizzled SMEM address: bank_group XOR (row % 8)
                    uint32_t bg = i ^ (lt % 8);
                    uint32_t addr = smem_base_addr + row_base + bg * 16;
                    st_s128(addr, p0, p1, p2, p3);
                }

                // Signal TMEM is free after last store block
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
        // tcgen05.dealloc has .sync.aligned — ALL threads in the warp must call it
        if (ew == 3)
            tmem_dealloc(tmem_base, TMEM_COLS);
    }
}

// =============================================================================
// Host Launch
// =============================================================================

extern "C" {

void launch_fp8_gemm(
    const void* A,   // FP8 E4M3 [M, K], K-contiguous
    const void* B,   // FP8 E4M3 [N, K], K-contiguous
    const void* SFA, // int32 [sf_k_dim, M], M-contiguous
    const void* SFB, // int32 [sf_k_dim, N], N-contiguous
    void* D,         // BF16 [M, N], N-contiguous
    int M, int N, int K)
{
    uint32_t sf_k = (K + GRAN_K * 4 - 1) / (GRAN_K * 4);

    // A: FP8 K-major. TMA dims = {K, M}, box = {BLOCK_K, BLOCK_M}
    CUtensorMap tma_a;
    {
        uint64_t d[2] = {(uint64_t)K, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)K};   // outer stride in bytes (FP8=1B)
        uint32_t b[2] = {BLOCK_K, BLOCK_M};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2, const_cast<void*>(A), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // B: FP8 K-major. TMA dims = {K, N}, box = {BLOCK_K, BLOCK_N}
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

    // SFA: int32, dims = {M, sf_k}, box = {BLOCK_M, 1}
    CUtensorMap tma_sfa;
    {
        uint64_t d[2] = {(uint64_t)M, (uint64_t)sf_k};
        uint64_t s[1] = {(uint64_t)M * 4};  // stride in bytes (int32=4B)
        uint32_t b[2] = {BLOCK_M, 1};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_sfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
            2, const_cast<void*>(SFA), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // SFB: int32, dims = {N, sf_k}, box = {BLOCK_N, 1}
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

    // D: BF16, dims = {N, M}, box = {STORE_N, STORE_M}
    CUtensorMap tma_d;
    {
        uint64_t d[2] = {(uint64_t)N, (uint64_t)M};
        uint64_t s[1] = {(uint64_t)N * 2};  // BF16=2B
        uint32_t b[2] = {STORE_N, STORE_M};
        uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, D, d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
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
