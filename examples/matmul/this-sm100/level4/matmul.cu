// level4/matmul.cu
// SM100 (GB200) BF16 GEMM — SMEM-Staged Coalesced Epilogue
//
// Improvements over level3:
//   1. Coalesced epilogue: TMEM → SMEM staging → coalesced vectorized global writes
//      Level3 epilogue: each thread writes to a DIFFERENT row → 32 independent memory
//      transactions per warp (completely uncoalesced, ~32x bandwidth waste)
//      Level4 epilogue: remap threads so each warp writes to the SAME row with
//      consecutive column addresses → 2 cache-line transactions per warp (fully coalesced)
//   2. Vectorized 8-byte (uint2) global stores: 4 BF16 per store instruction
//   3. Inherits all level3 optimizations (true warp specialization, 8-stage pipeline,
//      cta_group::2 TMA with PEER_BIT_MASK)
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
static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = 128;
static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t CLUSTER_SIZE = 2;
static constexpr uint32_t NUM_STAGES = 8;   // deep pipeline

static constexpr uint32_t LOAD_N_PER_CTA = BLOCK_N / CLUSTER_SIZE;
static constexpr uint32_t UMMA_M = 256;
static constexpr uint32_t UMMA_N = BLOCK_N;
static constexpr uint32_t UMMA_K = 16;

static constexpr uint32_t SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t SMEM_B_SIZE = LOAD_N_PER_CTA * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t SMEM_STAGE  = SMEM_A_SIZE + SMEM_B_SIZE;
static constexpr uint32_t TMA_BYTES   = SMEM_A_SIZE + SMEM_B_SIZE;

// ---- UMMA Stride Byte Offset (SBO) ----
// With SWIZZLE_128B, SMEM data is organized in repeating "swizzle atoms".
// Each atom spans 8 rows × 128 bytes (one swizzle period = 128B / 16B_per_bank_group = 8 rows).
// SBO = byte stride between consecutive atoms along the M (or N) dimension
//     = 8_rows × BLOCK_K × sizeof(BF16) = 8 × 64 × 2 = 1024 bytes
// Equivalently: SBO = swizzle_mode × (swizzle_mode / 16) = 128 × 8 = 1024
// (DeepGEMM uses: sbo = swizzle_ab_mode * 8)
static constexpr uint32_t UMMA_SBO = 8u * BLOCK_K * sizeof(__nv_bfloat16);  // = 1024 bytes

static constexpr uint32_t TMEM_COLS = 128;
static constexpr uint32_t NUM_THREADS = 128;

// SMEM layout: [barriers | pad to 1024 | stage data × NUM_STAGES]
static constexpr uint32_t OFF_DATA  = 1024;
static constexpr uint32_t SMEM_SIZE = OFF_DATA + NUM_STAGES * SMEM_STAGE;

// Peer bit mask: clears bit 24 to route barrier signal to leader CTA (rank 0)
static constexpr uint32_t PEER_BIT_MASK = 0xFEFFFFFF;

// ============================================================
//  PTX helpers
// ============================================================
__device__ __forceinline__ uint32_t get_warp_id() { return threadIdx.x >> 5; }
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

__device__ __forceinline__ void barrier_init(void* bar, uint32_t count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
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

__device__ __forceinline__ void prefetch_tma(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}
__device__ __forceinline__ void tma_load_2d(
    void* smem, const void* desc, void* bar, int32_t c0, int32_t c1) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t ba = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"(desc), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
// cta_group::2 TMA: loads into calling CTA's SMEM, signals leader CTA's barrier
// NOTE: PEER_BIT_MASK is applied ONLY to the barrier address (not destination!)
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

__device__ __forceinline__ void tmem_alloc_2sm(void* dst_smem, int ncols) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_2sm(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void tmem_load_4x(
    uint32_t col, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_2sm(void* bar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
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

// ---- SmemDescriptor for tcgen05.mma (64-bit) ----
//
// Tells the UMMA hardware where to find operand data in shared memory
// and how it's laid out (swizzle pattern, strides).
//
// Bit layout (from CUTLASS mma_sm100_desc.hpp: UMMA::SmemDescriptor):
//
//   Bits [13:0]  — start_address  : SMEM base address >> 4 (16-byte aligned)
//   Bits [15:14] — (reserved)
//   Bits [29:16] — leading_byte_offset (LBO) >> 4 : byte stride between atoms on K dim
//                  For K-major layout with BLOCK_K matching swizzle width, LBO = 0
//                  (only 1 atom along K, so no stride needed)
//   Bits [31:30] — (reserved)
//   Bits [45:32] — stride_byte_offset (SBO) >> 4 : byte stride between atoms on M/N dim
//                  = swizzle_period_rows × BLOCK_K × sizeof(elem) >> 4
//                  e.g. SWIZZLE_128B + BF16 + BLOCK_K=64: SBO = 8×64×2 = 1024 → field=64
//   Bits [47:46] — version : must be 1 for SM100 (was 0 for SM90)
//   Bit  [48]    — (unused)
//   Bits [51:49] — base_offset : base offset for address (0 for our case)
//   Bit  [52]    — lbo_mode : leading byte offset mode (0 = legacy/off)
//   Bits [60:53] — (reserved)
//   Bits [63:61] — layout_type : swizzle mode of SMEM data
//                  0 = SWIZZLE_NONE
//                  1 = SWIZZLE_128B_BASE32B (special for FP32 MN-major)
//                  2 = SWIZZLE_128B  ← we use this, matching CU_TENSOR_MAP_SWIZZLE_128B
//                  4 = SWIZZLE_64B
//                  6 = SWIZZLE_32B
//
// IMPORTANT: layout_type here MUST match the swizzle mode used in the TMA descriptor
// (cuTensorMapEncodeTiled). TMA loads data from global to SMEM with the specified swizzle,
// and the UMMA reads from SMEM expecting the same pattern.
//
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;

    // [13:0] start_address: SMEM pointer >> 4 (16-byte granularity)
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);

    // [29:16] leading_byte_offset: 0 for K-major with 1 atom on K (our case)
    //         (not set, stays 0)

    // [45:32] stride_byte_offset: SBO >> 4 (16-byte granularity)
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;

    // [47:46] version: 1 for SM100 (required, different from SM90 which uses 0)
    d |= (uint64_t)1 << 46;

    // [63:61] layout_type: 2 = SWIZZLE_128B
    //         Must match CU_TENSOR_MAP_SWIZZLE_128B in TMA descriptor
    d |= (uint64_t)2 << 61;

    return d;
}

// ---- InstrDescriptor for tcgen05.mma (32-bit) ----
//
// Encodes data types, dimensions, and layout for the UMMA instruction.
// This descriptor is the upper 32 bits of the 64-bit idescE register.
//
// Bit layout (from CUTLASS mma_sm100_desc.hpp: UMMA::InstrDescriptor):
//
//   Bits [1:0]  — sparse_id2   : 0 (dense, no sparsity)
//   Bit  [2]    — sparse_flag  : 0 = dense, 1 = sparse
//   Bit  [3]    — saturate     : 0 = no saturate
//   Bits [5:4]  — c_format     : accumulator (C/D) data type
//                 For kind::f16: 0 = FP16, 1 = FP32, 2 = BF16
//   Bit  [6]    — (reserved)
//   Bits [9:7]  — a_format     : A operand data type
//                 For kind::f16: 0 = FP16, 1 = BF16, 2 = TF32
//   Bits [12:10]— b_format     : B operand data type
//                 For kind::f16: 0 = FP16, 1 = BF16, 2 = TF32
//   Bit  [13]   — a_negate     : 0 = no negate
//   Bit  [14]   — b_negate     : 0 = no negate
//   Bit  [15]   — a_major      : 0 = K-major, 1 = MN-major
//   Bit  [16]   — b_major      : 0 = K-major, 1 = MN-major
//   Bits [22:17]— n_dim        : N / 8 (valid: 1..32 → N = 8..256)
//   Bit  [23]   — (reserved)
//   Bits [28:24]— m_dim        : M / 16 (valid: 4=M64, 8=M128, 16=M256)
//   Bit  [29]   — (reserved)
//   Bits [31:30]— max_shift    : 0 = no shift (for WS instructions)
//
// Note: The data type encoding in a_format/b_format depends on the instruction
// kind (f16, mxf8f6f4, s8, etc.). The values above are for kind::f16 only.
//
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;

    // [5:4] c_format = 1 → FP32 accumulator
    //       (our UMMA accumulates in FP32, stored in TMEM)
    d |= (1u << 4);

    // [9:7] a_format = 1 → BF16 for A operand
    d |= (1u << 7);

    // [12:10] b_format = 1 → BF16 for B operand
    d |= (1u << 10);

    // [15] a_major = 0 → K-major (A is row-major (M,K), K is fastest dim)
    // [16] b_major = 0 → K-major (B is row-major (N,K), K is fastest dim)
    //      (not set, stays 0)

    // [22:17] n_dim = N / 8 (e.g. N=128 → 16)
    d |= ((N / 8) << 17);

    // [28:24] m_dim = M / 16 (e.g. M=256 for 2SM → 16)
    d |= ((M / 16) << 24);

    return d;
}

// ============================================================
//  Kernel
// ============================================================
__global__ void
__cluster_dims__(2, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
bf16_gemm_2sm_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    __nv_bfloat16* __restrict__ D,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t num_n_blocks)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    const uint32_t warp_idx  = get_warp_id();
    const uint32_t cta_rank  = get_cluster_rank();
    const bool     is_leader = (cta_rank == 0);

    const uint32_t cluster_id = blockIdx.x / CLUSTER_SIZE;
    const uint32_t m_block    = cluster_id / num_n_blocks;
    const uint32_t n_block    = cluster_id % num_n_blocks;

    extern __shared__ __align__(1024) uint8_t smem_buf[];

    // ---- Barrier & TMEM pointer layout ----
    uint64_t* full_bar[NUM_STAGES];
    uint64_t* empty_bar[NUM_STAGES];
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        full_bar[s]  = reinterpret_cast<uint64_t*>(smem_buf + s * 8);
        empty_bar[s] = reinterpret_cast<uint64_t*>(smem_buf + NUM_STAGES * 8 + s * 8);
    }
    uint64_t* tmem_bar  = reinterpret_cast<uint64_t*>(smem_buf + NUM_STAGES * 16);
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(smem_buf + NUM_STAGES * 16 + 8);

    auto get_smem_a = [&](uint32_t s) -> void* { return smem_buf + OFF_DATA + s * SMEM_STAGE; };
    auto get_smem_b = [&](uint32_t s) -> void* { return smem_buf + OFF_DATA + s * SMEM_STAGE + SMEM_A_SIZE; };

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one()) {
        prefetch_tma(&tma_a);
        prefetch_tma(&tma_b);
    }

    // Initialize barriers
    if (warp_idx == 1 && elect_one()) {
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            barrier_init(full_bar[s],  CLUSTER_SIZE);
            barrier_init(empty_bar[s], 1);
        }
        barrier_init(tmem_bar, 1);
        fence_barrier_init();
    }

    // TMEM allocation
    if (warp_idx == 2) tmem_alloc_2sm(tmem_addr, TMEM_COLS);

    cluster_sync();

    // Compute constants
    const uint32_t num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    const int32_t  m_coord = m_block * BLOCK_M;
    const int32_t  n_coord = n_block * BLOCK_N + cta_rank * LOAD_N_PER_CTA;

    // ---- Build UMMA descriptors ----
    // InstrDescriptor: encodes data types (BF16×BF16→FP32), dimensions (M=256,N=128),
    //                  and layout (K-major for both A and B). Same for all stages.
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);

    // SmemDescriptor: encodes SMEM address + SBO + swizzle for each pipeline stage.
    // We split it into lo (32b) and hi (32b) parts because:
    //   - lo contains start_address which changes per stage (different SMEM offset)
    //   - hi contains SBO + version + layout_type which are IDENTICAL across stages
    // This lets us cheaply advance the K-dimension by just adding to lo:
    //   desc = (hi << 32) | (lo + k * 2)
    // where k*2 = k_slice_index * UMMA_K * sizeof(BF16) / 16 = k * 16 * 2 / 16 = k * 2
    uint32_t alo[NUM_STAGES], blo[NUM_STAGES], ahi, bhi;
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        uint64_t ad = make_smem_desc(get_smem_a(s), UMMA_SBO);
        uint64_t bd = make_smem_desc(get_smem_b(s), UMMA_SBO);
        alo[s] = static_cast<uint32_t>(ad);       // lo: start_address (different per stage)
        blo[s] = static_cast<uint32_t>(bd);
        if (s == 0) {
            ahi = static_cast<uint32_t>(ad >> 32); // hi: SBO + version + layout (same for all)
            bhi = static_cast<uint32_t>(bd >> 32);
        }
    }

    // ========== True Warp Specialization: independent loops ==========

    if (warp_idx == 0 && elect_one()) {
        // ==================== TMA Warp (both CTAs) ====================
        uint32_t tma_stage = 0, tma_phase = 0;

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            barrier_wait(empty_bar[tma_stage], tma_phase ^ 1);

            if (is_leader)
                barrier_arrive_expect_tx(full_bar[tma_stage], TMA_BYTES);
            else
                barrier_arrive_expect_tx_cluster(full_bar[tma_stage], TMA_BYTES, 0);

            int32_t kc = kb * BLOCK_K;
            tma_load_2d_cg2(get_smem_a(tma_stage), &tma_a, full_bar[tma_stage], kc, m_coord);
            tma_load_2d_cg2(get_smem_b(tma_stage), &tma_b, full_bar[tma_stage], kc, n_coord);

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            tma_phase ^= (tma_stage == 0);
        }

    } else if (warp_idx == 1 && is_leader && elect_one()) {
        // ==================== MMA Warp (leader CTA only) ====================
        uint32_t mma_stage = 0, mma_phase = 0;

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            barrier_wait(full_bar[mma_stage], mma_phase);
            tcgen05_fence_after();

            // Issue UMMA for each K-slice within this K block.
            // BLOCK_K=64, UMMA_K=16 → 4 UMMA instructions per K block.
            // Each UMMA reads 16 K-elements from swizzled SMEM via descriptors:
            //   desc = (hi << 32) | (stage_lo + k * 2)
            //   k*2: each K-slice is 16 BF16 = 32 bytes → 32/16 = 2 in 16B-granularity
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                uint64_t ad = ((uint64_t)ahi << 32) | (alo[mma_stage] + k * 2);
                uint64_t bd = ((uint64_t)bhi << 32) | (blo[mma_stage] + k * 2);
                // accum=0 on first slice of first K-block → zero accumulator
                // accum=1 otherwise → accumulate onto existing TMEM values
                uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;
                umma_f16_cg2(0, ad, bd, idesc, accum);
            }

            umma_commit_2sm(empty_bar[mma_stage]);

            if (kb == num_k_blocks - 1)
                umma_commit_2sm(tmem_bar);

            mma_stage = (mma_stage + 1) % NUM_STAGES;
            mma_phase ^= (mma_stage == 0);
        }
    }

    // ========== Epilogue: SMEM-staged coalesced writes ==========
    barrier_wait(tmem_bar, 0);
    tcgen05_fence_after();

    // Only leader CTA writes output
    if (is_leader) {
        // Reuse pipeline SMEM for output staging (128×128 BF16 = 32 KB)
        // Pipeline SMEM is free after K-loop completion
        __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_DATA);

        // ---- Phase 1: TMEM → SMEM ----
        // Each thread loads its row's data from TMEM and stores BF16 to SMEM
        // Thread t → row t, columns 0..127
        for (uint32_t col = 0; col < BLOCK_N; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(col, r0, r1, r2, r3);
            tmem_load_fence();

            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);

            uint32_t base = threadIdx.x * BLOCK_N + col;
            smem_out[base + 0] = __float2bfloat16(f0);
            smem_out[base + 1] = __float2bfloat16(f1);
            smem_out[base + 2] = __float2bfloat16(f2);
            smem_out[base + 3] = __float2bfloat16(f3);
        }

        __syncthreads();

        // ---- Phase 2: SMEM → Global (coalesced vectorized writes) ----
        // Remap threads: 4 warps process 4 DIFFERENT rows per step,
        // each warp's 32 threads write 4 consecutive BF16 each → 128 columns/row
        // Threads in a warp write to the SAME row at consecutive addresses → coalesced!
        //
        // Memory access pattern per warp per step:
        //   32 threads × 4 BF16 × 2 bytes = 256 bytes = 2 cache lines (fully coalesced)
        // Total: 32 steps × 4 warps × 2 cache lines = 256 cache line transactions
        // vs Level3: 128 threads × 32 iters × 1 cache line = 4096 transactions (uncoalesced)
        const uint32_t warp_id = threadIdx.x / 32;
        const uint32_t lane_id = threadIdx.x % 32;

        #pragma unroll 4
        for (uint32_t step = 0; step < BLOCK_M / 4; ++step) {
            uint32_t row = step * 4 + warp_id;
            uint32_t global_row = m_block * BLOCK_M + row;
            uint32_t col_start  = lane_id * 4;
            uint32_t global_col = n_block * BLOCK_N + col_start;

            if (global_row < M && global_col + 3 < N) {
                // Vectorized 8-byte read from SMEM
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BLOCK_N + col_start]);
                // Vectorized 8-byte write to global (coalesced within warp)
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            }
        }
    }

    cluster_sync();
    if (warp_idx == 2) tmem_dealloc_2sm(0, TMEM_COLS);
#endif
}

// ============================================================
//  Host — TMA Descriptor Creation
// ============================================================

// Creates a 2D TMA descriptor for loading tiles from global memory to SMEM.
//
// TMA (Tensor Memory Access) is SM90+ hardware that asynchronously copies
// 2D tiles from global memory to shared memory, automatically applying
// the specified swizzle pattern. This means the data arrives in SMEM
// already in the SWIZZLE_128B layout that UMMA expects.
//
// Parameters:
//   ptr  — pointer to the matrix in global memory (BF16)
//   dim0 — size of the fastest-changing dimension (num elements)
//   dim1 — size of the slowest-changing dimension (num elements)
//   box0 — tile size along dim0 (elements per tile load)
//   box1 — tile size along dim1 (elements per tile load)
//
// For matrix A (M×K, row-major, K-major in TMA terms):
//   dim0 = K (fastest), dim1 = M (slowest), box0 = BLOCK_K, box1 = BLOCK_M
//   stride = K × sizeof(BF16) bytes between consecutive M-rows
//
// For matrix B (N×K, row-major, K-major in TMA terms):
//   dim0 = K (fastest), dim1 = N (slowest), box0 = BLOCK_K, box1 = LOAD_N_PER_CTA
//   stride = K × sizeof(BF16) bytes between consecutive N-rows
//
// Swizzle: CU_TENSOR_MAP_SWIZZLE_128B
//   TMA rearranges data in SMEM using 128-byte swizzle atoms.
//   Within each atom (8 rows × 128 bytes), bank group indices are XOR'd
//   with the row index: swizzled_bank = bank XOR (row % 8).
//   This eliminates shared memory bank conflicts when UMMA reads the data.
//   CRITICAL: This swizzle must match layout_type=2 in SmemDescriptor.
//
static void create_tma_desc(
    CUtensorMap* map, const void* ptr,
    uint64_t dim0, uint64_t dim1,
    uint32_t box0, uint32_t box1) {
    uint64_t dims[2]    = {dim0, dim1};
    uint64_t strides[1] = {dim0 * sizeof(__nv_bfloat16)};  // row stride in bytes
    uint32_t box[2]     = {box0, box1};
    uint32_t estrides[2]= {1, 1};  // element strides (must be 1)
    cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(ptr), dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,        // ← 128-byte swizzle, matches SmemDesc layout_type=2
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

static void launch_kernel(
    __nv_bfloat16* D, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K) {
    CUtensorMap tma_a, tma_b;
    // A is (M,K) row-major → TMA sees: dim0=K (fast), dim1=M (slow), box=(BLOCK_K, BLOCK_M)
    create_tma_desc(&tma_a, A, (uint64_t)K, (uint64_t)M, BLOCK_K, BLOCK_M);
    // B is (N,K) row-major → TMA sees: dim0=K (fast), dim1=N (slow), box=(BLOCK_K, N_PER_CTA)
    // Each CTA loads LOAD_N_PER_CTA = BLOCK_N/2 = 64 rows of B (split across 2 CTAs in cluster)
    create_tma_desc(&tma_b, B, (uint64_t)K, (uint64_t)N, BLOCK_K, LOAD_N_PER_CTA);

    uint32_t num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t num_ctas     = num_m_blocks * num_n_blocks * CLUSTER_SIZE;

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

    uint32_t Mu = M, Nu = N, Ku = K, nnb = num_n_blocks;
    cudaLaunchKernelEx(&config, bf16_gemm_2sm_kernel,
                       tma_a, tma_b, D, Mu, Nu, Ku, nnb);
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
