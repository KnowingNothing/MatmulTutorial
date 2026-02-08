// level3/matmul.cu
// SM100 (GB200) BF16 GEMM — True Warp Specialization + 8-stage Deep Pipeline
//
// Improvements over level2:
//   1. True warp specialization: TMA warp and MMA warp run INDEPENDENT loops,
//      communicating only via full_bar / empty_bar barriers
//   2. Deep pipeline: 8 SMEM stages (vs 2), TMA can prefetch up to 8 K blocks
//      ahead of MMA, fully hiding memory latency behind compute
//   3. cta_group::2 TMA with PEER_BIT_MASK (inherited from level2)
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
static constexpr uint32_t NUM_STAGES = 8;   // deep pipeline (was 2)

static constexpr uint32_t LOAD_N_PER_CTA = BLOCK_N / CLUSTER_SIZE;
static constexpr uint32_t UMMA_M = 256;
static constexpr uint32_t UMMA_N = BLOCK_N;
static constexpr uint32_t UMMA_K = 16;

static constexpr uint32_t SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t SMEM_B_SIZE = LOAD_N_PER_CTA * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t SMEM_STAGE  = SMEM_A_SIZE + SMEM_B_SIZE;
static constexpr uint32_t TMA_BYTES   = SMEM_A_SIZE + SMEM_B_SIZE;

static constexpr uint32_t UMMA_SBO = 8u * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t TMEM_COLS = 128;
static constexpr uint32_t NUM_THREADS = 128;

// SMEM layout: [barriers | pad to 1024 | stage data × NUM_STAGES]
// Barriers: full[0..7](64B) + empty[0..7](64B) + tmem_bar(8B) + tmem_addr(4B) = 140B
static constexpr uint32_t OFF_DATA  = 1024;
// 8 stages: 1024 + 8 × 24576 = 197632 bytes (< 232448 SM100 SMEM capacity)
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
// Cluster-scoped arrive_expect_tx: signals a barrier in target_cta's shared memory
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
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
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
    // full_bar[0..7]  : offsets 0..63    (8 × 8 bytes)
    // empty_bar[0..7] : offsets 64..127  (8 × 8 bytes)
    // tmem_bar        : offset  128      (8 bytes)
    // tmem_addr       : offset  136      (4 bytes)
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
            barrier_init(full_bar[s],  CLUSTER_SIZE);  // 1 arrival per CTA
            barrier_init(empty_bar[s], 1);             // 1 UMMA commit arrival
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

    // UMMA descriptors per stage (only needed by MMA warp, but computed by all)
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);
    uint32_t alo[NUM_STAGES], blo[NUM_STAGES], ahi, bhi;
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        uint64_t ad = make_smem_desc(get_smem_a(s), UMMA_SBO);
        uint64_t bd = make_smem_desc(get_smem_b(s), UMMA_SBO);
        alo[s] = static_cast<uint32_t>(ad);
        blo[s] = static_cast<uint32_t>(bd);
        if (s == 0) { ahi = static_cast<uint32_t>(ad >> 32); bhi = static_cast<uint32_t>(bd >> 32); }
    }

    // ========== True Warp Specialization: independent loops ==========

    if (warp_idx == 0 && elect_one()) {
        // ==================== TMA Warp (both CTAs) ====================
        // Runs independently, can be up to NUM_STAGES K blocks ahead of MMA warp.
        // Waits on empty_bar (SMEM free) → arrive_expect_tx → issue TMA loads.
        uint32_t tma_stage = 0, tma_phase = 0;

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            // Wait for SMEM[tma_stage] to be free
            barrier_wait(empty_bar[tma_stage], tma_phase ^ 1);

            // Arrive on leader's full_bar with expected byte count
            if (is_leader)
                barrier_arrive_expect_tx(full_bar[tma_stage], TMA_BYTES);
            else
                barrier_arrive_expect_tx_cluster(full_bar[tma_stage], TMA_BYTES, 0);

            // Issue cta_group::2 TMA loads
            int32_t kc = kb * BLOCK_K;
            tma_load_2d_cg2(get_smem_a(tma_stage), &tma_a, full_bar[tma_stage], kc, m_coord);
            tma_load_2d_cg2(get_smem_b(tma_stage), &tma_b, full_bar[tma_stage], kc, n_coord);

            // Advance TMA pipeline
            tma_stage = (tma_stage + 1) % NUM_STAGES;
            tma_phase ^= (tma_stage == 0);
        }

    } else if (warp_idx == 1 && is_leader && elect_one()) {
        // ==================== MMA Warp (leader CTA only) ====================
        // Runs independently, waits on full_bar (TMA data ready) → UMMA → commit empty_bar.
        uint32_t mma_stage = 0, mma_phase = 0;

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            // Wait for TMA data to be ready in SMEM[mma_stage]
            barrier_wait(full_bar[mma_stage], mma_phase);
            tcgen05_fence_after();

            // Issue UMMA for this K block
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                uint64_t ad = ((uint64_t)ahi << 32) | (alo[mma_stage] + k * 2);
                uint64_t bd = ((uint64_t)bhi << 32) | (blo[mma_stage] + k * 2);
                uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;
                umma_f16_cg2(0, ad, bd, idesc, accum);
            }

            // Commit: multicast signals empty_bar on BOTH CTAs
            umma_commit_2sm(empty_bar[mma_stage]);

            // On last K block, also signal tmem_bar
            if (kb == num_k_blocks - 1)
                umma_commit_2sm(tmem_bar);

            // Advance MMA pipeline
            mma_stage = (mma_stage + 1) % NUM_STAGES;
            mma_phase ^= (mma_stage == 0);
        }
    }
    // Warps 2-3 and non-elected threads: idle, fall through to epilogue

    // ========== Epilogue ==========
    barrier_wait(tmem_bar, 0);
    tcgen05_fence_after();

    // Only leader CTA writes output
    if (is_leader) {
        const uint32_t m_idx = m_block * BLOCK_M + threadIdx.x;
        if (m_idx < M) {
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x(col, r0, r1, r2, r3);
                tmem_load_fence();

                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);

                uint32_t n_base = n_block * BLOCK_N + col;
                __nv_bfloat16* out = D + (uint64_t)m_idx * N + n_base;
                if (n_base     < N) out[0] = __float2bfloat16(f0);
                if (n_base + 1 < N) out[1] = __float2bfloat16(f1);
                if (n_base + 2 < N) out[2] = __float2bfloat16(f2);
                if (n_base + 3 < N) out[3] = __float2bfloat16(f3);
            }
        }
    }

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
    uint32_t box0, uint32_t box1) {
    uint64_t dims[2]    = {dim0, dim1};
    uint64_t strides[1] = {dim0 * sizeof(__nv_bfloat16)};
    uint32_t box[2]     = {box0, box1};
    uint32_t estrides[2]= {1, 1};
    cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(ptr), dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

static void launch_kernel(
    __nv_bfloat16* D, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K) {
    CUtensorMap tma_a, tma_b;
    create_tma_desc(&tma_a, A, (uint64_t)K, (uint64_t)M, BLOCK_K, BLOCK_M);
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
