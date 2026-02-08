// level0/matmul.cu
// SM100 (GB200) BF16 GEMM — Baseline: correct 2SM UMMA kernel
//
// Computes D = A @ B^T
//   A: (M, K) row-major BF16
//   B: (N, K) row-major BF16
//   D: (M, N) row-major BF16
// Accumulation in FP32, output converted back to BF16.
//
// All SM100 features (TMA, UMMA, TMEM) implemented from scratch with inline PTX.
// No cutlass/cute/deepgemm headers used.
//
// Build:
//   nvcc -gencode arch=compute_100a,code=sm_100a -O2 --shared
//        -Xcompiler -fPIC -o matmul.so matmul.cu -lcuda

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
static constexpr uint32_t CLUSTER_SIZE = 2;    // 2 CTAs per cluster (for 2SM UMMA)

// Each CTA loads half of B along N
static constexpr uint32_t LOAD_N_PER_CTA = BLOCK_N / CLUSTER_SIZE; // 64

// UMMA instruction parameters
// cta_group::2 → UMMA_M = 256 (128 rows per SM × 2 SMs)
// Both CTAs load the same A, so computation is redundant but correct.
static constexpr uint32_t UMMA_M = 256;
static constexpr uint32_t UMMA_N = BLOCK_N;    // 128
static constexpr uint32_t UMMA_K = 16;         // 32 bytes / sizeof(bf16)

// Shared memory sizes (bytes)
static constexpr uint32_t SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);        // 16384
static constexpr uint32_t SMEM_B_SIZE = LOAD_N_PER_CTA * BLOCK_K * sizeof(__nv_bfloat16); // 8192

// UMMA descriptor: K-major, SWIZZLE_128B
// atom_base = 16 for SWIZZLE_128B; num_non_contiguous = 128/16 = 8
// SBO = 8 * BLOCK_K * sizeof(bf16) = 1024;  LBO = 0
static constexpr uint32_t UMMA_SBO = 8u * BLOCK_K * sizeof(__nv_bfloat16); // 1024

// TMEM columns = BLOCK_N
static constexpr uint32_t TMEM_COLS = 128;

// Threads per CTA: 4 warps
static constexpr uint32_t NUM_THREADS = 128;

// TMA bytes per CTA per stage
static constexpr uint32_t TMA_BYTES_PER_CTA = SMEM_A_SIZE + SMEM_B_SIZE;

// Shared memory layout (bytes)
//   [0..7]    full_barrier   (TMA completion)
//   [8..15]   empty_barrier  (SMEM consumed by UMMA)
//   [16..23]  tmem_barrier   (TMEM result ready)
//   [24..27]  tmem_addr      (uint32_t from tcgen05.alloc)
//   [28..1023] padding
//   [1024..]  data: A then B
static constexpr uint32_t OFF_FULL_BAR  = 0;
static constexpr uint32_t OFF_EMPTY_BAR = 8;
static constexpr uint32_t OFF_TMEM_BAR  = 16;
static constexpr uint32_t OFF_TMEM_ADDR = 24;
static constexpr uint32_t OFF_DATA      = 1024;
static constexpr uint32_t SMEM_SIZE     = OFF_DATA + SMEM_A_SIZE + SMEM_B_SIZE;

// ============================================================
//  PTX helper functions
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

// --- Barrier ---
__device__ __forceinline__ void barrier_init(void* bar, uint32_t count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void barrier_arrive_expect_tx(void* bar, uint32_t tx) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx));
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

// --- TMA ---
__device__ __forceinline__ void prefetch_tma(const void* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(desc));
}
__device__ __forceinline__ void tma_load_2d(
    void* smem, const void* desc, void* bar, int32_t c0, int32_t c1)
{
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t ba = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"(desc), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

// --- TMEM ---
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
    uint32_t col, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// --- tcgen05 fences ---
__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// --- UMMA commit: cta_group::2, multicast to both CTAs ---
__device__ __forceinline__ void umma_commit_2sm(void* bar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

// --- UMMA MMA: cta_group::2, kind::f16 (BF16 × BF16 → FP32) ---
__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum)
{
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

// ============================================================
//  Descriptor builders
// ============================================================

// SmemDescriptor: K-major BF16, SWIZZLE_128B
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t sbo)
{
    uint64_t d = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);              // [0:14)  start_address
    // [16:30) leading_byte_offset = 0 (K-major, 1 atom on K axis)
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;  // [32:46) stride_byte_offset
    d |= (uint64_t)1 << 46;                       // [46:48) version = 1
    // base_offset = 0, lbo_mode = 0
    d |= (uint64_t)2 << 61;                       // [61:64) layout_type = SWIZZLE_128B
    return d;
}

// InstrDescriptor: BF16 × BF16 → FP32, K-major A/B
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N)
{
    // Bit layout (from CUTLASS InstrDescriptor union):
    //   [4:6)   c_format:  FP32 = 1
    //   [7:10)  a_format:  BF16 = 1
    //   [10:13) b_format:  BF16 = 1
    //   [17:23) n_dim = N/8
    //   [24:29) m_dim = M/16
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
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

    // Shared memory
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    uint64_t* full_bar  = reinterpret_cast<uint64_t*>(smem_buf + OFF_FULL_BAR);
    uint64_t* empty_bar = reinterpret_cast<uint64_t*>(smem_buf + OFF_EMPTY_BAR);
    uint64_t* tmem_bar  = reinterpret_cast<uint64_t*>(smem_buf + OFF_TMEM_BAR);
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(smem_buf + OFF_TMEM_ADDR);
    void* smem_a = smem_buf + OFF_DATA;
    void* smem_b = smem_buf + OFF_DATA + SMEM_A_SIZE;

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one()) {
        prefetch_tma(&tma_a);
        prefetch_tma(&tma_b);
    }

    // Initialize barriers
    if (warp_idx == 1 && elect_one()) {
        barrier_init(full_bar,  1);
        barrier_init(empty_bar, 1);
        barrier_init(tmem_bar,  1);
        fence_barrier_init();
    }

    // Allocate TMEM (warp 2)
    if (warp_idx == 2) {
        tmem_alloc_2sm(tmem_addr, TMEM_COLS);
    }

    cluster_sync();

    // Build descriptors (constant across K loop)
    const uint64_t a_desc = make_smem_desc(smem_a, UMMA_SBO);
    const uint64_t b_desc = make_smem_desc(smem_b, UMMA_SBO);
    const uint32_t a_hi = static_cast<uint32_t>(a_desc >> 32);
    const uint32_t b_hi = static_cast<uint32_t>(b_desc >> 32);
    const uint32_t a_lo = static_cast<uint32_t>(a_desc);
    const uint32_t b_lo = static_cast<uint32_t>(b_desc);
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);

    const int32_t m_coord = m_block * BLOCK_M;
    const int32_t n_coord = n_block * BLOCK_N + cta_rank * LOAD_N_PER_CTA;

    const uint32_t num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // ===================== Main K loop =====================
    uint32_t phase = 0;
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        // Wait for SMEM to be free (skip-free on first iteration via parity trick)
        barrier_wait(empty_bar, phase ^ 1);

        // TMA loads
        if (warp_idx == 0 && elect_one()) {
            const int32_t k_coord = kb * BLOCK_K;
            tma_load_2d(smem_a, &tma_a, full_bar, k_coord, m_coord);
            tma_load_2d(smem_b, &tma_b, full_bar, k_coord, n_coord);
            barrier_arrive_expect_tx(full_bar, TMA_BYTES_PER_CTA);
        }

        // Wait for TMA completion
        barrier_wait(full_bar, phase);

        // Ensure both CTAs have loaded
        cluster_sync();

        // UMMA compute (leader CTA, warp 1, elected thread)
        if (is_leader && warp_idx == 1 && elect_one()) {
            tcgen05_fence_after();

            for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                uint64_t ad = ((uint64_t)a_hi << 32) | (a_lo + k * 2);
                uint64_t bd = ((uint64_t)b_hi << 32) | (b_lo + k * 2);
                uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;
                umma_f16_cg2(0, ad, bd, idesc, accum);
            }

            // Signal: SMEM consumed
            umma_commit_2sm(empty_bar);

            // Signal: TMEM result ready (last K tile)
            if (kb == num_k_blocks - 1)
                umma_commit_2sm(tmem_bar);
        }

        phase ^= 1;
    }

    // ===================== Epilogue =====================
    barrier_wait(tmem_bar, 0);
    tcgen05_fence_after();

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

    cluster_sync();

    if (warp_idx == 2) {
        tmem_dealloc_2sm(0, TMEM_COLS);
    }
#endif
}

// ============================================================
//  Host helpers
// ============================================================
static void create_tma_desc(
    CUtensorMap* map, const void* ptr,
    uint64_t dim0, uint64_t dim1,
    uint32_t box0, uint32_t box1)
{
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
    int M, int N, int K)
{
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

// ============================================================
//  Exported C functions (called from Python via ctypes)
// ============================================================

// D(M,N) = A(M,K) @ B(N,K)^T   — all BF16, row-major
extern "C" void gemm_bf16_launch(
    const void* A, const void* B, void* D,
    int M, int N, int K)
{
    launch_kernel(
        reinterpret_cast<__nv_bfloat16*>(D),
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B),
        M, N, K);
}

// Returns average kernel time in milliseconds
extern "C" float benchmark_kernel(
    const void* A, const void* B, void* D,
    int M, int N, int K, int warmup, int iters)
{
    auto* Dp = reinterpret_cast<__nv_bfloat16*>(D);
    auto* Ap = reinterpret_cast<const __nv_bfloat16*>(A);
    auto* Bp = reinterpret_cast<const __nv_bfloat16*>(B);

    for (int i = 0; i < warmup; i++)
        launch_kernel(Dp, Ap, Bp, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        launch_kernel(Dp, Ap, Bp, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}
