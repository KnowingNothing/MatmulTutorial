// Minimal UMMA test: bypass TMA, manually store FP8 data in SMEM, run UMMA, check result
// Tests whether UMMA + SMEM descriptors work correctly in isolation.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = 128;
static constexpr uint32_t BLOCK_K = 128;
static constexpr uint32_t UMMA_K  = 32;

// SMEM layout: A[128][128] + B[128][128] + SFA[128] + SFB[128] + barriers + tmem_ptr
static constexpr uint32_t SMEM_A   = BLOCK_M * BLOCK_K;    // 16384
static constexpr uint32_t SMEM_B   = BLOCK_N * BLOCK_K;    // 16384
static constexpr uint32_t SMEM_SFA = 128 * sizeof(uint32_t); // 512
static constexpr uint32_t SMEM_SFB = 128 * sizeof(uint32_t); // 512

static constexpr uint32_t O_A   = 0;
static constexpr uint32_t O_B   = SMEM_A;
static constexpr uint32_t O_SFA = O_B + SMEM_B;
static constexpr uint32_t O_SFB = O_SFA + SMEM_SFA;
static constexpr uint32_t O_BAR = O_SFB + SMEM_SFB;

// Barriers: sf_full[1], tmem_full[1], tmem_empty[1] = 3 barriers
static constexpr uint32_t N_BARS = 3;
static constexpr uint32_t TOTAL_SMEM = O_BAR + N_BARS * 8 + 4; // +4 for tmem_ptr

// TMEM
static constexpr uint32_t ACCUM_COLS   = BLOCK_N;            // 128
static constexpr uint32_t SFA_TMEM_OFF = ACCUM_COLS;         // 128
static constexpr uint32_t SFB_TMEM_OFF = SFA_TMEM_OFF + 4;  // 132
static constexpr uint32_t TMEM_COLS    = 256;

// --- PTX helpers ---
__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "elect.sync _|P, 0xFFFFFFFF;\n\t"
                 "selp.b32 %0, 1, 0, P;\n\t}\n" : "=r"(p));
    return p != 0;
}

__device__ __forceinline__ void tmem_alloc(uint32_t* smem, uint32_t ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;");
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

#ifndef USE_SWIZZLE
#define USE_SWIZZLE 1
#endif

// SMEM descriptor for A/B with SWIZZLE_NONE
__device__ __forceinline__ uint64_t make_smem_desc_noswizzle(void* ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;    // version = 1
    // layout_type = 0 (SWIZZLE_NONE), no need to set
    return d;
}

// SMEM descriptor for A/B with SWIZZLE_128B
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;    // version = 1
    d |= (uint64_t)2 << 61;    // SWIZZLE_128B
    return d;
}

// SF descriptor: SWIZZLE_NONE, SBO=128
__device__ __forceinline__ uint64_t make_sf_desc(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((128u >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

// FP8 instruction descriptor
__device__ __forceinline__ uint32_t make_fp8_idesc() {
    uint32_t d = 0;
    d |= ((BLOCK_N / 8) << 17);
    d |= (1u << 23);             // E8M0
    d |= ((BLOCK_M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t set_sf_ids(uint32_t d, uint32_t a_id, uint32_t b_id) {
    d &= ~(0x3u << 29);
    d &= ~(0x3u << 4);
    d |= ((a_id & 0x3) << 29);
    d |= ((b_id & 0x3) << 4);
    return d;
}

// Barrier helpers
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

__device__ __forceinline__ void umma_commit(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void bar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

// Apply SWIZZLE_128B pattern: for row r, bank b -> physical bank (b XOR r%8)
__device__ __forceinline__ uint32_t swizzle_128b_addr(uint32_t row, uint32_t col) {
    // For FP8 (1 byte/element), 128 bytes per row, 16 bytes per bank
    uint32_t bank = col / 16;
    uint32_t in_bank = col % 16;
    uint32_t phys_bank = bank ^ (row % 8);
    return row * 128 + phys_bank * 16 + in_bank;
}

__global__ void __cluster_dims__(1, 1, 1)
__launch_bounds__(256, 1)
test_umma_kernel(
    const uint8_t* __restrict__ A_global,  // [M, K] FP8, K-contiguous
    const uint8_t* __restrict__ B_global,  // [N, K] FP8, K-contiguous
    float* __restrict__ D_global,          // [M, N] FP32 output for debugging
    uint32_t M, uint32_t N, uint32_t K)
{
    extern __shared__ __align__(1024) uint8_t smem[];

    uint64_t* bars = (uint64_t*)(smem + O_BAR);
    uint64_t* commit_bar = &bars[0];  // barrier for UMMA commit
    uint32_t* tmem_smem = (uint32_t*)(bars + N_BARS);

    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    // Initialize commit barrier (count=1 for one commit signal)
    if (threadIdx.x == 0) {
        bar_init(commit_bar, 1);
    }
    __syncthreads();
    bar_fence_init();

    // Warp 3: allocate TMEM
    if (wid == 3)
        tmem_alloc(tmem_smem, TMEM_COLS);
    __syncthreads();
    uint32_t tmem_base = *tmem_smem;

    if (threadIdx.x == 0)
        printf("test_umma: tmem_base=%u\n", tmem_base);

    // All threads cooperatively store A and B into SMEM
#if USE_SWIZZLE
    // With SWIZZLE_128B: A[m][k] stored at smem[O_A + swizzle_128b_addr(m, k)]
    for (uint32_t idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
        uint32_t m = idx / BLOCK_K;
        uint32_t k = idx % BLOCK_K;
        uint32_t addr = swizzle_128b_addr(m, k);
        smem[O_A + addr] = A_global[m * K + k];
    }
    for (uint32_t idx = threadIdx.x; idx < BLOCK_N * BLOCK_K; idx += blockDim.x) {
        uint32_t n = idx / BLOCK_K;
        uint32_t k = idx % BLOCK_K;
        uint32_t addr = swizzle_128b_addr(n, k);
        smem[O_B + addr] = B_global[n * K + k];
    }
#else
    // Without swizzle: linear layout A[m][k] at smem[O_A + m*128 + k]
    for (uint32_t idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
        smem[O_A + idx] = A_global[idx];
    }
    for (uint32_t idx = threadIdx.x; idx < BLOCK_N * BLOCK_K; idx += blockDim.x) {
        smem[O_B + idx] = B_global[idx];
    }
#endif

    // Store scale factors (all 0x7F7F7F7F = 1.0 in E8M0)
    for (uint32_t idx = threadIdx.x; idx < 128; idx += blockDim.x) {
        ((uint32_t*)(smem + O_SFA))[idx] = 0x7F7F7F7Fu;
        ((uint32_t*)(smem + O_SFB))[idx] = 0x7F7F7F7Fu;
    }
    __syncthreads();

    // Warp 2: transpose scale factors
    if (wid == 2) {
        utccp_transpose((uint32_t*)(smem + O_SFA), lane);
        fence_async_smem();
        utccp_transpose((uint32_t*)(smem + O_SFB), lane);
        fence_async_smem();
    }
    __syncthreads();

    // Warp 1: UTCCP copy + UMMA
    if (wid == 1) {
        tmem_fence_after();

        // UTCCP copy scale factors to TMEM
        if (elect_one()) {
            uint64_t sfa_d = make_sf_desc(smem + O_SFA);
            utccp_copy(sfa_d, tmem_base + SFA_TMEM_OFF);
            uint64_t sfb_d = make_sf_desc(smem + O_SFB);
            utccp_copy(sfb_d, tmem_base + SFB_TMEM_OFF);
        }
        __syncwarp();

        // Build SMEM descriptors
        // SBO for K-major: stride between 8-row groups = num_non_contiguous * BLOCK_K
        // num_non_contiguous = 128 / atom_base = 128 / 16 = 8
        static constexpr uint32_t SBO = 8 * BLOCK_K * 1;  // 1024
#if USE_SWIZZLE
        uint64_t a_desc = make_smem_desc(smem + O_A, SBO);
        uint64_t b_desc = make_smem_desc(smem + O_B, SBO);
#else
        uint64_t a_desc = make_smem_desc_noswizzle(smem + O_A, SBO);
        uint64_t b_desc = make_smem_desc_noswizzle(smem + O_B, SBO);
#endif
        // Split descriptor at bit 32 (NOT bit 14!)
        // Lower 32 bits contain start_address + LBO
        // Upper 32 bits contain SBO + version + layout_type
        uint32_t a_lo = (uint32_t)(a_desc);
        uint32_t a_hi = (uint32_t)(a_desc >> 32);
        uint32_t b_lo = (uint32_t)(b_desc);
        uint32_t b_hi = (uint32_t)(b_desc >> 32);

        uint32_t base_idesc = make_fp8_idesc();
        uint32_t idesc = set_sf_ids(base_idesc, 0, 0);  // sf_id = 0

        if (elect_one()) {
            #pragma unroll
            for (uint32_t kk = 0; kk < BLOCK_K / UMMA_K; kk++) {
                uint64_t da = ((uint64_t)a_hi << 32) | (uint64_t)(a_lo + kk * 2);
                uint64_t db = ((uint64_t)b_hi << 32) | (uint64_t)(b_lo + kk * 2);
                uint32_t accum = (kk > 0) ? 1u : 0u;
                // Update SF IDs for each K-step
                uint32_t kid = set_sf_ids(base_idesc, kk, kk);
                umma_fp8(tmem_base, da, db, kid, accum,
                         tmem_base + SFA_TMEM_OFF, tmem_base + SFB_TMEM_OFF);
            }
            // Commit all UMMA operations and signal the barrier
            umma_commit(commit_bar);
        }
    }

    // Wait for UMMA to complete
    if (wid == 0 && elect_one()) {
        bar_wait(commit_bar, 0);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    // All epilogue warps read TMEM and write to global
    // Each thread reads one FP32 value at a time
    // TMEM layout: column = N index, row = M index (implicit by warp/lane)
    // Thread mapping: threadIdx.x % 128 -> TMEM row -> M index
    uint32_t my_m = threadIdx.x % 128;  // wraps around for warps 4-7 same as 0-3
    for (uint32_t n_base = 0; n_base < BLOCK_N; n_base += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        tmem_ld_8x(tmem_base + n_base, r0, r1, r2, r3, r4, r5, r6, r7);
        tmem_wait_ld();

        // Only warps 0-3 write (they cover all 128 M-rows)
        if (wid < 4) {
            D_global[my_m * N + n_base + 0] = __uint_as_float(r0);
            D_global[my_m * N + n_base + 1] = __uint_as_float(r1);
            D_global[my_m * N + n_base + 2] = __uint_as_float(r2);
            D_global[my_m * N + n_base + 3] = __uint_as_float(r3);
            D_global[my_m * N + n_base + 4] = __uint_as_float(r4);
            D_global[my_m * N + n_base + 5] = __uint_as_float(r5);
            D_global[my_m * N + n_base + 6] = __uint_as_float(r6);
            D_global[my_m * N + n_base + 7] = __uint_as_float(r7);
        }
    }
    __syncthreads();

    // Dealloc TMEM
    if (wid == 3)
        tmem_dealloc(tmem_base, TMEM_COLS);
}

extern "C" {
void launch_test_umma(
    const void* A, const void* B, void* D,
    int M, int N, int K)
{
    cudaFuncSetAttribute(test_umma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3(1, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = TOTAL_SMEM;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    config.attrs    = attrs;
    config.numAttrs = 1;

    auto err = cudaLaunchKernelEx(&config, test_umma_kernel,
        (const uint8_t*)A, (const uint8_t*)B, (float*)D,
        (uint32_t)M, (uint32_t)N, (uint32_t)K);
    if (err != cudaSuccess)
        printf("Launch error: %s\n", cudaGetErrorString(err));
}
} // extern "C"
