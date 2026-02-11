// Debug kernel: same as Level 3 but writes TMEM values directly to global memory
// (bypassing SMEM swizzle + TMA store) to isolate correctness issues.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#ifndef TILE_N
#define TILE_N 160
#endif

static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = TILE_N;
static constexpr uint32_t BLOCK_K = 128;
static constexpr uint32_t UMMA_K  = 32;
static constexpr uint32_t GRAN_K = 128;
static constexpr uint32_t NUM_SF_PER_LOAD = 4;

static constexpr uint32_t SF_BLOCK_M = ((BLOCK_M + 127) / 128) * 128;
static constexpr uint32_t SF_BLOCK_N = ((BLOCK_N + 127) / 128) * 128;

static constexpr uint32_t SMEM_A   = BLOCK_M * BLOCK_K * 1;
static constexpr uint32_t SMEM_B   = BLOCK_N * BLOCK_K * 1;
static constexpr uint32_t SMEM_SFA = SF_BLOCK_M * sizeof(uint32_t);
static constexpr uint32_t SMEM_SFB = SF_BLOCK_N * sizeof(uint32_t);
static constexpr uint32_t SMEM_STAGE = SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB;

static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(uint32_t);
static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(uint32_t);

static constexpr uint32_t NUM_M_WAVES = 1;
static constexpr uint32_t NUM_EPI_STAGES = 2;
static constexpr uint32_t ACCUM_COLS = NUM_EPI_STAGES * NUM_M_WAVES * BLOCK_N;
static constexpr uint32_t SFA_TMEM_COLS = SF_BLOCK_M / 32;
static constexpr uint32_t SFB_TMEM_COLS = SF_BLOCK_N / 32;
static constexpr uint32_t SFA_TMEM_OFF = ACCUM_COLS;
static constexpr uint32_t SFB_TMEM_OFF = SFA_TMEM_OFF + SFA_TMEM_COLS;
static constexpr uint32_t TMEM_COLS_RAW = ACCUM_COLS + SFA_TMEM_COLS + SFB_TMEM_COLS;
static constexpr uint32_t TMEM_COLS =
    TMEM_COLS_RAW <=  32 ?  32 :
    TMEM_COLS_RAW <=  64 ?  64 :
    TMEM_COLS_RAW <= 128 ? 128 :
    TMEM_COLS_RAW <= 256 ? 256 : 512;

static constexpr uint32_t SMEM_CAPACITY = 232448;
// No CD staging needed for debug
static constexpr uint32_t O_A   = 0;
static constexpr uint32_t O_B   = SMEM_A * 4;  // 4 stages
static constexpr uint32_t O_SFA = O_B + SMEM_B * 4;
static constexpr uint32_t O_SFB = O_SFA + SMEM_SFA * 4;
static constexpr uint32_t O_BAR = O_SFB + SMEM_SFB * 4;
static constexpr uint32_t CAPPED_STAGES = 4;
static constexpr uint32_t N_BARS = CAPPED_STAGES * 3 + NUM_EPI_STAGES * 2;
static constexpr uint32_t TOTAL_SMEM = O_BAR + N_BARS * 8 + 4;

static constexpr uint32_t N_MMA_THREADS = 128;
static constexpr uint32_t N_THREADS = 256;

// -- PTX helpers (same as main kernel, abbreviated) --
__device__ __forceinline__ void bar_init(void* bar, uint32_t cnt) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt));
}
__device__ __forceinline__ void bar_wait(void* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done;
    do { asm volatile("{\n\t.reg .pred P;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P;\n\t}\n" : "=r"(done) : "r"(a), "r"(phase)); } while (!done);
}
__device__ __forceinline__ void bar_arrive(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}
__device__ __forceinline__ void bar_expect_tx(void* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void bar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;"); }
__device__ __forceinline__ bool elect_one() {
    uint32_t p; asm volatile("{\n\t.reg .pred P;\n\t"
    "elect.sync _|P, 0xFFFFFFFF;\n\t" "selp.b32 %0, 1, 0, P;\n\t}\n" : "=r"(p)); return p != 0;
}
__device__ __forceinline__ void tma_load(void* dst, const void* desc, void* bar, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"(desc), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
__device__ __forceinline__ void prefetch_tma(const void* d) { asm volatile("prefetch.tensormap [%0];" :: "l"(d)); }
__device__ __forceinline__ void tmem_alloc(uint32_t* s, uint32_t n) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(n));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr, uint32_t n) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(n));
}
__device__ __forceinline__ void tmem_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void tmem_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void tmem_ld_8x(uint32_t col,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t& r4, uint32_t& r5, uint32_t& r6, uint32_t& r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "r"(col));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void umma_commit(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}
__device__ __forceinline__ void umma_fp8(uint32_t tc, uint64_t da, uint64_t db, uint32_t id, uint32_t acc, uint32_t sfa, uint32_t sfb) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(id), "r"(acc), "r"(sfa), "r"(sfb));
}
__device__ __forceinline__ void utccp_copy(uint64_t sd, uint32_t tc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tc), "l"(sd));
}
__device__ __forceinline__ uint32_t ld_s32(const uint32_t* p) {
    uint32_t v, a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v;
}
__device__ __forceinline__ void st_s32(uint32_t* p, uint32_t v) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(v));
}
__device__ __forceinline__ void utccp_transpose(uint32_t* smem, uint32_t lane) {
    uint32_t vals[4];
    for (uint32_t i = 0; i < 4; i++) vals[i] = ld_s32(smem + (i ^ (lane >> 3)) * 32 + lane);
    __syncwarp();
    for (uint32_t i = 0; i < 4; i++) st_s32(smem + lane * 4 + (i ^ (lane >> 3)), vals[i]);
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
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
__device__ __forceinline__ uint32_t make_fp8_idesc() {
    uint32_t d = 0;
    d |= ((BLOCK_N / 8) << 17);
    d |= (1u << 23);
    d |= ((BLOCK_M / 16) << 24);
    return d;
}
__device__ __forceinline__ uint32_t set_sf_ids(uint32_t d, uint32_t a_id, uint32_t b_id) {
    d &= ~(0x3u << 29); d &= ~(0x3u << 4);
    d |= ((a_id & 0x3) << 29); d |= ((b_id & 0x3) << 4);
    return d;
}

// DEBUG: direct global memory write (no swizzle, no TMA store)
__global__ void __cluster_dims__(1, 1, 1)
__launch_bounds__(N_THREADS, 1)
fp8_debug_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_sfa,
    const __grid_constant__ CUtensorMap tma_sfb,
    __nv_bfloat16* D_out,
    uint32_t M, uint32_t N, uint32_t K)
{
    extern __shared__ __align__(1024) uint8_t smem[];

    uint64_t* full_bar    = (uint64_t*)(smem + O_BAR);
    uint64_t* empty_bar   = full_bar + CAPPED_STAGES;
    uint64_t* sf_full_bar = empty_bar + CAPPED_STAGES;
    uint64_t* tf_bar      = sf_full_bar + CAPPED_STAGES;
    uint64_t* te_bar      = tf_bar + NUM_EPI_STAGES;
    uint32_t* tmem_smem   = (uint32_t*)(te_bar + NUM_EPI_STAGES);

    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    if (wid == 1 && elect_one()) {
        for (uint32_t s = 0; s < CAPPED_STAGES; s++) {
            bar_init(&full_bar[s], 1);
            bar_init(&empty_bar[s], 1);
            bar_init(&sf_full_bar[s], 32);
        }
        for (uint32_t e = 0; e < NUM_EPI_STAGES; e++) {
            bar_init(&tf_bar[e], 1);
            bar_init(&te_bar[e], 128);
        }
        bar_fence_init();
    }
    if (wid == 3) tmem_alloc(tmem_smem, TMEM_COLS);
    __syncthreads();

    uint32_t tmem_base = *tmem_smem;
    uint32_t num_kb = (K + BLOCK_K - 1) / BLOCK_K;
    uint32_t num_mb = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t num_nb = (N + BLOCK_N - 1) / BLOCK_N;

    // Process only the first tile for simplicity
    uint32_t mb = blockIdx.x / num_nb;
    uint32_t nb = blockIdx.x % num_nb;
    if (mb >= num_mb) return;

    uint32_t stg = 0, ph = 0;
    auto adv = [&]() { stg = (stg + 1) % CAPPED_STAGES; ph ^= (stg == 0); };

    // Warp 0: Load
    if (wid == 0 && elect_one()) {
        for (uint32_t kb = 0; kb < num_kb; kb++) {
            tma_load(smem + O_A + stg * SMEM_A, &tma_a, &full_bar[stg],
                     (int32_t)(kb * BLOCK_K), (int32_t)(mb * BLOCK_M));
            tma_load(smem + O_B + stg * SMEM_B, &tma_b, &full_bar[stg],
                     (int32_t)(kb * BLOCK_K), (int32_t)(nb * BLOCK_N));
            uint32_t tx = SMEM_A + SMEM_B;
            if (kb % NUM_SF_PER_LOAD == 0) {
                uint32_t sfk = kb / NUM_SF_PER_LOAD;
                tma_load(smem + O_SFA + stg * SMEM_SFA, &tma_sfa, &full_bar[stg],
                         (int32_t)(mb * BLOCK_M), (int32_t)sfk);
                tma_load(smem + O_SFB + stg * SMEM_SFB, &tma_sfb, &full_bar[stg],
                         (int32_t)(nb * BLOCK_N), (int32_t)sfk);
                tx += TMA_SFA_BYTES + TMA_SFB_BYTES;
            }
            bar_expect_tx(&full_bar[stg], tx);
            adv();
        }
    }
    // Warp 2: Transpose
    else if (wid == 2) {
        for (uint32_t kb = 0; kb < num_kb; kb++) {
            bar_wait(&full_bar[stg], ph);
            if (kb % NUM_SF_PER_LOAD == 0) {
                for (uint32_t i = 0; i < SF_BLOCK_M / 128; i++)
                    utccp_transpose((uint32_t*)(smem + O_SFA + stg * SMEM_SFA) + i*128, lane);
                fence_async_smem();
                for (uint32_t i = 0; i < SF_BLOCK_N / 128; i++)
                    utccp_transpose((uint32_t*)(smem + O_SFB + stg * SMEM_SFB) + i*128, lane);
                fence_async_smem();
            }
            bar_arrive(&sf_full_bar[stg]);
            adv();
        }
    }
    // Warp 1: MMA
    else if (wid == 1) {
        static constexpr uint32_t SBO = 8 * BLOCK_K;
        uint32_t base_idesc = make_fp8_idesc();
        uint64_t a_base = make_smem_desc(smem + O_A, SBO);
        uint64_t b_base = make_smem_desc(smem + O_B, SBO);
        uint32_t a_hi = (uint32_t)(a_base >> 32);
        uint32_t b_hi = (uint32_t)(b_base >> 32);
        uint32_t my_a_lo = 0, my_b_lo = 0;
        if (lane < CAPPED_STAGES) {
            my_a_lo = (uint32_t)make_smem_desc(smem + O_A + lane * SMEM_A, SBO);
            my_b_lo = (uint32_t)make_smem_desc(smem + O_B + lane * SMEM_B, SBO);
        }

        uint32_t ai = 0;
        tmem_fence_after();

        for (uint32_t kb = 0; kb < num_kb; kb++) {
            bar_wait(&sf_full_bar[stg], ph);
            tmem_fence_after();

            uint32_t sf_idx = kb % NUM_SF_PER_LOAD;
            if (sf_idx == 0 && elect_one()) {
                for (uint32_t i = 0; i < SF_BLOCK_M / 128; i++) {
                    uint64_t d = make_sf_desc(smem + O_SFA + stg * SMEM_SFA + i * 128 * 4);
                    utccp_copy(d, tmem_base + SFA_TMEM_OFF + i * 4);
                }
                for (uint32_t i = 0; i < SF_BLOCK_N / 128; i++) {
                    uint64_t d = make_sf_desc(smem + O_SFB + stg * SMEM_SFB + i * 128 * 4);
                    utccp_copy(d, tmem_base + SFB_TMEM_OFF + i * 4);
                }
            }
            __syncwarp();

            uint32_t a_lo = __shfl_sync(0xFFFFFFFF, my_a_lo, stg);
            uint32_t b_lo = __shfl_sync(0xFFFFFFFF, my_b_lo, stg);
            uint32_t idesc = set_sf_ids(base_idesc, sf_idx, sf_idx);
            uint32_t accum = (kb > 0) ? 1u : 0u;

            if (elect_one()) {
                for (uint32_t kk = 0; kk < BLOCK_K / UMMA_K; kk++) {
                    uint64_t da = ((uint64_t)a_hi << 32) | (uint64_t)(a_lo + kk * 2);
                    uint64_t db = ((uint64_t)b_hi << 32) | (uint64_t)(b_lo + kk * 2);
                    umma_fp8(tmem_base + ai * NUM_M_WAVES * BLOCK_N, da, db, idesc, accum,
                             tmem_base + SFA_TMEM_OFF, tmem_base + SFB_TMEM_OFF);
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
    // Warp 4-7: DEBUG epilogue - direct global memory write
    else if (wid >= 4) {
        uint32_t lt = threadIdx.x - N_MMA_THREADS;  // 0..127
        uint32_t ai = 0;

        bar_wait(&tf_bar[ai], 0);
        tmem_fence_after();

        // Read TMEM values and write directly to D_out
        uint32_t out_row = mb * BLOCK_M + lt;
        if (out_row < M) {
            for (uint32_t col_start = 0; col_start < BLOCK_N; col_start += 8) {
                uint32_t tmem_col = tmem_base + ai * NUM_M_WAVES * BLOCK_N + col_start;
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                tmem_ld_8x(tmem_col, r0, r1, r2, r3, r4, r5, r6, r7);
                tmem_wait_ld();

                uint32_t out_col = nb * BLOCK_N + col_start;
                if (out_col + 7 < N) {
                    __nv_bfloat16* out = D_out + out_row * N + out_col;
                    out[0] = __float2bfloat16(__uint_as_float(r0));
                    out[1] = __float2bfloat16(__uint_as_float(r1));
                    out[2] = __float2bfloat16(__uint_as_float(r2));
                    out[3] = __float2bfloat16(__uint_as_float(r3));
                    out[4] = __float2bfloat16(__uint_as_float(r4));
                    out[5] = __float2bfloat16(__uint_as_float(r5));
                    out[6] = __float2bfloat16(__uint_as_float(r6));
                    out[7] = __float2bfloat16(__uint_as_float(r7));
                }
            }
        }

        tmem_fence_before();
        bar_arrive(&te_bar[ai]);

        if (wid == 7) tmem_dealloc(tmem_base, TMEM_COLS);
    }
}

extern "C" {
void launch_fp8_debug(
    const void* A, const void* B, const void* SFA, const void* SFB,
    void* D, int M, int N, int K)
{
    uint32_t sf_k = (K + GRAN_K * 4 - 1) / (GRAN_K * 4);

    CUtensorMap tma_a, tma_b, tma_sfa, tma_sfb;
    {
        uint64_t d[2] = {(uint64_t)K, (uint64_t)M}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {BLOCK_K, BLOCK_M}; uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(A), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    {
        uint64_t d[2] = {(uint64_t)K, (uint64_t)N}; uint64_t s[1] = {(uint64_t)K};
        uint32_t b[2] = {BLOCK_K, BLOCK_N}; uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(B), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    {
        uint64_t d[2] = {(uint64_t)M, (uint64_t)sf_k}; uint64_t s[1] = {(uint64_t)M * 4};
        uint32_t b[2] = {BLOCK_M, 1}; uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_sfa, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, const_cast<void*>(SFA), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    {
        uint64_t d[2] = {(uint64_t)N, (uint64_t)sf_k}; uint64_t s[1] = {(uint64_t)N * 4};
        uint32_t b[2] = {BLOCK_N, 1}; uint32_t e[2] = {1, 1};
        cuTensorMapEncodeTiled(&tma_sfb, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, const_cast<void*>(SFB), d, s, b, e,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    uint32_t num_mb = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t num_nb = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t num_ctas = num_mb * num_nb;

    cudaFuncSetAttribute(fp8_debug_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_ctas, 1, 1);
    config.blockDim = dim3(N_THREADS, 1, 1);
    config.dynamicSmemBytes = TOTAL_SMEM;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    config.attrs = attrs; config.numAttrs = 1;

    uint32_t Mu = M, Nu = N, Ku = K;
    cudaLaunchKernelEx(&config, fp8_debug_kernel, tma_a, tma_b, tma_sfa, tma_sfb, (__nv_bfloat16*)D, Mu, Nu, Ku);
}
}
