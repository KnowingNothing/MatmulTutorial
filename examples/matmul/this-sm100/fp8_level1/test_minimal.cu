// Minimal test kernel for FP8 GEMM debugging
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// Minimal barrier ops
__device__ __forceinline__ void bar_init(void* bar, uint32_t cnt) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt));
}
__device__ __forceinline__ void bar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}
__device__ __forceinline__ void bar_wait(void* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done;
    do {
        asm volatile("{\n\t.reg .pred P;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
            "selp.b32 %0, 1, 0, P;\n\t}\n"
            : "=r"(done) : "r"(a), "r"(phase));
    } while (!done);
}
__device__ __forceinline__ void bar_expect_tx(void* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void bar_arrive(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}
__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "elect.sync _|P, 0xFFFFFFFF;\n\t"
                 "selp.b32 %0, 1, 0, P;\n\t}\n" : "=r"(p));
    return p != 0;
}
__device__ __forceinline__ void tmem_alloc(uint32_t* smem, uint32_t ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

// TMA load
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

// UMMA commit
__device__ __forceinline__ void umma_commit(void* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

// FP8 UMMA
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

// SMEM desc
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

// SF SMEM desc
__device__ __forceinline__ uint64_t make_sf_desc(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((128u >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}

// UTCCP copy
__device__ __forceinline__ void utccp_copy(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(smem_desc));
}

// UTCCP transpose
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
    for (uint32_t i = 0; i < 4; i++)
        vals[i] = ld_s32(smem + (i ^ (lane >> 3)) * 32 + lane);
    __syncwarp();
    for (uint32_t i = 0; i < 4; i++)
        st_s32(smem + lane * 4 + (i ^ (lane >> 3)), vals[i]);
}

// TMEM load
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

// TMA Store
__device__ __forceinline__ void tma_store(void* src, const void* desc,
                                           int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(src);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                 " [%0, {%1, %2}], [%3];"
                 :: "l"(desc), "r"(c0), "r"(c1), "r"(sa) : "memory");
}

__device__ __forceinline__ void tmem_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tmem_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(uint32_t a, uint32_t b) {
    __nv_bfloat16 ha = __float2bfloat16(__uint_as_float(a));
    __nv_bfloat16 hb = __float2bfloat16(__uint_as_float(b));
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(uint16_t*)&ha), "h"(*(uint16_t*)&hb));
    return r;
}
__device__ __forceinline__ void st_s128(uint32_t addr,
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

// =================================================================
// STEP 1: Just test TMEM alloc/dealloc
// =================================================================
__global__ void __cluster_dims__(1, 1, 1)
__launch_bounds__(256, 1)
test_step1_kernel() {
    __shared__ uint32_t tmem_addr;
    uint32_t wid = threadIdx.x / 32;
    if (wid == 0 && elect_one()) {
        tmem_alloc(&tmem_addr, 512);
    }
    __syncthreads();
    uint32_t tb = tmem_addr;
    if (threadIdx.x == 0)
        printf("Step1: tmem_base=%u\n", tb);
    if (wid == 0 && elect_one())
        tmem_dealloc(tb, 512);
}

// =================================================================
// STEP 2: Test TMA load + barrier
// =================================================================
__global__ void __cluster_dims__(1, 1, 1)
test_step2_kernel(const __grid_constant__ CUtensorMap tma_a, int K) {
    __shared__ __align__(128) uint8_t smem[32768];
    uint64_t* bar = (uint64_t*)(smem + 16384);

    if (threadIdx.x == 0) {
        bar_init(bar, 1);
        bar_fence_init();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Load A[0:128, 0:128] via TMA
        tma_load(smem, &tma_a, bar, 0, 0);
        bar_expect_tx(bar, 128*128);  // 16384 bytes
    }

    // All threads wait for TMA
    if (threadIdx.x < 32)
        bar_wait(bar, 0);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Step2: TMA load done, first byte=%u\n", (uint32_t)smem[0]);
    }
}

// =================================================================
// STEP 3: Test UMMA + commit
// =================================================================
static constexpr uint32_t BK = 128;
static constexpr uint32_t BM = 128;
static constexpr uint32_t BN = 128;
static constexpr uint32_t UK = 32;

__global__ void __cluster_dims__(1, 1, 1)
__launch_bounds__(256, 1)
test_step3_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_sfa,
    const __grid_constant__ CUtensorMap tma_sfb)
{
    // SMEM: A(16K) + B(16K) + SFA(512) + SFB(512) + bar(24) + tmem_ptr(4)
    __shared__ __align__(1024) uint8_t smem[34000];

    uint8_t* s_a   = smem;
    uint8_t* s_b   = smem + 16384;
    uint32_t* s_sfa = (uint32_t*)(smem + 32768);
    uint32_t* s_sfb = (uint32_t*)(smem + 32768 + 512);
    uint64_t* full_bar  = (uint64_t*)(smem + 33792);
    uint64_t* empty_bar = (uint64_t*)(smem + 33800);
    uint64_t* tmem_bar  = (uint64_t*)(smem + 33808);
    uint32_t* tmem_ptr  = (uint32_t*)(smem + 33816);

    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;

    // Init
    if (wid == 0 && elect_one()) {
        bar_init(full_bar, 1);
        bar_init(empty_bar, 1);
        bar_init(tmem_bar, 32);
        bar_fence_init();
    }
    if (wid == 1 && elect_one())
        tmem_alloc(tmem_ptr, 512);
    __syncthreads();

    uint32_t tb = *tmem_ptr;

    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Step3: tmem_base=%u, starting TMA+UMMA test\n", tb);

    if (blockIdx.x != 0) {
        // Only block 0 does work
        if (wid == 1 && elect_one())
            tmem_dealloc(tb, 512);
        return;
    }

    // === TMA Load (warp 0, 1 thread) ===
    if (wid == 0 && elect_one()) {
        // Load A, B, SFA, SFB
        tma_load(s_a, &tma_a, full_bar, 0, 0);
        tma_load(s_b, &tma_b, full_bar, 0, 0);
        tma_load(s_sfa, &tma_sfa, full_bar, 0, 0);
        tma_load(s_sfb, &tma_sfb, full_bar, 0, 0);
        bar_expect_tx(full_bar, 16384 + 16384 + 512 + 512);
    }

    // === Warp 2: Wait for TMA, transpose SF, signal ===
    if (wid == 2) {
        bar_wait(full_bar, 0);
        utccp_transpose(s_sfa, lane);
        fence_async_smem();
        utccp_transpose(s_sfb, lane);
        fence_async_smem();
        bar_arrive(tmem_bar);
    }

    // === Warp 1: Wait for SF, UTCCP copy, UMMA ===
    if (wid == 1) {
        bar_wait(tmem_bar, 0);
        tmem_fence_after();

        // UTCCP copy SF to TMEM
        if (elect_one()) {
            uint64_t sfa_d = make_sf_desc(s_sfa);
            utccp_copy(sfa_d, tb + 256);
            uint64_t sfb_d = make_sf_desc(s_sfb);
            utccp_copy(sfb_d, tb + 260);
        }
        __syncwarp();

        // Build descriptors
        uint64_t a_desc = make_smem_desc(s_a, 1024);
        uint64_t b_desc = make_smem_desc(s_b, 1024);
        uint32_t a_lo = (uint32_t)(a_desc & 0x3FFF);
        uint32_t a_hi = (uint32_t)(a_desc >> 14);
        uint32_t b_lo = (uint32_t)(b_desc & 0x3FFF);
        uint32_t b_hi = (uint32_t)(b_desc >> 14);

        // Instruction descriptor: E4M3, M=128, N=128, K-major, E8M0 scale
        uint32_t idesc = 0;
        idesc |= ((BN / 8) << 17);
        idesc |= (1u << 23);
        idesc |= ((BM / 16) << 24);

        // Issue 4 UMMAs
        if (elect_one()) {
            for (uint32_t k = 0; k < BK / UK; k++) {
                uint64_t da = ((uint64_t)a_hi << 14) | (uint64_t)(a_lo + k * 2);
                uint64_t db = ((uint64_t)b_hi << 14) | (uint64_t)(b_lo + k * 2);
                uint32_t accum = (k > 0) ? 1u : 0u;
                umma_fp8(tb, da, db, idesc, accum, tb + 256, tb + 260);
            }
            // Commit to empty_bar
            umma_commit(empty_bar);
        }
    }

    // Wait for UMMA to finish via empty_bar
    if (wid == 3) {
        bar_wait(empty_bar, 0);
        tmem_fence_after();

        // Read first value from TMEM
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        tmem_ld_8x(tb, r0, r1, r2, r3, r4, r5, r6, r7);
        tmem_wait_ld();

        if (elect_one()) {
            printf("Step3: UMMA done! TMEM[0] = %f, %f, %f, %f\n",
                   __uint_as_float(r0), __uint_as_float(r1),
                   __uint_as_float(r2), __uint_as_float(r3));
        }
    }

    __syncthreads();
    if (wid == 1 && elect_one())
        tmem_dealloc(tb, 512);
}

extern "C" {

void test_step1() {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(1, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = a; cfg.numAttrs = 1;
    auto err = cudaLaunchKernelEx(&cfg, test_step1_kernel);
    if (err != cudaSuccess) printf("Step1 launch error: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Step1 sync error: %s\n", cudaGetErrorString(err));
}

void test_step2(const void* A, int M, int K) {
    CUtensorMap tma_a;
    uint64_t d[2] = {(uint64_t)K, (uint64_t)M};
    uint64_t s[1] = {(uint64_t)K};
    uint32_t b[2] = {128, 128};
    uint32_t e[2] = {1, 1};
    cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, const_cast<void*>(A), d, s, b, e,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(1, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = 32768;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs; cfg.numAttrs = 1;

    cudaFuncSetAttribute(test_step2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);

    auto err = cudaLaunchKernelEx(&cfg, test_step2_kernel, tma_a, K);
    if (err != cudaSuccess) printf("Step2 launch error: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Step2 sync error: %s\n", cudaGetErrorString(err));
}

void test_step3(const void* A, const void* B, const void* SFA, const void* SFB,
                int M, int N, int K) {
    CUtensorMap tma_a, tma_b, tma_sfa, tma_sfb;
    uint32_t sf_k = (K + 511) / 512;

    // A
    { uint64_t d[2]={K,M}; uint64_t s[1]={K}; uint32_t b[2]={128,128}; uint32_t e[2]={1,1};
      cuTensorMapEncodeTiled(&tma_a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, const_cast<void*>(A), d, s, b, e,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); }
    // B
    { uint64_t d[2]={K,N}; uint64_t s[1]={K}; uint32_t b[2]={128,128}; uint32_t e[2]={1,1};
      cuTensorMapEncodeTiled(&tma_b, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, const_cast<void*>(B), d, s, b, e,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); }
    // SFA
    { uint64_t d[2]={M,sf_k}; uint64_t s[1]={(uint64_t)M*4}; uint32_t b[2]={128,1}; uint32_t e[2]={1,1};
      cuTensorMapEncodeTiled(&tma_sfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
        2, const_cast<void*>(SFA), d, s, b, e,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); }
    // SFB
    { uint64_t d[2]={N,sf_k}; uint64_t s[1]={(uint64_t)N*4}; uint32_t b[2]={128,1}; uint32_t e[2]={1,1};
      cuTensorMapEncodeTiled(&tma_sfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
        2, const_cast<void*>(SFB), d, s, b, e,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); }

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3(1, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaFuncSetAttribute(test_step3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 34000);

    auto err = cudaLaunchKernelEx(&config, test_step3_kernel,
        tma_a, tma_b, tma_sfa, tma_sfb);
    if (err != cudaSuccess) printf("Step3 launch error: %s\n", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Step3 sync error: %s\n", cudaGetErrorString(err));
}

} // extern "C"
