/*
 * Flash Attention 3 forward, SM90-only, pure CUDA implementation.
 * Config: bf16, headdim=128, causal, varlen.
 * No CuTe/CUTLASS dependency - all code is hand-written CUDA.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>  // For cuTensorMapEncodeTiled (CUDA Driver API)
#include <cstdint>
#include <cstdio>
#include <cmath>

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define CHECK_CU(call)                                                                                    \
    do {                                                                                                  \
        CUresult status_ = call;                                                                          \
        if (status_ != CUDA_SUCCESS) {                                                                    \
            const char* err_str;                                                                          \
            cuGetErrorString(status_, &err_str);                                                          \
            fprintf(stderr, "CUDA driver error (%s:%d): %s\n", __FILE__, __LINE__, err_str);              \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

// ============================================================================
// Constants
// ============================================================================
static constexpr int kStages = 2;
static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;
static constexpr int kHeadDim = 128;
// static constexpr bool kIsCausal = true;

// Cluster shape (1x1x1 = no multi-CTA cluster)
// static constexpr int kClusterM = 1;
// static constexpr int kClusterN = 1;

using Element = __nv_bfloat16;

// ============================================================================
// Epilogue constants
// ============================================================================
// Smem size for epilogue O buffer (128x128 bf16 elements)
static constexpr int kSmemOSize = kBlockM * kHeadDim;  // 16384

// Number of epilogue threads (2 warpgroups = 256 threads)
static constexpr int kNumEpilogueThreads = 256;

// ============================================================================
// WGMMA descriptor creation
// ============================================================================

// Create WGMMA descriptor for K-major layout (A operand, row-major)
// ============================================================================
// WGMMA PTX operations
// ============================================================================

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N = 0>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// Fence operand to help compiler with register allocation (FA3 style)
__device__ __forceinline__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

__device__ __forceinline__ void warpgroup_fence_operand(uint32_t& reg) {
    asm volatile("" : "+r"(reg) :: "memory");
}

template<int N>
__device__ __forceinline__ void fence_operand_array(float (&arr)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        warpgroup_fence_operand(arr[i]);
    }
}

// Prefetch TMA descriptor (FA3 style)
__device__ __forceinline__ void prefetch_tma_descriptor(const void* desc_ptr) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    asm volatile("prefetch.tensormap [%0];" :: "l"(gmem_int_desc) : "memory");
}

// WGMMA m64n128k16: A[64,16] @ B[16,128] = D[64,128]
// Output: 64 floats per thread (64*128 / 128 threads)
// layoutB = 0 (row-major, N contiguous)
__device__ __forceinline__ void wgmma_m64n128k16_bf16_ss(
    float* d,           // Output: 64 floats in registers
    uint64_t desc_a,    // Descriptor for A in smem
    uint64_t desc_b,    // Descriptor for B in smem
    int scale_d = 1     // 1 = accumulate, 0 = overwrite
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " p,    1,    1,    0,    0;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d)
    );
}

// ============================================================================
// WGMMA RS (Register-Shared) operations for P@V
// ============================================================================
// RS mode: A operand from registers, B operand from shared memory
// This is more efficient than SS when A is already in registers (like P after softmax)
//
// For m64n128k16 RS with bf16:
// - A (P matrix): 4 x uint32_t registers per thread = 8 bf16 values
//   Layout: A is [64, 16], each thread holds 8 elements
// - B (V matrix): descriptor pointing to smem, [16, 128] tile
// - D (O accumulator): 64 floats per thread
//
// A register layout for wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16:
// Each of 128 threads holds 4 uint32_t (8 bf16) for the [64, 16] A matrix
// The layout matches the accumulator C layout from Q@K^T after conversion

// WGMMA m64n128k16 RS: A[64,16] from regs @ B[16,128] from smem = D[64,128]
// A is in registers (4 x uint32_t = 8 bf16 per thread)
// B is col-major in smem (K dimension contiguous)
__device__ __forceinline__ void wgmma_m64n128k16_bf16_rs(
    float* d,           // Output: 64 floats in registers
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,  // A in registers (4 x uint32_t)
    uint64_t desc_b,    // Descriptor for B in smem
    int scale_d = 1     // 1 = accumulate, 0 = overwrite
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " p,    1,    1,    1;\n"  // scaleA=1, scaleB=1, layoutB=1 (col-major/K-major)
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "l"(desc_b), "r"(scale_d)
    );
}

// ============================================================================
// TMA descriptor creation (host-side)
// ============================================================================

// Create a 3D TMA descriptor for K/V tensors
static void create_tma_descriptor_3d(
    CUtensorMap* tensor_map,
    void const* global_address,
    int64_t dim0,  // innermost (headdim)
    int64_t dim1,  // middle (seqlen)
    int64_t dim2,  // outermost (heads)
    int64_t stride1_bytes,
    int64_t stride2_bytes,
    int tile_dim0,
    int tile_dim1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B
) {
    uint64_t globalDim[3] = {(uint64_t)dim0, (uint64_t)dim1, (uint64_t)dim2};
    uint64_t globalStrides[2] = {(uint64_t)stride1_bytes, (uint64_t)stride2_bytes};
    uint32_t boxDim[3] = {(uint32_t)tile_dim0, (uint32_t)tile_dim1, 1};
    uint32_t elementStrides[3] = {1, 1, 1};
    
    CHECK_CU(cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        (void*)global_address,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE  // No OOB fill - we handle bounds via softmax mask
    ));
}

// ============================================================================
// TMA load operations (device-side PTX)
// ============================================================================

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// TMA load 3D tile
__device__ __forceinline__ void tma_load_3d(
    void const* desc_ptr,
    uint64_t* mbar_ptr,
    void* smem_ptr,
    int32_t crd0,
    int32_t crd1,
    int32_t crd2
) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
    
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory"
    );
}

// TMA cache hints for SM90
// EVICT_LAST: Keep data in L2 cache as long as possible (good for V which is reused)
// static constexpr uint64_t kCacheHintEvictNormal = 0x1000000000000000ULL;
// static constexpr uint64_t kCacheHintEvictFirst  = 0x12F0000000000000ULL;
static constexpr uint64_t kCacheHintEvictLast   = 0x14F0000000000000ULL;

// TMA load 3D tile with L2 cache hint
__device__ __forceinline__ void tma_load_3d_cache_hint(
    void const* desc_ptr,
    uint64_t* mbar_ptr,
    void* smem_ptr,
    int32_t crd0,
    int32_t crd1,
    int32_t crd2,
    uint64_t cache_hint
) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
    
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
        : "memory"
    );
}

// ============================================================================
// Utility: FastDivmod for efficient integer division
// Uses multiply-high + shift instead of expensive integer division
// ============================================================================
struct FastDivmod {
    int divisor;
    unsigned int multiplier;
    unsigned int shift_right;
    
    __host__ __device__ FastDivmod() : divisor(1), multiplier(0), shift_right(0) {}
    
    __host__ __device__ FastDivmod(int d) : divisor(d), multiplier(0), shift_right(0) {
        if (d != 1) {
            // Find ceil(log2(d))
            unsigned int log2_ceil = 0;
            unsigned int d_unsigned = (unsigned int)d;
            while ((1u << log2_ceil) < d_unsigned) ++log2_ceil;
            
            // p = 31 + ceil(log2(d))
            unsigned int p = 31 + log2_ceil;
            
            // Compute multiplier = ceil(2^p / d)
            multiplier = (unsigned int)(((1ull << p) + d_unsigned - 1) / d_unsigned);
            shift_right = p - 32;
        }
    }
    
    __device__ __forceinline__ int divide(int dividend) const {
        if (divisor == 1) return dividend;
        #if defined(__CUDA_ARCH__)
        return __umulhi((unsigned int)dividend, multiplier) >> shift_right;
        #else
        return (int)((((unsigned long long)dividend * multiplier) >> 32) >> shift_right);
        #endif
    }
    
    __device__ __forceinline__ int divmod(int &remainder, int dividend) const {
        int quotient = divide(dividend);
        remainder = dividend - quotient * divisor;
        return quotient;
    }
};

// ============================================================================
// Device info cache
// ============================================================================
static int s_cached_device = -1;
static int s_cached_num_sm = 0;

static void ensure_device_info(int &num_sm) {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        num_sm = 132;  // Default for H100
        return;
    }
    
    // Cache device properties (cudaGetDeviceProperties is very slow ~1ms)
    if (device == s_cached_device && s_cached_num_sm > 0) {
        num_sm = s_cached_num_sm;
        return;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        num_sm = 132;  // Default for H100
        return;
    }
    
    s_cached_device = device;
    s_cached_num_sm = prop.multiProcessorCount;
    num_sm = s_cached_num_sm;
}

// ============================================================================
// Scheduler metadata computation
// ============================================================================
static void compute_scheduler_meta(
    int batch, int nheads, int max_seqlen_q, int max_seqlen_k,
    int*& d_num_splits,
    int*& d_num_m_blocks,
    int*& d_batch_idx,
    int*& d_nheads_l2
) {
    int b_rounded = (batch + 3) & ~3;
    int num_m_blocks = (max_seqlen_q + kBlockM - 1) >> 7;  // / 128
    int num_n_blocks = (max_seqlen_k + kBlockN - 1) >> 7;  // / 128
    if (num_n_blocks < 1) num_n_blocks = 1;
    
    // Estimate how many heads fit in L2 cache
    int size_one_kvblock = kBlockN * kHeadDim * 2 * 2;  // K + V, bf16
    // TODO: 40MB is A100's L2 size, H100/H800 has 50MB. Check if using actual L2 size improves performance.
    int max_kv_in_l2 = 8 * 1024 * 1024 / size_one_kvblock;
    int nheads_l2 = num_n_blocks * 16 <= max_kv_in_l2 ? 16 
                  : num_n_blocks * 8 <= max_kv_in_l2 ? 8
                  : num_n_blocks * 4 <= max_kv_in_l2 ? 4 
                  : num_n_blocks * 2 <= max_kv_in_l2 ? 2 : 1;
    if (nheads_l2 > nheads) nheads_l2 = nheads;

    size_t meta_bytes = (size_t)(4 * b_rounded) * sizeof(int);
    
    // Static cache for metadata
    static int* s_d_meta = nullptr;
    static int* s_h_meta = nullptr;
    static int s_batch = -1, s_q = -1, s_k = -1, s_nh = -1;
    
    bool hit = s_d_meta && s_batch == batch && s_q == max_seqlen_q && s_k == max_seqlen_k && s_nh == nheads;
    if (!hit) {
        if (s_d_meta) cudaFree(s_d_meta);
        if (s_h_meta) free(s_h_meta);
        s_h_meta = (int*)malloc(meta_bytes);
        CHECK_CUDA(cudaMalloc(&s_d_meta, meta_bytes));
        s_batch = batch; s_q = max_seqlen_q; s_k = max_seqlen_k; s_nh = nheads;
        
        for (int i = 0; i < b_rounded; i++) {
            s_h_meta[i] = (i < batch) ? 1 : 0;                      // num_splits
            s_h_meta[b_rounded + i] = (i < batch) ? num_m_blocks : 0;  // num_m_blocks
            s_h_meta[2*b_rounded + i] = (i < batch) ? i : 0;           // batch_idx
            s_h_meta[3*b_rounded + i] = (i < batch) ? nheads_l2 : 0;   // nheads_l2
        }
        CHECK_CUDA(cudaMemcpy(s_d_meta, s_h_meta, meta_bytes, cudaMemcpyHostToDevice));
    }
    
    d_num_splits = s_d_meta;
    d_num_m_blocks = s_d_meta + b_rounded;
    d_batch_idx = s_d_meta + 2 * b_rounded;
    d_nheads_l2 = s_d_meta + 3 * b_rounded;
}

// ============================================================================
// Thread/Warp/Warpgroup constants
// ============================================================================
static constexpr int kWarpSize = 32;
static constexpr int kWarpsPerWarpGroup = 4;
static constexpr int kThreadsPerWarpGroup = kWarpSize * kWarpsPerWarpGroup;  // 128

// Thread configuration: 1 producer warpgroup + 2 consumer warpgroups
// static constexpr int kNumProducerThreads = kWarpSize;  // Single warp for TMA
static constexpr int kNumMmaWarpGroups = 2;
static constexpr int kNumMmaThreads = kNumMmaWarpGroups * kThreadsPerWarpGroup;  // 256
static constexpr int kNumThreads = kNumMmaThreads + kThreadsPerWarpGroup;  // 384

// ============================================================================
// Shared memory layout
// ============================================================================
// Q: [kBlockM, kHeadDim] with swizzle
// K: [kBlockN, kHeadDim, kStages] with swizzle  
// V: [kHeadDim, kBlockN, kStages] with swizzle (transposed for PV matmul)

static constexpr int kSmemQSize = kBlockM * kHeadDim;           // 128 * 128 = 16K elements
static constexpr int kSmemKSize = kBlockN * kHeadDim * kStages; // 128 * 128 * 2 = 32K elements
static constexpr int kSmemVSize = kHeadDim * kBlockN * kStages; // 128 * 128 * 2 = 32K elements
// kSmemVtSize removed - V is loaded with transposed TMA descriptor
// static constexpr int kSmemPSize = kBlockM * kBlockN;            // 128 * 128 = 16K elements (for P@V)

// TMA transaction bytes
// For 128B swizzle with bf16: tile inner dim must be <= 64 elements (128 bytes)
// So we load 128x64 tiles and need 2 TMA calls for 128x128
static constexpr int kTmaTileCols = 64;  // Max for 128B swizzle with bf16
static constexpr int kTmaTransactionBytesQ = kBlockM * kTmaTileCols * sizeof(Element);  // 16KB per call, 2 calls
static constexpr int kTmaTransactionBytesK = kBlockN * kTmaTileCols * sizeof(Element);  // 16KB per call, 2 calls
static constexpr int kTmaTransactionBytesV = kBlockN * kTmaTileCols * sizeof(Element);  // 16KB per call, 2 calls

struct SharedStorage {
    // Mainloop tensors
    alignas(128) Element smem_q[kSmemQSize];
    alignas(128) Element smem_k[kSmemKSize];
    alignas(128) Element smem_v[kSmemVSize];
    // smem_vt removed - V is loaded with transposed TMA descriptor, no need for separate buffer
    
    // smem_o: 64x128 = 8K elements with swizzle (used during epilogue)
    alignas(128) Element smem_o[kSmemOSize];    // O output buffer (epilogue, swizzled)
    
    // Pipeline barriers for K/V
    // full_barrier: Producer signals data ready, Consumer waits
    // empty_barrier: Consumer signals buffer free, Producer waits
    alignas(16) uint64_t full_barrier_K[kStages];
    alignas(16) uint64_t empty_barrier_K[kStages];
    alignas(16) uint64_t full_barrier_V[kStages];
    alignas(16) uint64_t empty_barrier_V[kStages];
    
    // Q barrier (single buffer, ping-pong phase)
    alignas(16) uint64_t full_barrier_Q;   // Producer signals Q ready
    alignas(16) uint64_t empty_barrier_Q;  // Consumer signals Q consumed
    
    // Scheduler communication: producer writes work info for consumers
    // Packed as int4 for single 128-bit load/store (like original FA3)
    // work_info.x = tile_idx (unused after decode, can be reused)
    // work_info.y = m_block
    // work_info.z = bidh
    // work_info.w = bidb
    alignas(16) int4 work_info;
};

// ============================================================================
// Mbarrier operations (PTX wrappers for SM90)
// ============================================================================

// Fence after mbarrier init
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
}

// Initialize mbarrier with expected arrival count
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t arrive_count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        "mbarrier.init.shared::cta.b64 [%1], %0;\n"
        "}\n"
        :: "r"(arrive_count), "r"(smem_addr)
    );
}

// Producer: arrive and expect transaction bytes (for TMA)
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n"
        "}\n"
        :: "r"(tx_bytes), "r"(smem_addr)
    );
}

// Producer: simple arrive (no transaction)
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        "}\n"
        :: "r"(smem_addr)
    );
}

// Consumer: non-blocking try_wait - returns true if barrier is ready
__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t ready;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(ready)
        : "r"(smem_addr), "r"(phase)
    );
    return ready != 0;
}

// Consumer: blocking wait for mbarrier phase (with timeout and retry)
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint32_t ticks = 0x989680;  // Large timeout value
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(smem_addr), "r"(phase), "r"(ticks)
    );
}

// Consumer: conditional wait - only waits if token indicates not ready
__device__ __forceinline__ void mbarrier_wait_if_needed(uint64_t* mbar, uint32_t phase, bool already_ready) {
    if (!already_ready) {
        mbarrier_wait(mbar, phase);
    }
}

// Invalidate mbarrier
__device__ __forceinline__ void mbarrier_invalidate(uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        "}\n"
        :: "r"(smem_addr)
    );
}

// ============================================================================
// Named barriers for producer-consumer sync
// ============================================================================
// Named barrier IDs (0-7 reserved, we use 8+)
// Ping-pong pattern:
// StreamkBarrier0: Consumer arrive after reading, Producer sync before writing next
// StreamkBarrier1: Producer arrive after writing, Consumer sync before reading
static constexpr int kBarrierConsumerDone = 8;   // StreamkBarrier0
static constexpr int kBarrierProducerDone = 9;   // StreamkBarrier1
static constexpr int kBarrierEpilogue = 10;      // For epilogue R2S -> S2G sync
static constexpr int kBarrierWarpSchedulerWG1 = 11;  // Warp scheduler for warpgroup 1
static constexpr int kBarrierWarpSchedulerWG2 = 12;  // Warp scheduler for warpgroup 2

// Number of threads participating in scheduler barriers
// = Producer warp (32) + Consumer warpgroups (256) = 288
// Only warp 0 of producer warpgroup + all consumer threads participate
static constexpr int kNumSchedulerThreads = kWarpSize + kNumMmaThreads;  // 32 + 256 = 288

// ============================================================================
// Pipeline state tracker
// ============================================================================
struct PipelineState {
    int index_;   // Current stage index [0, kStages)
    int phase_;   // Current phase (0 or 1) for mbarrier
    int count_;   // Number of iterations
    
    __device__ PipelineState() : index_(0), phase_(0), count_(0) {}
    __device__ PipelineState(int index, int phase, int count) : index_(index), phase_(phase), count_(count) {}
    
    __device__ int index() const { return index_; }
    __device__ int phase() const { return phase_; }
    __device__ int count() const { return count_; }
    
    __device__ PipelineState& operator++() {
        ++index_;
        ++count_;
        if (index_ >= kStages) {
            index_ = 0;
            phase_ ^= 1;  // Toggle phase
        }
        return *this;
    }
};

// Named barrier arrive (non-blocking signal)
__device__ __forceinline__ void named_barrier_arrive(int barrier_id, int num_threads) {
    asm volatile(
        "bar.arrive %0, %1;\n"
        :: "r"(barrier_id), "r"(num_threads)
    );
}

// Named barrier sync (blocking wait + arrive)
__device__ __forceinline__ void named_barrier_sync(int barrier_id, int num_threads) {
    asm volatile(
        "bar.sync %0, %1;\n"
        :: "r"(barrier_id), "r"(num_threads)
    );
}

// Warp scheduler barrier for overlapping QK and PV between warpgroups
// warp_group_idx: 1 or 2 (consumer warpgroups)
__device__ __forceinline__ void warp_scheduler_barrier_sync(int warp_group_idx) {
    // Each warpgroup syncs on its own barrier (256 threads = 2 warpgroups)
    int barrier_id = (warp_group_idx == 1) ? kBarrierWarpSchedulerWG1 : kBarrierWarpSchedulerWG2;
    named_barrier_sync(barrier_id, 2 * kThreadsPerWarpGroup);  // 256 threads
}

__device__ __forceinline__ void warp_scheduler_barrier_arrive(int warp_group_idx) {
    // Arrive on the OTHER warpgroup's barrier
    int next_wg = (warp_group_idx == 1) ? 2 : 1;
    int barrier_id = (next_wg == 1) ? kBarrierWarpSchedulerWG1 : kBarrierWarpSchedulerWG2;
    named_barrier_arrive(barrier_id, 2 * kThreadsPerWarpGroup);  // 256 threads
}


__device__ __forceinline__ uint32_t pack_acc(float acc1, float acc2) {
    auto tmp = __floats2bfloat162_rn(acc1, acc2);
    return *reinterpret_cast<uint32_t*>(&tmp);
}

// ============================================================================
// Epilogue helper functions (pure CUDA implementation with vectorized stores)
// ============================================================================
// FA3-style epilogue store: acc_o -> smem -> gmem
// thread_idx: 0-255 (256 threads = 2 warpgroups)

// R2S: Register to Shared Memory using stmatrix
// WGMMA m64n128k16 accumulator layout:
//   Thread t: base_row = ((t/4)%8) + (t/32)*16
//   acc[j]: row = base_row + ((j/2)%2)*8, col = (t%4)*2 + (j%2) + (j/4)*8
//
// For stmatrix.x2 storing 2 8x8 matrices (16 cols):
//   row_offset=0: r0=acc[col_group*8, col_group*8+1], r1=acc[col_group*8+4, col_group*8+5]
//   row_offset=8: r0=acc[col_group*8+2, col_group*8+3], r1=acc[col_group*8+6, col_group*8+7]
// where col_group = 0..7 covers cols 0-127
//
// thread_idx: 0-255 (256 threads = 2 warpgroups)
// - warpgroup 1 (0-127): acc_o for rows 0-63
// - warpgroup 2 (128-255): acc_o for rows 64-127
__device__ __forceinline__ void r2s_store_epilogue(
    float* acc,
    Element* smem,
    int thread_idx
) {
    // stmatrix.x4 version: 8 calls instead of 16 calls with stmatrix.x2
    //
    // thread_idx: 0-255 (256 threads = 2 warpgroups)
    // - warpgroup 1 (0-127): acc_o for rows 0-63
    // - warpgroup 2 (128-255): acc_o for rows 64-127
    //
    // Key insight: stmatrix.x4 writes a 16x16 tile using 32 threads (one warp)
    // Each warp in the warpgroup handles a different 16-row region
    // warp 0: rows 0-15, warp 1: rows 16-31, warp 2: rows 32-47, warp 3: rows 48-63
    //
    // stmatrix.x4 address pattern (per warp of 32 threads):
    //   matrix_id = lane_id / 8  (0,1,2,3)
    //   lane_in_matrix = lane_id % 8  (0-7)
    //   mat_row = matrix_id / 2  (0 for matrix 0,1; 1 for matrix 2,3)
    //   mat_col = matrix_id % 2  (0 for matrix 0,2; 1 for matrix 1,3)
    //   row_in_tile = mat_row * 8 + lane_in_matrix
    //   col_in_tile = mat_col * 8
    //
    // Data distribution (fixed by hardware):
    //   Threads 0-3 write row 0, threads 4-7 write row 1, ...
    //   reg0 -> matrix 0, reg1 -> matrix 1, reg2 -> matrix 2, reg3 -> matrix 3
    //
    // WGMMA output layout (per thread, 64 floats):
    //   acc[g*8 + 0,1] = mi=0, ni=0-1 (row offset 0, col offset 0)
    //   acc[g*8 + 2,3] = mi=1, ni=0-1 (row offset 8, col offset 0)
    //   acc[g*8 + 4,5] = mi=0, ni=8-9 (row offset 0, col offset 8)
    //   acc[g*8 + 6,7] = mi=1, ni=8-9 (row offset 8, col offset 8)
    //
    // Register mapping for stmatrix.x4:
    //   reg0 = bf16(acc[0], acc[1])   -> matrix 0 (row+0, col+0)
    //   reg1 = bf16(acc[4], acc[5])   -> matrix 1 (row+0, col+8)
    //   reg2 = bf16(acc[2], acc[3])   -> matrix 2 (row+8, col+0)
    //   reg3 = bf16(acc[6], acc[7])   -> matrix 3 (row+8, col+8)
    
    // Warpgroup index (0 or 1) and thread index within warpgroup (0-127)
    int const warpgroup_idx = thread_idx >> 7;  // 0 for threads 0-127, 1 for threads 128-255
    int const thread_in_wg = thread_idx & 127;  // 0-127 within warpgroup
    
    int const warp_in_wg = thread_in_wg >> 5;  // 0-3 within warpgroup
    int const lane_id = thread_in_wg & 31;
    
    // stmatrix.x4 address calculation
    int const matrix_id = lane_id >> 3;  // 0,1,2,3
    int const lane_in_matrix = lane_id & 7;  // 0-7
    int const mat_row = matrix_id >> 1;  // 0 for matrix 0,1; 1 for matrix 2,3
    int const mat_col = matrix_id & 1;   // 0 for matrix 0,2; 1 for matrix 1,3
    
    // Row within the 16x16 tile
    int const row_in_tile = (mat_row << 3) + lane_in_matrix;
    
    // Base row for this warp within warpgroup (warp 0->0, 1->16, 2->32, 3->48)
    int const warp_base_row = warp_in_wg << 4;
    
    // Base row for this warpgroup (warpgroup 0->0, warpgroup 1->64)
    int const warpgroup_base_row = warpgroup_idx << 6;
    
    int const row = warpgroup_base_row + warp_base_row + row_in_tile;
    
    // Col offset from matrix position (0 or 8)
    int const col_offset_matrix = mat_col << 3;
    
    // Precompute swizzle for this row
    int const swizzle_xor = (row << 3) & 0x70;
    
    // 8 iterations: 8 col_groups (each col_group is 16 columns = one stmatrix.x4)
    #pragma unroll
    for (int col_group = 0; col_group < 8; ++col_group) {
        int const acc_base = col_group * 8;
        
        // Pack registers in the correct order for stmatrix.x4
        // reg0 -> matrix 0, reg1 -> matrix 1, reg2 -> matrix 2, reg3 -> matrix 3
        uint32_t r0 = pack_acc(acc[acc_base + 0], acc[acc_base + 1]);
        uint32_t r1 = pack_acc(acc[acc_base + 4], acc[acc_base + 5]);
        uint32_t r2 = pack_acc(acc[acc_base + 2], acc[acc_base + 3]);
        uint32_t r3 = pack_acc(acc[acc_base + 6], acc[acc_base + 7]);
        
        // Compute column for this col_group
        // col_group * 16 gives the base column, plus col_offset_matrix (0 or 8)
        int const col = (col_group << 4) + col_offset_matrix;
        
        // Apply B128 swizzle layout: smem[row][col] with 128-byte swizzle
        // Layout: col_lo = col & 63, col_hi contributes 8192 per 64 cols
        // base = col_lo + (row << 6) + col_hi
        // result = base ^ swizzle_xor
        int const col_lo = col & 63;
        int const col_hi = (col >> 6) << 13;  // (col / 64) * 8192
        int const base = col_lo + (row << 6) + col_hi;
        int const smem_idx = base ^ swizzle_xor;
        
        // stmatrix.x4 requires 16-byte aligned address
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + smem_idx));
        asm volatile(
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        );
    }
}

// S2G: Shared Memory to Global Memory using vectorized 128-bit loads/stores
// Thread mapping: base_col = (t % 16) * 8, row_offset = t / 16
// Each thread handles 8 rows, 8 cols per row = 64 elements
// Note: headdim must be 128 (hardcoded for simplicity)
__device__ __forceinline__ void s2g_store_epilogue(
    Element* smem,
    Element* gmem,
    int thread_idx,
    int64_t stride,
    int rows_remaining
) {
    constexpr int headdim = 128;  // Hardcode to eliminate runtime checks
    int const base_col = (thread_idx & 15) << 3;  // * 8
    int const row_offset = thread_idx >> 4;
    
    // Skip if column is out of bounds (only threads 0-15 write for headdim=128)
    if (base_col >= headdim) return;
    
    // Precompute col contribution to base (only depends on thread_idx)
    // For base_col < 64: col_contrib = base_col, no 8192 offset
    // For base_col >= 64: col_contrib = base_col - 64, add 8192 offset
    int const col_contrib = (base_col < 64) ? base_col : (base_col - 64 + 8192);
    
    // Precompute swizzle_xor (only depends on row_offset, not m)
    // row = m * 16 + row_offset, swizzle_xor = (row << 3) & 0x70
    // Since (m * 16) << 3 = m << 7 has no bits in 0x70, swizzle_xor = (row_offset << 3) & 0x70
    int const swizzle_xor = (row_offset << 3) & 0x70;
    
    // Precompute base offset for row_offset
    int const base_row_offset = col_contrib + (row_offset << 6);
    
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int const row = m * 16 + row_offset;
        
        if (row < rows_remaining) {
            // base = col_contrib + (row << 6) = base_row_offset + (m << 10)
            int const base = base_row_offset + (m << 10);
            int const smem_idx = base ^ swizzle_xor;
            
            Element* smem_ptr = smem + smem_idx;
            Element* gmem_ptr = gmem + row * stride + base_col;
            
            // Full 128-bit vectorized load and store
            uint4 data = *reinterpret_cast<uint4*>(smem_ptr);
            *reinterpret_cast<uint4*>(gmem_ptr) = data;
        }
    }
}

__device__ __forceinline__ void epilogue_store(
    float* acc_o,                    // Accumulator in registers (64 floats)
    float const* row_sum,            // Row sums for normalization (2 floats)
    SharedStorage& shared_storage,
    int thread_idx,                  // Thread index for TiledMmaPV (0-255)
    int m_block,                     // M block index (128 rows per block)
    int bidh,                        // Head index
    int seqlen_q,                    // Sequence length
    int offset_q,                    // Offset in packed Q tensor
    Element* ptr_O,                  // Output pointer
    int64_t stride_O_row,            // Output row stride
    int headdim                      // Head dimension
) {
    // Normalize accumulator by row_sum
    float inv_sum[2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        float sum = row_sum[mi];
        inv_sum[mi] = (sum == 0.0f || sum != sum) ? 0.0f : 1.0f / sum;
    }
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        int mi = ((i >> 1) & 1);
        acc_o[i] *= inv_sum[mi];
    }
    
    // Note: No sync needed before R2S because:
    // 1. Each thread writes to its own smem locations (no conflicts)
    // 2. stmatrix is a warp-collective operation that handles intra-warp sync
    
    // R2S: Register -> Smem
    r2s_store_epilogue(acc_o, shared_storage.smem_o, thread_idx);
    
    // Sync after R2S, before S2G
    named_barrier_sync(kBarrierEpilogue, kNumEpilogueThreads);
    
    // S2G: Smem -> Gmem
    int rows_remaining = seqlen_q - m_block * kBlockM;
    Element* gmem_o = ptr_O + (offset_q + m_block * kBlockM) * stride_O_row + bidh * headdim;
    
    s2g_store_epilogue(shared_storage.smem_o, gmem_o, thread_idx, stride_O_row, rows_remaining);
}



static constexpr int kSharedMemSize = sizeof(SharedStorage);

// ============================================================================
// Sequence length info for varlen
// ============================================================================
struct SeqlenInfo {
    int offset_q;
    int offset_k;
    int seqlen_q;
    int seqlen_k;
    
    __device__ SeqlenInfo(
        int bidb,
        int const* cu_seqlens_q,
        int const* cu_seqlens_k,
        int const* seqused_q,
        int const* seqused_k,
        int max_seqlen_q,
        int max_seqlen_k
    ) {
        offset_q = cu_seqlens_q[bidb];
        offset_k = cu_seqlens_k[bidb];
        
        if (seqused_q) {
            seqlen_q = seqused_q[bidb];
        } else {
            seqlen_q = cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb];
        }
        
        if (seqused_k) {
            seqlen_k = seqused_k[bidb];
        } else {
            seqlen_k = cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb];
        }
    }
};

// ============================================================================
// Work tile info for scheduler
// ============================================================================
struct WorkTileInfo {
    int tile_idx;
    int m_block;
    int bidh;
    int bidb;
    
    __device__ bool is_valid(int num_batch) const {
        return bidb < num_batch;
    }
};

// ============================================================================
// Warp-level prefix sum for fast batch lookup
// ============================================================================
__device__ __forceinline__ int warp_prefix_sum(int val) {
    int lane = threadIdx.x & 31;  // % 32
    #pragma unroll
    for (int i = 1; i < kWarpSize; i <<= 1) {
        int partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) { val += partial_sum; }
    }
    return val;
}

// ============================================================================
// Fast tile-to-batch decoder using warp-level parallelism
// Each warp processes 31 batches at a time (lanes 0-30, lane 31 unused)
// Returns: bidb, bidh, m_block for the given tile_idx
// ============================================================================
__device__ __forceinline__ void decode_tile_varlen_fast(
    int tile_idx,
    int batch,
    int nheads,
    int const* cu_seqlens_q,
    int& out_bidb,
    int& out_bidh,
    int& out_m_block
) {
    int lane = threadIdx.x & 31;  // % 32
    
    // Start from batch 0
    int bidb = 0;
    int group_start_tile = 0;
    
    // Each lane reads num_m_blocks for batch (bidb + lane)
    // Lane 31 is unused (sentinel)
    auto get_num_m_blocks = [&](int bidb_start) -> int {
        int batch_idx = lane + bidb_start;
        if (batch_idx < batch && lane < kWarpSize - 1) {
            int seqlen_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
            return (seqlen_q + kBlockM - 1) >> 7;  // / 128
        }
        return 0;
    };
    
    int num_m_blocks = get_num_m_blocks(bidb);
    int num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
    int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, kWarpSize - 1);
    int group_end_tile = group_start_tile + m_blocks_in_group * nheads;
    
    // Skip groups of 31 batches until we find the one containing tile_idx
    while (group_end_tile <= tile_idx) {
        bidb += kWarpSize - 1;  // Move to next group of 31 batches
        if (bidb >= batch) {
            out_bidb = batch;  // Signal invalid
            out_bidh = 0;
            out_m_block = 0;
            return;
        }
        group_start_tile = group_end_tile;
        num_m_blocks = get_num_m_blocks(bidb);
        num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
        m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, kWarpSize - 1);
        group_end_tile = group_start_tile + m_blocks_in_group * nheads;
    }
    
    // Now tile_idx is within [group_start_tile, group_end_tile)
    // Use ballot to find which batch within the group
    // batch_idx_in_group = number of batches where (group_start_tile + cumsum * nheads <= tile_idx)
    int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, 
        group_start_tile + num_m_blocks_cumulative * nheads <= tile_idx));
    
    bidb += batch_idx_in_group;
    num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
    
    // Compute tile offset within this batch
    int batch_start_tile = group_start_tile;
    if (batch_idx_in_group > 0) {
        batch_start_tile += __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1) * nheads;
    }
    
    int tile_in_batch = tile_idx - batch_start_tile;
    out_bidb = bidb;
    // Note: This division is only called once per tile, so not worth optimizing
    out_bidh = tile_in_batch / num_m_blocks;
    out_m_block = tile_in_batch - out_bidh * num_m_blocks;  // Faster than %
}

// ============================================================================
// Get n_block range for causal masking
// ============================================================================
__device__ __forceinline__ void get_n_block_min_max(
    int& n_block_min,
    int& n_block_max,
    int m_block,
    int seqlen_q,
    int seqlen_k,
    bool is_causal
) {
    n_block_min = 0;
    n_block_max = (seqlen_k + kBlockN - 1) >> 7;  // / 128
    
    if (is_causal) {
        // For causal: n_idx <= m_idx + seqlen_k - seqlen_q
        int m_idx_max = (m_block + 1) << 7;  // * 128
        int n_idx_max = m_idx_max + seqlen_k - seqlen_q;
        int n_block_max_causal = (n_idx_max + kBlockN - 1) >> 7;  // / 128
        n_block_max = min(n_block_max, n_block_max_causal);
    }
}

// ============================================================================
// Kernel parameters struct (passed via __grid_constant__)
// ============================================================================
struct FlashFwdParams {
    // Q tensor
    Element const* ptr_Q;
    int64_t stride_Q_row;
    int64_t stride_Q_head;
    // K tensor
    Element const* ptr_K;
    int64_t stride_K_row;
    int64_t stride_K_head;
    // V tensor
    Element const* ptr_V;
    int64_t stride_V_row;
    int64_t stride_V_head;
    // O tensor
    Element* ptr_O;
    int64_t stride_O_row;
    int64_t stride_O_head;
    // Sequence info
    int const* cu_seqlens_q;
    int const* cu_seqlens_k;
    int const* seqused_q;
    int const* seqused_k;
    // Dimensions
    int total_q;
    int total_k;
    int max_seqlen_q;
    int max_seqlen_k;
    int batch;
    int nheads;
    int nkv_heads;
    int headdim;
    // Causal mask
    int is_causal;
    // Softmax
    float softmax_scale_log2;
    // LSE output
    float* ptr_LSE;
    int64_t stride_LSE;
    // Scheduler
    int* tile_count_semaphore;
    int const* num_splits_ptr;
    int const* num_m_blocks_ptr;
    int const* batch_idx_ptr;
    int const* nheads_l2_ptr;
    // Derived
    FastDivmod qhead_per_khead_divmod;
    FastDivmod num_m_blocks_divmod;  // For tile_in_batch / num_m_blocks
    // TMA descriptors (embedded directly, passed via kernel params)
    CUtensorMap tma_Q;
    CUtensorMap tma_K;
    CUtensorMap tma_V;
};



// ============================================================================
// Kernel body
// ============================================================================
__device__ void flash_fwd_kernel_body(FlashFwdParams const& params, char* smem_buf) {
    
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    
    int const thread_idx = threadIdx.x;
    int const warp_idx = thread_idx >> 5;           // / 32
    int const lane_idx = thread_idx & 31;           // % 32
    int const warp_group_idx = thread_idx >> 7;     // / 128
    int const warp_idx_in_warpgroup = warp_idx & 3; // % 4
    
    bool const is_warp0_lane0 = (warp_idx == 0 && lane_idx == 0);
    
    // Number of consumer warpgroups (for empty_barrier arrive count)
    int const num_consumer_warpgroups = kNumMmaWarpGroups;  // 2
    
    // ========== Initialization ==========
    if (is_warp0_lane0) {
        // Q barrier: producer arrives once
        mbarrier_init(&shared_storage.full_barrier_Q, 1);
        mbarrier_init(&shared_storage.empty_barrier_Q, kNumMmaWarpGroups);  // 2 consumer warpgroups
        
        // K/V pipeline barriers
        for (int s = 0; s < kStages; ++s) {
            // full_barrier: producer arrives once (with TMA transaction)
            mbarrier_init(&shared_storage.full_barrier_K[s], 1);
            mbarrier_init(&shared_storage.full_barrier_V[s], 1);
            // empty_barrier: consumer warpgroups arrive (one per warpgroup)
            mbarrier_init(&shared_storage.empty_barrier_K[s], num_consumer_warpgroups);
            mbarrier_init(&shared_storage.empty_barrier_V[s], num_consumer_warpgroups);
        }
    }
    fence_barrier_init();
    
    // Prefetch TMA descriptors (FA3 style) - only thread 0 in warp 0
    if (is_warp0_lane0) {
        prefetch_tma_descriptor(&params.tma_Q);
        prefetch_tma_descriptor(&params.tma_K);
        prefetch_tma_descriptor(&params.tma_V);
    }
    
    __syncthreads();
    
    // int const num_m_blocks = (params.max_seqlen_q + kBlockM - 1) >> 7;  // / 128
    
    // ========== Producer path (warp_group_idx == 0) ==========
    if (warp_group_idx == 0) {
        // Producer uses fewer registers - allows more concurrent warpgroups
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(24));
        
        // Only warp 0 in producer warpgroup participates in scheduler
        // Other warps (1-3) in warpgroup 0 just return
        if (warp_idx_in_warpgroup != 0) {
            return;
        }
        
        bool const is_leader = (thread_idx == 0);  // Only one thread writes to shared_storage
        
        // Get initial tile: use blockIdx.x for first tile (like original code)
        int tile_idx = static_cast<int>(blockIdx.x);
        
        // Producer starts with phase=1 (buffers are initially empty)
        PipelineState pipe_state_k(0, 1, 0);
        PipelineState pipe_state_v(0, 1, 0);
        
        // === Initial work (first iteration) ===
        {
            // Decode tile_idx for varlen using warp-level parallel lookup
            int bidb, bidh, m_block;
            decode_tile_varlen_fast(tile_idx, params.batch, params.nheads, 
                                    params.cu_seqlens_q, bidb, bidh, m_block);
            
            if (bidb < params.batch) {
                
                SeqlenInfo seqlen_info(
                    bidb, params.cu_seqlens_q, params.cu_seqlens_k,
                    params.seqused_q, params.seqused_k,
                    params.max_seqlen_q, params.max_seqlen_k
                );
                
                int n_block_min, n_block_max;
                get_n_block_min_max(n_block_min, n_block_max, m_block,
                                   seqlen_info.seqlen_q, seqlen_info.seqlen_k, params.is_causal);
                
                
                if (is_leader) {
                    shared_storage.work_info = make_int4(tile_idx, m_block, bidh, bidb);
                }
            } else {
                if (is_leader) {
                    shared_storage.work_info = make_int4(0, 0, 0, params.batch);  // Signal done
                }
            }
            
            // Signal consumers: work info ready
            // All 288 threads participate: producer arrive, consumer will sync
            named_barrier_arrive(kBarrierProducerDone, kNumSchedulerThreads);
        }
        
        // Main loop: process remaining tiles
        // Note: work_info already contains first tile's info from initial setup
        int tile_count = 0;
        int q_phase_producer = 0;  // Track Q barrier phase
        bool prev_tile_loaded_q = false;  // Track if previous tile loaded Q
        while (true) {
            // Use current work info (already set by previous iteration or initial setup)
            int4 work_info = shared_storage.work_info;
            int bidb = work_info.w;
            
            if (bidb >= params.batch) {
                // No more work - consumer already has the "done" signal from work_info
                // Just break, no need to arrive barrier again
                // (Consumer will read work_info, see bidb >= batch, and break)
                break;
            }
            
            ++tile_count;
            
            int m_block = work_info.y;
            int bidh = work_info.z;
            
            // Recompute seqlen_info from bidb (like original FA3)
            SeqlenInfo seqlen_info(
                bidb, params.cu_seqlens_q, params.cu_seqlens_k,
                params.seqused_q, params.seqused_k,
                params.max_seqlen_q, params.max_seqlen_k
            );
            
            int n_block_min, n_block_max;
            get_n_block_min_max(n_block_min, n_block_max, m_block,
                               seqlen_info.seqlen_q, seqlen_info.seqlen_k, params.is_causal);
            
            int n_blocks = n_block_max - n_block_min;
            int offset_q = seqlen_info.offset_q;
            int offset_k = seqlen_info.offset_k;
            
            // Prefetch next tile
            // Initial tiles are assigned by blockIdx.x (tiles 0 to gridDim.x-1)
            // Subsequent tiles are fetched via atomicAdd + gridDim.x
            // Semaphore starts at 0, so atomicAdd returns 0, 1, 2, ...
            // Adding gridDim.x gives us gridDim.x, gridDim.x+1, gridDim.x+2, ...
            if (is_leader) {
                tile_idx = atomicAdd(params.tile_count_semaphore, 1) + static_cast<int>(gridDim.x);
            }
            tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
            
            // Compute KV head index for GQA
            int kv_head_idx = params.qhead_per_khead_divmod.divide(bidh);
            
            if (n_blocks > 0) {
                // Wait for Q buffer to be free (only if previous tile loaded Q)
                if (prev_tile_loaded_q) {
                    mbarrier_wait(&shared_storage.empty_barrier_Q, q_phase_producer);
                    q_phase_producer ^= 1;  // Toggle phase for next wait
                }
                
                // Load Q using TMA (2 calls for 128 columns: 0-63 and 64-127)
                if (is_leader) {
                    mbarrier_arrive_expect_tx(&shared_storage.full_barrier_Q, kTmaTransactionBytesQ * 2);
                    
                    tma_load_3d(
                        &params.tma_Q,
                        &shared_storage.full_barrier_Q,
                        shared_storage.smem_q,
                        0,
                        offset_q + m_block * kBlockM,
                        bidh
                    );
                    tma_load_3d(
                        &params.tma_Q,
                        &shared_storage.full_barrier_Q,
                        shared_storage.smem_q + kBlockM * kTmaTileCols,
                        kTmaTileCols,
                        offset_q + m_block * kBlockM,
                        bidh
                    );
                }
                
                // Load K/V with pipeline (FA3-style IntraWGOverlap)
                // Key insight: V[i] is loaded with n_block[i-1] to overlap with QK[i]
                // This allows consumer to compute QK[i] while producer loads V[i-1]
                //
                // Loop structure:
                // - First iteration (i=0): load K[0] only (no V yet)
                // - Middle iterations (i=1 to n_blocks-1): load K[i] and V[i-1]
                // - After loop: load final V[n_blocks-1]
                
                int n_block_prev = n_block_max - 1;  // Track previous n_block for V loading
                
                // First iteration: load K[0] only
                if (n_blocks > 0) {
                    int n_block = n_block_max - 1;
                    int k_stage = pipe_state_k.index();
                    
                    if (is_leader) {
                        mbarrier_arrive_expect_tx(&shared_storage.full_barrier_K[k_stage], kTmaTransactionBytesK * 2);
                        
                        Element* smem_k_stage = shared_storage.smem_k + k_stage * kBlockN * kHeadDim;
                        tma_load_3d_cache_hint(&params.tma_K, &shared_storage.full_barrier_K[k_stage], smem_k_stage, 0, offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast);
                        tma_load_3d_cache_hint(&params.tma_K, &shared_storage.full_barrier_K[k_stage], smem_k_stage + kBlockN * kTmaTileCols, kTmaTileCols, offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast);
                    }
                    
                    n_block_prev = n_block;
                    ++pipe_state_k;
                }
                
                // Middle iterations: load K[i] and V[i-1]
                for (int i = 1; i < n_blocks; ++i) {
                    int n_block = n_block_max - 1 - i;
                    int k_stage = pipe_state_k.index();
                    int v_stage = pipe_state_v.index();
                    
                    if (i >= kStages) {
                        mbarrier_wait(&shared_storage.empty_barrier_K[k_stage], pipe_state_k.phase());
                    }
                    if (i - 1 >= kStages) {
                        mbarrier_wait(&shared_storage.empty_barrier_V[v_stage], pipe_state_v.phase());
                    }
                    
                    if (is_leader) {
                        // Load K[i]
                        mbarrier_arrive_expect_tx(&shared_storage.full_barrier_K[k_stage], kTmaTransactionBytesK * 2);
                        Element* smem_k_stage = shared_storage.smem_k + k_stage * kBlockN * kHeadDim;
                        tma_load_3d_cache_hint(&params.tma_K, &shared_storage.full_barrier_K[k_stage], smem_k_stage, 0, offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast);
                        tma_load_3d_cache_hint(&params.tma_K, &shared_storage.full_barrier_K[k_stage], smem_k_stage + kBlockN * kTmaTileCols, kTmaTileCols, offset_k + n_block * kBlockN, kv_head_idx, kCacheHintEvictLast);
                        
                        // Load V[i-1] (using n_block_prev)
                        mbarrier_arrive_expect_tx(&shared_storage.full_barrier_V[v_stage], kTmaTransactionBytesV * 2);
                        Element* smem_v_stage = shared_storage.smem_v + v_stage * kBlockN * kHeadDim;
                        tma_load_3d_cache_hint(&params.tma_V, &shared_storage.full_barrier_V[v_stage], smem_v_stage, 0, offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast);
                        tma_load_3d_cache_hint(&params.tma_V, &shared_storage.full_barrier_V[v_stage], smem_v_stage + kBlockN * kTmaTileCols, kTmaTileCols, offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast);
                    }
                    
                    n_block_prev = n_block;
                    ++pipe_state_k;
                    ++pipe_state_v;
                }
                
                // Final V: load V[n_blocks-1]
                if (n_blocks > 0) {
                    int v_stage = pipe_state_v.index();
                    
                    if (n_blocks - 1 >= kStages) {
                        mbarrier_wait(&shared_storage.empty_barrier_V[v_stage], pipe_state_v.phase());
                    }
                    
                    if (is_leader) {
                        mbarrier_arrive_expect_tx(&shared_storage.full_barrier_V[v_stage], kTmaTransactionBytesV * 2);
                        Element* smem_v_stage = shared_storage.smem_v + v_stage * kBlockN * kHeadDim;
                        tma_load_3d_cache_hint(&params.tma_V, &shared_storage.full_barrier_V[v_stage], smem_v_stage, 0, offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast);
                        tma_load_3d_cache_hint(&params.tma_V, &shared_storage.full_barrier_V[v_stage], smem_v_stage + kBlockN * kTmaTileCols, kTmaTileCols, offset_k + n_block_prev * kBlockN, kv_head_idx, kCacheHintEvictLast);
                    }
                    
                    ++pipe_state_v;
                }
                
                prev_tile_loaded_q = true;  // This tile loaded Q
            } else {
                prev_tile_loaded_q = false;  // This tile did NOT load Q
            }
            
            // Wait for consumers to finish reading current work info
            named_barrier_sync(kBarrierConsumerDone, kNumSchedulerThreads);
            
            // Decode next tile for varlen using warp-level parallel lookup
            {
                int next_bidb, next_bidh, next_m_block;
                decode_tile_varlen_fast(tile_idx, params.batch, params.nheads,
                                        params.cu_seqlens_q, next_bidb, next_bidh, next_m_block);
                
                if (is_leader) {
                    if (next_bidb < params.batch) {
                        shared_storage.work_info = make_int4(tile_idx, next_m_block, next_bidh, next_bidb);
                    } else {
                        shared_storage.work_info = make_int4(0, 0, 0, params.batch);  // Signal done
                    }
                }
            }
            
            // Signal consumers: next work info ready
            named_barrier_arrive(kBarrierProducerDone, kNumSchedulerThreads);
        }
        
    } else {
        // ========== Consumer path (warp_group_idx >= 1) ==========
        // Consumer needs more registers for WGMMA accumulators
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(240));
        
        bool const is_warpgroup_leader = ((thread_idx & 127) == 0);  // % 128
        
        PipelineState pipe_state_k;
        PipelineState pipe_state_v;
        int q_phase = 0;  // Track barrier_Q phase across tiles
        int consumer_tile_count = 0;
        
        // Scheduler barrier initialization (FA3 style)
        // WG1 arrives on WG1's own barrier so WG1's first sync doesn't block
        // FA3: if (warp_group_idx == 1) arrive on WG1_barrier
        if (warp_group_idx == 1) {
            named_barrier_arrive(kBarrierWarpSchedulerWG1, 2 * kThreadsPerWarpGroup);
        }
        
        while (true) {
            
            // Wait for producer to write work info
            // All 288 threads participate: consumer sync (= arrive + wait)
            named_barrier_sync(kBarrierProducerDone, kNumSchedulerThreads);
            
            // Read work_info as single int4 (128-bit load)
            int4 const work_info = shared_storage.work_info;
            int const bidb = work_info.w;
            
            if (bidb >= params.batch) {
                break;
            }
            
            ++consumer_tile_count;
            
            int const m_block = work_info.y;
            int const bidh = work_info.z;
            
            // Recompute seqlen_info from bidb (like original FA3)
            SeqlenInfo seqlen_info(
                bidb, params.cu_seqlens_q, params.cu_seqlens_k,
                params.seqused_q, params.seqused_k,
                params.max_seqlen_q, params.max_seqlen_k
            );
            
            int n_block_min, n_block_max;
            get_n_block_min_max(n_block_min, n_block_max, m_block,
                               seqlen_info.seqlen_q, seqlen_info.seqlen_k, params.is_causal);
            int const n_blocks = n_block_max - n_block_min;
            
            // Get sequence info
            int const seqlen_q = seqlen_info.seqlen_q;
            int const seqlen_k = seqlen_info.seqlen_k;
            int const offset_q = seqlen_info.offset_q;
            // int const offset_k = seqlen_info.offset_k;
            // Signal: work_info has been read, producer can write next work_info
            // This is done immediately after reading work_info, NOT after processing
            named_barrier_arrive(kBarrierConsumerDone, kNumSchedulerThreads);
            
            if (n_blocks > 0) {
                
                // Wait for Q (use tracked phase) - FA3-style two-step pattern
                bool q_ready = mbarrier_try_wait_parity(&shared_storage.full_barrier_Q, q_phase);
                mbarrier_wait_if_needed(&shared_storage.full_barrier_Q, q_phase, q_ready);
                
                q_phase ^= 1;  // Toggle phase for next tile
                
                // Initialize accumulators for O (64 floats per thread for m64n128)
                float acc_o[64];
                #pragma unroll
                for (int i = 0; i < 64; ++i) acc_o[i] = 0.0f;
                
                // Row max and sum for online softmax (per-row, distributed across threads)
                float row_max[2] = {-INFINITY, -INFINITY};  // 2 rows per thread in m64n128
                float row_sum[2] = {0.0f, 0.0f};
                
                // Scheduler barrier: initial sync before loop (FA3 style)
                warp_scheduler_barrier_sync(warp_group_idx);
                
                // Get softmax scale (constant across all iterations)
                float const softmax_scale_log2 = params.softmax_scale_log2;
                
                // Precompute thread's row/col bases (constant across all iterations)
                // Optimized: avoid & 127 by using thread_idx directly
                // thread_row_base = ((thread_idx >> 2) & 7) | ((thread_idx >> 1) & 48)
                // thread_col_offset = (thread_idx & 3) << 1
                int const thread_row_base = ((thread_idx >> 2) & 7) | ((thread_idx >> 1) & 48);
                int const thread_col_offset = (thread_idx & 3) << 1;
                int const m_offset = (warp_group_idx - 1) << 6;  // 0 or 64
                
                // Precomputed col_offset table (compile-time constants)
                static constexpr int col_offsets[32] = {
                    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
                    64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105, 112, 113, 120, 121
                };
                
                // Pre-compute descriptor bases (FA3 style optimization)
                // Descriptor format (64 bits):
                //   bits 0-13:  start_address (14 bits, in units of 16 bytes)
                //   bits 16-31: leading_byte_offset (16 bits) - in LOW 32 bits!
                //   bits 32-47: stride_byte_offset (16 bits) - in HIGH 32 bits
                //   bits 62-63: layout_type (2 bits) - bits 30-31 of HIGH 32 bits
                //
                // FA3 trick: split into {low32, high32}, only add to low32 in inner loop
                // WGMMA hardware only reads low 14 bits for address, overflow is ignored
                
                // Q descriptor: 64-col layout with 128B swizzle (no leading_byte_offset)
                uint32_t base_desc_q_lo = ((uint32_t)__cvta_generic_to_shared(shared_storage.smem_q + m_offset * 64)) >> 4;
                constexpr uint32_t desc_q_hi = (64U) | (1U << 30);  // stride=64, swizzle=128B
                
                // K descriptor: 128-row layout with 128B swizzle
                // leading_byte_offset=1 goes in LOW 32 bits (bits 16-31)!
                // Pre-compute desc_k_lead into base_desc_k_lo to avoid per-iteration OR
                // FA3 style: the lead offset is constant, so we can add it once at init
                uint32_t base_desc_k_lo = (((uint32_t)__cvta_generic_to_shared(shared_storage.smem_k)) >> 4) | (1U << 16);
                constexpr uint32_t desc_k_hi = (64U) | (1U << 30);  // stride=64, swizzle=128B
                
                // Offset constants (in units of 16 bytes)
                constexpr uint32_t k_stride = (16 * sizeof(Element)) >> 4;  // 16 elements = 32 bytes = 2 units
                constexpr uint32_t stage_stride = (kBlockN * kHeadDim * sizeof(Element)) >> 4;  // Stage offset
                constexpr uint32_t q_second_half = (kBlockM * 64 * sizeof(Element)) >> 4;  // Second half of Q
                constexpr uint32_t k_second_half = (kBlockM * 64 * sizeof(Element)) >> 4;  // Second half of K
                
                // Macro to construct descriptor from lo/hi parts using mov.b64
                #define MAKE_DESC(lo, hi) ({ \
                    uint64_t _desc; \
                    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(_desc) : "r"((uint32_t)(lo)), "r"((uint32_t)(hi))); \
                    _desc; \
                })
                
                // Lambda for QK matmul (shared between first and subsequent iterations)
                // Macros for compute operations (avoid lambda overhead for better unrolling)
                #define COMPUTE_QK(k_stage_arg) do { \
                    uint32_t s_off = (k_stage_arg) * stage_stride; \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) asm volatile("" : "+f"(acc_s[j]) :: "memory"); \
                    wgmma_fence(); \
                    /* FA3 optimization: no mask needed in inner loop - WGMMA only reads low 14 bits */ \
                    _Pragma("unroll") \
                    for (int ki = 0; ki < 4; ++ki) { \
                        uint32_t k_off = ki * k_stride; \
                        uint32_t q_lo = base_desc_q_lo + k_off; \
                        uint32_t k_lo = base_desc_k_lo + s_off + k_off; \
                        uint64_t desc_q = MAKE_DESC(q_lo, desc_q_hi); \
                        uint64_t desc_k = MAKE_DESC(k_lo, desc_k_hi); \
                        wgmma_m64n128k16_bf16_ss(acc_s, desc_q, desc_k, ki > 0 ? 1 : 0); \
                    } \
                    _Pragma("unroll") \
                    for (int ki = 0; ki < 4; ++ki) { \
                        uint32_t k_off = ki * k_stride; \
                        uint32_t q_lo = base_desc_q_lo + q_second_half + k_off; \
                        uint32_t k_lo = base_desc_k_lo + s_off + k_second_half + k_off; \
                        uint64_t desc_q = MAKE_DESC(q_lo, desc_q_hi); \
                        uint64_t desc_k = MAKE_DESC(k_lo, desc_k_hi); \
                        wgmma_m64n128k16_bf16_ss(acc_s, desc_q, desc_k, 1); \
                    } \
                    wgmma_commit_group(); \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) asm volatile("" : "+f"(acc_s[j]) :: "memory"); \
                } while(0)
                
                #define APPLY_MASK(n_block_arg) do { \
                    int const seqlenk_col_limit = seqlen_k - (n_block_arg) * kBlockN - thread_col_offset; \
                    int const causal_row_offset = 1 + seqlen_k - (n_block_arg) * kBlockN - seqlen_q - thread_col_offset; \
                    _Pragma("unroll") \
                    for (int mi = 0; mi < 2; ++mi) { \
                        int const row = thread_row_base + mi * 8; \
                        int const row_idx = m_block * kBlockM + m_offset + row; \
                        int col_limit_right; \
                        if (params.is_causal) { \
                            col_limit_right = __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit); \
                        } else { \
                            col_limit_right = seqlenk_col_limit; \
                        } \
                        bool const row_oob = (row_idx >= seqlen_q); \
                        _Pragma("unroll") \
                        for (int ni = 0; ni < 32; ++ni) { \
                            int const reg = mi * 2 + (ni & 1) + ((ni >> 1) << 2); \
                            int const col_offset = col_offsets[ni]; \
                            if (row_oob || col_offset >= col_limit_right) { \
                                acc_s[reg] = -INFINITY; \
                            } \
                        } \
                    } \
                } while(0)
                
                #define COMPUTE_PV_NO_WAIT(v_stage_arg) do { \
                    Element* smem_v_stage = shared_storage.smem_v + (v_stage_arg) * kBlockN * kHeadDim; \
                    uint32_t v_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_v_stage)); \
                    /* FA3 style: no mask needed - WGMMA only reads low 14 bits */ \
                    uint64_t desc_v_base = ((uint64_t)(v_addr >> 4)) \
                                         | (1024ULL << 16) \
                                         | (64ULL << 32) \
                                         | (1ULL << 62); \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) asm volatile("" : "+f"(acc_o[j]) :: "memory"); \
                    _Pragma("unroll") \
                    for (int j = 0; j < 32; ++j) asm volatile("" : "+r"(reinterpret_cast<uint32_t*>(acc_p_bf16)[j]) :: "memory"); \
                    wgmma_fence(); \
                    _Pragma("unroll") \
                    for (int ki = 0; ki < 8; ++ki) { \
                        uint32_t* a_ptr = reinterpret_cast<uint32_t*>(acc_p_bf16 + ki * 8); \
                        uint64_t desc_v = desc_v_base + ((ki * 2048) >> 4); \
                        wgmma_m64n128k16_bf16_rs(acc_o, a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3], desc_v, 1); \
                    } \
                    wgmma_commit_group(); \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) asm volatile("" : "+f"(acc_o[j]) :: "memory"); \
                } while(0)
                
                #define COMPUTE_PV(v_stage_arg) do { \
                    COMPUTE_PV_NO_WAIT(v_stage_arg); \
                    wgmma_wait_group<0>(); \
                } while(0)
                
                // ===== IntraWGOverlap: QK[i] and PV[i-1] execute concurrently =====
                // Key insight: PV uses the PREVIOUS iteration's P (softmax result)
                // This allows QK and PV to overlap, improving WGMMA utilization
                //
                // Loop structure:
                // - Iteration 0: QK[0] + softmax[0] -> P[0], NO PV (acc_o stays 0)
                // - Iteration i (1 to n_blocks-1): QK[i] + PV[i-1] + softmax[i] -> P[i]
                // - After loop: PV[n_blocks-1] (final PV with last P)
                
                // Pre-allocate all register arrays at the beginning to help compiler reuse
                Element acc_p_bf16[64];  // P matrix storage (bf16 to reduce register pressure)
                float acc_s[64];         // QK output / softmax workspace
                float local_max[2];      // Local max for softmax
                float prev_scores_scale[2] = {1.0f, 1.0f};  // Scale for RescaleOBeforeGemm
                
                // FA3-style: compute n_block where causal mask is needed
                // For n_block < n_block_min_causal_mask, no mask is needed
                int n_idx_right = m_block * kBlockM + seqlen_k - seqlen_q;
                int n_block_min_causal_mask = max(0, n_idx_right / kBlockN);
                
                #pragma unroll
                for (int j = 0; j < 32; ++j) reinterpret_cast<uint32_t*>(acc_p_bf16)[j] = 0;
                
                // ===== First iteration (i=0): QK + softmax only, no PV =====
                {
                    int n_block = n_block_max - 1;
                    int k_stage = pipe_state_k.index();
                    int k_phase = pipe_state_k.phase();
                    
                    // FA3-style: try_wait + wait two-step pattern
                    bool k_ready = mbarrier_try_wait_parity(&shared_storage.full_barrier_K[k_stage], k_phase);
                    mbarrier_wait_if_needed(&shared_storage.full_barrier_K[k_stage], k_phase, k_ready);
                    
                    #pragma unroll
                    for (int j = 0; j < 64; ++j) acc_s[j] = 0.0f;
                    
                    COMPUTE_QK(k_stage);
                    warp_scheduler_barrier_arrive(warp_group_idx);
                    wgmma_wait_group<0>();
                    
                    // Release K immediately after QK completes
                    if (is_warpgroup_leader) {
                        mbarrier_arrive(&shared_storage.empty_barrier_K[k_stage]);
                    }
                    ++pipe_state_k;
                    
                    APPLY_MASK(n_block);
                    
                    // First iteration softmax: simplified (no rescale needed)
                    local_max[0] = -INFINITY; local_max[1] = -INFINITY;
                    #pragma unroll
                    for (int j = 0; j < 64; ++j) {
                        int mi = (j >> 1) & 1;
                        local_max[mi] = fmaxf(local_max[mi], acc_s[j]);
                    }
                    
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        local_max[mi] = fmaxf(local_max[mi], __shfl_xor_sync(0xffffffff, local_max[mi], 2));
                        local_max[mi] = fmaxf(local_max[mi], __shfl_xor_sync(0xffffffff, local_max[mi], 1));
                        row_max[mi] = local_max[mi];
                    }
                    
                    // FA3-style: column-major accumulation order (ni outer, mi inner)
                    float max_scaled_0 = row_max[0] * softmax_scale_log2;
                    float max_scaled_1 = row_max[1] * softmax_scale_log2;
                    #pragma unroll
                    for (int ni = 0; ni < 32; ++ni) {
                        int reg0 = 0 * 2 + (ni & 1) + ((ni >> 1) << 2);
                        int reg1 = 1 * 2 + (ni & 1) + ((ni >> 1) << 2);
                        float val0 = exp2f(acc_s[reg0] * softmax_scale_log2 - max_scaled_0);
                        float val1 = exp2f(acc_s[reg1] * softmax_scale_log2 - max_scaled_1);
                        acc_s[reg0] = val0;
                        acc_s[reg1] = val1;
                        row_sum[0] += val0;
                        row_sum[1] += val1;
                    }
                    
                    // Store P[0] for next iteration's PV (convert float to bf16)
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        float4 v = *reinterpret_cast<float4*>(acc_s + j * 4);
                        __nv_bfloat162 bf2_0 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.x));
                        __nv_bfloat162 bf2_1 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.z));
                        *reinterpret_cast<uint32_t*>(acc_p_bf16 + j * 4) = *reinterpret_cast<uint32_t*>(&bf2_0);
                        *reinterpret_cast<uint32_t*>(acc_p_bf16 + j * 4 + 2) = *reinterpret_cast<uint32_t*>(&bf2_1);
                    }
                }
                
                // ===== Middle iterations: QK[i] + PV[i-1] overlapped =====
                // FA3-style: split into two loops - one with mask, one without
                // This allows compiler to generate more efficient code for no-mask case
                
                // Macro for single iteration (with or without mask)
                // FA3-style: use try_wait + wait two-step pattern for better overlap
                #define FWD_STEP(n_block_arg, apply_mask_flag) do { \
                    int k_stage = pipe_state_k.index(); \
                    int v_stage = pipe_state_v.index(); \
                    int k_phase = pipe_state_k.phase(); \
                    int v_phase = pipe_state_v.phase(); \
                    \
                    /* Step 1: try_wait for K (non-blocking check) */ \
                    bool k_ready = mbarrier_try_wait_parity(&shared_storage.full_barrier_K[k_stage], k_phase); \
                    /* Step 2: try_wait for V (non-blocking check) */ \
                    bool v_ready = mbarrier_try_wait_parity(&shared_storage.full_barrier_V[v_stage], v_phase); \
                    \
                    /* Step 3: blocking wait for K if not ready */ \
                    mbarrier_wait_if_needed(&shared_storage.full_barrier_K[k_stage], k_phase, k_ready); \
                    warp_scheduler_barrier_sync(warp_group_idx); \
                    \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) acc_s[j] = 0.0f; \
                    COMPUTE_QK(k_stage); \
                    \
                    /* Step 4: blocking wait for V if not ready */ \
                    mbarrier_wait_if_needed(&shared_storage.full_barrier_V[v_stage], v_phase, v_ready); \
                    \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) { \
                        int mi = (j >> 1) & 1; \
                        acc_o[j] *= prev_scores_scale[mi]; \
                    } \
                    \
                    COMPUTE_PV_NO_WAIT(v_stage); \
                    warp_scheduler_barrier_arrive(warp_group_idx); \
                    wgmma_wait_group<1>(); \
                    \
                    if (is_warpgroup_leader) { \
                        mbarrier_arrive(&shared_storage.empty_barrier_K[k_stage]); \
                    } \
                    ++pipe_state_k; \
                    \
                    if (apply_mask_flag) { APPLY_MASK(n_block_arg); } \
                    \
                    local_max[0] = -INFINITY; local_max[1] = -INFINITY; \
                    _Pragma("unroll") \
                    for (int j = 0; j < 64; ++j) { \
                        int mi = (j >> 1) & 1; \
                        local_max[mi] = fmaxf(local_max[mi], acc_s[j]); \
                    } \
                    \
                    _Pragma("unroll") \
                    for (int mi = 0; mi < 2; ++mi) { \
                        local_max[mi] = fmaxf(local_max[mi], __shfl_xor_sync(0xffffffff, local_max[mi], 2)); \
                        local_max[mi] = fmaxf(local_max[mi], __shfl_xor_sync(0xffffffff, local_max[mi], 1)); \
                    } \
                    \
                    _Pragma("unroll") \
                    for (int mi = 0; mi < 2; ++mi) { \
                        float old_max = row_max[mi]; \
                        float new_max = local_max[mi]; \
                        row_max[mi] = fmaxf(old_max, new_max); \
                        prev_scores_scale[mi] = exp2f((old_max - row_max[mi]) * softmax_scale_log2); \
                        row_sum[mi] *= prev_scores_scale[mi]; \
                    } \
                    \
                    /* FA3-style: column-major accumulation order (ni outer, mi inner) */ \
                    /* Direct accumulation to row_sum (same as CUTLASS reduce_sum) */ \
                    float max_scaled_0 = row_max[0] * softmax_scale_log2; \
                    float max_scaled_1 = row_max[1] * softmax_scale_log2; \
                    /* Wait for previous PV WGMMA early to overlap with softmax */ \
                    wgmma_wait_group<0>(); \
                    _Pragma("unroll") \
                    for (int ni = 0; ni < 32; ++ni) { \
                        int reg0 = 0 * 2 + (ni & 1) + ((ni >> 1) << 2); \
                        int reg1 = 1 * 2 + (ni & 1) + ((ni >> 1) << 2); \
                        float val0 = exp2f(acc_s[reg0] * softmax_scale_log2 - max_scaled_0); \
                        float val1 = exp2f(acc_s[reg1] * softmax_scale_log2 - max_scaled_1); \
                        acc_s[reg0] = val0; \
                        acc_s[reg1] = val1; \
                        row_sum[0] += val0; \
                        row_sum[1] += val1; \
                    } \
                    \
                    if (is_warpgroup_leader) { \
                        mbarrier_arrive(&shared_storage.empty_barrier_V[v_stage]); \
                    } \
                    ++pipe_state_v; \
                    \
                    _Pragma("unroll") \
                    for (int j = 0; j < 16; ++j) { \
                        float4 v = *reinterpret_cast<float4*>(acc_s + j * 4); \
                        __nv_bfloat162 bf2_0 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.x)); \
                        __nv_bfloat162 bf2_1 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.z)); \
                        *reinterpret_cast<uint32_t*>(acc_p_bf16 + j * 4) = *reinterpret_cast<uint32_t*>(&bf2_0); \
                        *reinterpret_cast<uint32_t*>(acc_p_bf16 + j * 4 + 2) = *reinterpret_cast<uint32_t*>(&bf2_1); \
                    } \
                } while(0)
                
                // Loop 1: iterations that need causal mask (n_block >= n_block_min_causal_mask)
                for (int n_block = n_block_max - 2; n_block >= n_block_min_causal_mask; --n_block) {
                    FWD_STEP(n_block, true);
                }
                
                // Loop 2: iterations that don't need mask (n_block < n_block_min_causal_mask)
                for (int n_block = n_block_min_causal_mask - 1; n_block >= 0; --n_block) {
                    FWD_STEP(n_block, false);
                }
                
                #undef FWD_STEP
                
                // ===== Final PV: process last P[n_blocks-1] =====
                {
                    int v_stage = pipe_state_v.index();
                    int v_phase = pipe_state_v.phase();
                    
                    // Two-step wait for final V (same as FA3)
                    bool v_ready = mbarrier_try_wait_parity(&shared_storage.full_barrier_V[v_stage], v_phase);
                    mbarrier_wait_if_needed(&shared_storage.full_barrier_V[v_stage], v_phase, v_ready);
                    warp_scheduler_barrier_sync(warp_group_idx);
                    
                    // Apply final scale before last PV (RescaleOBeforeGemm)
                    #pragma unroll
                    for (int j = 0; j < 64; ++j) {
                        int mi = (j >> 1) & 1;
                        acc_o[j] *= prev_scores_scale[mi];
                    }
                    
                    COMPUTE_PV(v_stage);
                    
                    if (is_warpgroup_leader) {
                        mbarrier_arrive(&shared_storage.empty_barrier_V[v_stage]);
                    }
                    ++pipe_state_v;
                }
                
                // Undefine macros
                #undef COMPUTE_QK
                #undef APPLY_MASK
                #undef COMPUTE_PV_NO_WAIT
                #undef COMPUTE_PV
                
                // Scheduler barrier: signal other WG can proceed
                warp_scheduler_barrier_arrive(warp_group_idx);
                
                // Signal: Q is no longer needed, producer can load next Q
                // Use mbarrier for Q synchronization (not named barrier)
                if (is_warpgroup_leader) {
                    mbarrier_arrive(&shared_storage.empty_barrier_Q);
                }
                
                // ===== Final softmax: do deferred allreduce on row_sum =====
                // FA3 optimization: only do quad_allreduce once at the end
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    row_sum[mi] += __shfl_xor_sync(0xffffffff, row_sum[mi], 2);
                    row_sum[mi] += __shfl_xor_sync(0xffffffff, row_sum[mi], 1);
                }
                
                // ===== Final softmax normalization and write O (FA3 style) =====
                // Use epilogue_store: stmatrix R2S + vectorized S2G
                // Both warpgroups must participate together (256 threads total)
                // thread_idx for TiledMmaPV: warpgroup 1 uses 0-127, warpgroup 2 uses 128-255
                // mma_thread_idx = (warp_group_idx - 1) * 128 + (thread_idx & 127)
                // Simplified: mma_thread_idx = thread_idx - 128 (works for both warp_group_idx 1 and 2)
                {
                    int const mma_thread_idx = thread_idx - 128;
                    
                    epilogue_store(
                        acc_o,
                        row_sum,
                        shared_storage,
                        mma_thread_idx,  // 0-255 for TiledMmaPV
                        m_block,
                        bidh,
                        seqlen_q,
                        offset_q,
                        params.ptr_O,
                        params.stride_O_row,
                        params.headdim
                    );
                }
            } else {
                // n_blocks == 0: no work for this tile
                // named_barrier_arrive already done above after reading work_info
            }
        }
    }
    
}



// ============================================================================
// Device kernel wrapper
// ============================================================================
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7)))
#define GRID_CONSTANT __grid_constant__
#else
#define GRID_CONSTANT
#endif

static constexpr int kMinBlocksPerSM = 1;

__global__ void __launch_bounds__(kNumThreads, kMinBlocksPerSM)
flash_fwd_varlen_kernel(GRID_CONSTANT FlashFwdParams const params) {
    extern __shared__ char smem_buf[];
    
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    flash_fwd_kernel_body(params, smem_buf);
#endif
}



// ============================================================================
// Entry function
// ============================================================================
extern "C" void flash_attn_fwd_varlen(
    void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr,
    void* cu_seqlens_q, void* cu_seqlens_k,
    int total_q, int total_k, int max_seqlen_q, int max_seqlen_k,
    int batch, int nheads, int nkv_heads, int headdim,
    float softmax_scale, int is_causal, void* stream_ptr,
    void* workspace_lse, void* workspace_semaphore,
    void* seqused_q, void* seqused_k, void* leftpad_k
) {
    // Validate inputs
    if (headdim != 128) {
        fprintf(stderr, "flash_attn_fwd_varlen: only headdim=128 supported\n");
        return;
    }
    if (!cu_seqlens_q || !cu_seqlens_k) {
        fprintf(stderr, "flash_attn_fwd_varlen: cu_seqlens_q and cu_seqlens_k required\n");
        return;
    }
    if (!workspace_lse || !workspace_semaphore) {
        fprintf(stderr, "flash_attn_fwd_varlen: workspace_lse and workspace_semaphore required\n");
        return;
    }
    
    // Cast pointers
    Element const* Q = static_cast<Element const*>(q_ptr);
    Element const* K = static_cast<Element const*>(k_ptr);
    Element const* V = static_cast<Element const*>(v_ptr);
    Element* O = static_cast<Element*>(o_ptr);
    int const* cu_seqlens_q_ptr = static_cast<int const*>(cu_seqlens_q);
    int const* cu_seqlens_k_ptr = static_cast<int const*>(cu_seqlens_k);
    int const* seqused_q_ptr = seqused_q ? static_cast<int const*>(seqused_q) : nullptr;
    int const* seqused_k_ptr = seqused_k ? static_cast<int const*>(seqused_k) : nullptr;
    float* lse_ptr = static_cast<float*>(workspace_lse);
    int* semaphore_ptr = static_cast<int*>(workspace_semaphore);
    
    // Compute strides (row-major layout: [total, headdim, nheads] -> stride = [nheads*headdim, 1, headdim])
    // Actually layout is [total, nheads, headdim] with stride [nheads*headdim, headdim, 1]
    // But the original uses stride_Q = [q_row, 1, headdim] where q_row = nheads * headdim
    int64_t q_row_stride = static_cast<int64_t>(nheads) * headdim;
    int64_t kv_row_stride = static_cast<int64_t>(nkv_heads) * headdim;
    int64_t head_stride = static_cast<int64_t>(headdim);
    int64_t lse_stride = static_cast<int64_t>(total_q);
    
    // Softmax scale
    float softmax_scale_log2 = softmax_scale * static_cast<float>(M_LOG2E);
    
    // FastDivmod for GQA and tile scheduling
    FastDivmod qhead_per_khead_divmod(nheads / nkv_heads);
    int num_m_blocks = (max_seqlen_q + kBlockM - 1) >> 7;  // / 128
    FastDivmod num_m_blocks_divmod(num_m_blocks);
    
    // Get device info
    int num_sm = 0;
    ensure_device_info(num_sm);
    
    // Compute scheduler metadata
    int* d_num_splits = nullptr;
    int* d_num_m_blocks = nullptr;
    int* d_batch_idx = nullptr;
    int* d_nheads_l2 = nullptr;
    compute_scheduler_meta(batch, nheads, max_seqlen_q, max_seqlen_k,
                          d_num_splits, d_num_m_blocks, d_batch_idx, d_nheads_l2);
    
    // Stream
    cudaStream_t stream = stream_ptr ? static_cast<cudaStream_t>(stream_ptr) : cudaStream_t(0);
    
    // Initialize semaphore to 0 (like original FA3)
    // Each block uses blockIdx.x as initial tile_idx
    // Then uses atomicAdd(semaphore, 1) + gridDim.x to get next tile
    // This avoids synchronous cudaMemcpy to compute total_tiles on host
    CHECK_CUDA(cudaMemsetAsync(semaphore_ptr, 0, sizeof(int), stream));
    
    // Create TMA descriptors on host (will be passed via kernel params)
    // Q: [total_q, nheads, headdim] -> load [kBlockM, headdim] tiles
    // K: [total_k, nkv_heads, headdim] -> load [kBlockN, headdim] tiles
    // V: [total_k, nkv_heads, headdim] -> load [kBlockN, headdim] tiles
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    // Q: 3D tensor [headdim, total_q, nheads] in TMA's column-major view
    // For 128B swizzle, tile_dim0 must be <= 64 for bf16
    // We load [kBlockM, 64] per TMA call, need 2 calls for full headdim=128
    create_tma_descriptor_3d(
        &tma_Q,
        Q,
        headdim,                    // dim0: headdim (innermost)
        total_q,                    // dim1: total_q
        nheads,                     // dim2: nheads
        q_row_stride * sizeof(Element),  // stride1: between rows
        head_stride * sizeof(Element),   // stride2: between heads
        kTmaTileCols,               // tile_dim0: 64 (max for 128B swizzle)
        kBlockM                     // tile_dim1: 128
    );
    
    // K: 3D tensor [headdim, total_k, nkv_heads]
    create_tma_descriptor_3d(
        &tma_K,
        K,
        headdim,
        total_k,
        nkv_heads,
        kv_row_stride * sizeof(Element),
        head_stride * sizeof(Element),
        kTmaTileCols,               // tile_dim0: 64
        kBlockN                     // tile_dim1: 128
    );
    
    // V: Load V with transposed view for WGMMA col-major B matrix
    // V in global memory is [seqlen, headdim] per head (headdim contiguous)
    // WGMMA B matrix is col-major, so B[k, n] reads from address n * K + k
    // We need V^T in smem so that B[k, n] = V^T[n, k] = V[k, n]
    //
    // To achieve this, we create TMA descriptor with swapped view:
    // - Treat V as [headdim, seqlen] instead of [seqlen, headdim]
    // - dim0 = headdim (innermost, stride = 1 element)
    // - dim1 = seqlen (stride = kv_row_stride)
    // - This way TMA loads V^T into smem
    //
    // After TMA load, smem_v has V^T[headdim, seqlen] layout
    // WGMMA reads B[k, n] = smem_v[n * seqlen + k] = V^T[n, k] = V[k, n] ✓
    create_tma_descriptor_3d(
        &tma_V,
        V,
        headdim,                    // dim0: headdim (treat as innermost)
        total_k,                    // dim1: seqlen
        nkv_heads,                  // dim2: heads
        kv_row_stride * sizeof(Element),  // stride1: seqlen stride
        head_stride * sizeof(Element),    // stride2: head stride
        kTmaTileCols,               // tile_dim0: 64 (headdim tile)
        kBlockN                     // tile_dim1: 128 (seqlen tile)
    );
    
    // Build params struct
    FlashFwdParams params;
    params.ptr_Q = Q;
    params.stride_Q_row = q_row_stride;
    params.stride_Q_head = head_stride;
    params.ptr_K = K;
    params.stride_K_row = kv_row_stride;
    params.stride_K_head = head_stride;
    params.ptr_V = V;
    params.stride_V_row = kv_row_stride;
    params.stride_V_head = head_stride;
    params.ptr_O = O;
    params.stride_O_row = q_row_stride;
    params.stride_O_head = head_stride;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    
    params.seqused_q = seqused_q_ptr;
    params.seqused_k = seqused_k_ptr;
    params.total_q = total_q;
    params.total_k = total_k;
    params.max_seqlen_q = max_seqlen_q;
    params.max_seqlen_k = max_seqlen_k;
    params.batch = batch;
    params.nheads = nheads;
    params.nkv_heads = nkv_heads;
    params.headdim = headdim;
    params.is_causal = is_causal;
    params.softmax_scale_log2 = softmax_scale_log2;
    params.ptr_LSE = lse_ptr;
    params.stride_LSE = lse_stride;
    params.tile_count_semaphore = semaphore_ptr;
    params.num_splits_ptr = d_num_splits;
    params.num_m_blocks_ptr = d_num_m_blocks;
    params.batch_idx_ptr = d_batch_idx;
    params.nheads_l2_ptr = d_nheads_l2;
    params.qhead_per_khead_divmod = qhead_per_khead_divmod;
    params.num_m_blocks_divmod = num_m_blocks_divmod;
    params.tma_Q = tma_Q;
    params.tma_K = tma_K;
    params.tma_V = tma_V;
    
    // Grid and block dimensions
    dim3 grid(num_sm);
    dim3 block(kNumThreads);  // 384 threads
    
    // Shared memory size
    int smem_size = kSharedMemSize;
    
    // Set max dynamic shared memory
    auto kernel = flash_fwd_varlen_kernel;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));
    }
    
    // Set cluster size (1x1x1 for this kernel)
    CHECK_CUDA(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1
    ));
    
    // Launch kernel
    kernel<<<grid, block, smem_size, stream>>>(params);
}
