// level5/matmul.cu
// SM100 (GB200) BF16 GEMM — Persistent Kernel + 2D Swizzle Tile Scheduler
//
// Improvements over level4:
//   1. Persistent kernel: launch gridDim = num_sms (not num_tiles × cluster_size).
//      Each cluster processes MULTIPLE output tiles in a while loop, reusing TMEM
//      allocation and keeping shared memory hot. Eliminates per-tile kernel launch overhead.
//   2. 2D swizzle tile scheduler: groups consecutive M blocks (group_size=8) and iterates
//      N blocks within each group. This keeps B-tile data in L2 cache across 8 consecutive
//      tiles that share the same N block → massive L2 cache reuse.
//   3. Inherits all level4 optimizations (SMEM-staged coalesced epilogue, true warp
//      specialization, 8-stage deep pipeline, cta_group::2 TMA with PEER_BIT_MASK).
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
static constexpr uint32_t NUM_STAGES = 8;

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
static constexpr uint32_t UMMA_SBO = 8u * BLOCK_K * sizeof(__nv_bfloat16);  // = 1024 bytes

static constexpr uint32_t TMEM_COLS = 128;
static constexpr uint32_t NUM_THREADS = 128;

// SMEM layout: [barriers | pad to 1024 | stage data × NUM_STAGES]
static constexpr uint32_t OFF_DATA  = 1024;
static constexpr uint32_t SMEM_SIZE = OFF_DATA + NUM_STAGES * SMEM_STAGE;

// Peer bit mask: clears bit 24 to route barrier signal to leader CTA (rank 0)
static constexpr uint32_t PEER_BIT_MASK = 0xFEFFFFFF;

// 2D swizzle group size: consecutive M blocks grouped for L2 B-data reuse
static constexpr uint32_t SWIZZLE_GROUP_SIZE = 8;

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
//  Descriptors (see level4 for detailed bit-layout comments)
// ============================================================

// SmemDescriptor (64-bit): tells UMMA where to find operand data in SMEM
//   [13:0]  start_address (>>4), [45:32] SBO (>>4), [47:46] version=1, [63:61] layout=2 (SWIZZLE_128B)
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = 2 (SWIZZLE_128B)
    return d;
}

// InstrDescriptor (32-bit): encodes data types + dimensions for tcgen05.mma
//   [5:4] c_format=1(FP32), [9:7] a_format=1(BF16), [12:10] b_format=1(BF16)
//   [22:17] n_dim=N/8, [28:24] m_dim=M/16
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);              // c_format = 1 → FP32 accumulator
    d |= (1u << 7);              // a_format = 1 → BF16
    d |= (1u << 10);             // b_format = 1 → BF16
    d |= ((N / 8) << 17);        // n_dim = N/8
    d |= ((M / 16) << 24);       // m_dim = M/16
    return d;
}

// ============================================================
//  2D Swizzle Tile Scheduler (Persistent Kernel)
// ============================================================
//
// Each cluster picks tiles in round-robin order:
//   linear_tile_idx = (++iter) * num_clusters + my_cluster_id
//
// 2D swizzle converts linear_tile_idx → (m_block, n_block):
//   Groups of SWIZZLE_GROUP_SIZE consecutive M blocks share the same N block,
//   keeping B-tile data hot in L2 cache.
//
//   Example: 64 M blocks, 64 N blocks, GROUP_SIZE=8 →
//     Group 0: M=[0..7], iterate over all N blocks  (B[n] stays in L2 for 8 tiles)
//     Group 1: M=[8..15], iterate over all N blocks
//     ...
//
// This is the same strategy as DeepGEMM's Scheduler::get_swizzled_block_idx().
//
struct TileScheduler {
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_tiles;
    uint32_t num_clusters;
    uint32_t cluster_id;
    int      current_iter;

    __device__ TileScheduler(uint32_t M, uint32_t N, uint32_t num_clusters_)
        : num_m_blocks((M + BLOCK_M - 1) / BLOCK_M),
          num_n_blocks((N + BLOCK_N - 1) / BLOCK_N),
          num_tiles(num_m_blocks * num_n_blocks),
          num_clusters(num_clusters_),
          cluster_id(blockIdx.x / CLUSTER_SIZE),
          current_iter(-1) {}

    __device__ bool get_next_block(uint32_t& m_block, uint32_t& n_block) {
        uint32_t tile_idx = static_cast<uint32_t>(++current_iter) * num_clusters + cluster_id;
        if (tile_idx >= num_tiles)
            return false;

        // ---- 2D swizzle for L2 cache reuse ----
        // Group SWIZZLE_GROUP_SIZE M blocks together.
        // Within a group, enumerate tiles as (m_in_group, n) in column-major:
        //   tile 0: (m=first+0, n=0), tile 1: (m=first+1, n=0), ...,
        //   tile G: (m=first+0, n=1), tile G+1: (m=first+1, n=1), ...
        // → consecutive tiles share the same n_block → B data reuse in L2
        uint32_t tiles_per_group = num_n_blocks * SWIZZLE_GROUP_SIZE;
        uint32_t group_idx       = tile_idx / tiles_per_group;
        uint32_t first_m         = group_idx * SWIZZLE_GROUP_SIZE;
        uint32_t in_group_idx    = tile_idx % tiles_per_group;
        uint32_t m_in_group      = min(SWIZZLE_GROUP_SIZE, num_m_blocks - first_m);

        m_block = first_m + in_group_idx % m_in_group;
        n_block = in_group_idx / m_in_group;
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
    __nv_bfloat16* __restrict__ D,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t num_clusters)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    const uint32_t warp_idx  = get_warp_id();
    const uint32_t cta_rank  = get_cluster_rank();
    const bool     is_leader = (cta_rank == 0);

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

    // Initialize barriers (first time)
    if (warp_idx == 1 && elect_one()) {
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            barrier_init(full_bar[s],  CLUSTER_SIZE);
            barrier_init(empty_bar[s], 1);
        }
        barrier_init(tmem_bar, 1);
        fence_barrier_init();
    }

    // TMEM allocation — persists across ALL tiles (allocated once, deallocated at end)
    if (warp_idx == 2) tmem_alloc_2sm(tmem_addr, TMEM_COLS);

    cluster_sync();

    // ---- UMMA descriptors (same for all tiles, only depends on SMEM layout) ----
    const uint32_t num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);
    uint32_t alo[NUM_STAGES], blo[NUM_STAGES], ahi, bhi;
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        uint64_t ad = make_smem_desc(get_smem_a(s), UMMA_SBO);
        uint64_t bd = make_smem_desc(get_smem_b(s), UMMA_SBO);
        alo[s] = static_cast<uint32_t>(ad);
        blo[s] = static_cast<uint32_t>(bd);
        if (s == 0) { ahi = static_cast<uint32_t>(ad >> 32); bhi = static_cast<uint32_t>(bd >> 32); }
    }

    // ---- Persistent tile loop ----
    TileScheduler scheduler(M, N, num_clusters);
    uint32_t m_block, n_block;

    while (scheduler.get_next_block(m_block, n_block)) {
        // Per-tile coordinates
        const int32_t m_coord = m_block * BLOCK_M;
        const int32_t n_coord = n_block * BLOCK_N + cta_rank * LOAD_N_PER_CTA;

        // ========== Warp-specialized K-loop (same as level4) ==========
        if (warp_idx == 0 && elect_one()) {
            // ---- TMA Warp (both CTAs) ----
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
            // ---- MMA Warp (leader CTA only) ----
            uint32_t mma_stage = 0, mma_phase = 0;
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                barrier_wait(full_bar[mma_stage], mma_phase);
                tcgen05_fence_after();

                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                    uint64_t ad = ((uint64_t)ahi << 32) | (alo[mma_stage] + k * 2);
                    uint64_t bd = ((uint64_t)bhi << 32) | (blo[mma_stage] + k * 2);
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

        if (is_leader) {
            __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_DATA);

            // Phase 1: TMEM → SMEM
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x(col, r0, r1, r2, r3);
                tmem_load_fence();

                uint32_t base = threadIdx.x * BLOCK_N + col;
                smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
                smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
                smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
                smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
            }

            __syncthreads();

            // Phase 2: SMEM → Global (coalesced vectorized writes)
            const uint32_t warp_id = threadIdx.x / 32;
            const uint32_t lane_id = threadIdx.x % 32;

            #pragma unroll 4
            for (uint32_t step = 0; step < BLOCK_M / 4; ++step) {
                uint32_t row = step * 4 + warp_id;
                uint32_t global_row = m_block * BLOCK_M + row;
                uint32_t col_start  = lane_id * 4;
                uint32_t global_col = n_block * BLOCK_N + col_start;

                if (global_row < M && global_col + 3 < N) {
                    uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BLOCK_N + col_start]);
                    *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
                }
            }
        }

        // ========== Re-initialize barriers for the next tile ==========
        // cluster_sync ensures epilogue is complete before we touch barriers
        cluster_sync();
        if (warp_idx == 1 && elect_one()) {
            for (uint32_t s = 0; s < NUM_STAGES; ++s) {
                barrier_init(full_bar[s],  CLUSTER_SIZE);
                barrier_init(empty_bar[s], 1);
            }
            barrier_init(tmem_bar, 1);
            fence_barrier_init();
        }
        // cluster_sync ensures barriers are visible before next tile's K-loop
        cluster_sync();

    } // end while (persistent tile loop)

    // ---- Cleanup: deallocate TMEM (once, at end) ----
    cluster_sync();
    if (warp_idx == 2) tmem_dealloc_2sm(0, TMEM_COLS);
#endif
}

// ============================================================
//  Host — TMA Descriptor Creation
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
    uint32_t num_tiles    = num_m_blocks * num_n_blocks;

    // Query number of SMs on this GPU
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    // Launch persistent kernel: one CTA per SM, grouped into clusters
    // Don't launch more clusters than tiles
    uint32_t num_clusters = min((uint32_t)(num_sms / CLUSTER_SIZE), num_tiles);
    uint32_t num_ctas     = num_clusters * CLUSTER_SIZE;

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
                       tma_a, tma_b, D, Mu, Nu, Ku, num_clusters);
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
