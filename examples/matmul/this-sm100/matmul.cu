// Referenced from DeepGEMM. For details, please see the README.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/container/tuple.hpp>

using bf16_t = cutlass::bfloat16_t;

#define UNROLL _Pragma("unroll")

template <typename T>
__device__ __forceinline__ int cast_into_bf16_and_pack(T& x, T& y) {
  auto bf16x2 = __float22bfloat162_rn(
      {*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
  return *reinterpret_cast<int*>(&bf16x2);
}

template <typename T>
__device__ __host__ __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <uint32_t kNumCols>
__device__ constexpr uint32_t get_num_aligned_tmem_cols() {
  static_assert(kNumCols <= 512, "Too many tensor memory columns");
  if (kNumCols <= 32) return 32;
  if (kNumCols <= 64) return 64;
  if (kNumCols <= 128) return 128;
  if (kNumCols <= 256) return 256;
  return 512;
}

__device__ __forceinline__ void st_shared(const void* ptr, uint32_t x,
                                          uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(x),
               "r"(y), "r"(z), "r"(w));
}

template <typename P>
__device__ __forceinline__ auto alloc(P& top_ptr, uint32_t size,
                                      uintptr_t align = 1) -> P {
  top_ptr = reinterpret_cast<P>(
      ceil_div(reinterpret_cast<uintptr_t>(top_ptr), align) * align);
  auto old_ptr = top_ptr;
  top_ptr += size;
  return old_ptr;
}

template <uint32_t NUM_STAGES>
struct PipelineState {
  uint32_t index{0}, stage{0}, phase{0};

  __device__ __forceinline__ void next() {
    this->index++;
    this->stage = this->index % NUM_STAGES;
    this->phase = (this->index / NUM_STAGES) & 1;
  }
};

// constants
constexpr uint32_t MAX_TMEM_ROWS = 128;
constexpr uint32_t MAX_TMEM_COLS = 512;
constexpr uint32_t SWIZZLE_SIZE = 128;
constexpr uint32_t CTA_PAIR_SIZE = 2;

constexpr uint32_t NUM_NON_EPI_THRDS = 128, NUM_EPI_THRDS = 128;
constexpr uint32_t NUM_THRDS = NUM_NON_EPI_THRDS + NUM_EPI_THRDS;

constexpr auto MAJOR_A = cute::UMMA::Major::K;
constexpr auto MAJOR_B = cute::UMMA::Major::K;
constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64;
constexpr uint32_t NUM_SMEM_AB_STAGES = 4;
constexpr uint32_t NUM_SMEM_C_STAGES = 2;

constexpr uint32_t NUM_UMMA_M_WAVES = BLOCK_M / MAX_TMEM_ROWS;  // 2
constexpr uint32_t NUM_TMEM_C_STAGES =
    MAX_TMEM_COLS / (NUM_UMMA_M_WAVES * BLOCK_N);  // 1
constexpr uint32_t NUM_TMEM_COLS =
    get_num_aligned_tmem_cols<NUM_TMEM_C_STAGES * NUM_UMMA_M_WAVES *
                              BLOCK_N>();                   // 512
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;                  // 256
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / CTA_PAIR_SIZE;  // 128
constexpr uint32_t STORE_BLOCK_M =
    cute::min<uint32_t>(BLOCK_M, MAX_TMEM_ROWS);                   // 128
constexpr uint32_t STORE_BLOCK_N = SWIZZLE_SIZE / sizeof(bf16_t);  // 128/2=64

constexpr uint32_t SMEM_C_SIZE_PER_STAGE =
    STORE_BLOCK_M * STORE_BLOCK_N * sizeof(bf16_t);  // 128*128=16KB
constexpr uint32_t SMEM_A_SIZE_PER_STAGE =
    LOAD_BLOCK_M * BLOCK_K * sizeof(bf16_t);  // 256*64*2=32KB
constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
    LOAD_BLOCK_N * BLOCK_K * sizeof(bf16_t);  // 128*64*2=16KB

constexpr uint32_t BAR_SIZE = sizeof(cutlass::arch::ClusterTransactionBarrier);
constexpr uint32_t SMEM_SIZE =
    (SMEM_C_SIZE_PER_STAGE + 2 * BAR_SIZE) * NUM_SMEM_C_STAGES +
    (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + 2 * BAR_SIZE) *
        NUM_SMEM_AB_STAGES +
    sizeof(uint32_t);

__global__ void __launch_bounds__(NUM_THRDS, 1)
    __cluster_dims__(CTA_PAIR_SIZE, 1, 1)
        matmul_sm100_bf16_2sm_256x256x64_kernel(
            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
            const __grid_constant__ cute::TmaDescriptor tensor_map_c,
            uint32_t shape_m, uint32_t shape_n, uint32_t shape_k) {
  // prefetch tma-descs
  if (threadIdx.x == 0) {
    cute::prefetch_tma_descriptor(&tensor_map_a);
    cute::prefetch_tma_descriptor(&tensor_map_b);
    cute::prefetch_tma_descriptor(&tensor_map_c);
  }

  // ids
  bool is_leader_cta = cute::block_rank_in_cluster() == 0;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = cutlass::canonical_lane_idx();

  // shared memory allocation
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  uint8_t* smem_top_ptr = smem_buffer;
  bf16_t* smem_c[NUM_SMEM_C_STAGES];
  UNROLL for (uint32_t i = 0; i < NUM_SMEM_C_STAGES; ++i) {
    smem_c[i] = reinterpret_cast<bf16_t*>(
        alloc(smem_top_ptr, SMEM_C_SIZE_PER_STAGE, 1024));
  }
  bf16_t* smem_a[NUM_SMEM_AB_STAGES];
  UNROLL for (uint32_t i = 0; i < NUM_SMEM_AB_STAGES; ++i) {
    smem_a[i] = reinterpret_cast<bf16_t*>(
        alloc(smem_top_ptr, SMEM_A_SIZE_PER_STAGE, 1024));
  }
  bf16_t* smem_b[NUM_SMEM_AB_STAGES];
  UNROLL for (uint32_t i = 0; i < NUM_SMEM_AB_STAGES; ++i) {
    smem_b[i] = reinterpret_cast<bf16_t*>(
        alloc(smem_top_ptr, SMEM_B_SIZE_PER_STAGE, 1024));
  }
  auto bar_top_ptr =
      reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_top_ptr);
  auto sab_ready_bars = alloc(bar_top_ptr, NUM_SMEM_AB_STAGES);
  auto sab_empty_bars = alloc(bar_top_ptr, NUM_SMEM_AB_STAGES);
  auto tc_ready_bars = alloc(bar_top_ptr, NUM_TMEM_C_STAGES);
  auto tc_empty_bars = alloc(bar_top_ptr, NUM_TMEM_C_STAGES);

  // tensor memory allocation
  if (warp_idx == 0) {
    cute::TMEM::Allocator2Sm().allocate(
        NUM_TMEM_COLS, reinterpret_cast<uint32_t*>(bar_top_ptr));
  }

  // barrier initialization
  if (threadIdx.x == 0) {
    UNROLL for (uint32_t i = 0; i < NUM_SMEM_AB_STAGES; ++i) {
      // arrived by CTA0/1-Warp0-Lane0
      sab_ready_bars[i].init(CTA_PAIR_SIZE);
      // arrived by CTA0-Warp1-Lane0
      sab_empty_bars[i].init(1);
    }
    UNROLL for (uint32_t i = 0; i < NUM_TMEM_C_STAGES; ++i) {
      // arrived by CTA0-Warp1-Lane0
      tc_ready_bars[i].init(1);
      // arrived by CTA0/1-Warp4/5/6/7
      tc_empty_bars[i].init(CTA_PAIR_SIZE * NUM_EPI_THRDS);
    }
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::fence_barrier_init();
  }

  // sync
  cute::cluster_sync();

  // scheduler
  constexpr uint32_t GROUP_SIZE = 16;
  const uint32_t num_m_blks = ceil_div(shape_m, BLOCK_M);
  const uint32_t num_n_blks = ceil_div(shape_n, BLOCK_N);
  const uint32_t num_k_blks = ceil_div(shape_k, BLOCK_K);
  const uint32_t num_mn_blks = num_m_blks * num_n_blks;
  uint32_t m_blk_idx, n_blk_idx;
  uint32_t cur_mn_blk_idx = blockIdx.x;
  auto get_next_mn_block = [&]() -> bool {
    if (cur_mn_blk_idx >= num_mn_blks) return false;
    const auto num_blks_per_grp = num_n_blks * GROUP_SIZE;
    const auto grp_idx = cur_mn_blk_idx / num_blks_per_grp;
    const auto fst_blk_idx = grp_idx * GROUP_SIZE;
    const auto in_grp_idx = cur_mn_blk_idx % num_blks_per_grp;
    const auto num_blks_in_grp = min(GROUP_SIZE, num_m_blks - fst_blk_idx);
    m_blk_idx = fst_blk_idx + in_grp_idx % num_blks_in_grp;
    n_blk_idx = in_grp_idx / num_blks_in_grp;
    cur_mn_blk_idx += gridDim.x;
    return true;
  };
  PipelineState<NUM_SMEM_AB_STAGES> sab_pipe;
  PipelineState<NUM_TMEM_C_STAGES> tc_pipe;

  if (warp_idx == 0) {  // TMA Load worker
    while (get_next_mn_block()) {
      uint32_t m_idx = m_blk_idx * BLOCK_M;
      uint32_t n_idx =
          n_blk_idx * BLOCK_N + cute::block_rank_in_cluster() * LOAD_BLOCK_N;
      for (uint32_t k_idx = 0; k_idx < shape_k; k_idx += BLOCK_K) {
        // SMEM-AB-pipeline: wait-consumer-empty
        sab_empty_bars[sab_pipe.stage].wait(sab_pipe.phase ^ 1);
        // SMEM-AB-pipeline: producer-execute
        if (cute::elect_one_sync()) {
          auto& bar = sab_ready_bars[sab_pipe.stage];
          cute::SM100_TMA_2SM_LOAD_2D::copy(
              &tensor_map_a, reinterpret_cast<uint64_t*>(&bar),
              static_cast<uint32_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_a[sab_pipe.stage], k_idx, m_idx);
          cute::SM100_TMA_2SM_LOAD_2D::copy(
              &tensor_map_b, reinterpret_cast<uint64_t*>(&bar),
              static_cast<uint32_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_b[sab_pipe.stage], k_idx, n_idx);
        }
        // SMEM-AB-pipeline: commit-producer-ready
        if (cute::elect_one_sync()) {
          auto& bar = sab_ready_bars[sab_pipe.stage];
          if (is_leader_cta) {
            bar.arrive_and_expect_tx(CTA_PAIR_SIZE * (SMEM_A_SIZE_PER_STAGE +
                                                      SMEM_B_SIZE_PER_STAGE));
          } else {
            bar.arrive(0u);
          }
        }
        // SMEM-AB-pipeline: next-stage
        sab_pipe.next();
      }
    }
  } else if (warp_idx == 1 and is_leader_cta) {  // UMMA worker

    constexpr uint32_t UMMA_M = MAX_TMEM_ROWS * CTA_PAIR_SIZE;  // 128*2=256
    constexpr uint32_t UMMA_N = BLOCK_N;                        // 256
    constexpr uint32_t UMMA_K = 32 / sizeof(bf16_t);            // 16

    while (get_next_mn_block()) {
      // TMEM-C-pipeline: wait-consumer-empty
      tc_empty_bars[tc_pipe.stage].wait(tc_pipe.phase ^ 1);
      // asm volatile("tcgen05.fence::after_thread_sync;");

      constexpr uint16_t CTA_MASK = (1 << CTA_PAIR_SIZE) - 1;
      auto make_8x128B_atom_smem_desc = []() {
        cute::UMMA::SmemDescriptor desc;
        desc.version_ = 1, desc.lbo_mode_ = 0, desc.base_offset_ = 0;
        desc.layout_type_ =
            static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
        desc.stride_byte_offset_ = (8 * SWIZZLE_SIZE) >> 4;
        desc.leading_byte_offset_ = 0;
        return desc;
      };
      auto a_desc = make_8x128B_atom_smem_desc();
      auto b_desc = make_8x128B_atom_smem_desc();
      auto instr_desc = cute::UMMA::make_runtime_instr_desc(
          cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, UMMA_M, UMMA_N,
                                      MAJOR_A, MAJOR_B>());
      bool inited = false;

      // TMEM-C-pipeline: producer-execute
      for (uint32_t k_idx = 0; k_idx < shape_k; k_idx += BLOCK_K) {
        // SMEM-AB-pipeline: wait-producer-ready
        sab_ready_bars[sab_pipe.stage].wait(sab_pipe.phase);
        // asm volatile("tcgen05.fence::after_thread_sync;");

        auto cur_smem_a = smem_a[sab_pipe.stage];
        auto cur_smem_b = smem_b[sab_pipe.stage];

        using cute_mma_t =
            cute::SM100_MMA_F16BF16_2x1SM_SS<bf16_t, bf16_t, float, UMMA_M,
                                             UMMA_N, MAJOR_A, MAJOR_B>;

        // SMEM-AB-pipeline: consumer-execute
        UNROLL for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
          UNROLL for (uint32_t w = 0; w < NUM_UMMA_M_WAVES; ++w) {
            a_desc.start_address_ = static_cast<uint16_t>(
                cute::cast_smem_ptr_to_uint(
                    cur_smem_a + w * MAX_TMEM_ROWS * BLOCK_K + k * UMMA_K) >>
                4);
            b_desc.start_address_ = static_cast<uint16_t>(
                cute::cast_smem_ptr_to_uint(cur_smem_b + k * UMMA_K) >> 4);
            cute_mma_t::fma(
                a_desc, b_desc,
                tc_pipe.stage * NUM_UMMA_M_WAVES * BLOCK_N + w * BLOCK_N,
                inited, instr_desc);
          }
          inited = true;
        }
        // SMEM-AB-pipeline: commit-consumer-empty
        // asm volatile("tcgen05.fence::before_thread_sync;");
        cutlass::arch::umma_arrive_multicast_2x1SM(
            reinterpret_cast<uint64_t*>(&sab_empty_bars[sab_pipe.stage]),
            CTA_MASK);
        // SMEM-AB-pipeline: next-stage
        sab_pipe.next();
      }
      // TMEM-C-pipeline: commit-producer-ready
      // asm volatile("tcgen05.fence::before_thread_sync;");
      cutlass::arch::umma_arrive_multicast_2x1SM(
          reinterpret_cast<uint64_t*>(&tc_ready_bars[tc_pipe.stage]), CTA_MASK);
      // TMEM-C-pipeline: next-stage
      tc_pipe.next();
    }
  } else if (threadIdx.x >= NUM_NON_EPI_THRDS) {  // Epilogue worker
    const auto epi_thrd_idx = threadIdx.x - NUM_NON_EPI_THRDS;
    const auto epi_warp_idx = warp_idx - (NUM_NON_EPI_THRDS / 32);

    constexpr uint32_t BANK_GROUP_SIZE = 16;
    constexpr uint32_t NUM_ELEMS_PER_BANK_GROUP =
        BANK_GROUP_SIZE / sizeof(bf16_t);  // 16/4=8

    PipelineState<NUM_SMEM_C_STAGES> sc_pipe;
    while (get_next_mn_block()) {
      // TMEM-C-pipeline: wait-producer-ready
      tc_ready_bars[tc_pipe.stage].wait(tc_pipe.phase);
      // asm volatile("tcgen05.fence::after_thread_sync;");

      // TMEM-C-pipeline: consumer-execute
      UNROLL for (uint32_t w = 0; w < NUM_UMMA_M_WAVES; ++w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;  // 256/64=4
        UNROLL for (uint32_t s = 0; s < kNumStores; ++s) {
          const uint32_t m_idx = m_blk_idx * BLOCK_M + w * MAX_TMEM_ROWS;
          const uint32_t n_idx = n_blk_idx * BLOCK_N + s * STORE_BLOCK_N;

          // SMEM-C-pipeline: wait-consumer-empty
          if (sc_pipe.index >= NUM_SMEM_C_STAGES) {
            if (epi_thrd_idx == 0)
              cute::tma_store_wait<NUM_SMEM_C_STAGES - 1>();
            cutlass::arch::NamedBarrier(NUM_EPI_THRDS).sync();
          }

          // SMEM-C-pipeline: producer-execute, commit-producer-ready
          UNROLL for (uint32_t i = 0;
                      i < STORE_BLOCK_N / NUM_ELEMS_PER_BANK_GROUP; ++i) {
            uint32_t tmem_addr = tc_pipe.stage * NUM_UMMA_M_WAVES * BLOCK_N +
                                 w * BLOCK_N + s * STORE_BLOCK_N +
                                 i * NUM_ELEMS_PER_BANK_GROUP;
            uint32_t values[NUM_ELEMS_PER_BANK_GROUP];
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(
                tmem_addr, values[0], values[1], values[2], values[3],
                values[4], values[5], values[6], values[7]);
            cutlass::arch::fence_view_async_tmem_load();

            auto row = lane_idx;
            auto col = i ^ (row % (SWIZZLE_SIZE / 16));
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_c[sc_pipe.stage]) +
                            epi_warp_idx * 32 * SWIZZLE_SIZE +
                            row * (BANK_GROUP_SIZE * 8) + col * BANK_GROUP_SIZE;
            st_shared(smem_ptr, cast_into_bf16_and_pack(values[0], values[1]),
                      cast_into_bf16_and_pack(values[2], values[3]),
                      cast_into_bf16_and_pack(values[4], values[5]),
                      cast_into_bf16_and_pack(values[6], values[7]));
          }

          // SMEM-C-pipeline: wait-producer-ready
          cute::tma_store_fence();
          cutlass::arch::NamedBarrier(NUM_EPI_THRDS).sync();

          // SMEM-C-pipeline: consumer-execute
          if (epi_thrd_idx == 0) {
            cute::SM90_TMA_STORE_2D::copy(&tensor_map_c, smem_c[sc_pipe.stage],
                                          n_idx, m_idx);
          }

          // SMEM-C-pipeline: commit-consumer-empty
          if (epi_thrd_idx == 0) {
            cute::tma_store_arrive();
          }

          // SMEM-C-pipeline: next-stage
          sc_pipe.next();
        }
      }

      // TMEM-C-pipeline: commit-consumer-empty
      // asm volatile("tcgen05.fence::before_thread_sync;");
      tc_empty_bars[tc_pipe.stage].arrive(0u);

      // TMEM-C-pipeline: next-stage
      tc_pipe.next();
    }

    if (epi_thrd_idx == 0) cute::tma_store_wait<0>();
    if (epi_warp_idx == 1) cute::TMEM::Allocator2Sm().free(0, NUM_TMEM_COLS);
  }
}

#define CRZ_STR_(x) #x
#define CRZ_STR(x) CRZ_STR_(x)
#define CRZ_CHECK(__cond, __action, __printf_args...)             \
  do {                                                            \
    if (!(__cond)) {                                              \
      printf(__FILE__ ":" CRZ_STR(__LINE__) ": check \"" CRZ_STR( \
          __cond) "\" failed: " __printf_args);                   \
      printf("\n");                                               \
      __action;                                                   \
    }                                                             \
  } while (0)
#define CUDA_CHECK(cmd, act)                                             \
  do {                                                                   \
    const auto& err = (cmd);                                             \
    CRZ_CHECK(err == cudaSuccess, act, "%s - %s", cudaGetErrorName(err), \
              cudaGetErrorString(err));                                  \
  } while (0)
#define CUDA_DRIVER_CHECK(cmd, act)                  \
  do {                                               \
    const auto& err = (cmd);                         \
    const char *name, *info;                         \
    CRZ_CHECK(err == CUDA_SUCCESS, act, "%s - %s",   \
              (cuGetErrorName(err, &name), name),    \
              (cuGetErrorString(err, &info), info)); \
  } while (0)

template <typename T>
static CUtensorMap make_2d_tensor_map(T* data_ptr, int gmem_inner_dim,
                                      int gmem_outer_dim, int smem_inner_dim,
                                      int smem_outer_dim,
                                      const int& gmem_outer_stride) {
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * sizeof(T)),
  };
  const cuuint32_t elem_strides[2] = {1, 1};
  CUDA_DRIVER_CHECK(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
          reinterpret_cast<void*>(data_ptr), gmem_dims, gmem_strides, smem_dims,
          elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      throw std::runtime_error("cuda driver check failed"));
  return tensor_map;
}

extern "C" int matmul_sm100_bf16_2sm_256x256x64(bf16_t* A, bf16_t* B, bf16_t* C,
                                                uint32_t shape_m,
                                                uint32_t shape_n,
                                                uint32_t shape_k) {
  static bool init_flag = false;
  if (!init_flag) {
    CUDA_CHECK(cudaFuncSetAttribute(matmul_sm100_bf16_2sm_256x256x64_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    SMEM_SIZE),
               return -1);
    init_flag = true;
  }

  const auto tensor_map_a =
      make_2d_tensor_map(A, shape_k, shape_m, BLOCK_K, LOAD_BLOCK_M, shape_k);
  const auto tensor_map_b =
      make_2d_tensor_map(B, shape_k, shape_n, BLOCK_K, LOAD_BLOCK_N, shape_k);
  const auto tensor_map_c = make_2d_tensor_map(
      C, shape_n, shape_m, STORE_BLOCK_N, STORE_BLOCK_M, shape_n);

  dim3 grid(ceil_div(shape_m, BLOCK_M) * ceil_div(shape_n, BLOCK_N), 1, 1);
  dim3 block(NUM_THRDS, 1, 1);
  matmul_sm100_bf16_2sm_256x256x64_kernel<<<grid, block, SMEM_SIZE>>>(
      tensor_map_a, tensor_map_b, tensor_map_c, shape_m, shape_n, shape_k);
  CUDA_CHECK(cudaGetLastError(), return -1);
  return 0;
}
