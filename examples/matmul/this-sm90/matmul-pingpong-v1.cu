#include "barrier.h"
#include "common.h"
#include "descriptor.h"
#include "pipeline.h"
#include "reference.h"
#include "scheduler.h"
#include "tma.h"

const int testM = 4096;
const int testN = 4096;
const int testK = 4096;
const int iters = 100;
static constexpr int CLUSTER_M = 2;
static constexpr int CLUSTER_N = 1;
static constexpr int WG_NUMBER = 3;
static constexpr int BLOCKM = 128;
static constexpr int BLOCKN = 128;
static constexpr int BLOCKK = 64;
static constexpr int STAGES = 7;

/// RUN:
/// nvcc -arch=sm_90a -I ../../../include -lcuda -std=c++17
/// matmul-pingpong-v1.cu -o test && ./test
/// |& tee trace.log

enum class Major : uint8_t {
  MajorK,
  MajorMN,
};

enum class ScaleOut {
  Zero = 0,
  One = 1,
};

template <class Pipeline>
DEVICE PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}

template<uint32_t RegCount>
DEVICE
void warpgroup_reg_dealloc(){
  asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
}

/*====================*/
/*=== WgMma  ====*/
struct WgMma {
  using DType = float;
  using AType = half_t;
  using BType = half_t;
  using CType = float;

  using AFrag = GmmaDescriptor;
  using BFrag = GmmaDescriptor;

  static constexpr int ShapeM = 64;
  static constexpr int ShapeN = 128;
  static constexpr int ShapeK = 16;
  static constexpr int NumThread = 128;

  ScaleOut accumulate_ = ScaleOut::One;

  /*
    tid_in_group : threadIdx.x % 128
    row_repeat_id : [0, 1]
    col_major_repeat_id : [0, ShapeN / 8 - 1]
    col_minor_repeat_id : [0, 1]
  */
  static int make_C_offset(int tid_in_group, int row_repeat_id,
                           int col_major_repeat_id, int col_minor_repeat_id) {
    static constexpr int row_repeat_number = 2;
    static constexpr int col_major_repeat_number = ShapeN / 8;
    static constexpr int col_minor_repeat_number = 2;

    static constexpr int col_minor_stride = 1;
    static constexpr int col_major_stride = 8;
    static constexpr int threads_per_row_in_warp = 8;
    static constexpr int threads_per_col_in_warp =
        WARP_SIZE / threads_per_row_in_warp;

    static constexpr int row_repeat_warp_stride =
        ShapeN * threads_per_row_in_warp;
    static constexpr int row_group_stride =
        row_repeat_warp_stride * row_repeat_number;
    static constexpr int row_warp_stride = ShapeN;
    static constexpr int col_warp_stride = col_minor_repeat_number;

    int row_id_in_group = tid_in_group / WARP_SIZE;
    int tid_in_warp = tid_in_group % WARP_SIZE;
    int row_id_in_warp = tid_in_warp / threads_per_col_in_warp;
    int col_id_in_warp = tid_in_warp % threads_per_col_in_warp;
    return (row_id_in_group * row_group_stride +
            row_repeat_id * row_repeat_warp_stride +
            row_id_in_warp * row_warp_stride +
            col_major_repeat_id * col_major_stride +
            col_id_in_warp * col_warp_stride +
            col_minor_repeat_id * col_minor_stride);
  }
};

template <int BlockM, int BlockN, int BlockK, int ClusterX, int ClusterY,
          int ClusterZ, int Stages>
struct BlockMma {
  using MainloopPipeline =
      PipelineTmaAsync<Stages, ClusterX, ClusterY, ClusterZ>;
  using PipelineState = PipelineState<Stages>;
  struct SharedStorage {
    struct __align__(128) TensorStorage {
      WgMma::AType smem_A[BlockM * BlockK * Stages];
      WgMma::BType smem_B[BlockN * BlockK * Stages];
    }
    tensors;
    typename MainloopPipeline::SharedStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename MainloopPipeline::SharedStorage;

  struct Arguments {
    WgMma::AType const* ptr_A;
    const int lda;
    WgMma::BType const* ptr_B;
    const int ldb;
  };

  struct Params {
    // swizzle requires at least 3D, with the outermost dim = 1
    // currently we don't consider multicast for A
    // and assume ClusterY == 1
    SM90_TMA_LOAD_3D tma_load_op_a;
    TmaDescriptor tma_desc_a;
    // currently we only consider multicast for B
    // and assume ClusterX > 1
    SM90_TMA_LOAD_MULTICAST_3D tma_load_op_b;
    TmaDescriptor tma_desc_b;
  };

  static constexpr int TmaTransactionBytes =
      BlockM * BlockK * sizeof(WgMma::AType) +
      BlockN * BlockK * sizeof(WgMma::BType);

  DEVICE
  static void prefetch_tma_descriptors(Params mainloop_params) {
    prefetch_tma_descriptor(&mainloop_params.tma_desc_a);
    prefetch_tma_descriptor(&mainloop_params.tma_desc_b);
  }

  template <typename DType>
  DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline,
                   PipelineState smem_pipeline_write, DType const* ptr_a,
                   DType const* ptr_b, int m_coord, int n_coord, int k_coord,
                   int k_tile_count, int thread_idx,
                   uint32_t block_rank_in_cluster,
                   TensorStorage& shared_tensors) {
    int warp_idx = warp_id_in_block();
    int warp_idx_in_warp_group = warp_id_in_warp_group();
    int lane_predicate = elect_one_sync();

    int global_major_offset_a = m_coord * BlockM;
    int global_major_offset_b = n_coord * BlockN;

    // for now only consider multicast for B
    uint16_t mcast_mask_b = 0;
    for (int i = 0; i < ClusterX; ++i) {
      mcast_mask_b |= (1 << (i));
    }
    global_major_offset_b +=
        block_rank_in_cluster % ClusterX * (BlockN / ClusterX);

    if (warp_idx_in_warp_group == 0 && lane_predicate) {
      DType* smem_a = shared_tensors.smem_A;
      DType* smem_b = shared_tensors.smem_B;

      for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipeline_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier =
            pipeline.producer_get_barrier(smem_pipeline_write);

        int write_stage = smem_pipeline_write.index();
        DType* smem_a_k_tile = smem_a + write_stage * BlockM * BlockK;
        DType* smem_b_k_tile = smem_b + write_stage * BlockN * BlockK;
        int global_minor_offset_a = k_coord * BlockK;
        int global_minor_offset_b = k_coord * BlockK;

        mainloop_params.tma_load_op_a.copy((void*)(&mainloop_params.tma_desc_a),
                                           *tma_barrier, (void*)smem_a_k_tile,
                                           global_minor_offset_a,
                                           global_major_offset_a, 0);

        mainloop_params.tma_load_op_b.copy(
            (void*)(&mainloop_params.tma_desc_b), *tma_barrier, mcast_mask_b,
            (void*)smem_b_k_tile, global_minor_offset_b, global_major_offset_b,
            mcast_mask_b);

        ++k_coord;
        ++smem_pipeline_write;
      }
    }
  }
};

using MmaBarrier = OrderedSequenceBarrier<2, 2>;
using LoadBarrier = OrderedSequenceBarrier<1, 2>;
using Mainloop =
    BlockMma<BLOCKM, BLOCKN, BLOCKK, CLUSTER_M, CLUSTER_N, 1, STAGES>;
using SmemSwizzle = Swizzle<3, 4, 3>;
using Scheduler =
    TileScheduler<CLUSTER_M, CLUSTER_N, 1, BLOCKM, BLOCKN, BLOCKK>;
static constexpr uint32_t LoadRegisterRequirement = 40;
static constexpr uint32_t MmaRegisterRequirement = 232;

struct KernelSharedStorage {
  typename Mainloop::SharedStorage mainloop;
  MmaBarrier::SharedStorage mma_order;
  LoadBarrier::SharedStorage load_order;
};

struct KernelParams {
  Mainloop::Params mainloop;
  Scheduler::Params scheduler;
};

template <class AType, class BType, class CType, class AccumType>
__global__ void gpu_gemm_kernel(
    GemmParams<AType, BType, CType, AccumType> gemm_params,
    KernelParams kernel_params) {
  int warp_group_id = threadIdx.x / WARP_GROUP_SIZE;
  int thread_idx_in_warp_group = threadIdx.x % WARP_GROUP_SIZE;
  int warp_idx = threadIdx.x / WARP_SIZE;
  int lane_idx = threadIdx.x % WARP_SIZE;
  int warp_idx_in_warp_group = thread_idx_in_warp_group / WARP_SIZE;
  int block_idx_in_cluster = block_rank_in_cluster();
  int lane_predicate = elect_one_sync();

  // Prefetch TMA descriptors
  if (warp_idx == 0 && lane_predicate) {
    Mainloop::prefetch_tma_descriptors(kernel_params.mainloop);
    // Epilogue...
  }

  extern __shared__ uint8_t raw_smem[];
  KernelSharedStorage& shared_storage =
      *reinterpret_cast<KernelSharedStorage*>(raw_smem);

  // Mma barrier
  MmaBarrier::Params mma_barrier_params;
  mma_barrier_params.group_id = warp_group_id - 1;
  mma_barrier_params.group_size = WARP_GROUP_SIZE;
  MmaBarrier mma_barrier(shared_storage.mma_order, mma_barrier_params);

  // Load barrier
  LoadBarrier::Params load_barrier_params;
  load_barrier_params.group_id = warp_idx_in_warp_group == 0 ? 0 : 1;
  load_barrier_params.group_size = WARP_SIZE;
  LoadBarrier load_barrier(shared_storage.load_order, load_barrier_params);

  // Mainloop pipeline
  using MainloopPipeline = typename Mainloop::MainloopPipeline;
  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (warp_group_id == 0 && warp_idx_in_warp_group == 0) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_id == 1 || warp_group_id == 2) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  mainloop_pipeline_params.is_leader = thread_idx_in_warp_group == 0;
  mainloop_pipeline_params.num_consumers = WARP_GROUP_SIZE;
  mainloop_pipeline_params.transaction_bytes = Mainloop::TmaTransactionBytes;
  MainloopPipeline mainloop_pipeline(shared_storage.mainloop.pipeline,
                                     mainloop_pipeline_params);

  // Epilogue pipeline load and store
  // TODO...

  // Pipeline state
  typename Mainloop::PipelineState mainloop_pipe_consumer_state;
  // Epilogue load pipe...

  PipelineState mainloop_pipe_producer_state =
      make_producer_start_state<MainloopPipeline>();
  // Epilogue pipe...

  auto cluster_wait_fn = [&]() {
    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer thread blocks in the Cluster
    cluster_arrive_relaxed();
    return []() { cluster_wait(); };
  }();

  // Mainloop
  Mainloop mainloop;
  // Eiplogue ...

  // Scheduler
  Scheduler scheduler{kernel_params.scheduler};
  
  // Tile counts
  int k_tile_count = gemm_params.K / BLOCKK;
  // Epilogue's ...

  if (warp_group_id == 2) {
    scheduler.advance_to_next_work();
    mainloop_pipe_consumer_state.advance(k_tile_count);
    // Epilogue ...
  }

  auto work_tile_info = scheduler.get_current_work();

  cluster_wait_fn();

  if (warp_group_id == 0) {
    warpgroup_reg_dealloc<LoadRegisterRequirement>();

    // producer
    if (warp_idx_in_warp_group == 0) {
      // iterate all tiles and load
      bool do_arrive = true;

      while (work_tile_info.is_valid()) {
        int m_coord = work_tile_info.M_idx;
        int n_coord = work_tile_info.N_idx;
        int k_coord = 0;
        
        mainloop.load(kernel_params.mainloop, mainloop_pipeline,
                      mainloop_pipe_producer_state, gemm_params.A, gemm_params.B,
                      m_coord, n_coord, k_coord, k_tile_count, lane_idx,
                      block_idx_in_cluster, shared_storage.mainloop.tensors);
        
        mainloop_pipe_producer_state.advance(k_tile_count);

        if (do_arrive) {
          do_arrive = false;
          load_barrier.arrive();
        }

        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      }
    } else if (warp_idx_in_warp_group == 2) {
      load_barrier.wait();
      // iterate all tiles and load
    }
  } else {
    // consumer
    for (int i = 0; i < 10; ++i) {
      mma_barrier.wait();
      // do
      mma_barrier.arrive();
      mma_barrier.wait();
      mma_barrier.arrive();
    }
  }
}

__global__ void dummy_kernel() { return; }

template <class AType, class BType, class CType, class AccumType>
void gpu_gemm(GemmParams<AType, BType, CType, AccumType> gemm_params) {
  dim3 grid(CLUSTER_M * CLUSTER_N, SM_NUMBER / (CLUSTER_M * CLUSTER_N), 1);
  dim3 block(WARP_GROUP_SIZE * WG_NUMBER, 1, 1);
  dim3 cluster(CLUSTER_M, CLUSTER_N, 1);
  auto* Kernel = gpu_gemm_kernel<AType, BType, CType, AccumType>;
  size_t smemSizeBytes = sizeof(KernelSharedStorage);
  // size_t smemSizeBytes;
  // cudaOccupancyAvailableDynamicSMemPerBlock(&smemSizeBytes, dummy_kernel, 1,
  // block.x); std::cout << "Available smem per block: " << smemSizeBytes << "
  // bytes\n";
  if (smemSizeBytes >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSizeBytes);
    CUDA_CHECK(result);
  }
  std::cout << "Launching kernel with grid " << grid.x << " " << grid.y << " "
            << grid.z << " and block " << block.x << " " << block.y << " "
            << block.z << " and cluster " << cluster.x << " " << cluster.y
            << " " << cluster.z << " and smem " << smemSizeBytes << " bytes\n";
  void const* kernel = (void const*)Kernel;

  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(status);

  /// Prepare kernel params
  KernelParams params;
  params.mainloop.tma_load_op_a = SM90_TMA_LOAD_3D();
  params.mainloop.tma_load_op_b = SM90_TMA_LOAD_MULTICAST_3D();

  auto smem_swizzle_a = SmemSwizzle{};
  auto smem_swizzle_b = SmemSwizzle{};
  params.mainloop.tma_desc_a = make_tma_copy_desc<BLOCKM, BLOCKK, 3>(
      gemm_params.A, gemm_params.M, gemm_params.K, smem_swizzle_a, cluster.y);
  params.mainloop.tma_desc_b = make_tma_copy_desc<BLOCKN, BLOCKK, 3>(
      gemm_params.B, gemm_params.N, gemm_params.K, smem_swizzle_b, cluster.x);
  params.scheduler.M = gemm_params.M;
  params.scheduler.N = gemm_params.N;
  params.scheduler.K = gemm_params.K;

  void* kernel_params[] = {&gemm_params, &params};
  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = {grid.x, grid.y, grid.z};
  launch_config.blockDim = {block.x, block.y, block.z};
  launch_config.dynamicSmemBytes = size_t(smemSizeBytes);
  launch_config.stream = nullptr;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;

  status = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
  cudaError_t launch_result = cudaGetLastError();
  CUDA_CHECK(launch_result);
}

int main(int argc, char** argv) {
  int M = testM;
  int N = testN;
  int K = testK;
  using AType = half_t;
  using BType = half_t;
  using CType = half_t;
  using AccumType = float;
  AccumType alpha = 0.9;
  AccumType beta = 0.1;

  std::vector<int> AShape = {M, K};
  std::vector<int> BShape = {N, K};
  std::vector<int> CShape = {M, N};
  auto hA = alloc_cpu_tensor<AType>(AShape);
  arange_fill(hA, AShape);
  auto hB = alloc_cpu_tensor<BType>(BShape);
  random_fill(hB, BShape);
  auto hC = alloc_cpu_tensor<CType>(CShape);
  random_fill(hC, CShape);
  auto goldenC = alloc_cpu_tensor<CType>(CShape);
  random_fill(goldenC, CShape);
  auto dA = alloc_gpu_tensor<AType>(AShape);
  auto dB = alloc_gpu_tensor<BType>(BShape);
  auto dgC = alloc_gpu_tensor<CType>(CShape);
  auto dC = alloc_gpu_tensor<CType>(CShape);

  /// timers
  CPUTimer cpu_timer;
  GPUTimer gpu_timer;

  /// copy data
  std::cout << "Copying data from CPU to GPU...\n";
  cpu_timer.tick();
  copy_to_gpu(hA, dA, AShape);
  copy_to_gpu(hB, dB, BShape);
  copy_to_gpu(hC, dC, CShape);
  copy_to_gpu(goldenC, dgC, CShape);
  cpu_timer.tick();
  std::cout << "Copy data done! Use " << cpu_timer.report_last_ms() << " ms.\n";

  /// compute gpu reference
  std::cout << "Computing gpu reference values...\n";
  GemmParams gpu_params(M, N, K, dA, dB, dgC, alpha, beta);
  gpu_timer.sync_all();
  gpu_timer.tick();
  reference_gpu_gemm(gpu_params);
  gpu_timer.tick();
  gpu_timer.sync_all();
  std::cout << "GPU reference done! Use " << gpu_timer.report_last_ms()
            << " ms.\n";

  /// copy results
  std::cout << "Copying results...\n";
  copy_to_cpu(goldenC, dgC, CShape);
  std::cout << "Copying results done!\n";

  /// compute gpu kernel
  std::cout << "Computing gpu kernel values...\n";
  GemmParams gpu_kernel_params(M, N, K, dA, dB, dC, alpha, beta);
  gpu_gemm(gpu_kernel_params);
  std::cout << "GPU kernel done!\n";

  /// copy results
  std::cout << "Copying results...\n";
  copy_to_cpu(hC, dC, CShape);
  std::cout << "Copying results done!\n";

  /// compare results
  assert_allclose(hC, goldenC, CShape, /*rtol=*/1e-3);
  std::cout << "Correct!\n";

  /// profile
  std::cout << "Profile performance...\n";
  gpu_timer.sync_all();
  gpu_timer.tick();
  for (int i = 0; i < iters; ++i) {
    gpu_gemm(gpu_params);
  }
  gpu_timer.tick();
  gpu_timer.sync_all();
  std::cout << "Profile done! Average latency is "
            << gpu_timer.report_last_ms() / float(iters) << " ms.\n";

  free_cpu_tensor(hA);
  free_cpu_tensor(hB);
  free_cpu_tensor(hC);
  free_cpu_tensor(goldenC);
  free_gpu_tensor(dA);
  free_gpu_tensor(dB);
  free_gpu_tensor(dC);
  return 0;
}