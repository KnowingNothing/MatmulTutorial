#include "barrier.h"
#include "common.h"
#include "reference.h"

const int testM = 4096;
const int testN = 4096;
const int testK = 4096;
const int iters = 100;

/// RUN:
/// nvcc -arch=sm_90a -lcuda -std=c++17 matmul-pingpong-v1.cu -o test && ./test
/// |& tee trace.log

using MmaBarrier = OrderedSequenceBarrier<2, 2>;
using LoadBarrier = OrderedSequenceBarrier<1, 2>;

struct KernelSharedStorage {
  MmaBarrier::SharedStorage mma_order;
  LoadBarrier::SharedStorage load_order;
};

template <class AType, class BType, class CType, class AccumType>
__global__ void gpu_gemm_kernel(
    GemmParams<AType, BType, CType, AccumType> params) {
  int warp_group_id = threadIdx.x / WARP_GROUP_SIZE;
  int thread_idx_in_warp_group = threadIdx.x % WARP_GROUP_SIZE;
  int warp_idx = threadIdx.x / WARP_SIZE;
  int warp_idx_in_warp_group = thread_idx_in_warp_group / WARP_SIZE;

  extern __shared__ uint8_t raw_smem[];
  KernelSharedStorage &shared_storage =
      *reinterpret_cast<KernelSharedStorage *>(raw_smem);

  MmaBarrier::Params mma_barrier_params;
  mma_barrier_params.group_id = warp_group_id - 1;
  mma_barrier_params.group_size = WARP_GROUP_SIZE;
  MmaBarrier mma_barrier(shared_storage.mma_order, mma_barrier_params);

  LoadBarrier::Params load_barrier_params;
  load_barrier_params.group_id = warp_idx_in_warp_group == 0 ? 0 : 1;
  load_barrier_params.group_size = WARP_SIZE;
  LoadBarrier load_barrier(shared_storage.load_order, load_barrier_params);

  auto cluster_wait_fn = [&]() {
    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer thread blocks in the Cluster
    cluster_arrive_relaxed();
    return []() { cluster_wait(); };
  }();

  cluster_wait_fn();
  if (warp_group_id == 0) {
    // producer
    bool do_arrive = true;
    if (warp_idx_in_warp_group == 0) {
      // iterate all tiles and load
      if (do_arrive) {
        do_arrive = false;
        load_barrier.arrive();
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

template <class AType, class BType, class CType, class AccumType>
void gpu_gemm(GemmParams<AType, BType, CType, AccumType> params) {
  dim3 grid(SM_NUMBER, 1, 1);
  dim3 block(WARP_GROUP_SIZE * WG_NUMBER, 1, 1);
  dim3 cluster(CLUSTER_M, CLUSTER_N, 1);
  int smemSizeBytes = sizeof(KernelSharedStorage);
  void const *kernel =
      (void const *)gpu_gemm_kernel<AType, BType, CType, AccumType>;
  void *kernel_params[] = {&params};
  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = {grid.x, grid.y, grid.z};
  launch_config.blockDim = {block.x, block.y, block.z};
  launch_config.dynamicSmemBytes = smemSizeBytes;
  launch_config.stream = nullptr;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;

  cudaError_t status =
      cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
  cudaError_t launch_result = cudaGetLastError();
  CUDA_CHECK(launch_result);
}

int main(int argc, char **argv) {
  int M = testM;
  int N = testN;
  int K = testK;
  using AType = half;
  using BType = half;
  using CType = half;
  using AccumType = float;
  AccumType alpha = 0.9;
  AccumType beta = 0.1;

  std::vector<int> AShape = {M, K};
  std::vector<int> BShape = {N, K};
  std::vector<int> CShape = {M, N};
  auto hA = alloc_cpu_tensor<AType>(AShape);
  random_fill(hA, AShape);
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