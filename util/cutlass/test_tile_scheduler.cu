#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "common.h"

using namespace cutlass;
using namespace cutlass::gemm::kernel::detail;
using namespace cute;

using Scheduler = PersistentTileSchedulerSm90;

/// nvcc -arch=sm_90a -I ../../include -I /home/jshao/zhengsz/cutlass/include -lcuda -std=c++17 test_tile_scheduler.cu -o test

struct KernelSharedStorage {

};

const int WG_NUMBER = 3;

struct KernelParams {
    int M;
    int N;
    int K;
    Scheduler::Params schedule_params;
    int* idx;
};

__global__ void test_kernel(KernelParams params) {
    Scheduler scheduler(params.schedule_params);
    auto tileinfo = scheduler.get_current_work();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int is_n = params.schedule_params.raster_order_ == Scheduler::RasterOrder::AlongN;
        printf("log swizzle %d is n %d\n", params.schedule_params.log_swizzle_size_, is_n);
    }
    if (threadIdx.x == 0) {
        printf("block %d maps to linear m %d n %d\n", blockIdx.x, tileinfo.M_idx, tileinfo.N_idx);
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    dim3 grid(SM_NUMBER, 1, 1);
    dim3 block(WARP_GROUP_SIZE * WG_NUMBER, 1, 1);
    const int CLUSTER_M = 2;
    const int CLUSTER_N = 1;
    dim3 cluster(CLUSTER_M, CLUSTER_N, 1);
    int smemSizeBytes = sizeof(KernelSharedStorage);
    void const *kernel =
        (void const *)test_kernel;

    auto idx = alloc_cpu_tensor<int>({(int)block.x});
    auto g_idx = alloc_gpu_tensor<int>({(int)block.x});

    using ShapeMNKL = Shape<int, int, int, int>;
    ShapeMNKL shape{M, N, K, 1};
    using TileShape = Shape<_128, _128, _64>;
    TileShape tile_shape{};
    using ClusterShape = Shape<_2, _1, _1>;
    ClusterShape cluster_shape{};
    KernelHardwareInfo info{};
    Scheduler::Arguments args{};
    Scheduler::Params schedule_params = Scheduler::to_underlying_arguments(shape, tile_shape, cluster_shape, info, args);

    KernelParams params{M, N, K, schedule_params, g_idx};
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

    copy_to_cpu(idx, g_idx, {(int)block.x});

    return 0;
}