#define _CG_ABI_EXPERIMENTAL
#include "helper.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T, int BlockSize>
__global__ void reduceKernel(T *A, T *B, int length) {
  cg::thread_block cta = cg::this_thread_block();
  T *smem = SharedMemory<T>();

  int tid = cta.thread_rank();

  T localSum = (T)0;

  for (int i = blockIdx.x * blockDim.x + tid; i < length;
       i += blockDim.x * gridDim.x) {
    localSum += A[i];
  }

  smem[tid] = localSum;
  cta.sync();

  if (BlockSize >= 512 && tid < 256) {
    smem[tid] = localSum = localSum + smem[tid + 256];
  }
  cta.sync();
  if (BlockSize >= 256 && tid < 128) {
    smem[tid] = localSum = localSum + smem[tid + 128];
  }
  cta.sync();
  if (BlockSize >= 128 && tid < 64) {
    smem[tid] = localSum = localSum + smem[tid + 64];
  }
  cta.sync();
  if (BlockSize >= 64 && tid < 32) {
    smem[tid] = localSum = localSum + smem[tid + 32];
  }
  cta.sync();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  if (cta.thread_rank() < 32) {
    for (int s = 16; s > 0; s >>= 1) {
      localSum += tile32.shfl_down(localSum, s);
    }
  }

  if (cta.thread_rank() == 0) {
    B[blockIdx.x] = localSum;
  }
}

int ceil(int a, int b) { return (a + b - 1) / b; }

template <class T>
void getLaunchConfig(int length, int &numBlocks, int &numThreads,
                     int &smemSize) {
  if (length > 512) {
    numThreads = 512;
  } else if (length > 128) {
    numThreads = 128;
  } else if (length > 64) {
    numThreads = 64;
  } else {
    numThreads = 32;
  }
  numBlocks = ceil(length, numThreads);
  smemSize = numThreads * sizeof(T);
}

template <class T> void reduce(T *A, T *B, int length) {
  int cpuReduceSize = 32;
  int numBlocks = 0;
  int numThreads = 0;
  int smemSize = 0;
  // init
  getLaunchConfig<T>(length, numBlocks, numThreads, smemSize);
  T *orgDevOutput = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&orgDevOutput, numBlocks * sizeof(T)));
  T *devOutput = orgDevOutput;
  T *devInput = A;
  T *hostOutput = B;
  T *hostTemp = (T *)malloc(cpuReduceSize * sizeof(T));

  T *debugBuf = (T *)malloc(length * sizeof(T));

  if (length <= cpuReduceSize) {
    devOutput = devInput;
  }
  while (length > cpuReduceSize) {
    // debug
    // std::cout << "length = " << length << "\n";
    // std::cout << "numBlocks = " << numBlocks << "\n";
    // std::cout << "numThreads = " << numThreads << "\n";
    // std::cout << "smemSize = " << smemSize << "\n";

    dim3 grid(numBlocks, 1, 1);
    dim3 block(numThreads, 1, 1);

    switch (numThreads) {
    case 512:
      reduceKernel<T, 512>
          <<<grid, block, smemSize, nullptr>>>(devInput, devOutput, length);
      break;
    case 256:
      reduceKernel<T, 256>
          <<<grid, block, smemSize, nullptr>>>(devInput, devOutput, length);
      break;
    case 128:
      reduceKernel<T, 128>
          <<<grid, block, smemSize, nullptr>>>(devInput, devOutput, length);
      break;
    case 64:
      reduceKernel<T, 64>
          <<<grid, block, smemSize, nullptr>>>(devInput, devOutput, length);
      break;
    case 32:
      reduceKernel<T, 32>
          <<<grid, block, smemSize, nullptr>>>(devInput, devOutput, length);
      break;
    }

    length = numBlocks;

    // debug
    // CUDA_CHECK(cudaMemcpy(debugBuf, devOutput, length * sizeof(T),
    // cudaMemcpyDeviceToHost)); for (int i = 0; i < length; ++i) {
    //     std::cout << debugBuf[i] << " ";
    // }
    // std::cout << "\n";

    // for next
    getLaunchConfig<T>(length, numBlocks, numThreads, smemSize);
    std::swap(devInput, devOutput);
  }
  std::swap(devInput, devOutput);

  CUDA_CHECK(cudaMemcpy(hostTemp, devOutput, length * sizeof(T),
                        cudaMemcpyDeviceToHost));
  hostOutput[0] = hostTemp[0];
  for (int i = 1; i < length; ++i) {
    hostOutput[0] += hostTemp[i];
  }

  free(hostTemp);
  CUDA_CHECK(cudaFree(orgDevOutput));
}

template void reduce<float>(float *A, float *B, int length);