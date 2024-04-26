#include <cooperative_groups.h>
#include <curand_kernel.h>

#include <cuda/barrier>
#include <iostream>

__device__ void device_sleep(int delay_time) {
  // 获取全局计时器的初始值
  unsigned long long start = clock64();
  unsigned long long int elapsed = 0;

  // 忙等循环，直到达到指定的cycle数
  while (elapsed < delay_time) {
    unsigned long long now = clock64();
    elapsed = (now - start);
  }
}

__device__ void compute(float* data, int curr_iteration) {
  data[threadIdx.x] += 1;
}

__global__ void split_arrive_wait(int iteration_count, float* data) {
  using barrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier bar;
  __shared__ float buffer[1024];
  buffer[threadIdx.x] = 0.0;
  __syncthreads();
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0) {
    init(&bar,
         block.size());  // Initialize the barrier with expected arrival count
  }
  block.sync();

  curandState state;
  curand_init(1234, threadIdx.x, 0, &state);

  for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
    /* code before arrive */
    compute(buffer, curr_iter);
    barrier::arrival_token token =
        bar.arrive(); /* this thread arrives. Arrival does not block a thread */

    int rand_cycle = (int)(curand_uniform(&state) * 1e15);
    device_sleep(rand_cycle);
    compute(buffer, curr_iter);
    bar.wait(std::move(token)); /* wait for all threads participating in the
                                   barrier to complete bar.arrive()*/
    /* code after wait */
    data[threadIdx.x] = buffer[(threadIdx.x + 32 + curr_iter) % 1024];
  }
}

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

int main() {
  int iteration = 10;
  int num_threads = 1024;
  float* data_host = (float*)malloc(num_threads * sizeof(float));
  float* data_device;
  CUDA_CHECK(cudaMalloc((void**)(&data_device), num_threads * sizeof(float)));
  split_arrive_wait<<<1, num_threads>>>(iteration, data_device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(data_host, data_device, num_threads * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::cout << "check results:\n";
  for (int i = 0; i < num_threads; ++i) {
    std::cout << data_host[i] << " ";
  }
  std::cout << "\n";
  return 0;
}