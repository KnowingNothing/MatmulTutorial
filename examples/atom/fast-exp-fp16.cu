#include "common.h"

DEVICE unsigned exp2(unsigned x) {
  unsigned ret;
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(ret) : "r"(x));
  return ret;
}

// thi is wrong, don't know how to fix
DEVICE unsigned exp2(half x) {
  half ret;
  asm volatile("ex2.approx.f16 %0, %1;" : "=r"(ret) : "r"(x));
  return ret;
}

__global__ void fast_exp(half* A, half* B, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  auto out = reinterpret_cast<unsigned*>(B);
  for (int i = x * 2; i * 2 < N; i += blockDim.x * gridDim.x * 2) {
    out[i / 2] = exp2((reinterpret_cast<unsigned*>(A))[i / 2]);
  }
  // for (int i = x; i < N; i += blockDim.x * gridDim.x) {
  //     B[i] = exp2(A[i]);
  // }
}

int main() {
  auto len = 100;
  auto A = alloc_cpu_tensor<half>({len});
  random_fill(A, {len});
  auto B = alloc_cpu_tensor<half>({len});
  auto dA = alloc_gpu_tensor<half>({len});
  auto dB = alloc_gpu_tensor<half>({len});
  copy_to_gpu(A, dA, {len});
  dim3 block(256);
  dim3 grid(ceil_div(len, block.x));
  fast_exp<<<grid, block>>>(dA, dB, len);
  copy_to_cpu(B, dB, {len});
  return 0;
}