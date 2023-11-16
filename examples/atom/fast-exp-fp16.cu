#include <cmath>

#include "common.h"

DEVICE unsigned nv_exp2(unsigned x) {
  unsigned ret;
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(ret) : "r"(x));
  return ret;
}

DEVICE half nv_exp2(half x) {
  unsigned short v = __half_as_ushort(0.0);
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(v) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(v);
}

__global__ void fast_exp2(half* A, half* B, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned* out = reinterpret_cast<unsigned*>(B);
  for (int i = x * 2; i < N; i += blockDim.x * gridDim.x * 2) {
    out[i / 2] = nv_exp2((reinterpret_cast<unsigned*>(A))[i / 2]);
  }
  //   for (int i = x; i < N; i += blockDim.x * gridDim.x) {
  //     B[i] = nv_exp2(A[i]);
  //   }
}

void cpu_exp2(half* A, half* B, int N) {
  for (int i = 0; i < N; ++i) {
    B[i] = (half)exp2((float)A[i]);
  }
}

int main() {
  auto len = 10000;
  auto A = alloc_cpu_tensor<half>({len});
  random_fill(A, {len});
  auto B = alloc_cpu_tensor<half>({len});
  auto golden = alloc_cpu_tensor<half>({len});
  cpu_exp2(A, golden, len);
  auto dA = alloc_gpu_tensor<half>({len});
  auto dB = alloc_gpu_tensor<half>({len});
  copy_to_gpu(A, dA, {len});
  dim3 block(256);
  dim3 grid(ceil_div(len, block.x));
  fast_exp2<<<grid, block>>>(dA, dB, len);
  copy_to_cpu(B, dB, {len});
  assert_allclose(B, golden, {len}, 1e-5, /*dump=*/false);
  std::cout << "Correct!\n";
  return 0;
}