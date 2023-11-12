// 2 mma + pipeline + double threading + smem crosswise + ldmatrix

// A100 PCIE 80GB
// Setting to 4 stages.
// Setting to 2x threading.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 1.10695ms
// TFLOPS: 106.942

// 3090
// Setting to 4 stages.
// Testing iters = 200.
// Setting to 2x threading.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 1.93203ms
// TFLOPS: 61.2723

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 32;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
// const int wmmaK = 16;

__device__ __forceinline__ void loadSmemA(half *smem, half *A, int M, int K,
                                          int ko) {
  // load 128 * 32
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 128 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int logic_row = i * 64 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 32 + tid / 8;
    int col = tid % 8 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                 "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]),
                 "n"(16));
  }
}

__device__ __forceinline__ void loadSmemB(half *smem, half *B, int N, int K,
                                          int ko) {
  // load 128 * 32
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 128 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int logic_row = i * 64 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 32 + tid / 8;
    int col = tid / 4 % 2 * 32 + tid % 4 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                 "l"(&B[(bx * 128 + logic_row) * K + (ko * KI + logic_col)]),
                 "n"(16));
  }
}

__device__ __forceinline__ void loadSmemC(float *smem, half *C, int M, int N) {
  // load 128 * 128
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 128 + ty * 32 + tx;
  for (int i = 0; i < 64; ++i) {
    int row = i * 2 + tid / 128;
    int col = tid % 128;
    smem[row * 128 + col] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
  }
}

__device__ __forceinline__ void storeSmemC(half *C, float *smem, int M, int N) {
  // load 128 * 128
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 128 + ty * 32 + tx;
  for (int i = 0; i < 64; ++i) {
    int row = i * 2 + tid / 128;
    int col = tid % 128;
    (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row * 128 + col];
  }
}

__device__ __forceinline__ void loadFragA(unsigned int *frag, half *smem,
                                          int ki) {
  // frag: [j, k]: [2, 2]
  // load 64x16
  int tx = threadIdx.x;
  int tz = threadIdx.z;

  // for (int i = 0; i < 4; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     for (int k = 0; k < 2; ++k) {
  //       int row = tz * 64 + i * 16 + j * 8 + tx / 4;
  //       int col = ki * KII + k * 8 + tx % 4 * 2;
  //       col = row % 2 * 32 + col;
  //       row = row / 2;
  //       col = col ^ ((row & 3) << 3);
  //       unsigned int *ptr =
  //           reinterpret_cast<unsigned int *>(smem + row * 64 + col);
  //       frag[i * 4 + j * 2 + k] = ptr[0];
  //     }
  //   }
  // }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
    int col = ki * KII + tx / 8 % 2 * 8;
    col = row % 2 * 32 + col;
    row = row / 2;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
          "=r"(frag[i * 4 + 3])
        : "r"(smem_ptr));
  }
}

__device__ __forceinline__ void loadFragB(unsigned int *frag, half *smem,
                                          int ki) {
  // frag: [j, k]: []
  // load 32x16
  // int tx = threadIdx.x;
  // int ty = threadIdx.y;
  // for (int i = 0; i < 2; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     for (int k = 0; k < 2; ++k) {
  //       int row = ty * 32 + i * 16 + j * 8 + tx / 4;
  //       int col = ki * KII + k * 8 + tx % 4 * 2;
  //       col = row % 2 * 32 + col;
  //       row = row / 2;
  //       col = col ^ ((row & 3) << 3);
  //       unsigned int *ptr =
  //           reinterpret_cast<unsigned int *>(smem + row * 64 + col);
  //       frag[i * 4 + j * 2 + k] = ptr[0];
  //     }
  //   }
  // }
  int tx = threadIdx.x;
  int ty = threadIdx.y;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int row = ty * 32 + i * 16 + tx / 16 * 8 + tx % 8;
    int col = ki * KII + tx / 8 % 2 * 8;
    col = row % 2 * 32 + col;
    row = row / 2;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
          "=r"(frag[i * 4 + 3])
        : "r"(smem_ptr));
  }
}

__device__ __forceinline__ void storeAccum(float *ptr, float *frag) {
  // frag [r, c, _]: [2, 2, 2]
  // store 64x32
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int row = tz * 64 + tx / 4;
  int col = ty * 32 + tx % 4 * 2;
  // float *dst = ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row %
  // 16 * 16 + col % 16;
  float *dst = ptr + row * 128 + col;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      dst[0] = frag[i * 16 + j * 8 + 0 * 4 + 0 * 2];
      dst[1] = frag[i * 16 + j * 8 + 0 * 4 + 0 * 2 + 1];

      dst[0 + 8] = frag[i * 16 + j * 8 + 0 * 4 + 1 * 2];
      dst[1 + 8] = frag[i * 16 + j * 8 + 0 * 4 + 1 * 2 + 1];

      dst[0 + 8 * 128] = frag[i * 16 + j * 8 + 1 * 4 + 0 * 2];
      dst[1 + 8 * 128] = frag[i * 16 + j * 8 + 1 * 4 + 0 * 2 + 1];

      dst[0 + 8 * 128 + 8] = frag[i * 16 + j * 8 + 1 * 4 + 1 * 2];
      dst[1 + 8 * 128 + 8] = frag[i * 16 + j * 8 + 1 * 4 + 1 * 2 + 1];

      dst += 16;
    }
    dst += 16 * 128 - 16 * 2;
  }
}

__device__ __forceinline__ void mmaSync(unsigned int *fragA,
                                        unsigned int *fragB, float *accum) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,  %1,  %2,  %3},"
               "{%4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11, %12, %13};\n"
               : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
               : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                 "r"(fragB[0]), "r"(fragB[1]), "f"(accum[0]), "f"(accum[1]),
                 "f"(accum[4]), "f"(accum[5]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,  %1,  %2,  %3},"
               "{%4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11, %12, %13};\n"
               : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
               : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                 "r"(fragB[2]), "r"(fragB[3]), "f"(accum[2]), "f"(accum[3]),
                 "f"(accum[6]), "f"(accum[7]));
}

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta) {
  // A is row-major
  // B is col-major
  // 128 threads [x, y, z] = [32, 2, 2]
  // threadblock mma: 128x128x32
  // warp mma: 64x64x16
  extern __shared__ uint8_t shared_storage[];
  half *SA1 = reinterpret_cast<half *>(shared_storage);
  half *SA2 = SA1 + MI * KI;
  half *SA3 = SA2 + MI * KI;
  half *SA4 = SA3 + MI * KI;
  half *SB1 = SA4 + MI * KI;
  half *SB2 = SB1 + NI * KI;
  half *SB3 = SB2 + NI * KI;
  half *SB4 = SB3 + NI * KI;
  float *SC = reinterpret_cast<float *>(shared_storage);

  unsigned int FragA[4 * 4]; // [4, 4]
  unsigned int FragB[2 * 4]; // [2, 4]

  float Accum[4 * 2 * 8] = {0.0}; // [4, 2, 8]

  // prologue
  loadSmemA(SA1, A, M, K, 0);
  loadSmemB(SB1, B, N, K, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  loadSmemA(SA2, A, M, K, 1);
  loadSmemB(SB2, B, N, K, 1);
  asm volatile("cp.async.commit_group;\n" ::);

  loadSmemA(SA3, A, M, K, 2);
  loadSmemB(SB3, B, N, K, 2);
  asm volatile("cp.async.commit_group;\n" ::);

  for (int ko = 0; ko < K / KI - 4; ko += 4) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 3 < K / KI) {
      loadSmemA(SA4, A, M, K, ko + 3);
      loadSmemB(SB4, B, N, K, ko + 3);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA1, ki);
      loadFragB(FragB, SB1, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 4 < K / KI) {
      loadSmemA(SA1, A, M, K, ko + 4);
      loadSmemB(SB1, B, N, K, ko + 4);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA2, ki);
      loadFragB(FragB, SB2, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 5 < K / KI) {
      loadSmemA(SA2, A, M, K, ko + 5);
      loadSmemB(SB2, B, N, K, ko + 5);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA3, ki);
      loadFragB(FragB, SB3, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 6 < K / KI) {
      loadSmemA(SA3, A, M, K, ko + 6);
      loadSmemB(SB3, B, N, K, ko + 6);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA4, ki);
      loadFragB(FragB, SB4, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }
  }

  // the last 4 iterations
  {
    int ko = (K / KI / 4 - 1) * 4;
    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 3 < K / KI) {
      loadSmemA(SA4, A, M, K, ko + 3);
      loadSmemB(SB4, B, N, K, ko + 3);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA1, ki);
      loadFragB(FragB, SB1, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 4 < K / KI) {
      loadSmemA(SA1, A, M, K, ko + 4);
      loadSmemB(SB1, B, N, K, ko + 4);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA2, ki);
      loadFragB(FragB, SB2, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
    if (ko + 5 < K / KI) {
      loadSmemA(SA2, A, M, K, ko + 5);
      loadSmemB(SB2, B, N, K, ko + 5);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA3, ki);
      loadFragB(FragB, SB3, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    if (ko + 6 < K / KI) {
      loadSmemA(SA3, A, M, K, ko + 6);
      loadSmemB(SB3, B, N, K, ko + 6);
    }
    for (int ki = 0; ki < KI / KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA4, ki);
      loadFragB(FragB, SB4, ki);
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1) {
          // 16x16x16 for each wmma
          mmaSync(&FragA[mii * 4], &FragB[nii * 4], &Accum[mii * 16 + nii * 8]);
        }
      }
    }
  }
  storeAccum(SC, Accum);
  __syncthreads();
  storeSmemC(C, SC, M, N);
}