// 2 mma + pipeline + smem crosswise + partial ldmatrix + no unroll + serpentine
// + ptx opt

// A100 PCIE 80GB

// 3090

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
// const int wmmaK = 16;

#define MIN(a, b) (a) < (b) ? (a) : (b)

__device__ __forceinline__ void loadSmemA(half *smem, half *A, int M, int K,
                                          int ko) {
  // load 128 * 32
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int logic_row = i * 32 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 16 + tid / 8;
    int col = tid % 8 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_ptr),
        "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16),
        "r"(16));

    // uint32_t smem_ptr;

    // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
    //     "%0, smem_ptr; }\n"
    //     : "=r"(smem_ptr)
    //     : "l"(ptr));

    // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
    // ::"r"(smem_ptr),
    //              "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]),
    //              "n"(16));
  }
}

__device__ __forceinline__ void predLoadSmemA(half *smem, half *A, int M, int K,
                                              int ko, bool pred_guard) {
  // load 128 * 32
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int logic_row = i * 32 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 16 + tid / 8;
    int col = tid % 8 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile("{\n"
                 " .reg .pred p;\n"
                 " setp.ne.b32 p, %0, 0;\n"
                 " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_ptr),
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
  int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int logic_row = i * 32 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 16 + tid / 8;
    int col = tid / 4 % 2 * 32 + tid % 4 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_ptr),
        "l"(&B[(bx * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16),
        "r"(16));
  }
}

__device__ __forceinline__ void predLoadSmemB(half *smem, half *B, int N, int K,
                                              int ko, bool pred_guard) {
  // load 128 * 32
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int logic_row = i * 32 + tid / 4;
    int logic_col = tid % 4 * 8;
    int row = i * 16 + tid / 8;
    int col = tid / 4 % 2 * 32 + tid % 4 * 8;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile("{\n"
                 " .reg .pred p;\n"
                 " setp.ne.b32 p, %0, 0;\n"
                 " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_ptr),
                 "l"(&B[(bx * 128 + logic_row) * K + (ko * KI + logic_col)]),
                 "n"(16));
  }
}

union Float4 {
  float4 f4;
  float2 f22[2];
};

__device__ __forceinline__ void storeSmemC(half *C, float *smem, int M, int N) {
  // load 128 * 128
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 64 + ty * 32 + tx;
  for (int i = 0; i < 128; ++i) {
    int row = i;
    int col = tid;
    int scol = col ^ ((row & 3) << 3);
    (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row * 128 + scol];
  }
}

__device__ __forceinline__ void loadFragA(unsigned int *frag, half *smem,
                                          int ki) {
  // frag: [j, k]: [2, 2]
  // load 64x16
  int tx = threadIdx.x;
  int tz = threadIdx.z;

  //   load 16x16 at a time
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
    int col = ki * KII + tx / 8 % 2 * 8;
    col = row % 2 * 32 + col;
    row = row / 2;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    // uint32_t smem_ptr;
    // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64"
    //     "%0, smem_ptr; }\n"
    //     : "=r"(smem_ptr)
    //     : "l"(ptr));
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
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
  // load 64x16
  int tx = threadIdx.x;
  int ty = threadIdx.y;

// load 16x16 at a time
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = ty * 64 + i * 16 + tx / 16 * 8 + tx % 8;
    int col = ki * KII + tx / 8 % 2 * 8;
    col = row % 2 * 32 + col;
    row = row / 2;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    // uint32_t smem_ptr;
    // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
    //     "%0, smem_ptr; }\n"
    //     : "=r"(smem_ptr)
    //     : "l"(ptr));
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
          "=r"(frag[i * 4 + 3])
        : "r"(smem_ptr));
  }
}

__device__ __forceinline__ void storeAccum(float *ptr, float *frag) {
  // frag [r, c, _]: [2, 2, 2]
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  // smem view is [128x128]
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          int row = tz * 64 + i * 16 + r * 8 + tx / 4;
          int col = ty * 64 + j * 16 + c * 8 + tx % 4 * 2;
          int scol = col ^ ((row & 3) << 3);
          ptr[row * 128 + scol] = frag[i * 32 + j * 8 + r * 4 + c * 2 + 0];
          ptr[row * 128 + (scol + 1)] =
              frag[i * 32 + j * 8 + r * 4 + c * 2 + 1];
        }
      }
    }
  }
}

template <int BM, int BN, int WM, int WN, int IM, int IN, int TX, int TY,
          int TZ>
struct Epilogue {
  static constexpr int WarpRepeatM = WM / IM;
  static constexpr int WarpRepeatN = WN / IN;
  static constexpr int TXY = TX * TY;
  static constexpr int storeElements = 16;
  static constexpr int storeRowThreads = BN / storeElements;
  int Bx = blockIdx.x;
  int By = blockIdx.y;
  int Bz = blockIdx.z;
  int Tx = threadIdx.x;
  int Ty = threadIdx.y;
  int Tz = threadIdx.z;

  __device__ __forceinline__ Epilogue(half *gmem_ptr, float *smem_ptr,
                                      float *frag_ptr, int M, int N,
                                      float alpha, float beta)
      : gmem_ptr(gmem_ptr), smem_ptr(smem_ptr), frag_ptr(frag_ptr), M(M), N(N),
        alpha(alpha), beta(beta) {
    store_smem_row_init = Tz * WM + Tx / 4;
    store_smem_col_init = Ty * WN + Tx % 4 * 2;
    int tid = Tz * TXY + Ty * TX + Tx;
    read_smem_row_init = tid / storeRowThreads;
    read_smem_col_init = (tid % storeRowThreads) * storeElements;
    store_smem_crosswise_factor = ((Tx / 4) & 3);
    read_smem_crosswise_factor = read_smem_row_init & 3;
  }

  __device__ __forceinline__ void operator()() {
    int store_smem_row = store_smem_row_init;
    int store_smem_col = store_smem_col_init;
    int read_smem_row = read_smem_row_init;
    int read_smem_col = read_smem_col_init;
    int store_gmem_row = By * BM + read_smem_row;
    int store_gmem_col = Bx * BN + read_smem_col;
    // store_smem_col = store_smem_col ^ (store_smem_crosswise_factor << 3);
    // int store_smem_minor_stride = store_smem_crosswise_factor & 1 ? -4 : 4;
    // int store_smem_minor_strides_1[] = {-8, 24, -8, 24};
    // int store_smem_minor_strides_2[] = {8, 8, 8, 8};
    // int* store_smem_major_strides = store_smem_crosswise_factor & 2 ?
    // store_smem_minor_strides_1 : store_smem_minor_strides_2; read_smem_col =
    // read_smem_col ^ (read_smem_crosswise_factor << 3); int
    // read_smem_minor_stride =
    float2 *store_smem_ptr = reinterpret_cast<float2 *>(
        smem_ptr + store_smem_row * BN + store_smem_col);
    float2 *read_frag_ptr = reinterpret_cast<float2 *>(frag_ptr);
    float4 *read_smem_ptr = reinterpret_cast<float4 *>(
        smem_ptr + read_smem_row * BN + read_smem_col);
    uint4 *store_gmem_ptr = reinterpret_cast<uint4 *>(
        gmem_ptr + store_gmem_row * N + store_gmem_col);
    for (int wm = 0; wm < WarpRepeatM; ++wm) {
      // store frag to smem
      float2 *tmp_store_smem_ptr = store_smem_ptr;
      float2 *tmp_read_frag_ptr = read_frag_ptr;
      for (int wn = 0; wn < WarpRepeatN; ++wn) {
        // thread row 0, col 0, 2 elements;
        tmp_store_smem_ptr[0] = tmp_read_frag_ptr[0];
        // thread row 0, col 1, 2 elements;
        tmp_store_smem_ptr[4] = tmp_read_frag_ptr[1];
        // thread row 1, col 0, 2 elements;
        tmp_store_smem_ptr[4 * BN] = tmp_read_frag_ptr[2];
        // thread row 1, col 1, 2 elements;
        tmp_store_smem_ptr[4 * BN + 4] = tmp_read_frag_ptr[3];
        tmp_store_smem_ptr += 8;
        tmp_read_frag_ptr += 4;
      }
      store_smem_ptr += 8 * BN;
      read_frag_ptr += 16;

      __syncthreads();

      // store smem to gmem
      float4 *tmp_read_smem_ptr = read_smem_ptr;
      uint4 *tmp_store_gmem_ptr = store_gmem_ptr;
      for (int i = 0; i < 2;
           ++i) { // how to better calculate this extend statically?
        // read
        float4 smem_read_values[4];
        float2 mul_values[8];
        half2 cast_values[8];
        uint4 *to_store = reinterpret_cast<uint4 *>(cast_values);
        // 1st 4 elements
        smem_read_values[0] = tmp_read_smem_ptr[0];
        // 2nd 4 elements
        smem_read_values[1] = tmp_read_smem_ptr[1];
        // 3rd 4 elements
        smem_read_values[2] = tmp_read_smem_ptr[2];
        // 4th 4 elements
        smem_read_values[3] = tmp_read_smem_ptr[3];
        // mul
        mul_values[0].x = smem_read_values[0].x * alpha;
        mul_values[0].y = smem_read_values[0].y * alpha;
        mul_values[1].x = smem_read_values[0].z * alpha;
        mul_values[1].y = smem_read_values[0].w * alpha;
        mul_values[2].x = smem_read_values[1].x * alpha;
        mul_values[2].y = smem_read_values[1].y * alpha;
        mul_values[3].x = smem_read_values[1].z * alpha;
        mul_values[3].y = smem_read_values[1].w * alpha;
        // cast
        cast_values[0] = __float22half2_rn(mul_values[0]);
        cast_values[1] = __float22half2_rn(mul_values[1]);
        cast_values[2] = __float22half2_rn(mul_values[2]);
        cast_values[3] = __float22half2_rn(mul_values[3]);
        // mul
        mul_values[4].x = smem_read_values[2].x * alpha;
        mul_values[4].y = smem_read_values[2].y * alpha;
        mul_values[5].x = smem_read_values[2].z * alpha;
        mul_values[5].y = smem_read_values[2].w * alpha;
        mul_values[6].x = smem_read_values[3].x * alpha;
        mul_values[6].y = smem_read_values[3].y * alpha;
        mul_values[7].x = smem_read_values[3].z * alpha;
        mul_values[7].y = smem_read_values[3].w * alpha;
        // store
        tmp_store_gmem_ptr[0] = to_store[0];
        // cast
        cast_values[4] = __float22half2_rn(mul_values[4]);
        cast_values[5] = __float22half2_rn(mul_values[5]);
        cast_values[6] = __float22half2_rn(mul_values[6]);
        cast_values[7] = __float22half2_rn(mul_values[7]);
        // store
        tmp_store_gmem_ptr[1] = to_store[1];
        // update ptr
        tmp_read_smem_ptr += 16 * BN;
        tmp_store_gmem_ptr += 8 * N;
      }
      read_smem_ptr += 4 * BN;
      store_gmem_ptr += 2 * N;
    }
  }

  half *gmem_ptr;
  float *smem_ptr;
  float *frag_ptr;
  int M;
  int N;
  float alpha;
  float beta;
  int store_smem_row_init;
  int store_smem_col_init;
  int read_smem_row_init;
  int read_smem_col_init;
  int store_smem_crosswise_factor;
  int read_smem_crosswise_factor;
};

__device__ __forceinline__ void epilogue(half *gmem_ptr, float *smem_ptr,
                                         float *frag_ptr, int M, int N,
                                         float alpha, float beta) {
  // frag [r, c, _]: [2, 2, 2]
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  // smem view is [128x128]
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int row = tz * 64 + i * 16 + 0 * 8 + tx / 4;
      for (int r = 0; r < 2; ++r) {
        int scol = ty * 64 + j * 16 + 0 * 8 + tx % 4 * 2;
        for (int c = 0; c < 2; ++c) {
          // scol = scol ^ ((row & 3) << 3);
          float2 *ptr = reinterpret_cast<float2 *>(smem_ptr + row * 128 + scol);
          float2 *frag = reinterpret_cast<float2 *>(frag_ptr + i * 32 + j * 8 +
                                                    r * 4 + c * 2 + 0);
          *ptr = *frag;
          scol += 8;
        }
        row += 8;
      }
    }
    __syncthreads();

    int p = tx / 2;
    int q = tx % 2;
    int srow = tz * 64 + i * 16 + p;
    int scol = ty * 64 + 0 * 16 + q * 8;
    for (int j = 0; j < 4; ++j) {
      half2 a[4];
      float4 value[2];
      // int scol = col ^ ((srow & 3) << 3);
      float4 *ptr = reinterpret_cast<float4 *>(smem_ptr + srow * 128 + scol);
      float2 *asFloat2 = reinterpret_cast<float2 *>(value);
      float *asFloat = reinterpret_cast<float *>(value + j * 2);
      for (int x = 0; x < 8; ++x) {
        asFloat[x] *= alpha;
        asFloat[x] += beta;
      }
      uint4 *asUInt4 = reinterpret_cast<uint4 *>(a);
      value[0] = *ptr;
      value[1] = *(ptr + 1);
      a[0] = __float22half2_rn(asFloat2[0]);
      a[1] = __float22half2_rn(asFloat2[1]);
      a[2] = __float22half2_rn(asFloat2[2]);
      a[3] = __float22half2_rn(asFloat2[3]);
      uint4 *dst = reinterpret_cast<uint4 *>(gmem_ptr + (by * 128 + srow) * N +
                                             bx * 128 + scol);
      *dst = *asUInt4;
      scol += 16;
    }
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

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K,
                       float alpha, float beta) {
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
  half *SA[] = {SA1, SA2, SA3, SA4};
  half *SB[] = {SB1, SB2, SB3, SB4};
  float *SC = reinterpret_cast<float *>(shared_storage);

  unsigned int FragA[2][4 * 4]; // [4, 4]
  unsigned int FragB[2][4 * 4]; // [4, 4]

  float Accum[4 * 4 * 8] = {0.0}; // [4, 4, 8]

  // FragIteratorA frag_iter_A(SA[0]);

  // prologue
  for (int i = 0; i < 3; ++i) {
    loadSmemA(SA[i], A, M, K, i);
    loadSmemB(SB[i], B, N, K, i);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
  __syncthreads();

  loadFragA(FragA[0], SA[(0) % 4], 0);
  // frag_iter_A(FragA[0]);
  // ++frag_iter_A;
  loadFragB(FragB[0], SB[(0) % 4], 0);

  for (int ko = 0; ko < K / KI; ko += 1) {

    // 64x64x16 mma for each warp
    loadFragA(FragA[1], SA[(ko) % 4], 1);
    // frag_iter_A(FragA[1]);
    // ++frag_iter_A;
    loadFragB(FragB[1], SB[(ko) % 4], 1);
#pragma unroll
    for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
        // 16x16x16 for each wmma
        int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
        mmaSync(&FragA[0][mii * 4], &FragB[0][n * 4], &Accum[mii * 32 + n * 8]);
      }
    }

    // if (ko + 3 < K / KI) {
    //     loadSmemA(SA[(ko+3)%4], A, M, K, ko + 3);
    //     loadSmemB(SB[(ko+3)%4], B, N, K, ko + 3);
    //     asm volatile("cp.async.commit_group;\n" ::);
    // }
    bool pred_guard = ko + 3 < K / KI;
    predLoadSmemA(SA[(ko + 3) % 4], A, M, K, ko + 3, pred_guard);
    predLoadSmemB(SB[(ko + 3) % 4], B, N, K, ko + 3, pred_guard);
    asm volatile("cp.async.commit_group;\n" ::);

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();

    // 64x64x16 mma for each warp
    loadFragA(FragA[0], SA[(ko + 1) % 4], 0);
    // frag_iter_A.set(SA[(ko+1)%4]);
    // frag_iter_A(FragA[0]);
    // ++frag_iter_A;
    loadFragB(FragB[0], SB[(ko + 1) % 4], 0);
#pragma unroll
    for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
      for (int nii = 0; nii < NII / wmmaN; nii += 1) {
        // 16x16x16 for each wmma
        int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
        mmaSync(&FragA[1][mii * 4], &FragB[1][n * 4], &Accum[mii * 32 + n * 8]);
      }
    }
  }

  storeAccum(SC, Accum);
  __syncthreads();
  storeSmemC(C, SC, M, N);
  // epilogue(C, SC, Accum, M, N, alpha, beta);
  // Epilogue<128, 128, 64, 64, 16, 16, 32, 2, 2> epi(C, SC, Accum, M, N, alpha,
  //                                                  beta);
  // epi();
}