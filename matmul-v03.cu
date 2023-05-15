// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 0.893115ms
// TFLOPS: 132.547

// 3090
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 2.1341ms
// TFLOPS: 55.4707

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
// const int wmmaK = 16;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 4; ++i)
    {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]

        void *ptr = (void *)(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));

        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                     "l"(&A[(by * 128 + row) * K + (ko * KI + col)]),
                     "n"(16));
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 4; ++i)
    {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]

        void *ptr = (void *)(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));

        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                     "l"(&B[(bx * 128 + row) * K + (ko * KI + col)]),
                     "n"(16));
    }
}

__device__ void loadSmemC(float *smem, half *C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
    }
}

__device__ void loadFragA(unsigned int *frag, half *smem, int ki)
{
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                int row = tz * 64 + i * 16 + j * 8 + tx / 4;
                int col = ki * KII + k * 8 + tx % 4 * 2;
                unsigned int *ptr = reinterpret_cast<unsigned int *>(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
                frag[j * 2 + k] = ptr[0];
            }
        }
    }
}

__device__ void loadFragB(unsigned int *frag, half *smem, int ki)
{
    // frag: [j, k]: []
    // load 64x16
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                int row = ty * 64 + i * 16 + j * 8 + tx / 4;
                int col = ki * KII + k * 8 + tx % 4 * 2;
                unsigned int *ptr = reinterpret_cast<unsigned int *>(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
                frag[j * 2 + k] = ptr[0];
            }
        }
    }
}

__device__ void storeAccum(float *ptr, float *frag)
{
    // frag [r, c, _]: [2, 2, 2]
    // store 64x64
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int r = 0; r < 2; ++r)
            {
                for (int c = 0; c < 2; ++c)
                {
                    int row = tz * 64 + i * 16 + r * 8 + tx / 4;
                    int col = ty * 64 + j * 16 + c * 8 + tx % 4 * 2;
                    float *dst = ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16;
                    dst[0] = frag[r * 4 + c * 2];
                    dst[1] = frag[r * 2 + c * 2 + 1];
                }
            }
        }
    }
}

__device__ void mmaSync(unsigned int *fragA, unsigned int *fragB, float *accum)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[0]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[1]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[2]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[3]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));
}

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K)
{
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

    unsigned int FragA[4];
    unsigned int FragB[4];
    float Accum[8] = {0.0};

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

    for (int ko = 0; ko < K / KI - 4; ko += 4)
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 3 < K / KI)
        {
            loadSmemA(SA4, A, M, K, ko + 3);
            loadSmemB(SB4, B, N, K, ko + 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA1, ki);
            loadFragB(FragB, SB1, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 4 < K / KI)
        {
            loadSmemA(SA1, A, M, K, ko + 4);
            loadSmemB(SB1, B, N, K, ko + 4);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA2, ki);
            loadFragB(FragB, SB2, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 5 < K / KI)
        {
            loadSmemA(SA2, A, M, K, ko + 5);
            loadSmemB(SB2, B, N, K, ko + 5);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA3, ki);
            loadFragB(FragB, SB3, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 6 < K / KI)
        {
            loadSmemA(SA3, A, M, K, ko + 6);
            loadSmemB(SB3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA4, ki);
            loadFragB(FragB, SB4, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }
    }

    // the last 4 iterations
    {
        int ko = (K / KI / 4 - 1) * 4;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 3 < K / KI)
        {
            loadSmemA(SA4, A, M, K, ko + 3);
            loadSmemB(SB4, B, N, K, ko + 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA1, ki);
            loadFragB(FragB, SB1, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 4 < K / KI)
        {
            loadSmemA(SA1, A, M, K, ko + 4);
            loadSmemB(SB1, B, N, K, ko + 4);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA2, ki);
            loadFragB(FragB, SB2, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        if (ko + 5 < K / KI)
        {
            loadSmemA(SA2, A, M, K, ko + 5);
            loadSmemB(SB2, B, N, K, ko + 5);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA3, ki);
            loadFragB(FragB, SB3, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        if (ko + 6 < K / KI)
        {
            loadSmemA(SA3, A, M, K, ko + 6);
            loadSmemB(SB3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA4, ki);
            loadFragB(FragB, SB4, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    mmaSync(FragA, FragB, Accum);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}