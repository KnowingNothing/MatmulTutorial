// wmma + pipeline

// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 1.15745ms
// TFLOPS: 102.277

// 3090
// Setting to 4 stages.
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 2.22491ms
// TFLOPS: 53.2068

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
const int wmmaK = 16;

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

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void storeAccum(float *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> *frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta)
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

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }

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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}