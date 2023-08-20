// A100 PCIE 80GB
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 4.46636ms
// TFLOPS: 26.5048

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

const int MI = 128;
// const int NI = 128;
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
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = A[(by * 128 + row) * K + ko * KI + col];
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
    for (int i = 0; i < 64; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
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
    half *SA = reinterpret_cast<half *>(shared_storage);
    half *SB = reinterpret_cast<half *>(shared_storage + MI * KI * sizeof(half));
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
    for (int ko = 0; ko < K / KI; ko += 1)
    {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA, ki);
            loadFragB(FragB, SB, ki);
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