#pragma once

#include "common.h"

template<class AType, class BType, class CType, class AccumType>
void reference_cpu_gemm(GemmParams<AType, BType, CType, AccumType> params) {
    /// pointers in params are guaranteed to be in cpu memory
    int M = params.M;
    int N = params.N;
    int K = params.K;
    const AType* hA = params.A;
    const BType* hB = params.B;
    CType* golden = params.C;
#pragma omp parallel for
    for (int i = 0; i < M; i += 64)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += 64)
        {
            AccumType accum[64 * 64] = {0};
            for (int k = 0; k < K; k += 32)
            {
                for (int kk = 0; kk < 32; ++kk)
                {
                    if (k + kk >= K) break;
                    for (int jj = 0; jj < 64; ++jj)
                    {
                        for (int ii = 0; ii < 64; ++ii)
                        {
                            accum[ii * 64 + jj] += ((AccumType)hA[(i + ii) % M * K + k + kk] * (AccumType)hB[(j + jj) % N * K + k + kk]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < 64; ++ii)
            {
                for (int jj = 0; jj < 64; ++jj)
                {
                    if (i + ii < M && j + jj < N) {
                        golden[(i + ii) * N + j + jj] = (CType)(accum[ii * 64 + jj] * params.alpha + params.beta);
                    }
                }
            }
        }
    }
}


template<class AType, class BType, class CType, class AccumType,
         int MTile, int NTile, int KTile>
__global__ void reference_gpu_gemm_kernel(GemmParams<AType, BType, CType, AccumType> params) {
    static_assert(MTile > 0);
    static_assert(NTile > 0);
    static_assert(KTile > 0);
    int m = (blockIdx.y * blockDim.y + threadIdx.y) * MTile;
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * NTile;
    AccumType accum[MTile * NTile] = {AccumType(0)};
    for (int k = 0; k < params.K; k += KTile) {
        for (int mi = 0; mi < MTile; ++mi) {
            for (int ki = 0; ki < KTile; ++ki) {
                for (int ni = 0; ni < NTile; ++ni) {
                    AType aval = k + ki < params.K ? params.A[(m + mi) % params.M * params.K + k + ki] : AType(0);
                    BType bval = k + ki < params.K ? params.B[(n + ni) % params.N * params.K + k + ki] : BType(0);
                    accum[mi * NTile + ni] += AccumType(aval) * AccumType(bval);
                }
            }
        }
    }
    for (int mi = 0; mi < MTile; ++mi) {
        for (int ni = 0; ni < NTile; ++ni) {
            if ((m + mi < params.M) && (n + ni < params.N)) {
                params.C[(m + mi) * params.N + n + ni] = (CType)(accum[mi * NTile + ni] * params.alpha + params.beta);
            }
        }
    }
}

template<class AType, class BType, class CType, class AccumType>
void reference_gpu_gemm(GemmParams<AType, BType, CType, AccumType> params) {
    /// don't know what tilings are good, but not important here
    const int mTile = 4;
    const int nTile = 4;
    const int kTile = 4;
    const int tx = 128;
    const int ty = 1;
    dim3 grid(ceil_div(params.N, nTile * tx), ceil_div(params.M, mTile * ty));
    dim3 block(tx, ty);
    reference_gpu_gemm_kernel<AType, BType, CType, AccumType, mTile, nTile, kTile><<<grid, block>>>(params);
    CUDA_CHECK(cudaGetLastError());
}