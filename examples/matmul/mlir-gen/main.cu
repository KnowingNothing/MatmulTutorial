#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>

int STAGES = 1;
int MULTI_THREADING = 1;
int ITERS = 20;

extern __global__ void matmul(half *A, half *B, half *C);

// #define DEBUG
// #define PRINT
#ifdef DEBUG
#include <omp.h>
int M = 1024;
int N = 1024;
int K = 1024;
#else
int M = 5376;
int N = 5376;
int K = 2048;
#endif
#define MAX(a, b) (a) > (b) ? (a) : (b)
float alpha = 1.0;
float beta = 0.0;

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main(int argc, char *argv[])
{

    if (argc > 1)
    {
        assert((argc - 1) % 2 == 0);
        for (int i = 1; i < argc; i += 2)
        {
            char *key = argv[i];
            char *value = argv[i + 1];
            std::string keys(key);
            if (keys == "stages")
            {
                STAGES = std::atoi(value);
                std::cout << "Setting to " << STAGES << " stages.\n";
            }
            else if (keys == "multi_threading")
            {
                MULTI_THREADING = std::atoi(value);
                std::cout << "Setting to " << MULTI_THREADING << "x threading.\n";
            }
            else if (keys == "iters") {
                ITERS = std::atoi(value);
                std::cout << "Testing iters = " << ITERS << ".\n";
            } else if (keys == "M") {
                M = std::atoi(value);
            } else if (keys == "N") {
                N = std::atoi(value);
            } else if (keys == "K") {
                K = std::atoi(value);
            }
        }
    }
#ifdef DEBUG
    std::cout << "Debugging using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
#else
    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
#endif
    srand(time(NULL));
    half *hA = (half *)malloc(M * K * 2);
    half *hB = (half *)malloc(K * N * 2);
    half *hC = (half *)malloc(M * N * 2);
    half *golden = (half *)malloc(M * N * 2);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            hA[i * K + j] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);
            golden[i * N + j] = (float)(0);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            hB[n * K + k] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
    }

#ifdef DEBUG
    std::cout << "Computing golden values...\n";
// simple tiling to make it a bit faster
#pragma omp parallel for
    for (int i = 0; i < M; i += 64)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += 64)
        {
            float accum[64 * 64] = {0};
            for (int k = 0; k < K; k += 32)
            {
                for (int kk = 0; kk < 32; ++kk)
                {
                    for (int jj = 0; jj < 64; ++jj)
                    {
                        for (int ii = 0; ii < 64; ++ii)
                        {
                            accum[ii * 64 + jj] += ((float)hA[(i + ii) * K + k + kk] * (float)hB[(j + jj) * K + k + kk]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < 64; ++ii)
            {
                for (int jj = 0; jj < 64; ++jj)
                {
                    for (int kk = 0; kk < 64; ++kk)
                    {
                        golden[(i + ii) * N + j + jj] = (half)accum[ii * 64 + jj];
                    }
                }
            }
        }
    }
    std::cout << "Golden values done!\n";
#endif

    half *dA;
    half *dB;
    half *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 2 * MULTI_THREADING, 2);
    dim3 dimGrid(N / 128, M / 128);

#ifndef DEBUG
    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 4);
    if (smem_size >= (48 << 10))
    {
        CUDA_CHECK(cudaFuncSetAttribute(matmul,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warmup
    for (int i = 0; i < ITERS / 20 - 1; ++i)
    {
        matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
    {
        matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost (ms) of CUDA kernel is " << double(ms) / ITERS << "\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (double(ms) / ITERS) * 1e3 / 1e12 << "\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CUDA kernel is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";
#endif

#ifdef DEBUG
    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 4);
    std::cout << "Using shared memory = " << (double)smem_size / 1e3 << " KB.\n";
    if (smem_size >= (48 << 10))
    {
        CUDA_CHECK(cudaFuncSetAttribute(matmul,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    std::cout << "Computing result values...\n";
    matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Computing results done!\n";
    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));

#ifdef PRINT
    std::cout << "Golden:" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << (float)golden[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Results:" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << (float)hC[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    int errors = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float diff = ((float)golden[i * N + j] - (float)hC[i * N + j]);
            if (diff < 0)
            {
                diff = -diff;
            }
            float maxv = MAX((float)golden[i * N + j], (float)hC[i * N + j]);
            if (maxv < 0)
            {
                maxv = -maxv;
            }
            if (diff / maxv > 1e-2)
            {
                errors += 1;
            }
        }
    }

    if (errors)
    {
        std::cout << "Wrong Answer! " << errors << " errors.\n";
    }
    else
    {
        std::cout << "Correctness Check Passed!\n";
    }
#endif

    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}