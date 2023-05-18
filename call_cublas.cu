// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 0.58983ms
// TFLOPS: 200.702

// 3090
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 1.97178ms
// TFLOPS: 60.0371

#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <cublas_v2.h>

inline const char*
cublas_get_error(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED -- The cuBLAS library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED -- Resource allocation failed inside the cuBLAS library.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE -- An unsupported value or parameter was passed to the function.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH -- The function requires a feature absent from the device architecture.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR -- An access to GPU memory space failed.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED -- The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR -- An internal cuBLAS operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED -- The functionality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR -- An error was detected when checking the current licensing.";
    default:
      return "CUBLAS_ERROR -- <unknown>";
  }
}

inline bool
cublas_is_error(cublasStatus_t status)
{
  return status != CUBLAS_STATUS_SUCCESS;
}

// hgemm
inline cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const half* alpha,
     const half* A, int ldA,
     const half* B, int ldB,
     const half* beta,
     half* C, int ldC)
{
  return cublasGemmEx(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const __half*>(alpha),
                      reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
                      reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
                      reinterpret_cast<const __half*>(beta),
                      reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


const int M = 5376;
const int N = 5376;
const int K = 2048;
#define MAX(a, b) (a) > (b) ? (a) : (b)

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
    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
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

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    half alpha = 1.0;
    half beta = 0.0;

    half *dA;
    half *dB;
    half *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    // warmup
    // for (int i = 0; i < 10; ++i)
    // {
    //     gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    // }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    for (int i = 0; i < 20; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost of CuBLAS is " << ms / 20.0 << "ms\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (ms / 20.0) * 1e3 / 1e12 << "\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CuBLAS is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";

    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}