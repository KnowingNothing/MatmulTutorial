#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include <type_traits>
#include <vector>
#include <curand_kernel.h>
#include <vector>
#include <numeric>
#include <utility>

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

const int SM_NUMBER = 144;
const int WARP_GROUP_SIZE = 128;
const int WG_NUMBER = 3;
const int WARP_SIZE = 32;
const int CLUSTER_M = 2;
const int CLUSTER_N = 1;

__forceinline__ __device__ __host__ int ceil_div(int a, int b) {
    return (a + (b - 1)) / b;
}

template<class DType>
__forceinline__ __host__ __device__ DType _abs(DType a) {
    return a < DType(0) ? -a : a;
}

template<class AType, class BType, class CType, class AccumType>
struct GemmParams {
    const int M;
    const int N;
    const int K;
    const AType* A;
    const BType* B;
    CType* C;
    const AccumType alpha;
    const AccumType beta;

    GemmParams(const int M, const int N, const int K, const AType* A, const BType* B, CType* C, const AccumType alpha, const AccumType beta) : M(M), N(N), K(K), A(A), B(B), C(C), alpha(alpha), beta(beta) {}
};


struct Timer {
    virtual void tick() = 0;
    virtual double report_last_ms() = 0;
};


struct CPUTimer : public Timer {
    void tick() final {
        trace_[cur_] = std::chrono::high_resolution_clock::now();
        cur_ = 1 - cur_;
    }

    double report_last_ms() final {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(trace_[1 - cur_] - trace_[cur_]);

        return duration.count() / 1e3;
    }

private:
    decltype(std::chrono::high_resolution_clock::now()) trace_[2];
    int cur_ = 0;
};

struct GPUTimer : public Timer {
    GPUTimer() {
        cudaEventCreate(&events_[0]);
        cudaEventCreate(&events_[1]);
    }

    ~GPUTimer() {
        cudaEventDestroy(events_[0]);
        cudaEventDestroy(events_[1]);
    }

    void tick() final {
        cudaEventRecord(events_[cur_]);
        cur_ = 1 - cur_;
    }

    double report_last_ms() final {
        float ms;
        cudaEventElapsedTime(&ms, events_[cur_], events_[1 - cur_]);
        return ms;
    }

    void sync_all() {
        cudaDeviceSynchronize();
    }
private:
    cudaEvent_t events_[2];
    int cur_ = 0;
};

template<class DType>
DType* alloc_cpu_tensor(std::vector<int> shape) {
    return (DType*)malloc(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(DType));
}

template<class DType>
void random_fill(DType* tensor, std::vector<int> shape) {
    int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    for (int i = 0; i < length; ++i) {
        tensor[i] = (DType)((rand() % 1000 * 1 / 100 % 10 - 5.0));
    }
}

template<class DType>
DType* alloc_gpu_tensor(std::vector<int> shape) {
    DType* dt;
    CUDA_CHECK(cudaMalloc((void**)&dt, std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(DType)));
    return dt;
}

template<class DType>
void free_cpu_tensor(DType* ptr) {
    free(ptr);
}

template<class DType>
void free_gpu_tensor(DType* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

template<class DType>
void copy_to_gpu(DType* hptr, DType* dptr, std::vector<int> shape) {
    CUDA_CHECK(cudaMemcpy(dptr, hptr, std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(DType), cudaMemcpyHostToDevice));
}

template<class DType>
void copy_to_cpu(DType* hptr, DType* dptr, std::vector<int> shape) {
    CUDA_CHECK(cudaMemcpy(hptr, dptr, std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(DType), cudaMemcpyDeviceToHost));
}


template<class DType>
void assert_allclose(DType* res_ptr, DType* golden_ptr, std::vector<int> shape, float rtol = 1e-5, bool dump = false) {
    int errors = 0;
    int length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    int row_size = shape[shape.size() - 1];
    double total_diff = 0.0;
    for (int i = 0; i < length; ++i) {
        float r = (float)res_ptr[i];
        float g = (float)golden_ptr[i];
        assert(!std::isinf(r) && "Result value contains inf.");
        assert(!std::isinf(g) && "Golden value contains inf.");
        if (_abs(r - g) > rtol * _abs(g)) {
            errors += 1;
            total_diff += _abs(r - g);
        }
        if (dump) {
            std::cout << "(" << r << " " << g << ") ";
            if ((i + 1) % row_size == 0) {
                std::cout << "\n";
            }
        }
    }
    if (errors > 0) {
        std::cout << "Wrong answer! " << errors << " errors! " << (float)errors/length * 100 << "%\n";
        std::cout << "Average diff = " << total_diff / errors << "\n";
    }
    assert(errors == 0);
}