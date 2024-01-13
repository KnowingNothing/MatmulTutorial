/***************************************************************************************************
 * Some code from barrier.h in Nvidia CUTLASS, the original copyright is:
 *
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using half_t = half;

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define HOST __forceinline__ __host__
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

static constexpr int SM_NUMBER = 114;
static constexpr int MAX_CLUSTER_SIZE = 16;
static constexpr int WARP_GROUP_SIZE = 128;
static constexpr int WARP_SIZE = 32;

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

HOST_DEVICE constexpr int ceil_div(int a, int b) { return (a + (b - 1)) / b; }

template <class DType>
HOST_DEVICE constexpr DType _abs(DType a) {
  return a < DType(0) ? -a : a;
}

template <class DType>
HOST_DEVICE constexpr DType _max(DType a, DType b) {
  return a < b ? b : a;
}

template <class DType>
HOST_DEVICE constexpr DType _min(DType a, DType b) {
  return a < b ? a : b;
}

// Computes the result of bitwise right-shift
template <class T>
HOST_DEVICE constexpr T shiftr(T x, int s) {
  return s >= 0 ? (x >> s) : (x << -s);
}

// Short name for fast compilation
template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
  HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

template <class T, T v>
using constant = C<v>;

#define LEFT_UNARY_OP(OP)                             \
  template <auto t>                                   \
  HOST_DEVICE constexpr C<(OP t)> operator OP(C<t>) { \
    return {};                                        \
  }
#define RIGHT_UNARY_OP(OP)                            \
  template <auto t>                                   \
  HOST_DEVICE constexpr C<(t OP)> operator OP(C<t>) { \
    return {};                                        \
  }
#define BINARY_OP(OP)                                         \
  template <auto t, auto u>                                   \
  HOST_DEVICE constexpr C<(t OP u)> operator OP(C<t>, C<u>) { \
    return {};                                                \
  }

#define BINARY_OP_RVAL(OP)                                         \
  template <auto t, class U, REQUIRES(std::is_integral<U>::value)> \
  HOST_DEVICE constexpr auto operator OP(C<t>, const U u) {        \
    return t OP u;                                                 \
  }

#define BINARY_OP_LVAL(OP)                                         \
  template <class T, auto u, REQUIRES(std::is_integral<T>::value)> \
  HOST_DEVICE constexpr auto operator OP(const T t, C<u>) {        \
    return t OP u;                                                 \
  }

LEFT_UNARY_OP(+);
LEFT_UNARY_OP(-);
LEFT_UNARY_OP(~);
LEFT_UNARY_OP(!);
LEFT_UNARY_OP(*);

BINARY_OP(+);
BINARY_OP(-);
BINARY_OP(*);
BINARY_OP(/);
BINARY_OP(%);
BINARY_OP(&);
BINARY_OP(|);
BINARY_OP(^);
BINARY_OP(<<);
BINARY_OP(>>);

BINARY_OP(&&);
BINARY_OP(||);

BINARY_OP(==);
BINARY_OP(!=);
BINARY_OP(>);
BINARY_OP(<);
BINARY_OP(>=);
BINARY_OP(<=);

BINARY_OP_RVAL(+);
BINARY_OP_RVAL(-);
BINARY_OP_RVAL(*);
BINARY_OP_RVAL(/);
BINARY_OP_RVAL(%);
BINARY_OP_RVAL(&);
BINARY_OP_RVAL(|);
BINARY_OP_RVAL(^);
BINARY_OP_RVAL(<<);
BINARY_OP_RVAL(>>);

BINARY_OP_RVAL(&&);
BINARY_OP_RVAL(||);

BINARY_OP_RVAL(==);
BINARY_OP_RVAL(!=);
BINARY_OP_RVAL(>);
BINARY_OP_RVAL(<);
BINARY_OP_RVAL(>=);
BINARY_OP_RVAL(<=);

BINARY_OP_LVAL(+);
BINARY_OP_LVAL(-);
BINARY_OP_LVAL(*);
BINARY_OP_LVAL(/);
BINARY_OP_LVAL(%);
BINARY_OP_LVAL(&);
BINARY_OP_LVAL(|);
BINARY_OP_LVAL(^);
BINARY_OP_LVAL(<<);
BINARY_OP_LVAL(>>);

BINARY_OP_LVAL(&&);
BINARY_OP_LVAL(||);

BINARY_OP_LVAL(==);
BINARY_OP_LVAL(!=);
BINARY_OP_LVAL(>);
BINARY_OP_LVAL(<);
BINARY_OP_LVAL(>=);
BINARY_OP_LVAL(<=);

#undef BINARY_OP
#undef LEFT_UNARY_OP
#undef RIGHT_UNARY_OP

template <class AType, class BType, class CType, class AccumType>
struct GemmParams {
  const int M;
  const int N;
  const int K;
  const AType* A;
  const BType* B;
  CType* C;
  const AccumType alpha;
  const AccumType beta;

  GemmParams(const int M, const int N, const int K, const AType* A,
             const BType* B, CType* C, const AccumType alpha,
             const AccumType beta)
      : M(M), N(N), K(K), A(A), B(B), C(C), alpha(alpha), beta(beta) {}
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
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        trace_[1 - cur_] - trace_[cur_]);

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

  void sync_all() { cudaDeviceSynchronize(); }

 private:
  cudaEvent_t events_[2];
  int cur_ = 0;
};

template <class DType>
DType* alloc_cpu_tensor(std::vector<int> shape) {
  return (DType*)malloc(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(DType));
}

template <class DType>
void random_fill(DType* tensor, std::vector<int> shape) {
  int length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < length; ++i) {
    tensor[i] = (DType)((rand() % 1000 * 1 / 100 % 10 - 5.0));
  }
}

template <class DType>
void arange_fill(DType* tensor, std::vector<int> shape, int bound = 1024) {
  int length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < length; ++i) {
    tensor[i] = (DType)((i % bound));
  }
}

template <class DType>
void constant_fill(DType* tensor, std::vector<int> shape, DType value) {
  int length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < length; ++i) {
    tensor[i] = value;
  }
}

template <class DType>
DType* alloc_gpu_tensor(std::vector<int> shape) {
  DType* dt;
  CUDA_CHECK(cudaMalloc(
      (void**)&dt,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          sizeof(DType)));
  return dt;
}

template <class DType>
void free_cpu_tensor(DType* ptr) {
  free(ptr);
}

template <class DType>
void free_gpu_tensor(DType* ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

template <class DType>
void copy_to_gpu(DType* hptr, DType* dptr, std::vector<int> shape) {
  CUDA_CHECK(cudaMemcpy(
      dptr, hptr,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          sizeof(DType),
      cudaMemcpyHostToDevice));
}

template <class DType>
void copy_to_gpu_async(DType* hptr, DType* dptr, std::vector<int> shape,
                       cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpyAsync(
      dptr, hptr,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          sizeof(DType),
      cudaMemcpyHostToDevice, stream));
}

template <class DType>
void copy_to_cpu(DType* hptr, DType* dptr, std::vector<int> shape) {
  CUDA_CHECK(cudaMemcpy(
      hptr, dptr,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          sizeof(DType),
      cudaMemcpyDeviceToHost));
}

template <class DType>
void copy_to_cpu_async(DType* hptr, DType* dptr, std::vector<int> shape,
                       cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpyAsync(
      hptr, dptr,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          sizeof(DType),
      cudaMemcpyDeviceToHost, stream));
}

template <class DType>
void assert_allclose(DType* res_ptr, DType* golden_ptr, std::vector<int> shape,
                     float rtol = 1e-5, bool dump = false) {
  int errors = 0;
  int length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
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
    std::cout << "Wrong answer! " << errors << " errors! "
              << (float)errors / length * 100 << "%\n";
    std::cout << "Average diff = " << total_diff / errors << "\n";
  }
  assert(errors == 0);
}

DEVICE uint32_t cast_smem_ptr_to_uint(void const* const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

DEVICE uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %rx;\n"
      ".reg .pred %px;\n"
      "     elect.sync %rx|%px, %2;\n"
      "@%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

DEVICE uint32_t block_rank_in_cluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
}

// Set the destination block-ID in cluster for a given SMEM Address
DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

// Returns the relative dim3 block rank local to the cluster.
DEVICE dim3 block_id_in_cluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the warp id in block
DEVICE int warp_id_in_block() { return threadIdx.x / WARP_SIZE; }

// Returns the warp id in warp group
DEVICE int warp_id_in_warp_group() {
  return warp_id_in_block() % (WARP_GROUP_SIZE / WARP_SIZE);
}

// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to
 * keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY
 * mask (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle {
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0, "MBase must be positive.");
  static_assert(num_bits >= 0, "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits,
                "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = constant<int, (1 << num_bits) - 1>;
  using yyy_msk = constant<int, bit_msk{} << (num_base + _max(0, num_shft))>;
  using zzz_msk = constant<int, bit_msk{} << (num_base - _min(0, num_shft))>;
  using msk_sft = constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk{} | zzz_msk{});

  template <class Offset>
  HOST_DEVICE constexpr static auto apply(Offset const& offset) {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});  // ZZZ ^= YYY
  }

  template <class Offset>
  HOST_DEVICE constexpr auto operator()(Offset const& offset) const {
    return apply(offset);
  }
};

/// Returns a warp index in the CTA. The threads in warp may not be convergent
/// As it doesn't sync the warp, it faster and allows forward progress
DEVICE int canonical_warp_idx() { return threadIdx.x / WARP_SIZE; }