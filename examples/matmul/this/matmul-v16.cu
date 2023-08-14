// CuTe
// A100 PCIE 80GB

#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#define CUTLASS_ENABLE_CUBLAS 0
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif
#include "cutlass/util/helper_cuda.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <cstdio>
#include <cassert>

// using half = cutlass::half_t;
using namespace cute;
using X = Underscore;

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define STAGES 4


template <class TA, class TB, class TC>
__global__ void matmul(TA *A, TB *B, TC *C, int M, int N, int K)
{
    auto dA = make_stride(Int<1>{}, M);
    auto dB = make_stride(Int<1>{}, N);
    auto dC = make_stride(Int<1>{}, M);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};

    auto blockA = make_layout(make_shape(bM,bK));
    auto blockB = make_layout(make_shape(bN,bK));
    auto blockC = make_layout(make_shape(bM,bN));

    auto threadA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
    auto threadB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
    auto threadC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    extern __shared__ uint8_t shared_storage[];
    TA* sA_ptr = reinterpret_cast<TA *>(shared_storage);
    TB* sB_ptr = reinterpret_cast<TB *>(sA_ptr + bM * bK);

    auto sA = make_tensor(make_smem_ptr(sA_ptr), blockA);               // (BLK_M,BLK_K)
    auto sB = make_tensor(make_smem_ptr(sB_ptr), blockB);               // (BLK_N,BLK_K)

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

    auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));// (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);            // (m,n,k)

    // auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    auto gA = local_tile(mA, make_shape(bM, bK), make_coord(blockIdx.x, _));
    // if (thread0()) {
    //   print("gA:\n");
    //   print(gA.shape());
    //   print("\n");
    //   print(gA.stride());
    //   print("\n");
    // }

    auto tAgA = local_partition(gA, threadA, threadIdx.x);                  // (THR_M,THR_K,k)
    auto tAsA = local_partition(sA, threadA, threadIdx.x);                  // (THR_M,THR_K)

    auto tBgB = local_partition(gB, threadB, threadIdx.x);                  // (THR_N,THR_K,k)
    auto tBsB = local_partition(sB, threadB, threadIdx.x);                  // (THR_N,THR_K)

    // Partition sA (M,K) by the rows of tC
    auto tCsA = local_partition(sA, threadC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    auto tCsB = local_partition(sB, threadC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC = local_partition(gC, threadC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);                              // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

    auto k_max = size<2>(tAgA);

    for (int k = 0; k < k_max; ++k)
    {
      // Copy gmem to smem
      copy(tAgA(_,_,k), tAsA);
      copy(tBgB(_,_,k), tBsB);

      cp_async_fence();
      cp_async_wait<0>();

      __syncthreads();

      // Compute gemm on smem
      gemm(tCsA, tCsB, tCrC);

      __syncthreads();
    }

    //
    // Epilogue
    //

    TC alpha = (TC)1.0;
    TC beta = (TC)0.0;
    axpby(alpha, tCrC, beta, tCgC);
}


void test_gemm(int m, int n, int k)
{
  cute::device_init(0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = half;
  using TB = half;
  using TC = float;
  using TI = float;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  TI alpha = 1.0;
  TI beta  = 0.0;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Run once
  d_C = h_C;
  blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                     m, n, k,
                     &alpha,
                     d_A.data().get(), m,
                     d_B.data().get(), n,
                     &beta,
                     d_C.data().get(), m);
  CUTE_CHECK_LAST();

  thrust::host_vector<TC> cublas_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       m, n, k,
                       &alpha,
                       d_A.data().get(), m,
                       d_B.data().get(), n,
                       &beta,
                       d_C.data().get(), m);
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);

#else

  std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
    "If you would like to enable verification with cuBLAS, "
    "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
    "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS

  //
  // CuTe
  //

  // Run once (and check)
  d_C = h_C;
  dim3 dimBlock(256);
  dim3 dimGrid(m / 128, n / 128);
  int smem_size = MAX(STAGES * 128 * 8 * 2 * 2, 128 * 128 * 4);
  std::cout << "Using shared memory = " << (double)smem_size / 1e3 << " KB.\n";
  if (smem_size >= (48 << 10))
  {
    cudaFuncSetAttribute(matmul<TA, TB, TC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  std::cout << "Computing result values...\n";
  matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(
    d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(
        d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

  auto host_matrix_to_const_column_major_cute_tensor =
    [](const auto& X, int num_rows, int num_cols, int LDX) {
      const auto shape = cute::Shape<int, int>{num_rows, num_cols};
      const auto strides = cute::Stride<int, int>{1, LDX};
      return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
    };

  const auto A_view = host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
  // B^T is k x n, so B is n x k.
  const auto B_view = host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
  const auto C_computed_view = host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
  const auto C_expected_view = host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
  print_matrix_multiply_mollified_relative_error("float", A_view, B_view, C_computed_view, C_expected_view);

#endif // CUTLASS_ENABLE_CUBLAS
}


int main(int argc, char** argv)
{
  int m = 5376;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5376;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 2048;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  test_gemm(m, n, k);

  return 0;
}