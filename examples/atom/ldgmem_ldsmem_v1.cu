#include "common.h"
// split copy async

// nvcc -arch=sm_90a -std=c++17 -I ../../include/ -lcuda ldgmem_ldsmem_v0.cu -o test

const int SM_LODA_BYTES = 128/8;

template <typename DType, int BLOCKM, int BLOCKN, int NUM_THREADS>
__global__ void split_matrix_ldsm(DType* source, int M, int N, DType* dummy_out, int split, int curr_split) {
    __shared__ DType smem[BLOCKM*BLOCKN];
    const int VEC_LEN = SM_LODA_BYTES / sizeof(DType);
    const int VEC_REPEAT = BLOCKN / VEC_LEN;
    const int THREAD_N = VEC_REPEAT;
    const int THREAD_M = NUM_THREADS / THREAD_N;
    const int ROW_REPEAT = BLOCKM / THREAD_M;
    static_assert(BLOCKN % VEC_LEN == 0);
    static_assert(NUM_THREADS % THREAD_N == 0);
    static_assert(ROW_REPEAT * THREAD_M == BLOCKM);

    dummy_out += M / split * curr_split * N;

    int mo = blockIdx.x * BLOCKM;
    int mi = threadIdx.x / THREAD_N;
    int ni = threadIdx.x % THREAD_N;
    int4* ld_source = reinterpret_cast<int4*>(source);
    int4* ld_smem = reinterpret_cast<int4*>(smem);
    for (int no = 0; no < N; no += BLOCKN) {
        for (int row_repeat = 0; row_repeat < ROW_REPEAT; ++row_repeat) {
            int m = mo + row_repeat * THREAD_M + mi;
            int n = no + ni * VEC_LEN;
            int idx = m * N + n;
            int sm = row_repeat * THREAD_M + mi;
            int sn = ni * VEC_LEN;
            int sm_idx = sm * BLOCKN + sn;
            ld_smem[sm_idx / VEC_LEN] = ld_source[idx / VEC_LEN];
        }
        __syncthreads();
        for (int x = 0; x < 256; ++x) {
            for (int row_repeat = 0; row_repeat < ROW_REPEAT; ++row_repeat) {
                int m = mo + row_repeat * THREAD_M + mi;
                int n = no + ni * VEC_LEN;
                int idx = m * N + n;
                int sm = row_repeat * THREAD_M + mi;
                int sn = ni * VEC_LEN;
                int sm_idx = sm * BLOCKN + sn;
                for (int i = 0; i < VEC_LEN; ++i) {
                    dummy_out[idx + i] = smem[sm_idx + i] + DType(1);
                }
            }
        }
    }
}


template<typename DType>
void cpu_dummy(DType* source, DType* dummy_out, int M, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            dummy_out[m * N + n] = (DType)((float)source[m * N + n] + (float)DType(1));
        }
    }
}


int main(int argc, char** argv) {
    const int M = 1024;
    const int N = 1024;
    int split = 4;
    using DType = half;
    const int BLOCKM = 128;
    const int BLOCKN = 128;
    const int NUM_THREADS = 128;
    std::vector<int> shape{M, N};
    std::vector<int> epoch_shape{M/split, N};
    auto A = alloc_cpu_tensor<DType>(shape);
    random_fill(A, shape);
    // constant_fill(A, shape, DType(1));
    auto B = alloc_cpu_tensor<DType>(shape);
    auto golden = alloc_cpu_tensor<DType>(shape);

    GPUTimer gpu_timer;

    std::vector<DType*> dAs;
    for (int i = 0; i < split; ++i) {
        dAs.push_back(alloc_gpu_tensor<DType>(epoch_shape));
    }
    auto dB = alloc_gpu_tensor<DType>(shape);
    
    dim3 block(NUM_THREADS);
    dim3 grid(ceil_div(M/split, BLOCKM));
    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < split; ++i) {
        copy_to_gpu_async(A + M/split * i * N, dAs[i], epoch_shape);
        split_matrix_ldsm<DType, BLOCKM, BLOCKN, NUM_THREADS><<<grid, block>>>(dAs[i], M, N, dB, split, i);
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    std::cout << "GPU split done! Use " << gpu_timer.report_last_ms() << " ms.\n";
    copy_to_cpu_async(B, dB, shape);
    

    std::cout << "Calculating golden...\n";
    cpu_dummy(A, golden, M, N);
    assert_allclose(B, golden, shape, 1e-5, /*dump=*/false);
    std::cout << "Correct!\n";

    
    return 0;
}