#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

using namespace nvcuda::wmma;

__global__ void warp_mma_16x16x16(half* A, half* B, float* C) {
    fragment<matrix_a, 16, 16, 16, half, row_major> FragA;
    fragment<matrix_b, 16, 16, 16, half, col_major> FragB;
    fragment<accumulator, 16, 16, 16, float> Accum;

    fill_fragment(Accum, 0.0);

    load_matrix_sync(FragA, A, 16);
    load_matrix_sync(FragB, B, 16);

    mma_sync(Accum, FragA, FragB, Accum);

    store_matrix_sync(C, Accum, 16, mem_row_major);
}

int main() {

    return 0;
}