#include "cute/algorithm/gemm.hpp"

/// RUN: nvcc -arch=sm_89 -std=c++17  -DDEBUG -Xcompiler -fopenmp -I /home/zhengsz/share/MatmulTutorial/3rdparty/cutlass/include matmul-v17.cu main.cu -o test && ./test stages 4

using namespace cute;

static constexpr int BLOCKM = 128;
static constexpr int BLOCKN = 128;
static constexpr int BLOCKK = 32;
static constexpr int Stages = 3;

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta) {
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{})
    );
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using GmemTiledCopyA = decltype(
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
            Layout<Shape<_32, _4>, Stride<_4, _1>>{},
            Layout<Shape<_1, _8>>{}
        )
    );
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        Shape<Int<BLOCKM>, Int<BLOCKK>, Int<Stages>>{}
    ));

    using SmemLayoutAtomB = decltype(
        composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{})
    );
    using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using GmemTiledCopyB = decltype(
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
            Layout<Shape<_32, _4>, Stride<_4, _1>>{},
            Layout<Shape<_1, _8>>{}
        )
    );
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        Shape<Int<BLOCKN>, Int<BLOCKK>, Int<Stages>>{}
    ));

    struct SharedStorage {
        cute::array_aligned<half_t, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<half_t, cute::cosize_v<SmemLayoutB>> smem_b;
    };

    /// TiledMMA

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_2, _2, _1>>,  // 2x2x1 thread group
        Layout<Shape<_1, _2, _1>>   // 1x2x1 or 1x2x2 value group for 16x16x16 MMA and LDSM
    >;

    auto tiled_mma = TiledMma{};

    /// Accumulator

    auto accum = partition_fragment_C(tiled_mma, Shape<Int<BLOCKM>, Int<BLOCKN>>{});

    clear(accum);

    /// Global Tensors
    int gA_offset = (blockIdx.y * BLOCKM) * K;
    auto gA = make_tensor(make_gmem_ptr(A), Shape<Int<BLOCKM>, Int<BLOCKK>>{}, Stride<Int<BLOCKK>, _1>{});
}