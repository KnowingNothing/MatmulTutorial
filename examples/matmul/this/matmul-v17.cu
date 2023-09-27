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

    /// Global Tensors before tiling
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    /// Global Tensors after block tiling
    auto blk_shape = Shape<Int<BLOCKM>, Int<BLOCKN>, Int<BLOCKK>>{};
    auto blk_coord = make_coord(blockIdx.y, blockIdx.x, _);

    auto gAblk = local_tile(gA, blk_shape, blk_coord, Step<_1, cute::Underscore, _1>{});
    auto gBblk = local_tile(gB, blk_shape, blk_coord, Step<cute::Underscore, _1, _1>{});
    auto gCblk = local_tile(gC, blk_shape, blk_coord, Step<_1, _1, cute::Underscore>{});

    /// Print debug
    if (thread0()) {
        print("SmemLayoutA: ");
        print(SmemLayoutA{});
        print("\n");
        print("SmemLayoutB: ");
        print(SmemLayoutB{});
        print("\n");
        print("gAblk: ");
        print(gAblk);
        print("\n");
        print("gBblk: ");
        print(gBblk);
        print("\n");
        print("gCblk: ");
        print(gCblk);
        print("\n");
        print("accum: ");
        print(accum);
        print("\n");
    }

    /// Shared Tensors
    extern __shared__ uint8_t raw_smem_ptr[];
    SharedStorage& smem_storage = *reinterpret_cast<SharedStorage*>(raw_smem_ptr);
    auto sAblk = make_tensor(make_smem_ptr(smem_storage.smem_a.data()), SmemLayoutA{});
    auto sBblk = make_tensor(make_smem_ptr(smem_storage.smem_b.data()), SmemLayoutB{});

    /// Global copy
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(threadIdx.x);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);

    auto gAthread = gmem_thr_copy_A.partition_S(gAblk);
    auto gBthread = gmem_thr_copy_B.partition_S(gBblk);
    auto sAthread = gmem_thr_copy_A.partition_D(sAblk);
    auto sBthread = gmem_thr_copy_B.partition_D(sBblk);

    /// Print debug
    if (thread0()) {
        print("gmem_thr_copy_A: ");
        print(gmem_thr_copy_A);
        print("\n");
        print("gmem_thr_copy_B: ");
        print(gmem_thr_copy_B);
        print("\n");
        print("gAthread: ");
        print(gAthread);
        print("\n");
        print("gBthread: ");
        print(gBthread);
        print("\n");
        print("sAthread: ");
        print(sAthread);
        print("\n");
        print("sBthread: ");
        print(sBthread);
        print("\n");
        print("gAthread(_,_,_,0): ");
        print(gAthread(_,_,_,0));
        print("\n");
        print("gBthread(_,_,_,0): ");
        print(gBthread(_,_,_,0));
        print("\n");
        print("size(recast<uint128_t>(gAthread(_,_,_,0))): ");
        static_assert(extent<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>::SRegisters>::value == 1);
        print(size(recast<uint128_t>(gAthread(_,_,_,0))));
        print("\n");
        print("group_modes<1,3>(gAthread(_,_,_,0)): ");
        print(group_modes<1,3>(gAthread(_,_,_,0)));
        print("\n");
        print("size<1>(group_modes<1,3>(gAthread(_,_,_,0))): ");
        print(size<1>(group_modes<1,3>(gAthread(_,_,_,0))));
        print("\n");
        print("group_modes<1,3>(gAthread(_,_,_,0))(_,0): ");
        print(group_modes<1,3>(gAthread(_,_,_,0))(_,0));
        print("\n");
        print("(recast<uint128_t>(group_modes<1,3>(gAthread(_,_,_,0))(_,0))): ");
        print((recast<uint128_t>(group_modes<1,3>(gAthread(_,_,_,0))(_,0))));
        print("\n");
        print("flatten(gAthread(_,_,_,0)): ");
        print(flatten(gAthread(_,_,_,0)));
        print("\n");
        print("group_modes<1,4>(flatten(gAthread(_,_,_,0)))(_,0): ");
        print(group_modes<1,4>(flatten(gAthread(_,_,_,0)))(_,0));
        print("\n");
        print("(recast<uint128_t>(group_modes<1,4>(flatten(gAthread(_,_,_,0)))(_,0))): ");
        print((recast<uint128_t>(group_modes<1,4>(flatten(gAthread(_,_,_,0)))(_,0))));
        print("\n");
        print("recast<uint128_t>(gAthread(_,_,_,0)): ");
        print((recast<uint128_t>(gAthread(_,_,_,0))));
        print("\n");
    }

    /// Pipeline begins
    int kouter = 0;
    int kmax = K / BLOCKK;
    #pragma unroll 4
    for (int k = 0; k < Stages - 1; ++k) {
        copy(gmem_tiled_copy_A, gAthread(_,_,_,kouter), sAthread(_,_,_,k));
        copy(gmem_tiled_copy_B, gBthread(_,_,_,kouter), sAthread(_,_,_,k));
        cp_async_fence();
        --kmax;
        if (kmax > 0) { ++kouter; }
    }

    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rAthread = thread_mma.partition_fragment_A(sAblk(_,_,0));
    auto rBthread = thread_mma.partition_fragment_B(sBblk(_,_,0));

    // Print debug
    if (thread0()) {
        print("rAthread: ");
        print(rAthread);
        print("\n");
        print("rBthread: ");
        print(rBthread);
        print("\n");
        print("size(tiled_mma): ");
        print(size(tiled_mma));
        print("\n");
        print("size(gmem_tiled_copy_A): ");
        print(size(gmem_tiled_copy_A));
        print("\n");
        print("size(gmem_tiled_copy_B): ");
        print(size(gmem_tiled_copy_B));
        print("\n");
    }
}