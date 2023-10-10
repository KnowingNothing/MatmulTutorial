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

/// RUN: nvcc -arch=sm_90a -std=c++17 -DDEBUG -Xcompiler -fopenmp matmul-fp8-v0.cu -o test && ./test
/// RUN: nvcc -arch=sm_90a matmul-fp8-v0.cu -o test && ./test stages 1 iters 200

typedef __nv_fp8_e5m2 e5m2;
static constexpr int BLOCKM = 128;
static constexpr int BLOCKN = 128;
static constexpr int BLOCKK = 32;
static constexpr int WGMMAM = 64;
static constexpr int WGMMAN = 128;
static constexpr int WGMMAK = 32;
static_assert(BLOCKK == WGMMAK, "WGMMA K should be the same as BLOCK K.");
static constexpr int WARP_SIZE = 32;
static constexpr int WARP_GROUP_SIZE = 4;
static constexpr int WARP_GROUP_NUMBER = 1;
static constexpr int WARP_NUMBER = WARP_GROUP_NUMBER * WARP_GROUP_SIZE;

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

int STAGES = 1;
int ITERS = 20;

template <class T> struct sizeof_bits {
  static constexpr int value = sizeof(T) * 8;
};

template <class T> static constexpr int sizeof_bits_v = sizeof_bits<T>::value;

union GmmaDescriptor {

  HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  HOST_DEVICE constexpr GmmaDescriptor &
  operator=(GmmaDescriptor const &t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2; // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }

  // Printer
  //   HOST_DEVICE friend void print(GmmaDescriptor const& t)
  //   {
  //     #if !defined(__CUDACC_RTC__)
  //     printf("GmmaDescriptor: 0x%016 %lli\n", static_cast<long
  //     long>(t.desc_)); printf("  start_addr :  0x%04x\n",
  //     t.bitfield.start_address_); printf("  leading_off:  0x%04x (%d)\n",
  //     t.bitfield.leading_byte_offset_, t.bitfield.leading_byte_offset_);
  //     printf("  stride_off :  0x%04x (%d)\n", t.bitfield.stride_byte_offset_,
  //     t.bitfield.stride_byte_offset_); printf("  base_offset:  0x%01x\n",
  //     t.bitfield.base_offset_); printf("  layout_type:  0x%01x (%s)\n",
  //     t.bitfield.layout_type_,
  //     to_string(static_cast<GMMA::LayoutType>(t.bitfield.layout_type_)));
  //     #endif
  //   }
};

struct MatmulKernel {
  using ADType = e5m2;
  using BDType = e5m2;
  using CDType = half;
  static_assert(std::is_same<ADType, BDType>::value);
  static constexpr int AElements = BLOCKM * BLOCKK;
  static constexpr int BElements = BLOCKN * BLOCKK;

  static constexpr int LD_SMEM_VEC_BYTES =
      16; /// shared memory vector load maximum size is 16 bytes (4 banks)
  static constexpr int LD_SMEM_VEC_ELEMS_A = LD_SMEM_VEC_BYTES / sizeof(ADType);
  static constexpr int LD_SMEM_VEC_ELEMS_B = LD_SMEM_VEC_BYTES / sizeof(BDType);
  /// fixed by WGMMA shared memory layout
  static constexpr int WGMMA_SMEM_BLOCK_ROW = 8;
  static constexpr int WGMMA_SMEM_BLOCK_COL = 128 / sizeof_bits_v<ADType>;
  static_assert(LD_SMEM_VEC_BYTES == WGMMA_SMEM_BLOCK_COL);
  /// fixed by WGMMA fragment layout: accumulator C/D
  static constexpr int WGMMA_FRGA_ELEMNTS_C =
      WGMMAM * WGMMAN / (WARP_NUMBER * WARP_SIZE);
  using WGMMA_ACCUM_ITEM_TYPE = uint32_t;
  static constexpr int WGMMA_FRAG_ITEM_BYTES = sizeof(WGMMA_ACCUM_ITEM_TYPE);
  static constexpr int WGMMA_FRAG_ELEMENTS_PER_ITEM =
      WGMMA_FRAG_ITEM_BYTES / sizeof(CDType);
  static constexpr int WGMMA_LANE_ROW_ELEMENT_STRIDE = 8;
  static constexpr int WGMMA_LANE_COL_ELEMENT_STRIDE = 8;
  static constexpr int WGMMA_LANES_PER_ROW =
      WGMMA_LANE_ROW_ELEMENT_STRIDE / WGMMA_FRAG_ELEMENTS_PER_ITEM; /// 4
  static constexpr int WGMMA_LANES_PER_COL =
      WARP_SIZE / WGMMA_LANES_PER_ROW; /// 8
  static_assert(WGMMA_LANE_COL_ELEMENT_STRIDE == WGMMA_LANES_PER_COL,
                "WGMMA fragment C/D layout mismatch.");
  static constexpr int WGMMA_WARPS_PER_ROW = 1;
  static constexpr int WGMMA_WARPS_PER_COL =
      WARP_GROUP_SIZE / WGMMA_WARPS_PER_ROW; /// 4
  static constexpr int WGMMA_WARP_REPEATS_PER_ROW = WGMMAN / WGMMA_WARPS_PER_ROW / WGMMA_LANE_ROW_ELEMENT_STRIDE; /// N / 8
      // WGMMAN / WGMMA_WARPS_PER_ROW / WGMMA_LANES_PER_ROW; /// N / 4
  static constexpr int WGMMA_WARP_REPEATS_PER_COL =
      WGMMAM / WGMMA_WARPS_PER_COL / WGMMA_LANES_PER_COL; /// 2
  static constexpr int WGMMA_WARP_ROW_ELEMENT_STRIDE =
      WGMMAN / WGMMA_WARP_REPEATS_PER_ROW / WGMMA_WARPS_PER_ROW; /// 8
  static constexpr int WGMMA_WARP_COL_ELEMENT_STRIDE =
      WGMMAM / WGMMA_WARP_REPEATS_PER_COL / WGMMA_WARPS_PER_COL; /// 8
  static_assert(WGMMA_LANE_ROW_ELEMENT_STRIDE == WGMMA_WARP_ROW_ELEMENT_STRIDE);
  static_assert(WGMMA_LANE_COL_ELEMENT_STRIDE == WGMMA_WARP_COL_ELEMENT_STRIDE);

  /// swizzle decisions
  /// matrix A swizzle: row-major
  static constexpr int LoadWarpsPerRowA = 2;
  static constexpr int LoadWarpsPerColA = WARP_NUMBER / LoadWarpsPerRowA;
  static constexpr int LoadElementsPerRowPerWarpA = BLOCKK / LoadWarpsPerRowA;
  static constexpr int LoadThreadsPerRowPerWarpA =
      LoadElementsPerRowPerWarpA / LD_SMEM_VEC_ELEMS_A;
  static constexpr int LoadElementsPerColPerWarpA =
      WARP_SIZE / LoadThreadsPerRowPerWarpA;
  static constexpr int LoadThreadsPerColPerWarpA =
      WARP_SIZE / LoadThreadsPerRowPerWarpA;
  static constexpr int LoadRepeatsPerRowPerWarpA =
      BLOCKK / LoadWarpsPerRowA / LoadElementsPerRowPerWarpA;
  static constexpr int LoadRepeatsPerColPerWarpA =
      BLOCKM / LoadWarpsPerColA / LoadElementsPerColPerWarpA;

  /// matrix B swizzle: column-major
  static constexpr int LoadWarpsPerColB = 2;
  static constexpr int LoadWarpsPerRowB = WARP_NUMBER / LoadWarpsPerColB;
  static constexpr int LoadElementsPerColPerWarpB = BLOCKK / LoadWarpsPerColB;
  static constexpr int LoadThreadsPerColPerWarpB =
      LoadElementsPerColPerWarpB / LD_SMEM_VEC_ELEMS_B;
  static constexpr int LoadElementsPerRowPerWarpB =
      WARP_SIZE / LoadThreadsPerColPerWarpB;
  static constexpr int LoadThreadsPerRowPerWarpB =
      WARP_SIZE / LoadThreadsPerColPerWarpB;
  static constexpr int LoadRepeatsPerColPerWarpB =
      BLOCKK / LoadWarpsPerColB / LoadElementsPerColPerWarpB;
  static constexpr int LoadRepeatsPerRowPerWarpB =
      BLOCKN / LoadWarpsPerRowB / LoadElementsPerRowPerWarpB;

  /// compute swizzle
  static constexpr int ComputeWarpsPerCol = 2;
  static constexpr int ComputeWarpsPerRow = WARP_NUMBER / ComputeWarpsPerCol;

  /// matrix C swizzle: row-major
  /// warpgroup organization
  static constexpr int WarpGroupsPerRow = 1;
  static constexpr int WarpGroupsPerCol =
      WARP_GROUP_NUMBER / WarpGroupsPerRow; /// 1
  static constexpr int WarpGroupRepeatsPerRowC =
      BLOCKN / WarpGroupsPerRow / WGMMAN; /// 1
  static constexpr int WarpGroupRepeatsPerColC =
      BLOCKM / WarpGroupsPerCol / WGMMAM; /// 2

  /// constructor
  DEVICE MatmulKernel(int M, int N, int K, uint8_t *smem) {
    GlobalM = M;
    GlobalN = N;
    GlobalK = K;
    smem_ = smem;
    /// currently, no pipeline
    SA_ = reinterpret_cast<e5m2 *>(smem_);
    SB_ = SA_ + AElements;
    /// different views of accumulator
    accum_item_ = reinterpret_cast<WGMMA_ACCUM_ITEM_TYPE *>(accum_);
  }

  /// PTX wrapper
  template <class PointerType>
  DEVICE void async_copy(void *smem, PointerType global) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_ptr),
        "l"(global), "n"(LD_SMEM_VEC_BYTES), "r"(LD_SMEM_VEC_BYTES));
  }

  DEVICE void async_copy_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
  }

  template <int Group> DEVICE void async_copy_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Group));
  }

  /// load functions
  DEVICE void ldGlobalSharedA(ADType *globalA, int ko) {
    for (int i = 0; i < LoadRepeatsPerColPerWarpA; ++i) {
      int globalRowIdx = globalRowElementIdxA(i);
      int sharedRowIdx = sharedRowElementIdxA(i);
      for (int j = 0; j < LoadRepeatsPerRowPerWarpA; ++j) {
        int globalColIdx = globalColElementIdxA(ko, j);
        int sharedColIdx = sharedColElementIdxA(j);
        int globalElementIdx = globalElementIdxA(globalRowIdx, globalColIdx);
        int sharedElementIdx = sharedElementIdxA(sharedRowIdx, sharedColIdx);
        void *ptr = (void *)(SA_ + sharedElementIdx);
        async_copy(ptr, &globalA[globalElementIdx]);
      }
    }
  }

  DEVICE void ldGlobalSharedB(BDType *globalB, int ko) {
    for (int i = 0; i < LoadRepeatsPerRowPerWarpB; ++i) {
      int globalColIdx = globalColElementIdxB(i);
      int sharedColIdx = sharedColElementIdxB(i);
      for (int j = 0; j < LoadRepeatsPerColPerWarpB; ++j) {
        int globalRowIdx = globalRowElementIdxB(ko, j);
        int sharedRowIdx = sharedRowElementIdxB(j);
        int globalElementIdx = globalElementIdxB(globalRowIdx, globalColIdx);
        int sharedElementIdx = sharedElementIdxB(sharedRowIdx, sharedColIdx);
        void *ptr = (void *)(SB_ + sharedElementIdx);
        async_copy(ptr, &globalB[globalElementIdx]);
      }
    }
  }

  /// compute functions
  /// call wgmma.mma_async
  DEVICE void MMA(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t &d00,
                  uint32_t &d01, uint32_t &d02, uint32_t &d03, uint32_t &d04,
                  uint32_t &d05, uint32_t &d06, uint32_t &d07, uint32_t &d08,
                  uint32_t &d09, uint32_t &d10, uint32_t &d11, uint32_t &d12,
                  uint32_t &d13, uint32_t &d14, uint32_t &d15, uint32_t &d16,
                  uint32_t &d17, uint32_t &d18, uint32_t &d19, uint32_t &d20,
                  uint32_t &d21, uint32_t &d22, uint32_t &d23, uint32_t &d24,
                  uint32_t &d25, uint32_t &d26, uint32_t &d27, uint32_t &d28,
                  uint32_t &d29, uint32_t &d30, uint32_t &d31) {
    int scale_D = 1; /// use D=A*B+C format
    constexpr int32_t scaleA = 1;
    constexpr int32_t scaleB = 1;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k32.f16.e5m2.e5m2 "
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
                 " %8,  %9,  %10, %11, %12, %13, %14, %15, "
                 " %16, %17, %18, %19, %20, %21, %22, %23, "
                 " %24, %25, %26, %27, %28, %29, %30, %31},"
                 " %32,"
                 " %33,"
                 " p,   %35, %36;\n"
                 "}\n"
                 : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03), "+r"(d04),
                   "+r"(d05), "+r"(d06), "+r"(d07), "+r"(d08), "+r"(d09),
                   "+r"(d10), "+r"(d11), "+r"(d12), "+r"(d13), "+r"(d14),
                   "+r"(d15), "+r"(d16), "+r"(d17), "+r"(d18), "+r"(d19),
                   "+r"(d20), "+r"(d21), "+r"(d22), "+r"(d23), "+r"(d24),
                   "+r"(d25), "+r"(d26), "+r"(d27), "+r"(d28), "+r"(d29),
                   "+r"(d30), "+r"(d31)
                 : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)),
                   "n"(int32_t(scaleA)), "n"(int32_t(scaleB)));
  }

  /// make shared memory descriptor
  template <class PointerType>
  DEVICE GmmaDescriptor make_desc_a(PointerType smem_ptr) {
    GmmaDescriptor desc;
    uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = 0;          /// no swizzle
    desc.bitfield.leading_byte_offset_ = 8; /// 8 bytes
    desc.bitfield.stride_byte_offset_ = 16;   /// 16 bytes
    /// base_offset_ is not valid for non-swizzle
    return desc;
  }

  template <class PointerType>
  DEVICE GmmaDescriptor make_desc_b(PointerType smem_ptr) {
    GmmaDescriptor desc;
    uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = 0;          /// no swizzle
    desc.bitfield.leading_byte_offset_ = 8; /// 8 bytes
    desc.bitfield.stride_byte_offset_ = 16;   /// 16 bytes
    /// base_offset_ is not valid for non-swizzle
    return desc;
  }

  DEVICE void warpgroup_fence_operand(WGMMA_ACCUM_ITEM_TYPE &reg) {
    asm volatile("" : "+r"(reg)::"memory");
  }

  DEVICE
  void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  DEVICE
  void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  template <int N> DEVICE void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
  }

  /// naive gemm without pipeline
  DEVICE void gemm_no_pipeline(ADType *globalA, BDType *globalB,
                               CDType *globalC) {
    for (int ko = 0; ko < GlobalK / BLOCKK; ++ko) {
      ldGlobalSharedA(globalA, ko);
      ldGlobalSharedB(globalB, ko);
      async_copy_fence();
      async_copy_wait<0>();
      __syncthreads();
      for (int i = 0; i < WarpGroupRepeatsPerColC; ++i) {
        ADType *smemA = SA_ + i * WGMMAM * WGMMAK;
        for (int j = 0; j < WarpGroupRepeatsPerRowC; ++j) {
          BDType *smemB = SB_ + j * WGMMAN * WGMMAK;
          GmmaDescriptor desc_a = make_desc_a(smemA);
          GmmaDescriptor desc_b = make_desc_b(smemB);
          WGMMA_ACCUM_ITEM_TYPE *curr_accum_item =
              accum_item_ + (i * WarpGroupRepeatsPerRowC + j) *
                                WGMMA_FRGA_ELEMNTS_C /
                                WGMMA_FRAG_ELEMENTS_PER_ITEM;
          // clear
          // for (int i = 0; i < WGMMA_FRGA_ELEMNTS_C / WGMMA_FRAG_ELEMENTS_PER_ITEM; ++i) {
          //   curr_accum_item[i] = (WGMMA_ACCUM_ITEM_TYPE)0;
          // }
          for (int i = 0; i < WGMMA_FRGA_ELEMNTS_C / WGMMA_FRAG_ELEMENTS_PER_ITEM; ++i) {
            warpgroup_fence_operand(curr_accum_item[i]);
          }
          warpgroup_arrive();
          MMA(desc_a, desc_b, curr_accum_item[0], curr_accum_item[1],
              curr_accum_item[2], curr_accum_item[3], curr_accum_item[4],
              curr_accum_item[5], curr_accum_item[6], curr_accum_item[7],
              curr_accum_item[8], curr_accum_item[9], curr_accum_item[10],
              curr_accum_item[11], curr_accum_item[12], curr_accum_item[13],
              curr_accum_item[14], curr_accum_item[15], curr_accum_item[16],
              curr_accum_item[17], curr_accum_item[18], curr_accum_item[19],
              curr_accum_item[20], curr_accum_item[21], curr_accum_item[22],
              curr_accum_item[23], curr_accum_item[24], curr_accum_item[25],
              curr_accum_item[26], curr_accum_item[27], curr_accum_item[28],
              curr_accum_item[29], curr_accum_item[30], curr_accum_item[31]);
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          for (int i = 0; i < WGMMA_FRGA_ELEMNTS_C / WGMMA_FRAG_ELEMENTS_PER_ITEM; ++i) {
            warpgroup_fence_operand(curr_accum_item[i]);
          }
        }
      }
    }
    stAccumGlobalC(globalC);
  }

  /// store functions
  /// store directly to global, bypassing shared memory
  DEVICE void stAccumGlobalC(CDType *globalC) {
    /// store one item at a time
    WGMMA_ACCUM_ITEM_TYPE *globalC_item =
        reinterpret_cast<WGMMA_ACCUM_ITEM_TYPE *>(globalC);
    for (int i = 0; i < WarpGroupRepeatsPerColC; ++i) {
      for (int j = 0; j < WarpGroupRepeatsPerRowC; ++j) {
        for (int r = 0; r < WGMMA_WARP_REPEATS_PER_COL; ++r) {
          for (int c = 0; c < WGMMA_WARP_REPEATS_PER_ROW; ++c) {
            int rowIdx = globalRowElementIdxC(i, r);
            int colIdx = globalColElementIdxC(
                j, c, 0); /// 0 for the start position of each item
            int idx = globalElementIdxC(rowIdx, colIdx);
            idx /= WGMMA_FRAG_ELEMENTS_PER_ITEM; /// get the item idx
            int accum_idx = (i * WarpGroupRepeatsPerRowC + j) * WGMMA_FRGA_ELEMNTS_C / WGMMA_FRAG_ELEMENTS_PER_ITEM + c * WGMMA_WARP_REPEATS_PER_COL + r;
                // i * WarpGroupRepeatsPerRowC * WGMMA_WARP_REPEATS_PER_COL *
                //     WGMMA_WARP_REPEATS_PER_ROW +
                // j * WGMMA_WARP_REPEATS_PER_COL * WGMMA_WARP_REPEATS_PER_ROW +
                // r * WGMMA_WARP_REPEATS_PER_ROW + c;
            // accum_idx /= WGMMA_FRAG_ELEMENTS_PER_ITEM; /// get the item idx
            globalC_item[idx] = accum_item_[accum_idx];
          }
        }
      }
    }
  }

  /// this is used to make sure the load and store are correct
  /// this function is hard-coded
  DEVICE void naive_gemm(ADType *globalA, BDType *globalB, CDType *globalC) {
    /// 8x4 thread organization
    /// each thread 8x16 outputs
    ADType fragA[8 * 8];
    BDType fragB[16 * 8];
    CDType accum[8 * 16] = {CDType(0.0)};
    /// 4 warps for 128x128 block
    /// each warp: 64x64
    /// each thread: 8x16
    int sharedRowIdx = warpId() / 2 * 64 + lane() / 4 * 8;
    int sharedColIdx = warpId() % 2 * 64 + lane() % 4 * 16;
    for (int ko = 0; ko < GlobalK / BLOCKK; ++ko) {
      ldGlobalSharedA(globalA, ko);
      ldGlobalSharedB(globalB, ko);
      async_copy_fence();
      async_copy_wait<0>();
      __syncthreads();
      /// have to load from shared to register
      /// KII = 8
      for (int ki = 0; ki < BLOCKK / 8; ++ki) {
        /// load frag
        for (int i = 0; i < 8; ++i) {
          for (int kii = 0; kii < 8; ++kii) {
            int SA_row = sharedRowIdx + i;
            int SA_col = ki * 8 + kii;
            fragA[i * 8 + kii] = SA_[sharedElementIdxA(SA_row, SA_col)];
          }
        }
        for (int j = 0; j < 16; ++j) {
          for (int kii = 0; kii < 8; ++kii) {
            int SB_row = ki * 8 + kii;
            int SB_col = sharedColIdx + j;
            fragB[j * 8 + kii] = SB_[sharedElementIdxB(SB_row, SB_col)];
          }
        }
        for (int kii = 0; kii < 8; ++kii) {
          for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 16; ++j) {
              accum[i * 16 + j] +=
                  (half)fragA[i * 8 + kii] * (half)fragB[j * 8 + kii];
            }
          }
        }
      }
    }
    /// store accum to global
    int globalRowIdx = blockY() * BLOCKM + sharedRowIdx;
    int globalColIdx = blockX() * BLOCKN + sharedColIdx;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 16; ++j) {
        globalC[(globalRowIdx + i) * GlobalN + (globalColIdx + j)] =
            accum[i * 16 + j];
      }
    }
  }

  DEVICE void run(ADType *globalA, BDType *globalB, CDType *globalC) {
    // naive_gemm(globalA, globalB, globalC);
    gemm_no_pipeline(globalA, globalB, globalC);
  }

  /// idex related calculations
  DEVICE int blockX() { return blockIdx.x; }

  DEVICE int blockY() { return blockIdx.y; }

  DEVICE int warpId() { return threadIdx.x / WARP_SIZE; }

  DEVICE int warpGroupId() { return warpId() / WARP_GROUP_SIZE; }

  DEVICE int lane() { return threadIdx.x % WARP_SIZE; }

  /// the No. of warp within one warp group
  DEVICE int warpIdInGroup() { return warpId() % WARP_GROUP_SIZE; }

  /// load A from global to shared swizzle
  /// global: row-major A load
  DEVICE int globalBlockRowElementIdxA() { return blockY() * BLOCKM; }

  DEVICE int globalBlockColElementIdxA(int ko) { return ko * BLOCKK; }

  DEVICE int globalWarpRowElementIdxA() {
    return warpId() / LoadWarpsPerRowA * LoadElementsPerColPerWarpA;
  }

  DEVICE int globalWarpColElementIdxA() {
    return warpId() % LoadWarpsPerRowA * LoadElementsPerRowPerWarpA;
  }

  DEVICE int globalLaneRowElementIdxA() {
    return lane() / LoadThreadsPerRowPerWarpA;
  }

  DEVICE int globalLaneColElementIdxA() {
    return lane() % LoadThreadsPerRowPerWarpA * LD_SMEM_VEC_ELEMS_A;
  }

  DEVICE int globalRowElementIdxA(int warpRepeatRow) {
    return globalBlockRowElementIdxA() + globalWarpRowElementIdxA() +
           warpRepeatRow * LoadElementsPerColPerWarpA * LoadWarpsPerColA +
           globalLaneRowElementIdxA();
  }

  DEVICE int globalColElementIdxA(int ko, int warpRepeatCol) {
    return globalBlockColElementIdxA(ko) + globalWarpColElementIdxA() +
           warpRepeatCol * LoadElementsPerRowPerWarpA * LoadWarpsPerRowA +
           globalLaneColElementIdxA();
  }

  DEVICE int globalElementIdxA(int globalRowIdx, int globalColIdx) {
    return globalRowIdx * GlobalK + globalColIdx;
  }

  /// shared: row-major A store
  DEVICE int sharedRowElementIdxA(int warpRepeatRow) {
    return globalWarpRowElementIdxA() +
           warpRepeatRow * LoadElementsPerColPerWarpA * LoadWarpsPerColA +
           globalLaneRowElementIdxA();
  }

  DEVICE int sharedColElementIdxA(int warpRepeatCol) {
    return globalWarpColElementIdxA() +
           warpRepeatCol * LoadElementsPerRowPerWarpA * LoadWarpsPerRowA +
           globalLaneColElementIdxA();
  }

  /// WGMMA layout for A
  DEVICE int sharedElementIdxA(int sharedRowIdx, int sharedColIdx) {
    return sharedRowIdx / WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_ROW * BLOCKK +
           sharedColIdx / WGMMA_SMEM_BLOCK_COL * WGMMA_SMEM_BLOCK_ROW *
               WGMMA_SMEM_BLOCK_COL +
           sharedRowIdx % WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_COL +
           sharedColIdx % WGMMA_SMEM_BLOCK_COL;
      // return sharedRowIdx * BLOCKK + sharedColIdx;
  }

  /// load B from global to shared swizzle
  /// global: col-major B load
  DEVICE int globalBlockColElementIdxB() { return blockX() * BLOCKN; }

  DEVICE int globalBlockRowElementIdxB(int ko) { return ko * BLOCKK; }

  DEVICE int globalWarpColElementIdxB() {
    return warpId() / LoadWarpsPerColB * LoadElementsPerRowPerWarpB;
  }

  DEVICE int globalWarpRowElementIdxB() {
    return warpId() % LoadWarpsPerColB * LoadElementsPerColPerWarpB;
  }

  DEVICE int globalLaneColElementIdxB() {
    return lane() / LoadThreadsPerColPerWarpB;
  }

  DEVICE int globalLaneRowElementIdxB() {
    return lane() % LoadThreadsPerColPerWarpB * LD_SMEM_VEC_ELEMS_B;
  }

  DEVICE int globalColElementIdxB(int warpRepeatCol) {
    return globalBlockColElementIdxB() + globalWarpColElementIdxB() +
           warpRepeatCol * LoadElementsPerRowPerWarpB * LoadWarpsPerRowB +
           globalLaneColElementIdxB();
  }

  DEVICE int globalRowElementIdxB(int ko, int warpRepeatRow) {
    return globalBlockRowElementIdxB(ko) + globalWarpRowElementIdxB() +
           warpRepeatRow * LoadElementsPerColPerWarpB * LoadWarpsPerColB +
           globalLaneRowElementIdxB();
  }

  DEVICE int globalElementIdxB(int globalRowIdx, int globalColIdx) {
    return globalRowIdx + globalColIdx * GlobalK;
  }

  /// shared: col-major B store
  DEVICE int sharedColElementIdxB(int warpRepeatCol) {
    return globalWarpColElementIdxB() +
           warpRepeatCol * LoadElementsPerRowPerWarpB * LoadWarpsPerRowB +
           globalLaneColElementIdxB();
  }

  DEVICE int sharedRowElementIdxB(int warpRepeatRow) {
    return globalWarpRowElementIdxB() +
           warpRepeatRow * LoadElementsPerColPerWarpB * LoadWarpsPerColB +
           globalLaneRowElementIdxB();
  }

  /// WGMMA layout for B
  DEVICE int sharedElementIdxB(int sharedRowIdx, int sharedColIdx) {
    return sharedRowIdx % WGMMA_SMEM_BLOCK_COL +
           sharedColIdx % WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_COL +
           sharedRowIdx / WGMMA_SMEM_BLOCK_COL * WGMMA_SMEM_BLOCK_ROW *
               WGMMA_SMEM_BLOCK_COL +
           sharedColIdx / WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_ROW * BLOCKK;
  }

  /// store C from register to global
  /// global: row-major C store
  DEVICE int globalBlockRowElementIdxC() { return blockY() * BLOCKM; }

  DEVICE int globalBlockColElementIdxC() { return blockX() * BLOCKN; }

  DEVICE int globalWarpGroupRowElementIdxC(int warpGroupRepeatRow) {
    return warpGroupId() / WarpGroupsPerRow * WarpGroupRepeatsPerColC * WGMMAM +
           warpGroupRepeatRow * WGMMAM;
  }

  DEVICE int globalWarpGroupColElementIdxC(int warpGroupRepeatCol) {
    return warpGroupId() % WarpGroupsPerRow * WarpGroupRepeatsPerRowC * WGMMAN +
           warpGroupRepeatCol * WGMMAN;
  }

  DEVICE int globalWarpRowElementIdxC(int warpRepeatRow) {
    return warpIdInGroup() / WGMMA_WARPS_PER_ROW * WGMMA_WARP_REPEATS_PER_COL *
               WGMMA_WARP_COL_ELEMENT_STRIDE +
           warpRepeatRow * WGMMA_WARP_COL_ELEMENT_STRIDE;
  }

  DEVICE int globalWarpColElementIdxC(int warpRepeatCol) {
    return warpIdInGroup() % WGMMA_WARPS_PER_ROW * WGMMA_WARP_REPEATS_PER_ROW *
               WGMMA_WARP_ROW_ELEMENT_STRIDE +
           warpRepeatCol * WGMMA_WARP_ROW_ELEMENT_STRIDE;
  }

  DEVICE int globalLaneRowElementIdxC() { return lane() / WGMMA_LANES_PER_ROW; }

  DEVICE int globalLaneColElementIdxC(int i) {
    return lane() % WGMMA_LANES_PER_ROW * WGMMA_FRAG_ELEMENTS_PER_ITEM + i;
  }

  DEVICE int globalRowElementIdxC(int warpGroupRepeatRow, int warpRepeatRow) {
    return globalBlockRowElementIdxC() +
           globalWarpGroupRowElementIdxC(warpGroupRepeatRow) +
           globalWarpRowElementIdxC(warpRepeatRow) + globalLaneRowElementIdxC();
  }

  DEVICE int globalColElementIdxC(int warpGroupRepeatCol, int warpRepeatCol,
                                  int i) {
    return globalBlockColElementIdxC() +
           globalWarpGroupColElementIdxC(warpGroupRepeatCol) +
           globalWarpColElementIdxC(warpRepeatCol) +
           globalLaneColElementIdxC(i);
  }

  DEVICE int globalElementIdxC(int globalRowIdx, int globalColIdx) {
    return globalRowIdx * GlobalN + globalColIdx;
  }

private:
  uint8_t *smem_;
  ADType *SA_;
  BDType *SB_;
  int GlobalM;
  int GlobalN;
  int GlobalK;
  CDType accum_[WarpGroupRepeatsPerRowC * WarpGroupRepeatsPerColC *
                WGMMA_FRGA_ELEMNTS_C] = {CDType(0.0)}; /// 1 * 2 * 64
  WGMMA_ACCUM_ITEM_TYPE *accum_item_;
};

__global__ void matmul_fp8(e5m2 *A, e5m2 *B, half *C, int M, int N, int K) {
  extern __shared__ uint8_t smem[];
  MatmulKernel kernel(M, N, K, smem);
  kernel.run(A, B, C);
}

// #define DEBUG
// #define PRINT
#ifdef DEBUG
#include <omp.h>
const int M = 256;
const int N = 256;
const int K = 256;
#else
const int M = 4096;
const int N = 4096;
const int K = 4096;
#endif
#define MAX(a, b) (a) > (b) ? (a) : (b)

// float alpha = 1.0;
// float beta = 0.0;

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int main(int argc, char *argv[]) {

  int runtime_version;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::cout << "CUDA Runtime Version: " << runtime_version << "\n";
  int driver_version;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  std::cout << "CUDA Driver Version: " << driver_version << "\n";

  if (argc > 1) {
    assert((argc - 1) % 2 == 0);
    for (int i = 1; i < argc; i += 2) {
      char *key = argv[i];
      char *value = argv[i + 1];
      std::string keys(key);
      if (keys == "stages") {
        STAGES = std::atoi(value);
        std::cout << "Setting to " << STAGES << " stages.\n";
      } else if (keys == "iters") {
        ITERS = std::atoi(value);
        std::cout << "Testing iters = " << ITERS << ".\n";
      }
    }
  }
#ifdef DEBUG
  std::cout << "Debugging using shape M=" << M << ", N=" << N << ", K=" << K
            << "\n";
#else
  std::cout << "Test performance using shape M=" << M << ", N=" << N
            << ", K=" << K << "\n";
#endif
  srand(time(NULL));
  e5m2 *hA = (e5m2 *)malloc(M * K);
  e5m2 *hB = (e5m2 *)malloc(K * N);
  half *hC = (half *)malloc(M * N * sizeof(half));
  half *golden = (half *)malloc(M * N * sizeof(half));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      hA[i * K + j] = (e5m2)(rand() % 1000 * 1 / 100 % 10 + 0.0);
    }
    for (int j = 0; j < N; ++j) {
      hC[i * N + j] = (float)(0);
      golden[i * N + j] = (float)(0);
    }
  }

  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      hB[n * K + k] = (e5m2)(rand() % 1000 * 1 / 100 % 10 + 0.0);
    }
  }

#ifdef DEBUG
  std::cout << "Computing golden values...\n";
// simple tiling to make it a bit faster
#pragma omp parallel for
  for (int i = 0; i < M; i += 64) {
#pragma omp parallel for
    for (int j = 0; j < N; j += 64) {
      half accum[64 * 64] = {0};
      for (int k = 0; k < K; k += 32) {
        for (int kk = 0; kk < 32; ++kk) {
          for (int jj = 0; jj < 64; ++jj) {
            for (int ii = 0; ii < 64; ++ii) {
              auto tmp = (float)accum[ii * 64 + jj] + ((float)hA[(i + ii) * K + k + kk] *
                                      (float)hB[(j + jj) * K + k + kk]);
              accum[ii * 64 + jj] = (half)tmp;
            }
          }
        }
      }
      for (int ii = 0; ii < 64; ++ii) {
        for (int jj = 0; jj < 64; ++jj) {
          for (int kk = 0; kk < 64; ++kk) {
            golden[(i + ii) * N + j + jj] = (half)accum[ii * 64 + jj];
          }
        }
      }
    }
  }
  std::cout << "Golden values done!\n";
#endif

  e5m2 *dA;
  e5m2 *dB;
  half *dC;

  CUDA_CHECK(cudaMalloc(&dA, M * K));
  CUDA_CHECK(cudaMalloc(&dB, K * N));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dA, hA, M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(half), cudaMemcpyHostToDevice));

  dim3 dimBlock(WARP_SIZE * WARP_NUMBER, 1, 1);
  dim3 dimGrid(N / BLOCKN, M / BLOCKM, 1);

#ifndef DEBUG
  int smem_size = MAX(STAGES * (BLOCKM + BLOCKN) * BLOCKK * sizeof(e5m2),
                      BLOCKM * BLOCKN * sizeof(half));
  if (smem_size >= (48 << 10)) {
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // warmup
  for (int i = 0; i < ITERS / 20 - 1; ++i) {
    matmul_fp8<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K);
  }
  cudaDeviceSynchronize();
  // auto start = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start);
  for (int i = 0; i < ITERS; ++i) {
    matmul_fp8<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "Running cost of CUDA kernel is " << double(ms) / ITERS
            << "ms\n";
  std::cout << "TFLOPS: "
            << (float)M * N * K * 2 / (double(ms) / ITERS) * 1e3 / 1e12 << "\n";
  // cudaDeviceSynchronize();
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -
  // start);

  // std::cout << "Running cost of CUDA kernel is " << duration.count() / 1e3 /
  // 200.0 << "ms\n"; std::cout << "TFLOPS: " << (float)M * N * K * 2 /
  // ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";
#endif

#ifdef DEBUG
  int smem_size = MAX(STAGES * (BLOCKM + BLOCKN) * BLOCKK * sizeof(e5m2),
                      BLOCKM * BLOCKN * sizeof(half));
  std::cout << "Using shared memory = " << (double)smem_size / 1e3 << " KB.\n";
  if (smem_size >= (48 << 10)) {
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  std::cout << "Computing result values...\n";
  matmul_fp8<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  std::cout << "Computing results done!\n";
  CUDA_CHECK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));

#ifdef PRINT
  std::cout << "Golden:" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << (float)golden[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Results:" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << (float)hC[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
#endif

  int errors = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float diff = ((float)golden[i * N + j] - (float)hC[i * N + j]);
      if (diff < 0) {
        diff = -diff;
      }
      float maxv = MAX((float)golden[i * N + j], (float)hC[i * N + j]);
      if (maxv < 0) {
        maxv = -maxv;
      }
      if (diff / maxv > 1e-1) {
        errors += 1;
      }
    }
  }

  if (errors) {
    std::cout << "Wrong Answer! " << errors << " errors.\n";
  } else {
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