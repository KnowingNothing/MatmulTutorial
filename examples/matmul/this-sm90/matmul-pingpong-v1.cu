#include "common.h"
#include "reference.h"

const int testM = 5120;
const int testN = 4096;
const int testK = 2048;
const int iters = 100;
static constexpr int CLUSTER_M = 2;
static constexpr int CLUSTER_N = 1;
static constexpr int WG_NUMBER = 3;
static constexpr int BLOCKM = 128;
static constexpr int BLOCKN = 128;
static constexpr int BLOCKK = 64;
static constexpr int STAGES = 7;
#define DEBUG 1
#ifdef DEBUG
#define PRINT_BT(BX, BY, TX, ...)                                    \
  {                                                                  \
    if (blockIdx.x == BX && blockIdx.y == BY && threadIdx.x == TX) { \
      printf(__VA_ARGS__);                                           \
    }                                                                \
  }
#else
#define PRINT_BT(BX, BY, TX, ...)
#endif
#ifdef DEBUG
#define PRINT_B(BX, BY, ...)                    \
  {                                             \
    if (blockIdx.x == BX && blockIdx.y == BY) { \
      printf(__VA_ARGS__);                      \
    }                                           \
  }
#else
#define PRINT_B(BX, BY, ...)
#endif

/// RUN:
/// nvcc -arch=sm_90a -I ../../../include -lcuda -std=c++17
/// matmul-pingpong-v1.cu -o test && ./test
/// |& tee trace.log

namespace utils {

using TmaDescriptor = CUtensorMap;

template <class T>
inline CUtensorMapDataType to_CUtensorMapDataType() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else
    //   if constexpr (std::is_same<T, float_e4m3_t>::value) { return
    //   CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else if constexpr (std::is_same<T,
    //   float_e5m2_t>::value) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
    if constexpr (std::is_same<T, uint16_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else if constexpr (std::is_same<T, uint32_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same<T, uint64_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_INT64;
    } else if constexpr (std::is_same<T, half_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same<T, float>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same<T, double>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    } else
    //   if constexpr (std::is_same<T,   bfloat16_t>::value) { return
    //   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; } else if constexpr (std::is_same<T,
    //   tfloat32_t>::value) { return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32; } else
    {
      static_assert(sizeof(T) < 0, "Unknown TMA Format!");
    }
}

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

template <int B, int M, int S>
HOST_DEVICE constexpr SmemSwizzleBits get_tma_swizzle_bits(Swizzle<B, M, S>) {
  if constexpr (M == 4) {
    switch (B) {
      default:
        static_assert(0 <= B && B <= 3,
                      "Expected B = 0,1,2, or 3 when M == 4. Unsupported "
                      "layout swizzle.");
      case 3:
        return SmemSwizzleBits::B128;
      case 2:
        return SmemSwizzleBits::B64;
      case 1:
        return SmemSwizzleBits::B32;
      case 0:
        return SmemSwizzleBits::DISABLE;
    }
  } else {
    static_assert(M < 0, "Unsupported layout swizzle.");
  }
}

inline CUtensorMapSwizzle to_CUtensorMapSwizzle(SmemSwizzleBits const& t) {
  switch (t) {
    default:
      assert(false && "Unknown SmemSwizzleBits!");
    case SmemSwizzleBits::DISABLE:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}

/// In this function, minor dimension moves faster than major dimension
template <int BlockMajorSize, int BlockMinorSize, int TmaDim, typename DType,
          int B, int M, int S>
TmaDescriptor make_tma_copy_desc(DType* gmem_ptr, int shape_major,
                                 int shape_minor,
                                 Swizzle<B, M, S> const& swizzle,
                                 uint32_t num_multicast) {
  void* gmem_address = (void*)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)shape_minor, (uint64_t)shape_major,
                                 1, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(DType), sizeof(DType) * shape_minor, 0,
                                  0, 0};

  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);
  assert(gmem_prob_shape[0] >= (uint64_t(1)));
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[1] >= (uint64_t(1)));
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[2] >= (uint64_t(1)));
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[3] >= (uint64_t(1)));
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[4] >= (uint64_t(1)));
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));

  assert(gmem_prob_stride[0] == sizeof(DType));
  assert(gmem_prob_stride[1] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[1] & 0b1111) == 0);
  assert(gmem_prob_stride[2] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[2] & 0b1111) == 0);
  assert(gmem_prob_stride[3] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[3] & 0b1111) == 0);
  assert(gmem_prob_stride[4] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[4] & 0b1111) == 0);

  assert(BlockMajorSize % num_multicast == 0);
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize / num_multicast), 1, 1,
                                1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  assert(smem_box_shape[0] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[0] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[1] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[2] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[3] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[4] <=
         (uint32_t(1) << 8));  // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));  // Stride must be max 2^3 = 8

  TmaDescriptor tma_desc = {0};

  CUtensorMapDataType tma_format =
      to_CUtensorMapDataType<typename std::remove_cv<DType>::type>();
  CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill tma_oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  CUtensorMapSwizzle smem_swizzle =
      to_CUtensorMapSwizzle(get_tma_swizzle_bits(swizzle));
  CUresult result = cuTensorMapEncodeTiled(
      &tma_desc, tma_format, TmaDim, gmem_address, gmem_prob_shape,
      gmem_prob_stride + 1, smem_box_shape, smem_box_stride, tma_interleave,
      smem_swizzle, tma_l2Promotion, tma_oobFill);

  if (result != CUDA_SUCCESS) {
    std::cerr << "TMA Desc Addr:   " << &tma_desc << "\nformat         "
              << tma_format << "\ndim            " << TmaDim
              << "\ngmem_address   " << gmem_address << "\nglobalDim      "
              << gmem_prob_shape << "\nglobalStrides  " << gmem_prob_stride
              << "\nboxDim         " << smem_box_shape << "\nelementStrides "
              << smem_box_stride << "\ninterleave     " << tma_interleave
              << "\nswizzle        " << smem_swizzle << "\nl2Promotion    "
              << tma_l2Promotion << "\noobFill        " << tma_oobFill
              << std::endl;
    std::cerr << "Error: Failed to initialize the TMA descriptor " << result
              << std::endl;
    assert(false);
  }

  return tma_desc;
}

HOST_DEVICE
void prefetch_tma_descriptor(TmaDescriptor const* desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state
  // space: const or param)
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

DEVICE void fence_barrier_init() {
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}" ::);
}

DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

template <uint32_t RegCount>
DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

DEVICE void tma_copy_2d(TmaDescriptor const* const desc_ptr,
                        uint64_t& smem_mbar, void const* const smem_ptr,
                        int32_t const& crd0, int32_t const& crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1)
      : "memory");
}

DEVICE void tma_copy_2d_multicast(TmaDescriptor const* const desc_ptr,
                                  uint64_t& smem_mbar, uint16_t multicast_mask,
                                  void const* const smem_ptr,
                                  int32_t const& crd0, int32_t const& crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster"
      " [%0], [%1, {%4, %5}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask), "r"(crd0), "r"(crd1)
      : "memory");
}

DEVICE void tma_copy_3d(TmaDescriptor const* const desc_ptr,
                        uint64_t& smem_mbar, void const* const smem_ptr,
                        int32_t const& crd0, int32_t const& crd1,
                        int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "r"(crd2)
      : "memory");
}

DEVICE void tma_copy_3d_multicast(TmaDescriptor const* const desc_ptr,
                                  uint64_t& smem_mbar, uint16_t multicast_mask,
                                  void const* const smem_ptr,
                                  int32_t const& crd0, int32_t const& crd1,
                                  int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster"
      " [%0], [%1, {%4, %5, %6}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask), "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
}

template <typename T>
DEVICE void cp_async(void* ptr, const T* gmem_ptr) {
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_ptr),
               "l"(gmem_ptr), "n"(16), "r"(16));
}

// GMMA 64x128x16 F32+=F16*F16
template <int ScaleA, int ScaleB, int ScaleD, int TransA, int TransB>
struct SM90_64x128x16_F32F16F16_SS {
  DEVICE static void wgmma(
      uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01,
      float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13,
      float& d14, float& d15, float& d16, float& d17, float& d18, float& d19,
      float& d20, float& d21, float& d22, float& d23, float& d24, float& d25,
      float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      float& d32, float& d33, float& d34, float& d35, float& d36, float& d37,
      float& d38, float& d39, float& d40, float& d41, float& d42, float& d43,
      float& d44, float& d45, float& d46, float& d47, float& d48, float& d49,
      float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
      float& d56, float& d57, float& d58, float& d59, float& d60, float& d61,
      float& d62, float& d63) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " p,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05),
          "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17),
          "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
          "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
          "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41),
          "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
          "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53),
          "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
          "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }
};

DEVICE void warpgroup_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg)::"memory");
}

DEVICE
void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

DEVICE
void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
DEVICE void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

union GmmaDescriptor {
  HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor const& t) noexcept
      : desc_(t.desc_) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor&& t) noexcept
      : desc_(t.desc_) {}

  HOST_DEVICE constexpr GmmaDescriptor& operator=(
      GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr GmmaDescriptor& operator=(GmmaDescriptor&& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4;  // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;  // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }
};

/// make shared memory descriptor
template <class PointerType>
DEVICE GmmaDescriptor make_smem_desc(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ =
      0x1;  /// swizzle 128B because we use Swizzle<3,4,3>
  desc.bitfield.leading_byte_offset_ = 0x1;  /// no use
  desc.bitfield.stride_byte_offset_ =
      64;  /// how many 128bits-rows needed between two core matrices
  desc.bitfield.base_offset_ = 0x0;
  return desc;
}

}  // namespace utils

// Cluster-wide barrier. CUDA barrier doesn't support cluster scope. Have to
// follow CUTLASS. CUDA doesn't support barrier because cluster-wide barrier
// arrive can't return phase token. So CUTLASS doesn't use phase token as return
// value. But wait still need the phase token.
struct Barrier {
  uint64_t barrier_;
  DEVICE Barrier() = delete;

  DEVICE void init(uint32_t arrive_count) const {
    uint64_t const* smem_ptr = &barrier_;
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
  }

  // local arrive
  DEVICE void arrive() const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
  }

  // remote arrive
  DEVICE void arrive(uint32_t cta_id, uint32_t pred = true) const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred));
  }

  DEVICE void wait(uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));
  }

  DEVICE uint32_t try_wait(uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return waitComplete;
  }

  DEVICE void invalidate() {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.ival.shared.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
  }

  // These are TMA related barrier methods.
  // CULTASS implements it in another barrier.
  // We put them together.
  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }

  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id,
                                   uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], "
        "%3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
  }

  DEVICE void expect_transaction(uint32_t transaction_bytes) const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }
};

enum class BarrierStatus : uint32_t {
  WaitAgain = 0u,
  WaitDone = 1u,
  WaitOnly = 2u
};

struct ArrivalToken {
  HOST_DEVICE ArrivalToken(BarrierStatus barrier_status)
      : barrier_status(barrier_status) {}

  HOST_DEVICE ArrivalToken() = delete;

  HOST_DEVICE BarrierStatus get() const { return barrier_status; }

  HOST_DEVICE bool operator==(ArrivalToken const& other) const {
    return barrier_status == other.get();
  }

  HOST_DEVICE bool operator!=(ArrivalToken const& other) const {
    return !(*this == other);
  }

  BarrierStatus barrier_status;
};

struct ProducerToken : public ArrivalToken {};

struct ConsumerToken : public ArrivalToken {};

template <int Stages>
struct PipelineState {
  int index = 0;
  uint32_t phase = 0;
  uint32_t count = 0;

  DEVICE PipelineState() : index(), phase(), count() {}

  DEVICE PipelineState(int index, uint32_t phase, uint32_t count)
      : index(index), phase(phase), count(count) {}

  DEVICE void operator++() {
    if constexpr (Stages > 0) {
      ++index;
      ++count;
      if (index == Stages) {
        index = 0;
        phase ^= 1;
      }
    }
  }

  DEVICE PipelineState advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      if ((num_iterations < Stages) && (index + num_iterations) >= Stages) {
        phase ^= 1;
      }
      if ((num_iterations >= Stages) &&
          (((index + num_iterations) / Stages) % 2) == 1) {
        phase ^= 1;
      }
      index = (index + num_iterations) % Stages;
      count += num_iterations;
    }
    return *this;
  }

  DEVICE static PipelineState make_pipeline_state(PipelineState start_state,
                                                  uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template <int Depth, int Length>
struct OrderedBarrierSharedStorage {
  Barrier barrier[Depth][Length];
};

template <int Depth, int Length>
struct OrderedBarrierParams {
  uint32_t group_id;
  uint32_t group_size;
};

template <int Depth, int Length>
struct OrderedBarrier {
  OrderedBarrierParams<Depth, Length> params;
  Barrier* barrier_ptr;
  PipelineState<Depth> state;

  DEVICE OrderedBarrier() = delete;

  DEVICE OrderedBarrier(OrderedBarrierSharedStorage<Depth, Length>& storage,
                        OrderedBarrierParams<Depth, Length>& params)
      : params(params),
        barrier_ptr(&storage.barrier[0][0]),
        state({0, params.group_id == 0, 0}) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Depth; ++i) {
        for (int j = 0; j < Length; ++j) {
          barrier_ptr[i * Length + j].init(params.group_size);
        }
      }
    }
    utils::fence_barrier_init();
  }

  DEVICE void wait() {
    get_barrier_for_current_stage(params.group_id).wait(state.phase);
  }

  // This will the next slot's barrier. Gurantee -> order.
  DEVICE void arrive() {
    int signaling_id = (params.group_id + 1) % Length;
    get_barrier_for_current_stage(signaling_id).arrive();
    ++state;
  }

  DEVICE void advance() { ++state; }

  DEVICE Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr[state.index * Length + group_id];
  }
};

template <int Stages>
DEVICE PipelineState<Stages> make_producer_start_state() {
  // start from the next phase, so that the barrier wait doesn't block
  // execution.
  return {0, 1, 0};
}

template <int Stages>
struct TmaPipelineSharedStorage {
  Barrier full_barrier[Stages];
  Barrier empty_barrier[Stages];
};

template <int Stages>
struct TmaPipelineParams {
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  uint32_t transaction_bytes = 0;
  ThreadCategory role = ThreadCategory::NonParticipant;
  uint32_t is_leader = 0;
  uint32_t num_consumers = 0;
};

// TmaPipeline structure. Follow CUTLASS impl.
template <int Stages, int ClusterM, int ClusterN>
struct TmaPipeline {
  uint32_t dst_blockid = 0;
  uint32_t is_signaling_thread = 0;
  Barrier* full_barrier_ptr = nullptr;
  Barrier* empty_barrier_ptr = nullptr;
  TmaPipelineParams<Stages> params;

  DEVICE TmaPipeline(TmaPipelineSharedStorage<Stages>& storage,
                     TmaPipelineParams<Stages> p)
      : full_barrier_ptr(&storage.full_barrier[0]),
        empty_barrier_ptr(&storage.empty_barrier[0]),
        params(p) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();

    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr[i].init(1);
      }
      // Question: why num_consumers = WARP_GROUP_SIZE?
      uint32_t const num_consumer_warpgroups_per_cluster =
          params.num_consumers / WARP_GROUP_SIZE;
      // Question: why this? I guess it's the same row and col.
      uint32_t const multicast_consumer_arrival_count =
          (ClusterM + ClusterN - 1) * num_consumer_warpgroups_per_cluster;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr[i].init(multicast_consumer_arrival_count);
      }
    }
    utils::fence_barrier_init();

    // CUTLASS says the following logic is used to equally spread the duty of
    // SYNCS Empty Arriveal to 128 threads.
    dim3 block_id = block_id_in_cluster();
    static constexpr uint32_t cluster_size = ClusterM * ClusterN;
    static_assert(cluster_size <= MAX_CLUSTER_SIZE, "Cluster size too large!");
    if (params.num_consumers % WARP_GROUP_SIZE == 0) {
      int thread_idx = threadIdx.x % WARP_GROUP_SIZE;
      is_signaling_thread =
          (thread_idx % (WARP_GROUP_SIZE / MAX_CLUSTER_SIZE)) == 0;
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      auto swizzle = Swizzle<2, 0, -2>{};
      dst_blockid = swizzle(thread_row * 4 + thread_col);
    } else if (params.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signaling_thread = (thread_idx % (32 / MAX_CLUSTER_SIZE)) == 0;
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid = thread_row * 4 + thread_col;
    } else {
      is_signaling_thread = 0;
      // Should not arrive there.
      assert(false);
    }

    is_signaling_thread &= dst_blockid < cluster_size;
    is_signaling_thread &= is_same_row_or_col(dst_blockid, block_id);
  }

  DEVICE bool is_same_row_or_col(int dst_block_id, dim3 block_id) {
    return ((dst_block_id % ClusterM) == block_id.x) ||
           ((dst_block_id / ClusterM) == block_id.y);
  }

  DEVICE void producer_acquire(PipelineState<Stages> state,
                               ProducerToken barrier_token = {
                                   BarrierStatus::WaitAgain}) {
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr[state.index].wait(state.phase);
    }
    if (barrier_token == BarrierStatus::WaitOnly) {
      return;
    }

    if (params.is_leader) {
      full_barrier_ptr[state.index].arrive_and_expect_tx(
          params.transaction_bytes);
    }
  }

  DEVICE void producer_tail(PipelineState<Stages> state) {
    for (int i = 0; i < Stages; ++i) {
      producer_acquire(state, {BarrierStatus::WaitOnly});
      ++state;
    }
  }

  DEVICE uint64_t* producer_get_barrier(PipelineState<Stages> state) {
    return reinterpret_cast<uint64_t*>(&full_barrier_ptr[state.index]);
  }

  DEVICE ConsumerToken consumer_try_wait(PipelineState<Stages> state,
                                         uint32_t skip_wait = false) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status =
        full_barrier_ptr[state.index].try_wait(state.phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  DEVICE void consumer_wait(PipelineState<Stages> state) {
    full_barrier_ptr[state.index].wait(state.phase);
  }

  DEVICE void consumer_wait(PipelineState<Stages> state,
                            ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr[state.index].wait(state.phase);
    }
  }

  DEVICE void consumer_release(PipelineState<Stages> state,
                               uint32_t skip = false) {
    empty_barrier_ptr[state.index].arrive(dst_blockid,
                                          is_signaling_thread & (!skip));
  }
};

template <int Stages>
struct EpilogueLoadPipelineSharedStorage {
  Barrier full_barrier[Stages];
  Barrier empty_barrier[Stages];
};

template <int Stages>
struct EpilgoueLoadPipelineParams {
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  ThreadCategory role = ThreadCategory::NonParticipant;
  uint32_t transaction_bytes = 0;
  uint32_t producer_arv_count = 1;
  uint32_t consumer_arv_count = 1;
  uint32_t dst_blockid = block_rank_in_cluster();
};

// This seems not necessary in this example, so I didn't finish it.
template <int Stages>
struct EpilogueLoadPipeline {
  Barrier* full_barrier_ptr = nullptr;
  Barrier* empty_barrier_ptr = nullptr;
  EpilgoueLoadPipelineParams<Stages> params;

  DEVICE EpilogueLoadPipeline(
      EpilogueLoadPipelineSharedStorage<Stages>& storage,
      EpilgoueLoadPipelineParams<Stages>& params)
      : full_barrier_ptr(&storage.full_barrier[0]),
        empty_barrier_ptr(&storage.empty_barrier[0]),
        params(params) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();

    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr[i].init(params.producer_arv_count);
        empty_barrier_ptr[i].init(params.consumer_arv_count);
      }
    }
    utils::fence_barrier_init();
  }

  DEVICE void producer_acquire(PipelineState<Stages> state,
                               ProducerToken barrier_token = {
                                   BarrierStatus::WaitAgain}) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr[state.index].wait(state.phase);
    }
  }

  DEVICE void producer_expect_transaction(PipelineState<Stages> state) {
    full_barrier_ptr[state.index].expect_transaction(params.transaction_bytes);
  }
};

template <typename AType, typename BType, typename AccumType, int MTile,
          int NTile, int KTile>
struct WgMMA;

template <int MTile, int NTile, int KTile>
struct WgMMA<half_t, half_t, float, MTile, NTile, KTile> {
  static constexpr int elements_per_thread = 2;
  static constexpr int threads_per_row = 4;
  static constexpr int threads_per_col = WARP_SIZE / threads_per_row;
  static constexpr int warp_elements_per_row =
      elements_per_thread * threads_per_row;
  static constexpr int warp_repeats_per_row = NTile / warp_elements_per_row;
  static constexpr int warp_repeats_per_col = 2;
  static constexpr int num_warps = WARP_NUMBER_IN_WARP_GROUP;
  static constexpr int WGMMA_M = 64;
  static constexpr int WGMMA_K = 16;
  static constexpr int num_warp_groups_m = MTile / WGMMA_M;
  static constexpr int k_iter = KTile / WGMMA_K;

  static constexpr int num_elements_accumulators =
      num_warp_groups_m * warp_repeats_per_row * warp_repeats_per_col *
      elements_per_thread;

  DEVICE WgMMA() {
    // PRINT_BT(0, 0, 0, "Use WgMMA: MTile=%d, NTile=%d, KTile=%d\n", MTile,
    // NTile,
    //          KTile);
    // PRINT_BT(0, 0, 0, "warp_repeats_per_row=%d\n", warp_repeats_per_row);
    // PRINT_BT(0, 0, 0, "warp_repeats_per_col=%d\n", warp_repeats_per_col);
    // PRINT_BT(0, 0, 0, "elements_per_thread=%d\n", elements_per_thread);
  }

  DEVICE static void get_m_n_idx_fragment(int& m, int& n, int thread_id,
                                          int k_wgmma, int row_id, int col_id,
                                          int item_id) {
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    m = k_wgmma * WGMMA_M + warp_id * threads_per_col * warp_repeats_per_col +
        row_id * threads_per_col + lane_id / threads_per_row;
    n = col_id * warp_elements_per_row +
        lane_id % threads_per_row * elements_per_thread + item_id;
  }

  DEVICE static void get_4d_idx_from_linear(int& k_wgmma, int& row_id,
                                            int& col_id, int& item_id,
                                            int linear_id) {
    item_id = linear_id % elements_per_thread;
    row_id = linear_id / elements_per_thread % warp_repeats_per_col;
    col_id = linear_id / elements_per_thread / warp_repeats_per_col %
             warp_repeats_per_row;
    k_wgmma = linear_id / (num_elements_accumulators / num_warp_groups_m);
  }

  template <int ScaleA, int ScaleB, int ScaleD, int TransA, int TransB>
  DEVICE static void wgmma(half_t* smem_A, half_t* smem_B, float* accum) {
    float* accum_ = accum;
    {
      int k = 0;
      accum = accum_;
      auto desc_b = utils::make_smem_desc(smem_B + k * WGMMA_K);
      for (int m = 0; m < num_warp_groups_m; ++m) {
        accum = accum + m * (num_elements_accumulators / num_warp_groups_m);
        auto desc_a =
            utils::make_smem_desc(smem_A + k * WGMMA_K + m * WGMMA_M * KTile);
        // the first is ScaleD
        utils::SM90_64x128x16_F32F16F16_SS<
            ScaleA, ScaleB, ScaleD, TransA,
            TransB>::wgmma(desc_a, desc_b, accum[0], accum[1], accum[2],
                           accum[3], accum[4], accum[5], accum[6], accum[7],
                           accum[8], accum[9], accum[10], accum[11], accum[12],
                           accum[13], accum[14], accum[15], accum[16],
                           accum[17], accum[18], accum[19], accum[20],
                           accum[21], accum[22], accum[23], accum[24],
                           accum[25], accum[26], accum[27], accum[28],
                           accum[29], accum[30], accum[31], accum[32],
                           accum[33], accum[34], accum[35], accum[36],
                           accum[37], accum[38], accum[39], accum[40],
                           accum[41], accum[42], accum[43], accum[44],
                           accum[45], accum[46], accum[47], accum[48],
                           accum[49], accum[50], accum[51], accum[52],
                           accum[53], accum[54], accum[55], accum[56],
                           accum[57], accum[58], accum[59], accum[60],
                           accum[61], accum[62], accum[63]);
      }
    }
    for (int k = 1; k < k_iter; ++k) {
      accum = accum_;
      auto desc_b = utils::make_smem_desc(smem_B + k * WGMMA_K);
      for (int m = 0; m < num_warp_groups_m; ++m) {
        auto desc_a =
            utils::make_smem_desc(smem_A + k * WGMMA_K + m * WGMMA_M * KTile);
        // the remaining must be ScaleD = 1
        utils::SM90_64x128x16_F32F16F16_SS<ScaleA, ScaleB, 1, TransA, TransB>::
            wgmma(desc_a, desc_b, accum[0], accum[1], accum[2], accum[3],
                  accum[4], accum[5], accum[6], accum[7], accum[8], accum[9],
                  accum[10], accum[11], accum[12], accum[13], accum[14],
                  accum[15], accum[16], accum[17], accum[18], accum[19],
                  accum[20], accum[21], accum[22], accum[23], accum[24],
                  accum[25], accum[26], accum[27], accum[28], accum[29],
                  accum[30], accum[31], accum[32], accum[33], accum[34],
                  accum[35], accum[36], accum[37], accum[38], accum[39],
                  accum[40], accum[41], accum[42], accum[43], accum[44],
                  accum[45], accum[46], accum[47], accum[48], accum[49],
                  accum[50], accum[51], accum[52], accum[53], accum[54],
                  accum[55], accum[56], accum[57], accum[58], accum[59],
                  accum[60], accum[61], accum[62], accum[63]);
        accum = accum + (num_elements_accumulators / num_warp_groups_m);
      }
    }
  }
};

// A simpilified tile scheduler that always takes AlongN and non swizzle
template <int BlockM, int BlockN, int ClusterM, int ClusterN>
struct TileScheduler {
  int linear_idx;
  int m_blocks;
  int n_blocks;

  struct WorkInfo {
    int m_idx;
    int n_idx;
    bool valid;
  };

  DEVICE TileScheduler(int M, int N) { init(M, N); }

  DEVICE void init(int M, int N) {
    linear_idx = blockIdx.x + blockIdx.y * gridDim.x;
    get_blocks_m_n(M, N);
  }

  DEVICE WorkInfo get_current_work_info() {
    int m_idx, n_idx;
    get_current_m_n_idx(m_idx, n_idx, m_blocks, n_blocks);
    return {m_idx, n_idx, linear_idx < m_blocks * n_blocks};
  }

  DEVICE void advance(int number = 1) {
    linear_idx += number * gridDim.x * gridDim.y;
  }

  DEVICE void get_current_m_n_idx(int& m_idx, int& n_idx, int m_blocks,
                                  int n_blocks) {
    int div_cluster_x = linear_idx / ClusterM;
    int mod_cluster_x = linear_idx % ClusterM;
    int div_cluster_xy = div_cluster_x / ClusterN;
    int mod_cluster_xy = div_cluster_x % ClusterN;
    int clusters_per_row = n_blocks / ClusterN;
    int cluster_row = div_cluster_xy / clusters_per_row;
    int cluster_col = div_cluster_xy % clusters_per_row;
    m_idx = cluster_row * ClusterM + mod_cluster_x;
    n_idx = cluster_col * ClusterN + mod_cluster_xy;
  }

  DEVICE void get_blocks_m_n(int M, int N) {
    m_blocks = ((M + BlockM - 1) / BlockM + ClusterM - 1) / ClusterM * ClusterM;
    n_blocks = ((N + BlockN - 1) / BlockN + ClusterN - 1) / ClusterN * ClusterN;
  }
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct MainloopSharedStorage {
  alignas(128) AType smem_A[BlockM * BlockK * Stages];
  alignas(128) BType smem_B[BlockN * BlockK * Stages];
  TmaPipelineSharedStorage<Stages> pipeline;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct MainloopParams {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct Mainloop {
  static_assert(std::is_same<AType, BType>::value);
  static constexpr uint32_t TmaTransactionBytes =
      BlockM * BlockK * sizeof(AType) + BlockN * BlockK * sizeof(BType);

  DEVICE static void prefetch_tma_descriptor(
      const utils::TmaDescriptor* tensormap_a,
      const utils::TmaDescriptor* tensormap_b) {
    utils::prefetch_tma_descriptor(tensormap_a);
    utils::prefetch_tma_descriptor(tensormap_b);
  }

  DEVICE void load(const utils::TmaDescriptor& tensormap_a,
                   const utils::TmaDescriptor& tensormap_b,
                   TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
                   PipelineState<Stages> mainloop_pipeline_state, int m_idx,
                   int n_idx, int k_tile_count, uint32_t block_rank_in_cluster,
                   MainloopSharedStorage<AType, BType, CType, AccumType, BlockM,
                                         BlockN, BlockK, ClusterM, ClusterN,
                                         Stages>& shared_storage) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = elect_one_sync();

    if (warp_idx_in_warp_group == 0 && lane_predicate == 1) {
      int block_id_x_in_cluster = block_rank_in_cluster % ClusterM;
      int block_id_y_in_cluster = block_rank_in_cluster / ClusterM;
      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;
      constexpr int multicast_stride_a = BlockM / ClusterN;
      constexpr int multicast_stride_b = BlockN / ClusterM;
      if constexpr (ClusterM > 1) {
        // multicast B
        for (int i = 0; i < ClusterM; ++i) {
          mcast_mask_b |=
              (uint16_t(1) << (block_id_y_in_cluster * ClusterM + i));
        }
      }
      if constexpr (ClusterN > 1) {
        // multicast A
        for (int i = 0; i < ClusterN; ++i) {
          mcast_mask_a |=
              (uint16_t(1) << (block_id_x_in_cluster + i * ClusterN));
        }
      }

      // PRINT_B(1, 0, "mcast_mask_b=%d, stride_b=%d, block_id_x=%d\n",
      // mcast_mask_b, multicast_stride_b, block_id_x_in_cluster);

      for (int i = 0; i < k_tile_count; ++i) {
        // PRINT_B(0, 0, "producer acuquire stage %d phase %d\n",
        // mainloop_pipeline_state.index, mainloop_pipeline_state.phase);
        mainloop_pipeline.producer_acquire(mainloop_pipeline_state);
        // PRINT_B(0, 0, "get!");

        int stage = mainloop_pipeline_state.index;
        AType* smem_ptr_A = (shared_storage.smem_A + stage * BlockM * BlockK);
        BType* smem_ptr_B = (shared_storage.smem_B + stage * BlockN * BlockK);

        // load A and B using the same barrier
        if constexpr (ClusterN > 1) {
          // multicast copy A
          utils::tma_copy_3d_multicast(
              &tensormap_a,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              mcast_mask_a,
              reinterpret_cast<void*>(smem_ptr_A + block_id_y_in_cluster *
                                                       multicast_stride_a *
                                                       BlockK),
              i * BlockK,  // innermost dim moves fastest
              m_idx * BlockM + block_id_y_in_cluster * multicast_stride_a, 0);
        } else {
          // normal copy A
          utils::tma_copy_3d(
              &tensormap_a,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              smem_ptr_A, i * BlockK, m_idx * BlockM, 0);
        }

        if constexpr (ClusterM > 1) {
          // multicast copy B
          utils::tma_copy_3d_multicast(
              &tensormap_b,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              mcast_mask_b,
              reinterpret_cast<void*>(smem_ptr_B + block_id_x_in_cluster *
                                                       multicast_stride_b *
                                                       BlockK),
              i * BlockK,  // innermost dim moves fastest
              n_idx * BlockN + block_id_x_in_cluster * multicast_stride_b, 0);
          // PRINT_B(0, 0, "crd0=%d, crd1=%d\n", i * BlockK, n_idx * BlockN +
          // block_id_x_in_cluster * multicast_stride_b);
        } else {
          // normal copy B
          utils::tma_copy_3d(
              &tensormap_b,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              smem_ptr_B, i * BlockK, n_idx * BlockN, 0);
        }

        // PRINT_B(0, 0, "TMA load at stage %d issued\n",
        //         mainloop_pipeline_state.index);
        // PRINT_B(0, 0, "TMA load bytes %d\n",
        // mainloop_pipeline.params.transaction_bytes);

        // this moves to next stage, but doesn't affect the outer state
        // because this state is passed by copy, not reference.
        ++mainloop_pipeline_state;
      }
    }
  }

  DEVICE void load_tail(
      TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
      PipelineState<Stages> mainloop_pipeline_state) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = elect_one_sync();

    if (warp_idx_in_warp_group == 0 && lane_predicate == 1) {
      mainloop_pipeline.producer_tail(mainloop_pipeline_state);
    }
  }

  template <typename WGMMA>
  DEVICE void mma(TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
                  PipelineState<Stages> mainloop_pipeline_state, WGMMA wgmma,
                  AccumType* accum, int k_tile_count,
                  MainloopSharedStorage<AType, BType, CType, AccumType, BlockM,
                                        BlockN, BlockK, ClusterM, ClusterN,
                                        Stages>& shared_tensors) {
    PipelineState<Stages> mainloop_pipeline_state_release =
        mainloop_pipeline_state;

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }

    // PRINT_BT(0, 0, 128, "hi\n");

    auto barrier_token =
        mainloop_pipeline.consumer_try_wait(mainloop_pipeline_state);
    mainloop_pipeline.consumer_wait(mainloop_pipeline_state, barrier_token);

    // PRINT_BT(0, 0, 128, "here\n");

    int read_stage = mainloop_pipeline_state.index;
    utils::warpgroup_arrive();

    WGMMA::wgmma<1, 1, 0, 0, 0>(
        shared_tensors.smem_A +
            read_stage * BlockM * BlockK,  // half_t* smem_A,
        shared_tensors.smem_B +
            read_stage * BlockN * BlockK,  // half_t* smem_B,
        accum                              // float* accum,
    );

    utils::warpgroup_commit_batch();
    // move to the next stage
    ++mainloop_pipeline_state;

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }

    // PRINT_BT(0, 0, 128, "1\n");

    // PRINT_BT(0, 0, 0, "issue the first wgmma at stage %d\n", read_stage);

    // start from 1 because the first wgmma was done
    for (; k_tile_count > 1; --k_tile_count) {
      // PRINT_BT(0, 0, 128, "work\n");
      auto barrier_token =
          mainloop_pipeline.consumer_try_wait(mainloop_pipeline_state);
      mainloop_pipeline.consumer_wait(mainloop_pipeline_state, barrier_token);

      int read_stage = mainloop_pipeline_state.index;
      for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
        utils::warpgroup_fence_operand(accum[i]);
      }
      utils::warpgroup_arrive();

      WGMMA::wgmma<1, 1, 1, 0, 0>(
          shared_tensors.smem_A +
              read_stage * BlockM * BlockK,  // half_t* smem_A,
          shared_tensors.smem_B +
              read_stage * BlockN * BlockK,  // half_t* smem_B,
          accum                              // float* accum,
      );

      utils::warpgroup_commit_batch();
      utils::warpgroup_wait<1>();
      for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
        utils::warpgroup_fence_operand(accum[i]);
      }

      // PRINT_BT(0, 0, 128, "issue wgmma at stage %d\n", read_stage);

      mainloop_pipeline.consumer_release(mainloop_pipeline_state_release);

      // PRINT_BT(0, 0, 128, "release pipeline at stage %d\n",
      // mainloop_pipeline_state_release.index);

      ++mainloop_pipeline_state;
      ++mainloop_pipeline_state_release;
    }

    // PRINT_BT(0, 0, 128, "2\n");

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }
  }

  DEVICE void mma_tail(
      TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
      PipelineState<Stages> mainloop_pipeline_state, int k_tile_count) {
    mainloop_pipeline_state.advance(k_tile_count - 1);
    utils::warpgroup_wait<0>();

    mainloop_pipeline.consumer_release(mainloop_pipeline_state);
    ++mainloop_pipeline_state;
  }
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct EpilogueSharedStorage {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct EpilogueParams {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct Epilogue {
  DEVICE static void prefetch_tma_descriptor(
      [[maybe_unused]] const utils::TmaDescriptor* tensormap_a,
      [[maybe_unused]] const utils::TmaDescriptor* tensormap_b) {}

  template <typename WGMMA>
  DEVICE void store(CType* dst, WGMMA wgmma, AccumType* accum, int m_idx,
                    int n_idx, int M, int N) {
    // this store is specialized for WgMMA M64NnK16
    int m = m_idx * BlockM;
    int n = n_idx * BlockN;

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      AccumType value = accum[i];
      int m_frag, n_frag, k_wgmma, row_id, col_id, item_id;
      WGMMA::get_4d_idx_from_linear(k_wgmma, row_id, col_id, item_id, i);
      WGMMA::get_m_n_idx_fragment(m_frag, n_frag, threadIdx.x % WARP_GROUP_SIZE,
                                  k_wgmma, row_id, col_id, item_id);
      if ((m + m_frag < M) && (n + n_frag < N)) {
        dst[(m + m_frag) * N + (n + n_frag)] = (CType)value;
      }
    }
  }
};

using LoadWarpOrderBarrier = OrderedBarrier<1, 2>;
using LoadWarpOrderBarrierSharedStorage = OrderedBarrierSharedStorage<1, 2>;
using LoadWarpOrderBarrierParams = OrderedBarrierParams<1, 2>;

using MathWarpGroupOrderBarrier = OrderedBarrier<2, 2>;
using MathWarpGroupOrderBarrierSharedStorage =
    OrderedBarrierSharedStorage<2, 2>;
using MathWarpGroupOrderBarrierParams = OrderedBarrierParams<2, 2>;

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct KernelSharedStorage {
  alignas(128)
      MainloopSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN,
                            BlockK, ClusterM, ClusterN, Stages> mainloop;
  // epilogue: no shared storage
  alignas(16) MathWarpGroupOrderBarrierSharedStorage math_wg_order;
  alignas(16) LoadWarpOrderBarrierSharedStorage load_order;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct GemmKernelParams {
  GemmParams<AType, BType, CType, AccumType> gemm_params;
  MainloopParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                 ClusterM, ClusterN, Stages>
      mainloop_params;
  EpilogueParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                 ClusterM, ClusterN, Stages>
      epilogue_params;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
__global__ void gpu_gemm_kernel(
    // GemmParams<AType, BType, CType, AccumType> gemm_params,
    GemmKernelParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                     ClusterM, ClusterN, Stages>
        kernel_params,
    const __grid_constant__ utils::TmaDescriptor tensormap_a,
    const __grid_constant__ utils::TmaDescriptor tensormap_b) {
  // we follow CUTLASS warp specialization
  enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };

  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    Warp3 = 3
  };

  extern __shared__ uint8_t raw_shared_mem[];
  // this is CUTLASS manner shared storage cast
  KernelSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                      ClusterM, ClusterN, Stages>& shared_storage =
      *reinterpret_cast<
          KernelSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN,
                              BlockK, ClusterM, ClusterN, Stages>*>(
          raw_shared_mem);

  // get useful ids:
  // int thread_idx = threadIdx.x;
  // int lane_idx = threadIdx.x % WARP_SIZE;
  int warp_idx = threadIdx.x / WARP_SIZE;
  int warp_idx_in_warp_group =
      threadIdx.x / WARP_SIZE % WARP_NUMBER_IN_WARP_GROUP;
  int warp_group_thread_idx = threadIdx.x % WARP_GROUP_SIZE;
  uint32_t block_idx_in_cluster = block_rank_in_cluster();

  // get roles
  auto warp_group_role = WarpGroupRole(threadIdx.x / WARP_GROUP_SIZE);
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
  int lane_predicate = elect_one_sync();

  // only the first thread in a block launch tma prefetch
  if ((warp_idx == 0) && lane_predicate) {
    Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
             ClusterN, Stages>::prefetch_tma_descriptor(&tensormap_a,
                                                        &tensormap_b);
    Epilogue<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
             ClusterN, Stages>::prefetch_tma_descriptor(&tensormap_a,
                                                        &tensormap_b);
  }

  // PRINT_BT(0, 0, 0, "TMA prefetch issued\n");

  // mainloop pipeline
  TmaPipelineParams<Stages> mainloop_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    mainloop_pipeline_params.role =
        TmaPipelineParams<Stages>::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    mainloop_pipeline_params.role =
        TmaPipelineParams<Stages>::ThreadCategory::Consumer;
  }
  mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
  mainloop_pipeline_params.num_consumers = WARP_GROUP_SIZE;
  mainloop_pipeline_params.transaction_bytes =
      Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
               ClusterN, Stages>::TmaTransactionBytes;
  TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline(
      shared_storage.mainloop.pipeline, mainloop_pipeline_params);

  // epilogue pipeline: load and store
  // seems not necessary in this example.

  // barriers used to control warpgroups
  LoadWarpOrderBarrierParams load_order_params;
  load_order_params.group_id =
      producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
  load_order_params.group_size = WARP_SIZE;
  LoadWarpOrderBarrier load_order(shared_storage.load_order, load_order_params);

  MathWarpGroupOrderBarrierParams math_wg_order_params;
  math_wg_order_params.group_id = threadIdx.x / WARP_GROUP_SIZE - 1;
  math_wg_order_params.group_size = WARP_GROUP_SIZE;
  MathWarpGroupOrderBarrier math_wg_order(shared_storage.math_wg_order,
                                          math_wg_order_params);

  PipelineState<Stages> mainloop_pipeline_consumer_state;
  PipelineState<Stages> mainloop_pipeline_producer_state =
      make_producer_start_state<Stages>();

  Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
           ClusterN, Stages>
      mainloop;

  WgMMA<AType, BType, AccumType, BlockM, BlockN, BlockK> wgmma;
  Epilogue<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
           ClusterN, Stages>
      epilogue;

  auto cluster_wait_fn = [&]() {
    if constexpr (ClusterM * ClusterN > 1) {
      utils::cluster_arrive_relaxed();
      return []() { utils::cluster_wait(); };
    } else {
      __syncthreads();
      return []() {};
    }
  }();

  TileScheduler<BlockM, BlockN, ClusterM, ClusterN> scheduler(
      kernel_params.gemm_params.M, kernel_params.gemm_params.N);
  int k_tile_count = (kernel_params.gemm_params.K + BlockK - 1) / BlockK;

  if (warp_group_role == WarpGroupRole::Consumer1) {
    scheduler.advance();
    mainloop_pipeline_consumer_state.advance(k_tile_count);
  }

  auto work_tile_info = scheduler.get_current_work_info();

  cluster_wait_fn();

  // PRINT_BT(0, 0, 0, "Ready to enter producer-consuemr loop\n");

  if (warp_group_role == WarpGroupRole::Producer) {
    // you can't only dealloc without alloc in consumer!
    // Don't know why the magic number 40
    utils::warpgroup_reg_dealloc<40>();

    if (producer_warp_role == ProducerWarpRole::Mainloop) {
      bool first_arrive = true;
      while (work_tile_info.valid) {
        // PRINT_BT(0, 0, 0, "current m=%d, n=%d\n", work_tile_info.m_idx,
        // work_tile_info.n_idx);
        mainloop.load(tensormap_a, tensormap_b, mainloop_pipeline,
                      mainloop_pipeline_producer_state, work_tile_info.m_idx,
                      work_tile_info.n_idx, k_tile_count, block_idx_in_cluster,
                      shared_storage.mainloop);

        // PRINT_BT(0, 0, 0, "done!\n");

        mainloop_pipeline_producer_state.advance(k_tile_count);
        if (first_arrive) {
          load_order.arrive();
          first_arrive = false;
        }
        scheduler.advance();
        work_tile_info = scheduler.get_current_work_info();
      }

      mainloop.load_tail(mainloop_pipeline, mainloop_pipeline_producer_state);
    } else if (producer_warp_role == ProducerWarpRole::Epilogue && false) {
      // no need to do epilogue load, so in this example the epilogue producer
      // is empty
      load_order.wait();
    }

  } else if (warp_group_role == WarpGroupRole::Consumer0 ||
             warp_group_role == WarpGroupRole::Consumer1) {
    // you can't only alloc without dealloc in producer!
    // Don't know why the magic number 232
    utils::warpgroup_reg_alloc<232>();

    while (work_tile_info.valid) {
      // PRINT_BT(0, 0, 128, "current m=%d, n=%d\n", work_tile_info.m_idx,
      // work_tile_info.n_idx);
      AccumType accumulators[WgMMA<AType, BType, AccumType, BlockM, BlockN,
                                   BlockK>::num_elements_accumulators];

      // consuemr 0 doens't block at the first wait
      math_wg_order.wait();

      mainloop.mma(mainloop_pipeline, mainloop_pipeline_consumer_state, wgmma,
                   accumulators, k_tile_count, shared_storage.mainloop);

      // PRINT_BT(0, 0, 128, "done mma!\n");

      math_wg_order.arrive();

      mainloop.mma_tail(mainloop_pipeline, mainloop_pipeline_consumer_state,
                        k_tile_count);
      mainloop_pipeline_consumer_state.advance(k_tile_count * 2);

      // PRINT_BT(0, 0, 128, "done mma_tail!\n");

      math_wg_order.wait();

      epilogue.store(kernel_params.gemm_params.C, wgmma, accumulators,
                     work_tile_info.m_idx, work_tile_info.n_idx,
                     kernel_params.gemm_params.M, kernel_params.gemm_params.N);

      math_wg_order.arrive();

      // do nothing for epilogue store tail

      // PRINT_BT(0, 0, 128, "done!\n");

      scheduler.advance(2);
      work_tile_info = scheduler.get_current_work_info();
    }
  }
}

template <class AType, class BType, class CType, class AccumType>
void gpu_gemm(GemmParams<AType, BType, CType, AccumType> gemm_params,
              bool verbose = true) {
  int sm_number = get_sm_count();
  dim3 grid(CLUSTER_M * CLUSTER_N, sm_number / (CLUSTER_M * CLUSTER_N), 1);
  dim3 block(WARP_GROUP_SIZE * WG_NUMBER, 1, 1);
  dim3 cluster(CLUSTER_M, CLUSTER_N, 1);
  if (verbose) {
    std::cout << "sm_number: " << sm_number << "\n";
  }
  auto* Kernel = gpu_gemm_kernel<AType, BType, CType, AccumType, BLOCKM, BLOCKN,
                                 BLOCKK, CLUSTER_M, CLUSTER_N, STAGES>;
  size_t smemSizeBytes =
      sizeof(KernelSharedStorage<AType, BType, CType, AccumType, BLOCKM, BLOCKN,
                                 BLOCKK, CLUSTER_M, CLUSTER_N, STAGES>);
  if (smemSizeBytes >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSizeBytes);
    CUDA_CHECK(result);
  }
  if (verbose) {
    std::cout << "Launching kernel with grid " << grid.x << " " << grid.y << " "
              << grid.z << " and block " << block.x << " " << block.y << " "
              << block.z << " and cluster " << cluster.x << " " << cluster.y
              << " " << cluster.z << " and smem " << smemSizeBytes
              << " bytes\n";
  }
  void const* kernel = (void const*)Kernel;

  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(status);

  // tensor_map
  utils::TmaDescriptor tensormap_a =
      utils::make_tma_copy_desc<BLOCKM, BLOCKK, 3>(
          gemm_params.A, gemm_params.M, gemm_params.K, Swizzle<3, 4, 3>{},
          CLUSTER_N);
  utils::TmaDescriptor tensormap_b =
      utils::make_tma_copy_desc<BLOCKN, BLOCKK, 3>(
          gemm_params.B, gemm_params.N, gemm_params.K, Swizzle<3, 4, 3>{},
          CLUSTER_M);
  MainloopParams<AType, BType, CType, AccumType, BLOCKM, BLOCKN, BLOCKK,
                 CLUSTER_M, CLUSTER_N, STAGES>
      mainloop_params{};
  /// Prepare kernel params
  GemmKernelParams<AType, BType, CType, AccumType, BLOCKM, BLOCKN, BLOCKK,
                   CLUSTER_M, CLUSTER_N, STAGES>
      params{gemm_params, mainloop_params, {}};

  void* kernel_params[] = {&params, &tensormap_a, &tensormap_b};
  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = {grid.x, grid.y, grid.z};
  launch_config.blockDim = {block.x, block.y, block.z};
  launch_config.dynamicSmemBytes = size_t(smemSizeBytes);
  launch_config.stream = nullptr;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;

  status = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
  cudaError_t launch_result = cudaGetLastError();
  CUDA_CHECK(launch_result);
}

int main(int argc, char** argv) {
  int M = testM;
  int N = testN;
  int K = testK;
  using AType = half_t;
  using BType = half_t;
  using CType = half_t;
  using AccumType = float;
  AccumType alpha = 1.0;
  AccumType beta = 0.0;

  std::vector<int> AShape = {M, K};
  std::vector<int> BShape = {N, K};
  std::vector<int> CShape = {M, N};
  auto hA = alloc_cpu_tensor<AType>(AShape);
  random_fill(hA, AShape);
  // constant_fill(hA, AShape, (AType)1.0);
  auto hB = alloc_cpu_tensor<BType>(BShape);
  random_fill(hB, BShape);
  // constant_fill(hB, BShape, (BType)1.0);
  auto hC = alloc_cpu_tensor<CType>(CShape);
  random_fill(hC, CShape);
  // constant_fill(hC, CShape, (CType)(-13.0));
  auto goldenC = alloc_cpu_tensor<CType>(CShape);
  random_fill(goldenC, CShape);
  auto dA = alloc_gpu_tensor<AType>(AShape);
  auto dB = alloc_gpu_tensor<BType>(BShape);
  auto dgC = alloc_gpu_tensor<CType>(CShape);
  auto dC = alloc_gpu_tensor<CType>(CShape);

  /// timers
  CPUTimer cpu_timer;
  GPUTimer gpu_timer;

  /// copy data
  std::cout << "Copying data from CPU to GPU...\n";
  cpu_timer.tick();
  copy_to_gpu(hA, dA, AShape);
  copy_to_gpu(hB, dB, BShape);
  copy_to_gpu(hC, dC, CShape);
  copy_to_gpu(goldenC, dgC, CShape);
  cpu_timer.tick();
  std::cout << "Copy data done! Use " << cpu_timer.report_last_ms() << " ms.\n";

  /// compute gpu reference
  std::cout << "Computing gpu reference values...\n";
  GemmParams gpu_params(M, N, K, dA, dB, dgC, alpha, beta);
  gpu_timer.sync_all();
  gpu_timer.tick();
  reference_gpu_gemm(gpu_params);
  gpu_timer.tick();
  gpu_timer.sync_all();
  std::cout << "GPU reference done! Use " << gpu_timer.report_last_ms()
            << " ms.\n";

  /// copy results
  std::cout << "Copying results...\n";
  copy_to_cpu(goldenC, dgC, CShape);
  std::cout << "Copying results done!\n";

  /// compute gpu kernel
  std::cout << "Computing gpu kernel values...\n";
  GemmParams gpu_kernel_params(M, N, K, dA, dB, dC, alpha, beta);
  gpu_gemm(gpu_kernel_params);
  std::cout << "GPU kernel done!\n";

  /// copy results

  std::cout << "Copying results...\n";
  copy_to_cpu(hC, dC, CShape);
  std::cout << "Copying results done!\n";

  /// compare results
  assert_allclose(hC, goldenC, CShape, /*rtol=*/1e-3, /*dump=*/false);
  std::cout << "Correct!\n";

  /// profile
  std::cout << "Profile performance...\n";
  gpu_timer.sync_all();
  gpu_timer.tick();
  for (int i = 0; i < iters; ++i) {
    gpu_gemm(gpu_params, /*verbose=*/false);
  }
  gpu_timer.tick();
  gpu_timer.sync_all();
  float latency = gpu_timer.report_last_ms() / float(iters);
  std::cout << "Profile done! Average latency is " << latency << " ms.\n";
  std::cout << "TFLOPS: "
            << ((double)M * (double)N * (double)K * 2.0) / (latency / 1000.0) /
                   1e12
            << "\n";

  free_cpu_tensor(hA);
  free_cpu_tensor(hB);
  free_cpu_tensor(hC);
  free_cpu_tensor(goldenC);
  free_gpu_tensor(dA);
  free_gpu_tensor(dB);
  free_gpu_tensor(dC);
  return 0;
}