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

/// RUN: nvcc -arch=sm_90a -std=c++17 test_wgmma_f8.cu -o test_wgmma_f8 && ./test_wgmma_f8 |& tee trace.log

typedef __nv_fp8_e5m2 e5m2;
// typedef int8_t e5m2;

__device__ void MMA(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t &d00,
    uint32_t &d01, uint32_t &d02, uint32_t &d03, uint32_t &d04,
    uint32_t &d05, uint32_t &d06, uint32_t &d07, uint32_t &d08,
    uint32_t &d09, uint32_t &d10, uint32_t &d11, uint32_t &d12,
    uint32_t &d13, uint32_t &d14, uint32_t &d15, uint32_t &d16,
    uint32_t &d17, uint32_t &d18, uint32_t &d19, uint32_t &d20,
    uint32_t &d21, uint32_t &d22, uint32_t &d23, uint32_t &d24,
    uint32_t &d25, uint32_t &d26, uint32_t &d27, uint32_t &d28,
    uint32_t &d29, uint32_t &d30, uint32_t &d31)
{
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

// __device__ void MMA(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t &d00,
//     uint32_t &d01, uint32_t &d02, uint32_t &d03, uint32_t &d04,
//     uint32_t &d05, uint32_t &d06, uint32_t &d07, uint32_t &d08,
//     uint32_t &d09, uint32_t &d10, uint32_t &d11, uint32_t &d12,
//     uint32_t &d13, uint32_t &d14, uint32_t &d15)
// {
//     int scale_D = 1; /// use D=A*B+C format
//     constexpr int32_t scaleA = 1;
//     constexpr int32_t scaleB = 1;
//     asm volatile("{\n"
//     ".reg .pred p;\n"
//     "setp.ne.b32 p, %18, 0;\n"
//     "wgmma.mma_async.sync.aligned.m64n64k32.f16.e5m2.e5m2 "
//     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
//     " %8,  %9,  %10, %11, %12, %13, %14, %15}, "
//     " %16,"
//     " %17,"
//     " p,   %19, %20;\n"
//     "}\n"
//     : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03), "+r"(d04),
//         "+r"(d05), "+r"(d06), "+r"(d07), "+r"(d08), "+r"(d09),
//         "+r"(d10), "+r"(d11), "+r"(d12), "+r"(d13), "+r"(d14),
//         "+r"(d15)
//     : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)),
//         "n"(int32_t(scaleA)), "n"(int32_t(scaleB)));
// }

__device__ void warpgroup_fence_operand(uint32_t &reg) {
    asm volatile("" : "+r"(reg)::"memory");
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

union GmmaDescriptor {

    __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}
    __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept
        : desc_(t.desc_) {}
    __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept
        : desc_(t.desc_) {}
  
    __device__ constexpr GmmaDescriptor &
    operator=(GmmaDescriptor const &t) noexcept {
      desc_ = t.desc_;
      return *this;
    }
  
    __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
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
    __device__ constexpr operator uint64_t() const noexcept { return desc_; }
  
    // Printer
    //   __device__ friend void print(GmmaDescriptor const& t)
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
  

template <class PointerType>
__device__ GmmaDescriptor make_desc_a(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = 0;          /// no swizzle
  desc.bitfield.leading_byte_offset_ = 8; /// 16 bytes
  desc.bitfield.stride_byte_offset_ = 16;   /// 8 bytes
  /// base_offset_ is not valid for non-swizzle
  desc.bitfield.base_offset_ = 0;
  return desc;
}

template <class PointerType>
__device__ GmmaDescriptor make_desc_b(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = 0;          /// no swizzle
  desc.bitfield.leading_byte_offset_ = 8; /// 16 bytes
  desc.bitfield.stride_byte_offset_ = 16;   /// 8 bytes
  /// base_offset_ is not valid for non-swizzle
  desc.bitfield.base_offset_ = 0; //(uint_ptr >> 7) & 7;
  return desc;
}

__global__ void test_wgmma_fp8(half* C, half* gold_C) {
    auto C_u32 = reinterpret_cast<uint32_t*>(C);
    __shared__ e5m2 A[64*32];
    __shared__ e5m2 B[128*32];
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lid = tid % 32;

    curandState state;
    curand_init(tid, tid, 0, &state);
    for (int i = 0; i < 16; ++i) {
        int r = tid / 2;
        int c = tid % 2 * 16 + i;
        // A[r * 32 + c] = (e5m2)(float)(r);
        float value = curand_uniform(&state);
        A[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(value);
        // A[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(r / 8 == 7 && c / 16 == 1 ? 2 : 0);
        // A[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(r / 8 * 2 + c / 16 == 31);
        // A[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(r / 8 == 7 && c / 16 == 1 ? 1 : 0);
    }
    for (int i = 0; i < 32; ++i) {
        int r = tid;
        int c = i;
        float value = curand_uniform(&state);
        B[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(value);
        // B[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(1);
        // B[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(r / 8 == 7);
        // B[r / 8 * 8 * 32 + c / 16 * 8 * 16 + r % 8 * 16 + c % 16] = (e5m2)(float)(r / 8 == 7 && c / 16 == 0 && (r % 8 + 1 == c % 16));
    }
    __syncthreads();
    GmmaDescriptor desc_a = make_desc_a(A);
    GmmaDescriptor desc_b = make_desc_b(B);
    uint32_t accum_u32[32];
    for (int i = 0; i < 32; ++i) {
        accum_u32[i] = 0;
    }
    for (int i = 0; i < 32; ++i) {
        // accum_u32[i] = 0;
        warpgroup_fence_operand(accum_u32[i]);
    }
    warpgroup_arrive();
    MMA(desc_a, desc_b, accum_u32[0], accum_u32[1],
        accum_u32[2], accum_u32[3], accum_u32[4],
        accum_u32[5], accum_u32[6], accum_u32[7],
        accum_u32[8], accum_u32[9], accum_u32[10],
        accum_u32[11], accum_u32[12], accum_u32[13],
        accum_u32[14], accum_u32[15], accum_u32[16],
        accum_u32[17], accum_u32[18], accum_u32[19],
        accum_u32[20], accum_u32[21], accum_u32[22],
        accum_u32[23], accum_u32[24], accum_u32[25],
        accum_u32[26], accum_u32[27], accum_u32[28],
        accum_u32[29], accum_u32[30], accum_u32[31]);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    for (int i = 0; i < 32; ++i) {
        warpgroup_fence_operand(accum_u32[i]);
    }
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 16; ++c) {
            // C_u32[(wid * 16 + r * 8 + lid / 4) * 64 + c * 4 + lid % 4] = accum_u32[r * 16 + c];
            C_u32[(wid * 16 + r * 8 + lid / 4) * 64 + c * 4 + lid % 4] = accum_u32[c * 2 + r];
        }
    }

    half gaccum[64] = {0.0};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int r = wid / 2 * 32 + lid / 8 * 8 + i;
            int c = wid % 2 * 64 + lid % 8 * 8 + j;
            for (int k = 0; k < 32; ++k) {
                auto valueA = A[r / 8 * 8 * 32 + k / 16 * 8 * 16 + r % 8 * 16 + k % 16];
                auto valueB = B[c / 8 * 8 * 32 + k / 16 * 8 * 16 + c % 8 * 16 + k % 16];
                gaccum[i * 8 + j] += (half)((float)valueA * (float)valueB);
            }
            gold_C[r * 128 + c] = gaccum[i * 8 + j];
        }
    }
}

int main() {
    dim3 grid(1);
    dim3 thread(128);

    half* hC = (half*)malloc(64*128*2);
    half* dC;
    cudaMalloc((void**)&dC, 64*128*2);
    half* hgC = (half*)malloc(64*128*2);
    half* dgC;
    cudaMalloc((void**)&dgC, 64*128*2);
    test_wgmma_fp8<<<grid, thread>>>(dC, dgC);
    cudaMemcpy(hC, dC, 64*128*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hgC, dgC, 64*128*2, cudaMemcpyDeviceToHost);

    std::cout << "Results:\n";
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 128; ++j) {
            std::cout << (float)hC[i * 128 + j] << " ";
        }
        std::cout << "\n";
    }

    int errors = 0;
    std::cout << "Gold:\n";
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 128; ++j) {
            if (std::abs((float)hC[i * 128 +j] - (float)hgC[i * 128 + j]) > 1e-1) {
                errors += 1;
            }
            std::cout << (float)hgC[i * 128 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Errors= " << errors << "\n";
    return 0;
}