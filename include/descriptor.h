/***************************************************************************************************
 * Some code from mma_sm90_desc.hpp, copy_sm90_desc.hpp
 *  in Nvidia CUTLASS, the original copyright is:
 *
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "common.h"

enum class LayoutType : uint8_t {
  INTERLEAVE = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

union GmmaDescriptor
{

  HOST_DEVICE constexpr
  GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  HOST_DEVICE constexpr
  GmmaDescriptor(GmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr
  GmmaDescriptor& operator=(GmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType, int MorN, int K>
DEVICE GmmaDescriptor make_smem_desc_major_k(PointerType smem_ptr, LayoutType layout_type) {
    static_assert(MorN % 8 == 0);
    static_assert(K == 2);
    GmmaDescriptor desc;
    uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = static_cast<uint8_t>(layout_type);          /// no swizzle
    desc.bitfield.leading_byte_offset_ = 8; /// 8 bytes
    desc.bitfield.stride_byte_offset_ = 16; /// 16 bytes
    desc.bitfield.base_offset_ = 0;
    return desc;
}

using TmaDescriptor = CUtensorMap;

HOST_DEVICE
void
prefetch_tma_descriptor(TmaDescriptor const* desc_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}