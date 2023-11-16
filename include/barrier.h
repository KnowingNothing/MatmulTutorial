/***************************************************************************************************
 * Some code from barrier.h in Nvidia CUTLASS, the original copyright is:
 *
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once
#include "common.h"

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

struct ClusterBarrier {
  using ValueType = uint64_t;

 protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

 public:
  DEVICE ClusterBarrier() = delete;

  DEVICE void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  DEVICE uint32_t test_wait(uint32_t phase, uint32_t pred = true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  DEVICE uint32_t try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  DEVICE void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  DEVICE void arrive() const { ClusterBarrier::arrive(&this->barrier_); }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing
  // the arrive)
  DEVICE void arrive(uint32_t cta_id, uint32_t pred = true) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  //
  //  Static Versions
  //
  DEVICE static void init(ValueType const *smem_ptr, uint32_t arrive_count) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
  }

  // Static version of wait - in case we don't want to burn a register
  DEVICE static void wait(ValueType const *smem_ptr, uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
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

  DEVICE static uint32_t test_wait(ValueType const *smem_ptr, uint32_t phase,
                                   uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "setp.eq.u32 P2, %3, 1;\n\t"
        "@P2 mbarrier.test_wait.parity.shared.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred));

    return waitComplete;
  }

  DEVICE static uint32_t try_wait(ValueType const *smem_ptr, uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
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

  // Static Predicated version of the above - in case we know the address.
  DEVICE static void arrive(ValueType const *smem_ptr, uint32_t cta_id,
                            uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
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

  // Barrier arrive on local smem
  DEVICE static void arrive(ValueType const *smem_ptr) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    uint64_t state = 0;
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared.b64 %1, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr), "l"(state));
  }

  DEVICE static void invalidate(ValueType const *smem_ptr) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.ival.shared.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
  }
};

template <uint32_t Stages_>
struct PipelineState {
  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  DEVICE PipelineState() : index_{}, phase_{}, count_{} {}

  DEVICE PipelineState(int index, uint32_t phase, uint32_t count)
      : index_(index), phase_(phase), count_(count) {}

  DEVICE int index() const { return index_; }

  DEVICE uint32_t phase() const { return phase_; }

  DEVICE uint32_t count() const { return count_; }

  DEVICE void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  DEVICE PipelineState &operator=(const PipelineState &other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  DEVICE PipelineState advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) &&
          (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  DEVICE static PipelineState make_pipeline_state(PipelineState start_state,
                                                  uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template <int SequenceDepth, int SequenceLength>
class OrderedSequenceBarrier {
 public:
  using Barrier = ClusterBarrier;

  struct SharedStorage {
    Barrier barrier_[SequenceDepth][SequenceLength];
  };

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
    int active_warps = 0;
  };

 private:
  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier *barrier_ptr_;
  PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

 public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier &) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier &&) = delete;
  OrderedSequenceBarrier &operator=(const OrderedSequenceBarrier &) = delete;
  OrderedSequenceBarrier &operator=(OrderedSequenceBarrier &&) = delete;
  ~OrderedSequenceBarrier() = default;

  DEVICE OrderedSequenceBarrier(SharedStorage &storage, Params const &params)
      : params_(params),
        barrier_ptr_(&storage.barrier_[0][0]),
        // Group 0 - starts with an opposite phase
        stage_({0, (params.group_id == 0), 0}) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by the one elected thread of the block
    if (warp_idx == params.active_warps && lane_predicate == 1) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(params.group_size);
        }
      }
    }
    fence_barrier_init();
  }

  // Wait on a stage to be unlocked
  DEVICE void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  DEVICE void check_phase(int val) {
    if (threadIdx.x % WARP_GROUP_SIZE == 0) {
      printf("round %d group %d phase is %d\n", val,
             threadIdx.x / WARP_GROUP_SIZE, stage_.phase());
    }
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  DEVICE void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  DEVICE void advance() { ++stage_; }

 private:
  DEVICE Barrier &get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

// SM90 also introduces a new type of cluster-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct ClusterTransactionBarrier : public ClusterBarrier {
  DEVICE ClusterTransactionBarrier() = delete;

  // Performs an arrive operation + expected transaction bytes increment
  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_,
                                                    transaction_bytes);
  }

  // Performs an arrive operation + expected transaction bytes increment
  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes,
                                   uint32_t cta_id) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(
        &this->barrier_, transaction_bytes, cta_id, true);
  }

  // Performs an expected transaction bytes increment without doing an arrive
  // operation
  DEVICE void expect_transaction(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::expect_transaction(&this->barrier_,
                                                  transaction_bytes);
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  DEVICE void complete_transaction(uint32_t transaction_bytes,
                                   uint32_t pred = 1) const {
    uint32_t cta_rank = block_rank_in_cluster();
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, cta_rank,
                                                    transaction_bytes, pred);
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  DEVICE void complete_transaction(uint32_t dst_cta_id,
                                   uint32_t transaction_bytes,
                                   uint32_t pred) const {
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id,
                                                    transaction_bytes, pred);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + expected transaction bytes increment
  DEVICE static void arrive_and_expect_tx(ValueType const *smem_ptr,
                                          uint32_t transaction_bytes) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }

  // Performs an arrive operation + expected transaction bytes increment for a
  // remote cta_id in a Cluster
  DEVICE static void arrive_and_expect_tx(ValueType const *smem_ptr,
                                          uint32_t transaction_bytes,
                                          uint32_t cta_id, uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
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

  // Performs an expected transaction bytes increment without doing an arrive
  // operation
  DEVICE static void expect_transaction(ValueType const *smem_ptr,
                                        uint32_t transaction_bytes) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  DEVICE static void complete_transaction(ValueType const *smem_ptr,
                                          uint32_t dst_cta_id,
                                          uint32_t transaction_bytes,
                                          uint32_t pred = 1) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64   [%1], "
        "%0;"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
  }

  //
  // DEPRECATED APIs
  //
  [[deprecated("Use arrive_and_expect_tx instead")]] DEVICE void
  arrive_and_reset_bytes(uint32_t transaction_bytes) const {
    arrive_and_expect_tx(transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] DEVICE void
  arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
    arrive_and_expect_tx(transaction_bytes, cta_id);
  }
  [[deprecated("Use expect_transaction instead")]] DEVICE void reset_bytes(
      uint32_t transaction_bytes) const {
    expect_transaction(transaction_bytes);
  }
  [[deprecated("Use complete_transaction instead")]] DEVICE void commit(
      uint32_t transaction_bytes, uint32_t pred = 1) const {
    complete_transaction(transaction_bytes, pred);
  }
  [[deprecated("Use complete_transaction instead")]] DEVICE void commit(
      uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    complete_transaction(dst_cta_id, transaction_bytes, pred);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] DEVICE static void
  arrive_and_reset_bytes(ValueType const *smem_ptr,
                         uint32_t transaction_bytes) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] DEVICE static void
  arrive_and_reset_bytes(ValueType const *smem_ptr, uint32_t transaction_bytes,
                         uint32_t cta_id, uint32_t pred) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
  }
  [[deprecated("Use expect_transaction instead")]] DEVICE static void
  reset_bytes(ValueType const *smem_ptr, uint32_t transaction_bytes) {
    expect_transaction(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use complete_transaction instead")]] DEVICE static void commit(
      ValueType const *smem_ptr, uint32_t dst_cta_id,
      uint32_t transaction_bytes, uint32_t pred = 1) {
    complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
  }
};