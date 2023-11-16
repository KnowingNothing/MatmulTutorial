/***************************************************************************************************
 * Some code from barrier.h in Nvidia CUTLASS, the original copyright is:
 *
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once
#include "barrier.h"

enum class BarrierStatus : uint32_t { WaitAgain = 0u, WaitDone = 1u };

class ArrivalToken {
 public:
  HOST_DEVICE ArrivalToken(BarrierStatus barrier_status)
      : barrier_status_(barrier_status) {}

  HOST_DEVICE ArrivalToken() = delete;

  HOST_DEVICE BarrierStatus get() const {
    return barrier_status_;
    ;
  }

  HOST_DEVICE bool operator==(ArrivalToken const& other) const {
    return barrier_status_ == other.get();
  }

 private:
  BarrierStatus barrier_status_;

  HOST_DEVICE friend bool operator==(const ArrivalToken& left,
                                             const BarrierStatus& right) {
    return left.get() == right;
  }

  HOST_DEVICE friend bool operator==(const BarrierStatus& left,
                                             const ArrivalToken& right) {
    return left == right.get();
  }
};

class ProducerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

class ConsumerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA load (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is visible Cluster-wide (as it needs a
// Cluster-Sync) We have exactly one thread elected in the Producer as the
// "leader" Currently, it is optional to elect a leader for the Consumers
template <int Stages_, int ClusterX_, int ClusterY_, int ClusterZ_>
class PipelineTmaAsync {
 public:
  using FullBarrier = ClusterTransactionBarrier;
  using EmptyBarrier = ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = PipelineState<Stages>;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
    int active_warps = 0;
  };

  // Constructor
  DEVICE PipelineTmaAsync(SharedStorage& storage, Params params)
      : params_(params),
        full_barrier_ptr_(&storage.full_barrier_[0]),
        empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    int warp_idx = canonical_warp_idx();
    int lane_predicate = elect_one_sync();
    if (warp_idx == params.active_warps && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(1);
      }
      uint32_t const num_consumer_warpgroups_per_cluster =
          params_.num_consumers / WARP_GROUP_SIZE;
      uint32_t const multicast_consumer_arrival_count =
          (ClusterX_ + ClusterY_ - 1) * num_consumer_warpgroups_per_cluster;
      // Barrier EMPTY init
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[i].init(multicast_consumer_arrival_count);
      }
    }
    // Logic to optimally schedule Empty Arrives
    // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group
    // (128 threads)
    dim3 block_id = block_id_in_cluster();
    auto cluster_size = ClusterX_ * ClusterY_ * ClusterZ_;
    static_assert(cluster_size <= MAX_CLUSTER_SIZE,
                  "ERROR : Cluster size too large !");

    // STEP 1 : Use Cute Layout function to generate an optimal dst block-id
    // (0-15)
    if (params_.num_consumers % WARP_GROUP_SIZE == 0) {
      int thread_idx = threadIdx.x % WARP_GROUP_SIZE;
      is_signalling_thread_ =
          (thread_idx % (WARP_GROUP_SIZE / MAX_CLUSTER_SIZE)) == 0;
      auto swizzle = Swizzle<2, 0, -2>{};
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      dst_blockid_ = swizzle(thread_row * 4 + thread_col);
    } else if (params_.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signalling_thread_ = (thread_idx % (32 / MAX_CLUSTER_SIZE)) == 0;
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid_ = thread_row * 4 + thread_col;
    } else {
      is_signalling_thread_ = 0;
    }

    // STEP 2: Find if this dst block-id needs an arrival for this problem
    is_signalling_thread_ &= dst_blockid_ < cluster_size;
    is_signalling_thread_ &= is_same_row_or_col(
        dst_blockid_, block_id, ClusterX_, ClusterY_, ClusterZ_);

    fence_barrier_init();
  }

  DEVICE bool is_same_row_or_col(int dst_block_id, dim3 block_id,
                                     int cluster_x, int cluster_y,
                                     int cluster_z) {
    return (((dst_block_id % cluster_x) == block_id.x) ||
            (((dst_block_id / cluster_x) == block_id.y)));
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the
  // barrier to flip. It opportunistically waits for an implementation-dependent
  // timeout. Whether or not the barrier has flipped yet, the try function will
  // return a token. If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  DEVICE ProducerToken producer_try_acquire(PipelineState state,
                                                uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  DEVICE void producer_acquire(PipelineState state,
                                   ProducerToken barrier_token = {
                                       BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  DEVICE void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  DEVICE void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  DEVICE ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  DEVICE ConsumerToken consumer_try_wait(PipelineState state,
                                             uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  DEVICE void consumer_wait(PipelineState state) {
    consumer_wait(state.index(), state.phase());
  }

  DEVICE void consumer_wait(PipelineState state,
                                ConsumerToken barrier_token) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  DEVICE void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

 private:
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier* full_barrier_ptr_ = nullptr;
  EmptyBarrier* empty_barrier_ptr_ = nullptr;
  Params params_;

  DEVICE ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase,
                                                uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  DEVICE void producer_acquire(uint32_t stage, uint32_t phase,
                                   ProducerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
  }

  // NOP for TMA based mainloop
  DEVICE void producer_commit(uint32_t stage, uint32_t bytes) {}

  DEVICE ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase,
                                             uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  DEVICE void consumer_wait(uint32_t stage, uint32_t phase) {
    full_barrier_ptr_[stage].wait(phase);
  }

  // Wait for producer to commit transactions (done by TMA)
  DEVICE void consumer_wait(uint32_t stage, uint32_t phase,
                                ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  DEVICE void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(dst_blockid_,
                                     is_signalling_thread_ & (!skip));
  }

  DEVICE ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
};