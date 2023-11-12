#pragma once
#include "common.h"

__device__
    uint32_t
    cast_smem_ptr_to_uint(void const *const ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %rx;\n"
        ".reg .pred %px;\n"
        "     elect.sync %rx|%px, %2;\n"
        "@%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ void fence_barrier_init()
{
    asm volatile(
        "{\n\t"
        "fence.mbarrier_init.release.cluster; \n"
        "}" ::);
}

__device__ void cluster_arrive_relaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

__device__ void cluster_wait()
{
    asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

struct ClusterBarrier
{

    using ValueType = uint64_t;

protected:
    // Can never be initialized - can only be aliased to smem
    ValueType barrier_;

public:
    __device__
    ClusterBarrier() = delete;

    __device__ void init(uint32_t arrive_count) const
    {
        ClusterBarrier::init(&this->barrier_, arrive_count);
    }

    __device__
        uint32_t
        test_wait(uint32_t phase, uint32_t pred = true) const
    {
        return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
    }

    __device__
        uint32_t
        try_wait(uint32_t phase) const
    {
        return ClusterBarrier::try_wait(&this->barrier_, phase);
    }

    __device__ void wait(uint32_t phase) const
    {
        ClusterBarrier::wait(&this->barrier_, phase);
    }

    // Barrier arrive on local smem
    __device__ void arrive() const
    {
        ClusterBarrier::arrive(&this->barrier_);
    }

    // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
    __device__ void arrive(uint32_t cta_id, uint32_t pred = true) const
    {
        ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
    }

    //
    //  Static Versions
    //
    __device__ static void init(ValueType const *smem_ptr, uint32_t arrive_count)
    {
        uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "{\n\t"
            "mbarrier.init.shared.b64 [%1], %0; \n"
            "}"
            :
            : "r"(arrive_count), "r"(smem_addr));
    }

    // Static version of wait - in case we don't want to burn a register
    __device__ static void wait(ValueType const *smem_ptr, uint32_t phase)
    {
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

    __device__ static uint32_t test_wait(ValueType const *smem_ptr, uint32_t phase, uint32_t pred)
    {
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

    __device__ static uint32_t try_wait(ValueType const *smem_ptr, uint32_t phase)
    {
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
    __device__ static void arrive(ValueType const *smem_ptr, uint32_t cta_id, uint32_t pred)
    {
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
    __device__ static void arrive(ValueType const *smem_ptr)
    {
        uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        uint64_t state = 0;
        asm volatile(
            "{\n\t"
            "mbarrier.arrive.shared.b64 %1, [%0];\n\t"
            "}"
            :
            : "r"(smem_addr), "l"(state));
    }

    __device__ static void invalidate(ValueType const *smem_ptr)
    {
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
struct PipelineState
{

    static constexpr uint32_t Stages = Stages_;

    int index_ = 0;
    uint32_t phase_ = 0;
    uint32_t count_ = 0;

    __device__
    PipelineState() : index_{}, phase_{}, count_{} {}

    __device__
    PipelineState(int index, uint32_t phase, uint32_t count)
        : index_(index), phase_(phase), count_(count) {}

    __device__ int index() const
    {
        return index_;
    }

    __device__
        uint32_t
        phase() const
    {
        return phase_;
    }

    __device__
        uint32_t
        count() const
    {
        return count_;
    }

    __device__ void operator++()
    {
        if constexpr (Stages > 0)
        {
            ++index_;
            ++count_;
            if (index_ == Stages)
            {
                index_ = 0;
                phase_ ^= 1;
            }
        }
    }

    __device__
        PipelineState &
        operator=(const PipelineState &other)
    {
        index_ = other.index();
        phase_ = other.phase();
        count_ = other.count();
        return *this;
    }

    __device__
        PipelineState
        advance(uint32_t num_iterations)
    {
        if constexpr (Stages > 0)
        {
            // Number of iterations cross over the stage boundary => flipped phase
            if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages)
            {
                phase_ ^= 1;
            }
            // How many times number of iterations cross over the stage boundary and
            // end up on a odd number => flipped phase
            if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1)
            {
                phase_ ^= 1;
            }
            index_ = (index_ + num_iterations) % Stages;
            count_ += num_iterations;
        }
        return *this;
    }

    __device__ static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations)
    {
        return start_state.advance(num_iterations);
    }
};

template <int SequenceDepth, int SequenceLength>
class OrderedSequenceBarrier
{
public:
    using Barrier = ClusterBarrier;

    struct SharedStorage
    {
        Barrier barrier_[SequenceDepth][SequenceLength];
    };

    struct Params
    {
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

    __device__
    OrderedSequenceBarrier(SharedStorage &storage, Params const &params) : params_(params),
                                                                           barrier_ptr_(&storage.barrier_[0][0]),
                                                                           // Group 0 - starts with an opposite phase
                                                                           stage_({0, (params.group_id == 0), 0})
    {
        int warp_idx = threadIdx.x / WARP_SIZE;
        int lane_predicate = elect_one_sync();

        // Barrier FULL, EMPTY init
        // Init is done only by the one elected thread of the block
        if (warp_idx == params.active_warps && lane_predicate == 1)
        {
            for (int d = 0; d < Depth; ++d)
            {
                for (int l = 0; l < Length; ++l)
                {
                    barrier_ptr_[d * Length + l].init(params.group_size);
                }
            }
        }
        fence_barrier_init();
    }

    // Wait on a stage to be unlocked
    __device__ void wait()
    {
        get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
    }

    __device__ void check_phase(int val)
    {
        if (threadIdx.x % WARP_GROUP_SIZE == 0)
        {
            printf("round %d group %d phase is %d\n", val, threadIdx.x / WARP_GROUP_SIZE, stage_.phase());
        }
    }

    // Signal completion of Stage and move to the next stage
    // (group_id) signals to (group_id+1)
    __device__ void arrive()
    {
        int signalling_id = (params_.group_id + 1) % Length;
        get_barrier_for_current_stage(signalling_id).arrive();
        ++stage_;
    }

    __device__ void advance()
    {
        ++stage_;
    }

private:
    __device__
        Barrier &
        get_barrier_for_current_stage(int group_id)
    {
        return barrier_ptr_[stage_.index() * Length + group_id];
    }
};