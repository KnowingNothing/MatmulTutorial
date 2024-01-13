#pragma once
#include "common.h"

template<int ClusterX, int ClusterY, int ClusterZ,
         int BlockX, int BlockY, int BlockZ>
struct TileScheduler {
    int64_t current_idx;
    int problem_m;
    int problem_n;
    int total_problem_mn;

    struct Params {
        int M;
        int N;
        int K;
    };

    struct TileInfo {
        int m = 0;
        int n = 0;
        bool valid = false;
    };

    DEVICE TileScheduler(Params params) {
        static_assert(ClusterZ == 1 && "Cluster Z should be 1.");
        current_idx = blockIdx.x + blockIdx.y * gridDim.x;
        problem_m = ceil_div(params.M, BlockX);
        problem_n = ceil_div(params.N, BlockY);
        problem_m = ceil_div(problem_m, ClusterX) * ClusterX;
        problem_n = ceil_div(problem_n, ClusterY) * ClusterY;
        total_problem_mn = problem_m * problem_n;
    }

    DEVICE TileInfo get_current_tile() {
        return get_current_tile_for_idx(current_idx);
    }

    DEVICE TileInfo get_current_tile_for_idx(int64_t idx) {
        int64_t cluster_id = idx / (ClusterX * ClusterY); // ClusterZ should be 1
        auto [block_m_in_cluster, block_n_in_cluster, _] = block_id_in_cluster();
        int cluster_m = cluster_id / (problem_n / ClusterY);
        int cluster_n = cluster_id - cluster_m * (problem_n / ClusterY);
        int work_m = cluster_m * ClusterX + block_m_in_cluster;
        int work_n = cluster_n * ClusterY + block_n_in_cluster; 
        return {work_m, work_n, idx < total_problem_mn};
    }

    DEVICE void advance(int count = 1) {
        current_idx += gridDim.z * gridDim.y * gridDim.x * count;
    }
};