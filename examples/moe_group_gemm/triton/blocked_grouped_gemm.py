"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

import triton
import triton.language as tl

import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
from functools import reduce


@triton.jit
def grouped_matmul_kernel(
    A,
    B,
    C,
    scatter_idx,
    expert_idx,
    M,
    N,
    K,
    E,
    num_valid_tokens,
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    
    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size
    
    offs_token_id = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_token = tl.load(scatter_idx + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_token[:, None] * A_stride_m + offs_k[None, :] * A_stride_k
    
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_be = tl.load(expert_idx + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K))
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * A_stride_k
        b_ptrs += BLOCK_K * B_stride_k
        
    accumulator = accumulator.to(tl.float16)
    
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    c_ptrs = C + offs_token[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(A, B, C, scatter_idx, expert_idx, M, N, K, E, num_valid_tokens, BLOCK_M):
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grouped_matmul_kernel[grid](
        A,
        B,
        C,
        scatter_idx,
        expert_idx,
        M,
        N,
        K,
        E,
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M
    )

def cdiv(a, b):
    return (a + b - 1) // b

def main(M_list, N, K, trials):
    M = sum(M_list)
    E = len(M_list)
    BLOCK_M = 128
    M_list_pad = [cdiv(x, BLOCK_M) * BLOCK_M for x in M_list]
    EM = sum(M_list_pad)
    num_blocks = cdiv(EM, BLOCK_M)
    scatter_idx = torch.zeros((EM, ), dtype=torch.int32, device="cuda")
    expert_idx_full = torch.zeros((EM, ), dtype=torch.int32, device="cuda")
    expert_idx = torch.zeros([num_blocks, ], dtype=torch.int32, device="cuda")
    offset = 0
    offset_pad = 0
    for i, (org, pad) in enumerate(zip(M_list, M_list_pad)):
        scatter_idx[offset_pad: offset_pad + org] = (torch.arange(org, dtype=torch.int32, device="cuda") + offset)
        scatter_idx[offset_pad + org: offset_pad + pad] = 0
        expert_idx_full[offset_pad: offset_pad + pad] = i
        offset += org
        offset_pad += pad
    expert_idx[torch.arange(num_blocks)] = expert_idx_full[torch.arange(num_blocks) * BLOCK_M]
        
    A = torch.rand((M, K), device="cuda", dtype=torch.float16)
    B = torch.rand((E, K, N), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # warm-up
    try:
        triton_output = triton_perf_fn(A, B, C, scatter_idx, expert_idx, EM, N, K, E, M, BLOCK_M)
        # Create CUDA events for measuring time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Record the start event
        start_event.record()
        for i in range(trials):
            triton_output = triton_perf_fn(A, B, C, scatter_idx, expert_idx, EM, N, K, E, M, BLOCK_M)
        end_event.record()
        end_event.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / trials
    except Exception as e:
        print(M_list, N, K, flush=True)
        print(e, flush=True)
        raise RuntimeError()


shape_M = [8, 32, 1024, 4096, 16*1024]
top_K = [2, 4, 8]
n_experts = [8]
expert_freq = [0.1, 0.2, 0.1, 0.1, 0.05, 0.15, 0.25, 0.05]
shape_N_K_pairs = [
[
    14336,
    4096,
],
[
    4096,
    14336,
]
]


@dataclass
class GroupGemmShape:
    M_list: List[int]
    N: int
    K: int
    
shapes = []
for M in shape_M:
    for topk in top_K:
        for n_expert in n_experts:
            for shape_N_K in shape_N_K_pairs:
                N, K = shape_N_K
                distribution = expert_freq
                M_list = [int(M * topk * distribution[i]) for i in range(n_expert)]
                M_list[-1] = M * topk - reduce(lambda x, y: x + y, M_list[:-1], 0)
                shapes.append(GroupGemmShape(M_list, N, K))
                    


example_text = """
 example:
    python grouped_gemm.py --begin 0 --num 1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    with open(f"triton_blocked_grouped_gemm.csv", "w") as fout:
        print("Triton Performance", file=fout)
        print("M_list,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M_list, N, K = shapes[i].M_list, shapes[i].N, shapes[i].K
        cost = main(M_list, N, K, args.trials)
        costs.append((shapes[i], cost))

        with open(f"triton_blocked_grouped_gemm.csv", "a") as fout:
            M_list, N, K = shapes[i].M_list, shapes[i].N, shapes[i].K
            tflops = 2 * sum(M_list) * N * K / (cost / 1e3) / 1e12
            print(f"{';'.join(map(str, M_list))},{N},{K},{'float16'},{'float32'},{cost},{tflops}", file=fout)
    print("Done!")
