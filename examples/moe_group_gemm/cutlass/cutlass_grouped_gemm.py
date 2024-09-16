import cutlass
import torch

import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
from functools import reduce

dtype = torch.float16
plan = cutlass.op.GroupedGemm(element=dtype, layout=cutlass.LayoutType.RowMajor)
op = plan.construct()
grouped_gemm = cutlass.emit.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=True)


def main(M_list, N, K, trials):
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for M in M_list:
        A = torch.rand((M, K), device="cuda", dtype=torch.float16)
        B = torch.rand((K, N), device="cuda", dtype=torch.float16)
        C = torch.empty((M, N), device="cuda", dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [K, N, N]

    # warm-up
    try:
        cutlass_output = grouped_gemm.run(group_A, group_B)
        # Create CUDA events for measuring time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Record the start event
        start_event.record()
        for i in range(trials):
            cutlass_output = grouped_gemm.run(group_A, group_B)
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
    with open(f"cutlass_grouped_gemm.csv", "w") as fout:
        print("Triton Performance", file=fout)
        print("M_list,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M_list, N, K = shapes[i].M_list, shapes[i].N, shapes[i].K
        cost = main(M_list, N, K, args.trials)
        costs.append((shapes[i], cost))

        with open(f"cutlass_grouped_gemm.csv", "a") as fout:
            M_list, N, K = shapes[i].M_list, shapes[i].N, shapes[i].K
            tflops = 2 * sum(M_list) * N * K / (cost / 1e3) / 1e12
            print(f"{';'.join(map(str, M_list))},{N},{K},{'float16'},{'float32'},{cost},{tflops}", file=fout)
    print("Done!")