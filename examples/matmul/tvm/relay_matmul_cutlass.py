import tvm
from tvm import relay
import numpy as np
import argparse
import time
from tqdm import tqdm
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules,
    finalize_modules_vm,
)


def relay_matmul(
    M, N, K, in_dtype="float16", acc_dtype="float32", target="cuda"
):
    A = relay.var("A", shape=[M, K], dtype=in_dtype)
    B = relay.var("B", shape=[N, K], dtype=in_dtype)
    C = relay.nn.matmul(A, B, out_dtype=acc_dtype,
                        transpose_a=False, transpose_b=True)
    D = relay.cast(C, dtype=in_dtype)
    args = relay.analysis.free_vars(C)
    func = relay.Function(args, C)
    return func


def profile_and_build(mod, params, sm, tmp_dir="./tmp", lib_path="compile.so", use_fast_math=False):
    mod = partition_for_cutlass(mod)
    use_3xtf32 = False
    split_k_slices = [1]
    num_cutlass_partition = num_cutlass_partitions(mod)
    cuda = tvm.target.Target("cuda")
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": sm,
            "use_3xtf32": use_3xtf32,
            "split_k_slices": split_k_slices,
            "profile_all_alignments": False,
            "find_first_valid": True,
            "use_multiprocessing": True,
            "use_fast_math": use_fast_math,
            "tmp_dir": tmp_dir,
        }
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=params)
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def verify_matmul(
    func, M, N, K, ref_target="cuda", sm=80, atol=1e-5, rtol=1e-5
):
    if not has_cutlass():
        raise RuntimeError("No CUTLASS!")
    mod = tvm.IRModule.from_expr(func)
    rt_mod, dev, num_partition = profile_and_build(mod, {}, sm)

    cost = rt_mod.benchmark(dev, number=200).mean * 1e3
    return cost


def main(M, N, K, in_dtype, acc_dtype, only_once):
    return verify_matmul(relay_matmul(M, N, K, in_dtype=in_dtype, acc_dtype=acc_dtype), M, N, K)


example_text = """
 example:
    python relay_matmul_cutlass.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
"""


dims = [128 * i for i in range(1, 11)]
base_shape = [5376, 5376, 2048]
shapes = [base_shape]
for dim in range(3):
    for f in [-1, 1]:
        for delta in dims:
            new_shape = [*base_shape]
            new_shape[dim] += delta * f
            shapes.append(new_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--acc_dtype",
        type=str,
        choices=["float16", "float32", "int32"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M, N, K = shapes[i]
        cost = main(M, N, K, args.in_dtype, args.acc_dtype, args.only_once)
        costs.append((shapes[i], cost))

    with open("relay_matmul_cutlass_results.csv", "w") as fout:
        print("Relay+CUTLASS Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
        for cc in costs:
            M, N, K = cc[0]
            runtime = cc[1]
            tflops = 2 * M * N * K / (runtime/1e3) / 1e12
            print(
                f"{M},{N},{K},{args.in_dtype},{args.acc_dtype},{runtime},{tflops}", file=fout)
    print("Done!")
