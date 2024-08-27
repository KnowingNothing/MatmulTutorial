import subprocess
import argparse
from tqdm import tqdm
import os

example_text = """
 example:
    python test_cutlass_perf.py --begin 0 --num 1
"""


def build():
    for file in ["pingpong-gemm.cu"]:
        assert os.path.exists(file) and os.path.isfile(
            file), "CUDA files not exist"
    command = "nvcc -arch=sm_90a -std=c++17 --expt-relaxed-constexpr -I ../../../3rdparty/cutlass/include/ -I ../../../3rdparty/cutlass/examples/common/ -I ../../../3rdparty/cutlass/tools/util/include/ pingpong-gemm.cu -lcuda -lcudart -o test_cutlass"

    p = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert p.returncode == 0


def main(M, N, K):
    command = f"./test_cutlass --m={M} --n={N} --k={K} --iterations=200"
    p = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)

    cost = -1
    if p.returncode == 0:
        print(p.stdout)
        for line in p.stdout.splitlines():
            print(line)
            key = "  Avg runtime:"
            if key in line:
                print("YES")
                cost = float(line[len(key)+1:][:-3])
        return cost
    else:
        print("Error!")
        return cost


dims = [512 * i for i in range(1, 11)]
base_shape = [8192, 8192, 8192]
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
    # parser.add_argument("--only_once", action="store_true")
    # parser.add_argument(
    #     "--in_dtype",
    #     type=str,
    #     choices=["float16", "int8"],
    #     default="float16",
    # )
    # parser.add_argument(
    #     "--acc_dtype",
    #     type=str,
    #     choices=["float16", "float32", "int32"],
    #     default="float16",
    # )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    build()
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M, N, K = shapes[i]
        cost = main(M, N, K)
        costs.append((shapes[i], cost))

    with open("cutlass_results.csv", "w") as fout:
        print("CUTLASS Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
        for cc in costs:
            M, N, K = cc[0]
            runtime = cc[1]
            tflops = 2 * M * N * K / (runtime/1e3) / 1e12
            print(
                f"{M},{N},{K},{'float16'},{'float32'},{runtime},{tflops}", file=fout)
    print("Done!")
