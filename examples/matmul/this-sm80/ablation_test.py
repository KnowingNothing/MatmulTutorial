import subprocess
import argparse
from tqdm import tqdm
import os

example_text = """
 example:
    python test_this_perf.py --begin 0 --num 1
"""


def build(version):
    version = str(version).zfill(2)
    kernel = "matmul-v{0:>2}.cu".format(version)
    for file in [kernel, "test_this_perf.cu"]:
        assert os.path.exists(file) and os.path.isfile(
            file), f"CUDA files {file} not exist"
    command = f"nvcc -arch=sm_80  {kernel} test_this_perf.cu -o ablation_test"

    p = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert p.returncode == 0


def main(M, N, K, version):
    if version == 0:
        command = f"./ablation_test M {M} N {N} K {K} iters 200"
    elif version >= 8 and version <= 11:
        command = f"./ablation_test M {M} N {N} K {K} stages 4 multi_threading 2 iters 200"
    else:
        command = f"./ablation_test M {M} N {N} K {K} iters 200 stages 4"
    p = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)

    cost = -1
    if p.returncode == 0:
        print(p.stdout)
        for line in p.stdout.splitlines():
            print(line)
            key = "Running cost (ms) of CUDA kernel is"
            if key in line:
                print("YES")
                cost = float(line[len(key)+1:])
        return cost
    else:
        print("Error!")
        return cost


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
    parser.add_argument("--version", type=int, default=-1)
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    if args.version < 0:
        for v in range(16):
            costs = []
            build(v)
            for i in tqdm(range(args.begin, args.begin + args.num)):
                M, N, K = shapes[i]
                cost = main(M, N, K, v)
                costs.append((shapes[i], cost))

            with open(f"v{v}_results.csv", "w") as fout:
                print(f"Version {v} Performance", file=fout)
                print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
                for cc in costs:
                    M, N, K = cc[0]
                    runtime = cc[1]
                    tflops = 2 * M * N * K / (runtime/1e3) / 1e12
                    print(
                        f"{M},{N},{K},{'float16'},{'float32'},{runtime},{tflops}", file=fout)
    else:
        costs = []
        build(args.version)
        for i in tqdm(range(args.begin, args.begin + args.num)):
            M, N, K = shapes[i]
            cost = main(M, N, K, args.version)
            costs.append((shapes[i], cost))

        with open(f"v{args.version}_results.csv", "w") as fout:
            print(f"Version {args.version} Performance", file=fout)
            print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
            for cc in costs:
                M, N, K = cc[0]
                runtime = cc[1]
                tflops = 2 * M * N * K / (runtime/1e3) / 1e12
                print(
                    f"{M},{N},{K},{'float16'},{'float32'},{runtime},{tflops}", file=fout)
    print("Done!")
