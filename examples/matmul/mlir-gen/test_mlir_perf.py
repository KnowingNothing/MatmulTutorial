import subprocess
import argparse
from tqdm import tqdm
import os

example_text = """
 example:
    python test_mlir_perf.py --begin 0 --num 1
"""


def build(M, N, K):
    matmul = f"matmul_m{M}_n{N}_k{K}_fp16fp16fp16_tensorop.cu"
    for file in [matmul, "main.cu"]:
        assert os.path.exists(file) and os.path.isfile(
            file), "CUDA files not exist"
    command = f"nvcc -arch=sm_80  tmp_matmul.cu main.cu -o test_perf"
    with open("tmp_matmul.cu", "w") as fout:
        fout.writelines(("#include <cuda.h>\n"
                         "#include <cuda_fp16.h>\n"
                         "#include <mma.h>\n"
                         "#include \"mlir_mma.h\"\n"
                         "using namespace byteir_mma;\n\n"))
        with open(matmul, "r") as fin:
            for line in fin:
                fout.write(line)
    p = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(p.stderr)

    assert p.returncode == 0


def main(M, N, K):
    build(M, N, K)
    command = f"./test_perf M {M} N {N} K {K} iters 200 stages 4"
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
        cost = main(M, N, K)
        costs.append((shapes[i], cost))

    with open("mlir_results.csv", "w") as fout:
        print("MLIR Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
        for cc in costs:
            M, N, K = cc[0]
            runtime = cc[1]
            tflops = 2 * M * N * K / (runtime/1e3) / 1e12
            print(
                f"{M},{N},{K},{'float16'},{'float16'},{runtime},{tflops}", file=fout)
    print("Done!")
