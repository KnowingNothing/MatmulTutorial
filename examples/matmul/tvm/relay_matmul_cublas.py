import tvm
from tvm import relay
import numpy as np
import argparse
import time
from tqdm import tqdm


def relay_matmul(
    M, N, K, in_dtype="float16", acc_dtype="float32", target="cuda"
):
    A = relay.var("A", shape=[M, K], dtype=in_dtype)
    B = relay.var("B", shape=[N, K], dtype=in_dtype)
    C = relay.nn.matmul(A, B, out_dtype=acc_dtype,
                        transpose_a=False, transpose_b=True)
    D = relay.cast(C, dtype=in_dtype)
    args = relay.analysis.free_vars(D)
    func = relay.Function(args, D)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    return [[M, K], [N, K]], [[M, N]], module


def main(M, N, K, in_dtype, acc_dtype, only_once):
    in_dtype = in_dtype
    acc_dtype = acc_dtype
    target = "cuda -libs=cublas,cudnn"
    ins, outs, module = relay_matmul(
        M, N, K, in_dtype=in_dtype, acc_dtype=acc_dtype, target=target
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in outs
    ]
    ctx = tvm.cuda()
    dev = tvm.device(str(target), 0)
    if only_once:
        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
        module.set_input(key=0, value=inputs_tvm[0])
        module.set_input(key=1, value=inputs_tvm[1])
        module.set_input(key=2, value=inputs_tvm[2])
        # module.set_input(key=3, value=outputs_tvm[0])
        module.run()
        cost = -1
    else:
        ret = module.benchmark(dev, number=200)
        cost = ret.mean * 1e3
    return cost


example_text = """
 example:
    python relay_matmul_cublas.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
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

    with open("relay_matmul_cublas_results.csv", "w") as fout:
        print("Relay+CUDNN/CuBLAS Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
        for cc in costs:
            M, N, K = cc[0]
            runtime = cc[1]
            tflops = 2 * M * N * K / (runtime/1e3) / 1e12
            print(
                f"{M},{N},{K},{args.in_dtype},{args.acc_dtype},{runtime},{tflops}", file=fout)
    print("Done!")
