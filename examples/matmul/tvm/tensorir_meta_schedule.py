import tempfile
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm

import tvm
from tvm import te
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.meta_schedule.builder import LocalBuilder
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import Trace

# get tensor intrin
from tvm.tir.tensor_intrin import cuda  # pylint: disable=unused-import

import tvm.testing


def matmul_fp16(N: int, M: int, K: int, in_dtype: str, out_dtype: str):
    x = te.placeholder((N, K), name="X", dtype=in_dtype)
    y = te.placeholder((K, M), name="Y", dtype=in_dtype)
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(  # pylint: disable=invalid-name
        (N, M),
        lambda i, j: te.sum(x[i][k].astype(out_dtype) * \
                            y[k][j].astype(out_dtype), axis=[k]),
        name="C",
    )
    d = c
    # d = te.compute(
    #     (N, M),
    #     lambda i, j: c[i][j].astype(in_dtype),
    #     name="D"
    # )
    return (x, y, d)


def initializer():
    @register_func("meta_schedule.builder.async_build")
    def async_build(mod, target, _params):  # pylint: disable=unused-variable, unused-argument
        # pylint: disable=import-outside-toplevel
        from tvm.driver import build as tvm_build
        from tvm.tir.transform import RemoveWeightLayoutRewriteBlock

        # re-import here for local builder to register index_map_m16n8k8_matrixC
        # pylint: disable=import-outside-toplevel, unused-import
        from tvm.tir.tensor_intrin import cuda

        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            rt_mod = tvm_build(mod, target=target)
        return rt_mod


def multi_level_tiling_mma(out_dtype):
    simplify_dict = {"float32": "f32", "float16": "f16"}
    out_dtype = simplify_dict[out_dtype]
    return ms.schedule_rule.MultiLevelTilingTensorCore(
        intrin_groups=[
            {
                "init": f"mma_init_m16n8k8_{out_dtype}",
                "load_a": "mma_load_m16n8k8_f16_A_shared_dyn",
                "load_b": "mma_load_m16n8k8_f16_B_shared_dyn",
                "compute": f"mma_sync_m16n8k8_f16f16{out_dtype}",
                "store": f"mma_store_m16n8k8_{out_dtype}_global",
            },
        ],
        structure="SSSRRSRS",
        tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
        max_innermost_factor=4,  # 64 // tensor intrin size
        vector_load_lens=[1, 2, 3, 4, 8, 16],
        reuse_read=ms.schedule_rule.ReuseType(
            req="must",
            levels=[4],
            scope="shared.dyn",
        ),
        reuse_write=ms.schedule_rule.ReuseType(
            req="no",
            levels=[2],
            scope="shared.dyn",
        ),
        use_software_pipeline=True,
    )


def main(M, N, K, in_dtype, out_dtype, only_once, trials):
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    if major < 8:
        # At least sm80 is required
        return

    # pylint: disable=import-outside-toplevel
    from tvm.contrib import cublas

    def tune(out_dtype):
        # M, N, K = 1024, 1024, 1024
        target = Target("nvidia/nvidia-a100")
        func = te.create_prim_func(matmul_fp16(N=N, M=M, K=K, in_dtype=in_dtype, out_dtype=out_dtype)).with_attr(
            {"global_symbol": f"main_{M}_{N}_{K}"}
        )
        mod = tvm.IRModule({f"main": func})

        with tempfile.TemporaryDirectory() as work_dir:
            db = ms.tir_integration.tune_tir(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=trials,
                builder=LocalBuilder(
                    f_build="meta_schedule.builder.async_build", initializer=initializer
                ),
                space=ms.space_generator.PostOrderApply(
                    sch_rules=[multi_level_tiling_mma(out_dtype=out_dtype)],
                ),
            )
            sch = db.query_schedule(
                mod, target=target, workload_name=f"main")
            with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
                rt_mod = tvm.build(sch.mod, target=target)
                dev = tvm.cuda(0)
                a_np = np.random.uniform(0, 1, size=(N, K)).astype("float16")
                b_np = np.random.uniform(0, 1, size=(K, M)).astype("float16")
                a_tvm = tvm.nd.array(a_np, device=tvm.cuda(0))
                b_tvm = tvm.nd.array(b_np, device=tvm.cuda(0))
                c_tvm = tvm.nd.array(np.empty((N, M)).astype(
                    out_dtype), device=tvm.cuda(0))
                evaluator = rt_mod.time_evaluator(rt_mod.entry_name, dev, number=200)
                cost = evaluator(a_tvm, b_tvm, c_tvm).mean * 1e3
                return cost

    return tune(out_dtype)


dims = [128 * i for i in range(1, 11)]
base_shape = [5376, 5376, 2048]
shapes = [base_shape]
for dim in range(3):
    for f in [-1, 1]:
        for delta in dims:
            new_shape = [*base_shape]
            new_shape[dim] += delta * f
            shapes.append(new_shape)

example_text = """
 example:
    python tensorir_meta_schedule.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
"""

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
        "--trials", type=int, default=10
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    with open(f"tensorir_meta_schedule_results_{args.trials}_trials.csv", "w") as fout:
        print("TensorIR+MetaSchedule Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M, N, K = shapes[i]
        cost = main(M, N, K, args.in_dtype, args.acc_dtype, args.only_once, args.trials)
        costs.append((shapes[i], cost))

        with open(f"tensorir_meta_schedule_results_{args.trials}_trials.csv", "a") as fout:
            M, N, K = shapes[i]
            tflops = 2 * M * N * K / (cost/1e3) / 1e12
            print(
                f"{M},{N},{K},{args.in_dtype},{args.acc_dtype},{cost},{tflops}", file=fout)
    print("Done!")
