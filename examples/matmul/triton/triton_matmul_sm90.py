"""
Persistent FP8 Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
It includes various matmul methods, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches, and only supports GPUs with compute capability >= 9.0.
Triton and CuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.
"""

import argparse
import time
from tqdm import tqdm

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2.0 * M * N * K
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret

# Autotuner does not work with TMA. Use manual config.
configs = {
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
}


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    # TODO(embg) remove TMA fence after __grid_constant__ lands
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [a_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [b_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
        "=r, l",
        [c_desc_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(desc_a, desc_b, desc_c, dtype, NUM_SMS):

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_tma_persistent[grid](
        desc_a,
        desc_b,
        desc_c,  #
        M,
        N,
        K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )


def main(M, N, K, trials):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Check constraints.
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        a.data_ptr(),
        M,
        K,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_K"],
        a.element_size(),
    )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        b.data_ptr(),
        N,
        K,
        configs[dtype]["BLOCK_SIZE_N"],
        configs[dtype]["BLOCK_SIZE_K"],
        b.element_size(),
    )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        c.data_ptr(),
        M,
        N,
        configs[dtype]["BLOCK_SIZE_M"],
        configs[dtype]["BLOCK_SIZE_N"],
        c.element_size(),
    )
    # warm-up
    triton_output = matmul_tma_persistent(desc_a, desc_b, desc_c, dtype, NUM_SMS)
    # Create CUDA events for measuring time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record the start event
    start_event.record()
    for i in range(200):
        triton_output = matmul_tma_persistent(desc_a, desc_b, desc_c, dtype, NUM_SMS)
    end_event.record()
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / 200


dims = [512 * i for i in range(1, 11)]
base_shape = [8192, 8192, 8192]
shapes = [base_shape]
for dim in range(3):
    for f in [-1, 1]:
        for delta in dims:
            new_shape = [*base_shape]
            new_shape[dim] += delta * f
            shapes.append(new_shape)

example_text = """
 example:
    python triton_matmul_sm90.py --begin 0 --num 1
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
    with open(f"triton_results_sm90.csv", "w") as fout:
        print("Triton Performance", file=fout)
        print("M,N,K,in_dtype,acc_dtype,cost(ms),TFLOPS", file=fout)
    for i in tqdm(range(args.begin, args.begin + args.num)):
        M, N, K = shapes[i]
        cost = main(M, N, K, args.trials)
        costs.append((shapes[i], cost))

        with open(f"triton_results_sm90.csv", "a") as fout:
            M, N, K = shapes[i]
            tflops = 2 * M * N * K / (cost / 1e3) / 1e12
            print(f"{M},{N},{K},{'float16'},{'float32'},{cost},{tflops}", file=fout)
    print("Done!")
