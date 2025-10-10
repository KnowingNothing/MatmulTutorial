# Referenced from DeepGEMM.

import torch
import random
import os
import sys


def generate_normal(
    m: int,
    n: int,
    k: int,
    accumulate: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).uniform_(0, 1)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).uniform_(0, 1)
    d = (
        torch.randn((m, n), device="cuda", dtype=out_dtype) * 32
        if accumulate
        else torch.empty((m, n), device="cuda", dtype=out_dtype).uniform_(0, 1)
    )
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    return a, b, c, d, ref_d


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get("DG_NSYS_PROFILING", 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert (
                sum([name in line for line in prof_lines]) == 1
            ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


if torch.cuda.is_available():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


CWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(CWD)
CU_SRC = "matmul.cu"
CU_OBJ = "matmul.so"
CUTLASS_HOME = os.environ.get(
    "CUTLASS_HOME", os.path.join(CWD, "../../../3rdparty/cutlass")
)


def compile_matmul_kernel():
    import os
    import subprocess

    if not os.path.exists(CU_OBJ) or os.path.getmtime(CU_OBJ) < os.path.getmtime(
        CU_SRC
    ):
        cmd = (
            f"nvcc ./{CU_SRC} -x cu -shared -o ./{CU_OBJ} -std=c++20 "
            f"-I {CUTLASS_HOME}/include "
            "--diag-suppress=39,161,174,177,186,940 "
            "--ptxas-options=--register-usage-level=10 "
            "--ptxas-options=--verbose "
            "--gpu-architecture=sm_100a "
            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
            "-O3 --expt-relaxed-constexpr --expt-extended-lambda "
            "-lcuda "
        )
        print(cmd)
        subprocess.run(cmd, check=True, cwd=CWD, shell=True)


def load_matmul_kernel():
    import ctypes

    compile_matmul_kernel()

    lib = ctypes.CDLL(f"./{CU_OBJ}")
    kernel = lib.matmul_sm100_bf16_2sm_256x256x64

    kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    kernel.restype = ctypes.c_int

    return lib, kernel


def test_matmul() -> None:
    lib, kernel = load_matmul_kernel()

    print("Testing Matmul:")
    for m, n, k in [(8192, 9728, 16384)]:
        major_opt = "NT"
        out_opt = "BF16"
        acc_opt = f"acc=0"

        a, b, c, d, ref_d = generate_normal(m, n, k)
        kernel(a.data_ptr(), b.data_ptr(), d.data_ptr(), m, n, k)
        diff = calc_diff(d, ref_d)
        assert diff < 0.0001, f"{m=}, {n=}, {k=}, {diff:.5f}"

        a, b, c, d, ref_d = generate_normal(m, n, k)

        cublas_t = 0
        t = bench_kineto(
            lambda: kernel(a.data_ptr(), b.data_ptr(), d.data_ptr(), m, n, k),
            "matmul_sm100_bf16",
            suppress_kineto_output=True,
        )

        # noinspection PyBroadException
        try:
            cublas_t = bench_kineto(
                lambda: a @ b.T, "nvjet", suppress_kineto_output=True
            )
        except Exception:
            import traceback

            traceback.print_exc()
        print(
            f" > Perf (m={m:5}, n={n:5}, k={k:5}, layout={major_opt}, {out_opt}, {acc_opt}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s | "
            f"{cublas_t / t:.2f}x cuBLAS"
        )
    print()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)
    test_matmul()
