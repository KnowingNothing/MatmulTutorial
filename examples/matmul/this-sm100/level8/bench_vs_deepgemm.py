#!/usr/bin/env python3
"""Benchmark: Level 8 vs DeepGEMM vs cuBLAS (torch.matmul) on same GPU."""
import torch
import ctypes
import subprocess
import os
import sys
import time

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(KERNEL_DIR, "matmul.cu")
SO_PATH = os.path.join(KERNEL_DIR, "gemm_bf16_sm100_level8.so")

def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a", "-O3",
           "-DTC_UTIL_PERCENT=100",
           KERNEL_PATH, "-o", SO_PATH, "-lcuda"]
    print(f"Compiling level8: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        sys.exit(1)
    print("OK\n")

def bench_ours(M, N, K, warmup=10, iters=100):
    lib = ctypes.CDLL(SO_PATH)
    lib.benchmark_kernel.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*5
    lib.benchmark_kernel.restype = ctypes.c_float
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    ms = lib.benchmark_kernel(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                              M, N, K, warmup, iters)
    return ms

def bench_cublas(M, N, K, warmup=10, iters=100):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.matmul(A, B.t())
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def bench_deepgemm(M, N, K, warmup=10, iters=100):
    try:
        import deep_gemm
    except ImportError:
        return None
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    # warmup (also triggers JIT compilation)
    for _ in range(warmup):
        deep_gemm.bf16_gemm_nt(A, B, D)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        deep_gemm.bf16_gemm_nt(A, B, D)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

if __name__ == "__main__":
    compile_kernel()

    sizes = [4096, 6144, 8192, 10240, 12288]

    print(f"{'Size':>8} | {'Ours (TFLOPS)':>14} | {'DeepGEMM':>14} | {'cuBLAS':>14} | {'Ours/DG':>8} | {'Ours/cuBLAS':>12}")
    print("-" * 90)

    for sz in sizes:
        M = N = K = sz
        flops = 2 * M * N * K

        ms_ours = bench_ours(M, N, K)
        tflops_ours = flops / (ms_ours * 1e-3) / 1e12

        ms_dg = bench_deepgemm(M, N, K)
        tflops_dg = flops / (ms_dg * 1e-3) / 1e12 if ms_dg else 0

        ms_cublas = bench_cublas(M, N, K)
        tflops_cublas = flops / (ms_cublas * 1e-3) / 1e12

        dg_str = f"{tflops_dg:.1f}" if ms_dg else "N/A"
        ratio_dg = f"{tflops_ours/tflops_dg:.3f}x" if tflops_dg > 0 else "N/A"
        ratio_cb = f"{tflops_ours/tflops_cublas:.3f}x"

        print(f"{sz:>8} | {tflops_ours:>13.1f} | {dg_str:>14} | {tflops_cublas:>13.1f} | {ratio_dg:>8} | {ratio_cb:>12}")
