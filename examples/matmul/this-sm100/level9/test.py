#!/usr/bin/env python3
"""Test script for Level 9 — Matching DeepGEMM architecture on SM100."""
import torch
import ctypes
import subprocess
import os
import sys

LEVEL = "level9"
KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(KERNEL_DIR, "matmul.cu")
SO_PATH = os.path.join(KERNEL_DIR, f"gemm_bf16_sm100_{LEVEL}.so")


def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a", "-O3",
           KERNEL_PATH, "-o", SO_PATH, "-lcuda"]
    print(f"Compiling: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        sys.exit(1)
    print("OK")


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    return torch.nn.functional.cosine_similarity(
        x_flat.unsqueeze(0), y_flat.unsqueeze(0)).item()


def test_correctness(M=4096, N=4096, K=4096,
                     atol=1e-2, rtol=1e-2):
    lib = ctypes.CDLL(SO_PATH)
    lib.gemm_bf16_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    lib.gemm_bf16_launch(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    D_ref = torch.matmul(A.float(), B.float().t())

    max_err = (D.float() - D_ref).abs().max().item()
    mean_err = (D.float() - D_ref).abs().mean().item()
    cos_sim = cosine_similarity(D, D_ref)
    passed = torch.allclose(D.float(), D_ref, atol=atol, rtol=rtol)

    print(f"Correctness ({M}x{N}x{K}): "
          f"max_err={max_err:.6f}, mean_err={mean_err:.6f}, "
          f"cos_sim={cos_sim:.6f}")
    print(f"  allclose(atol={atol}, rtol={rtol})="
          f"{'PASSED' if passed else 'FAILED'}")
    return passed, max_err, cos_sim


def benchmark(M, N, K, warmup=10, iters=100):
    lib = ctypes.CDLL(SO_PATH)
    lib.benchmark_kernel.argtypes = ([ctypes.c_void_p] * 3
                                     + [ctypes.c_int] * 5)
    lib.benchmark_kernel.restype = ctypes.c_float

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    ms = lib.benchmark_kernel(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                              M, N, K, warmup, iters)
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12

    # Also benchmark DeepGEMM if available
    dg_tflops = 0
    try:
        import deep_gemm
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
        dg_ms = start.elapsed_time(end) / iters
        dg_tflops = 2 * M * N * K / (dg_ms * 1e-3) / 1e12
    except ImportError:
        pass

    # cuBLAS
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
    torch_ms = start.elapsed_time(end) / iters
    torch_tflops = 2 * M * N * K / (torch_ms * 1e-3) / 1e12

    dg_str = f"{dg_tflops:.1f}" if dg_tflops > 0 else "N/A"
    ratio_dg = f"{tflops/dg_tflops:.3f}x" if dg_tflops > 0 else "N/A"
    ratio_cb = f"{tflops/torch_tflops:.3f}x"

    print(f"Benchmark ({M}x{N}x{K}):")
    print(f"  Ours:     {ms:.3f} ms, {tflops:.1f} TFLOPS")
    print(f"  DeepGEMM: {dg_str} TFLOPS  ({ratio_dg})")
    print(f"  cuBLAS:   {torch_tflops:.1f} TFLOPS  ({ratio_cb})")
    return tflops, dg_tflops, torch_tflops


if __name__ == "__main__":
    compile_kernel()
    test_correctness()
    print()
    for sz in [4096, 6144, 8192, 10240, 12288]:
        benchmark(M=sz, N=sz, K=sz)
        print()
