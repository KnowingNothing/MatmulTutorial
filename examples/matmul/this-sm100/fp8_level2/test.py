#!/usr/bin/env python3
"""
FP8 GEMM Level 2 — Test & Benchmark

Level 2 improvements over Level 1:
  1. Removed debug printf from kernel
  2. SMEM descriptor precomputation via __shfl_sync (avoids per-stage recompute)
  3. Maximized pipeline stages (auto-computed from SMEM budget)
  4. Cleaner SMEM layout calculations
"""

import os
import torch
import ctypes
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul_n128.so")
GRAN_K = 128

def compile_kernel():
    if os.path.exists(LIB) and os.path.getmtime(LIB) > os.path.getmtime(SRC):
        print("Kernel already compiled.")
        return
    cmd = [
        "nvcc", "--shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3", "-std=c++17",
        "-DTILE_N=128",
        "-lcuda",
        SRC, "-o", LIB
    ]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fp8_gemm.restype = None
    return lib

def create_test_data(M, N, K, device="cuda"):
    A = (torch.randn(M, K, device=device) * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device=device) * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device=device).contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device=device).contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device=device).contiguous()
    return A, B, SFA, SFB, D

def run_kernel(lib, A, B, SFA, SFB, D, M, N, K):
    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

def check_correctness(lib, M=1024, N=1024, K=1024):
    print(f"\n=== Correctness: M={M}, N={N}, K={K} ===")
    A, B, SFA, SFB, D = create_test_data(M, N, K)
    ref = (A.float() @ B.float().T).bfloat16()
    run_kernel(lib, A, B, SFA, SFB, D, M, N, K)

    max_err = (D.float() - ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        D.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0)).item()

    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")
    ok = cos_sim > 0.99
    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok

def benchmark(lib, M, N, K, warmup=5, iters=20):
    A, B, SFA, SFB, D = create_test_data(M, N, K)
    for _ in range(warmup):
        run_kernel(lib, A, B, SFA, SFB, D, M, N, K)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_kernel(lib, A, B, SFA, SFB, D, M, N, K)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / iters * 1000
    tflops = 2.0 * M * N * K / (avg_ms / 1000) / 1e12
    return avg_ms, tflops

def run_benchmarks(lib):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== Benchmarks (SMs={num_sms}, BLOCK_N=128, 5 stages) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} {'TFLOPS':>10}")
    print("-" * 52)

    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 7168, 4096),
        (4096, 7168, 7168),
        (8192, 8192, 8192),
    ]

    for M, N, K in sizes:
        try:
            ms, tflops = benchmark(lib, M, N, K)
            print(f"{M:>6} {N:>6} {K:>6} | {ms:>10.3f} {tflops:>10.1f}")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} | ERROR: {e}")

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    all_pass = True
    for m, n, k in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        if not check_correctness(lib, m, n, k):
            all_pass = False

    if all_pass:
        run_benchmarks(lib)
    else:
        print("\nSkipping benchmarks due to correctness failures.")
