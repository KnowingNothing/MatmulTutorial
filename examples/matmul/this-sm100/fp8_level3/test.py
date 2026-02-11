#!/usr/bin/env python3
"""
FP8 GEMM Level 3 — Test & Benchmark

Level 3 improvements over Level 2:
  1. BLOCK_N=224 (vs 128): higher compute density, matches DeepGEMM config
  2. Fixed epilogue swizzle for SWIZZLE_CD != 128B (was causing hangs for BLOCK_N > 128)
  3. Generalized CD STORE_N and BANK_GROUPS for any BLOCK_N
"""

import os
import torch
import ctypes
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul_n224.so")
GRAN_K = 128
BLOCK_N = 224

def compile_kernel():
    if os.path.exists(LIB) and os.path.getmtime(LIB) > os.path.getmtime(SRC):
        print("Kernel already compiled.")
        return
    cmd = [
        "nvcc", "--shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3", "-std=c++17",
        f"-DTILE_N={BLOCK_N}",
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

def check_correctness(lib, M, N, K):
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
    print(f"\n=== FP8 Level 3 Benchmarks (BLOCK_N={BLOCK_N}, cta_group::1) ===")
    print(f"  SMs={num_sms}")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} {'TFLOPS':>10}")
    print("-" * 52)

    # Use N values that are multiples of BLOCK_N=224 for best results
    # 7168 = 224 * 32, 4480 = 224 * 20, 2240 = 224 * 10
    sizes = [
        (1024, 1024, 1024),      # small test (N not aligned to 224, relies on OOB fill)
        (2048, 2240, 2048),      # N=2240 = 224*10
        (4096, 4480, 4096),      # N=4480 = 224*20
        (4096, 7168, 4096),      # N=7168 = 224*32  (DeepGEMM benchmark size)
        (4096, 7168, 7168),      # N=7168 = 224*32  (DeepGEMM benchmark size)
        (8192, 7168, 8192),      # N=7168 = 224*32
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

    # Correctness: use N multiples of 224, M multiples of 128, K multiples of 128
    all_pass = True
    for m, n, k in [(896, 896, 896), (1024, 2240, 1024), (2048, 2240, 2048)]:
        if not check_correctness(lib, m, n, k):
            all_pass = False

    if all_pass:
        run_benchmarks(lib)
    else:
        print("\nSkipping benchmarks due to correctness failures.")
