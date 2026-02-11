#!/usr/bin/env python3
"""
FP8 GEMM Level 1 — Test & Benchmark

D (BF16, M×N) = A (FP8 E4M3, M×K) × B^T (FP8 E4M3, N×K)

Scale factors: UE8M0 format (4 E8M0 values packed per int32).
For this level, all scale factors are set to 1.0 (E8M0 value 0x7F = 127).
"""

import os
import torch
import ctypes
import time
import subprocess

# ============================================================================
# Compilation
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul.so")

def compile_kernel():
    if os.path.exists(LIB) and os.path.getmtime(LIB) > os.path.getmtime(SRC):
        print("Kernel already compiled, skipping.")
        return
    cmd = [
        "nvcc", "--shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3", "-std=c++17",
        "-lcuda",
        SRC, "-o", LIB
    ]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Compilation done.")

# ============================================================================
# Load library
# ============================================================================

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # SFA
        ctypes.c_void_p,  # SFB
        ctypes.c_void_p,  # D
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
    ]
    lib.launch_fp8_gemm.restype = None
    return lib

# ============================================================================
# Test data generation
# ============================================================================

GRAN_K = 128

def create_test_data(M, N, K, device="cuda"):
    """Create FP8 matrices and scale factors.

    For Level 1, all scale factors = 1.0 (UE8M0 value 127 = 0x7F).
    This means the GEMM output should match float(A_fp8) @ float(B_fp8)^T.
    """
    # Random FP8 E4M3 matrices
    # Generate small float values, then cast to FP8
    A_fp32 = torch.randn(M, K, device=device) * 0.1
    B_fp32 = torch.randn(N, K, device=device) * 0.1
    A = A_fp32.to(torch.float8_e4m3fn).contiguous()
    B = B_fp32.to(torch.float8_e4m3fn).contiguous()

    # Scale factors: all 1.0
    # UE8M0 value 0x7F = 127 means 2^(127-127) = 1.0
    # Pack 4 E8M0 values per int32: 0x7F7F7F7F
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device=device).contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device=device).contiguous()

    # Output
    D = torch.zeros(M, N, dtype=torch.bfloat16, device=device).contiguous()

    return A, B, SFA, SFB, D

def reference_matmul(A_fp8, B_fp8):
    """Compute reference: float(A) @ float(B)^T"""
    return torch.matmul(A_fp8.float(), B_fp8.float().T)

# ============================================================================
# Run kernel
# ============================================================================

def run_kernel(lib, A, B, SFA, SFB, D, M, N, K):
    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

# ============================================================================
# Correctness check
# ============================================================================

def check_correctness(lib, M=1024, N=1024, K=1024):
    print(f"\n=== Correctness Test: M={M}, N={N}, K={K} ===")

    A, B, SFA, SFB, D = create_test_data(M, N, K)
    ref = reference_matmul(A, B).bfloat16()

    run_kernel(lib, A, B, SFA, SFB, D, M, N, K)

    # Check
    max_err = (D.float() - ref.float()).abs().max().item()
    mean_err = (D.float() - ref.float()).abs().mean().item()

    # Cosine similarity
    d_flat = D.float().flatten()
    r_flat = ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        d_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()

    print(f"  Max error:  {max_err:.4f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Cosine sim: {cos_sim:.6f}")

    # FP8 has low precision, so allow larger errors
    if cos_sim > 0.99:
        print("  ✓ PASSED")
        return True
    else:
        print("  ✗ FAILED")
        # Show some samples
        print("  First 5 elements:")
        print(f"    Kernel: {D.flatten()[:5].tolist()}")
        print(f"    Ref:    {ref.flatten()[:5].tolist()}")
        return False

# ============================================================================
# Benchmark
# ============================================================================

def benchmark(lib, M, N, K, warmup=5, iters=20):
    A, B, SFA, SFB, D = create_test_data(M, N, K)

    # Warmup
    for _ in range(warmup):
        run_kernel(lib, A, B, SFA, SFB, D, M, N, K)

    # Timing
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
    print("\n=== Benchmarks ===")
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

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    # Correctness tests
    all_pass = True
    for m, n, k in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        if not check_correctness(lib, m, n, k):
            all_pass = False

    if all_pass:
        run_benchmarks(lib)
    else:
        print("\nSkipping benchmarks due to correctness failures.")
