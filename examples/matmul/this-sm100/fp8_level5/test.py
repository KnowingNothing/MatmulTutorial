#!/usr/bin/env python3
"""
FP8 GEMM Level 5 — Dynamic BLOCK_N + 2-CTA Multicast
Compiles multiple BLOCK_N variants, selects best per problem size.

Implements the same heuristic as DeepGEMM:
  1. Minimize the number of SM waves
  2. Among equal-wave configs, maximize last-wave SM utilization
  3. Among equal-util configs, prefer smaller BLOCK_N (less wasted compute)
"""
import os, sys, torch, ctypes, subprocess, time, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
GRAN_K = 128
NUM_MULTICAST = 2

# =============================================================================
# Compile multiple BLOCK_N variants
# =============================================================================

# Candidate BLOCK_N values (must be even for multicast on B, LOAD_BLOCK_N = N/2)
# BLOCK_N=256 excluded: TMEM would need 2*1*256 + 4 + 4 = 524 > 512
BLOCK_N_CANDIDATES = [64, 96, 128, 192, 224]

libs = {}
print("=== Compiling BLOCK_N variants ===")
for bn in BLOCK_N_CANDIDATES:
    lib_path = os.path.join(SCRIPT_DIR, f"matmul_n{bn}.so")
    print(f"  Compiling BLOCK_N={bn} ...", end=" ", flush=True)
    subprocess.check_call([
        "nvcc", "--shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3", "-std=c++17", f"-DTILE_N={bn}", "-lcuda",
        SRC, "-o", lib_path,
    ], stderr=subprocess.STDOUT)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fp8_gemm.restype = None
    libs[bn] = lib
    print("OK")
print()

# =============================================================================
# Get number of SMs
# =============================================================================
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
print(f"GPU: {torch.cuda.get_device_name(0)}, SMs: {num_sms}")

# =============================================================================
# DeepGEMM-style heuristic for BLOCK_N selection
# =============================================================================

def is_block_size_legal(block_n, block_m=128):
    """Check TMEM and basic constraints."""
    if block_n % 16 != 0:
        return False
    # TMEM check: 2 * block_n + sf_cols <= 512
    sf_block_m = ((block_m + 127) // 128) * 128
    sf_block_n = ((block_n + 127) // 128) * 128
    tmem_cols = 2 * block_n + sf_block_m // 32 + sf_block_n // 32
    if tmem_cols > 512:
        return False
    return True

def select_best_block_n(M, N, K, block_m=128):
    """
    Select BLOCK_N that minimizes waves, then maximizes last-wave utilization.
    Matches DeepGEMM's get_best_config heuristic.
    """
    best_bn = None
    best_waves = float('inf')
    best_last_util = 0

    for bn in BLOCK_N_CANDIDATES:
        if not is_block_size_legal(bn, block_m):
            continue

        num_m_blocks = (M + block_m - 1) // block_m
        num_n_blocks = (N + bn - 1) // bn
        total_tiles = num_m_blocks * num_n_blocks

        # With 2-CTA clusters, effective SMs = num_sms (each cluster uses 2 SMs)
        waves = (total_tiles + num_sms - 1) // num_sms
        last_wave_blocks = total_tiles % num_sms
        last_util = last_wave_blocks if last_wave_blocks > 0 else num_sms

        # Multicast requires even num_m_blocks
        if num_m_blocks % NUM_MULTICAST != 0:
            continue

        if best_bn is None or waves < best_waves:
            best_bn = bn
            best_waves = waves
            best_last_util = last_util
        elif waves == best_waves:
            if last_util > best_last_util:
                best_bn = bn
                best_waves = waves
                best_last_util = last_util
            elif last_util == best_last_util:
                # Same waves and util: prefer smaller block_n (less waste)
                if bn < best_bn:
                    best_bn = bn
                    best_waves = waves
                    best_last_util = last_util

    return best_bn if best_bn is not None else 224

def run_gemm(lib, M, N, K, block_n):
    """Run GEMM with the given library variant."""
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    return A, B, D

# =============================================================================
# Correctness
# =============================================================================
print("\n=== Correctness ===")
for bn in BLOCK_N_CANDIDATES:
    lib = libs[bn]
    sizes = [
        (256, max(bn, 256), 512),
        (512, max(bn, 256), 1024),
        (1024, max(bn*2, 512), 2048),
    ]
    for M, N, K in sizes:
        # Ensure M is multiple of 256 (for 2-CTA with BLOCK_M=128)
        # and N is a reasonable multiple
        A, B, D = run_gemm(lib, M, N, K, bn)
        ref = (A.float() @ B.float().T).bfloat16()
        cos = torch.nn.functional.cosine_similarity(
            D.flatten().float(), ref.flatten().float(), dim=0).item()
        status = "PASS" if cos > 0.99 else "FAIL"
        print(f"  BLOCK_N={bn:3d}  {M:5d}x{N:5d}x{K:5d}  cos={cos:.4f}  [{status}]")

# =============================================================================
# Benchmark with dynamic BLOCK_N selection
# =============================================================================
print("\n=== Benchmark (Dynamic BLOCK_N) ===")
print(f"{'M':>6} {'N':>6} {'K':>6} | {'BN':>4} {'Time(ms)':>10} {'TFLOPS':>10}")
print("-" * 56)

benchmark_sizes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
    (8192, 7168, 8192),
]

for M, N, K in benchmark_sizes:
    bn = select_best_block_n(M, N, K)
    lib = libs[bn]

    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    warmup, iters = 10, 50
    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    tflops = 2.0 * M * N * K / (ms / 1000) / 1e12
    print(f"{M:>6} {N:>6} {K:>6} | {bn:>4} {ms:>10.3f} {tflops:>10.1f}")

# =============================================================================
# Also benchmark all BLOCK_N for comparison on 4096^3
# =============================================================================
print("\n=== BLOCK_N Sweep on 4096x4096x4096 ===")
M, N, K = 4096, 4096, 4096
for bn in BLOCK_N_CANDIDATES:
    lib = libs[bn]
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    warmup, iters = 10, 50
    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    tflops = 2.0 * M * N * K / (ms / 1000) / 1e12
    print(f"  BLOCK_N={bn:3d}: {ms:.3f} ms  {tflops:.1f} TFLOPS")
