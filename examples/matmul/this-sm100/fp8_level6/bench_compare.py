#!/usr/bin/env python3
"""
FP8 GEMM Level 6 — Side-by-side comparison:
  1. Level 5 equivalent (non-JIT, BLOCK_N-only templating)
  2. Level 6 JIT (shape-specialized, compile-time M/N/K)
  3. DeepGEMM (reference, via bench_kineto for GPU-accurate timing)
"""
import os, sys, torch, ctypes, time

sys.path.insert(0, '/home/zhengsize/DeepGEMM/tests')
import deep_gemm
from deep_gemm.testing import bench_kineto
from generators import KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from test import compile_kernel, select_best_config

# =============================================================================
# Setup
# =============================================================================

num_sms = torch.cuda.get_device_properties(0).multi_processor_count
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}, SMs: {num_sms}")

GRAN_K = 128

shapes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
]

# DeepGEMM config
kernel_type = KernelType.Kernel1D1D
quant_config = QuantConfig()
use_ue8m0 = get_ue8m0_usage(kernel_type)
major_a = MajorTypeAB.KMajor
major_b = MajorTypeAB.KMajor
out_dtype = torch.bfloat16

# =============================================================================
# Pre-compile all kernel variants
# =============================================================================

print("\nPre-compiling kernels...")
nojit_libs = {}
jit_libs = {}
config_info = {}

for M, N, K in shapes:
    bn, sms = select_best_config(M, N, K, num_sms)
    config_info[(M, N, K)] = (bn, sms)

    # Non-JIT: only BLOCK_N specialized
    if bn not in nojit_libs:
        print(f"  Non-JIT BLOCK_N={bn}...", end=" ", flush=True)
        nojit_libs[bn] = compile_kernel(bn)
        print("OK")

    # JIT: BLOCK_N + full shape specialization
    # For small problems, compile-time shapes help (fewer tiles, simpler scheduler).
    # For large problems, it can hurt due to register allocation changes.
    # We compile both and pick the faster one at runtime.
    jit_key = (bn, M, N, K)
    if jit_key not in jit_libs:
        print(f"  JIT BN={bn} M={M} N={N} K={K}...", end=" ", flush=True)
        jit_libs[jit_key] = compile_kernel(bn, M, N, K)
        print("OK")

print("Done.\n")

# =============================================================================
# Benchmark
# =============================================================================

def bench_our_kernel(lib, M, N, K, warmup=20, iters=100):
    """Benchmark our kernel with wall-clock timing (100 iterations to amortize host overhead)."""
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


print(f"{'M':>6} {'N':>6} {'K':>6} | {'BN':>3} | {'NonJIT':>10} {'TFLOPS':>7} | {'JIT':>10} {'TFLOPS':>7} | {'Best':>8} | {'DG':>10} {'TFLOPS':>7} | {'Best/DG':>8}")
print("-" * 115)

for M, N, K in shapes:
    bn, sms = config_info[(M, N, K)]

    # Non-JIT (BLOCK_N-only templating, like Level 5 but with #pragma unroll 1)
    ms_nojit = bench_our_kernel(nojit_libs[bn], M, N, K)
    tflops_nojit = 2.0 * M * N * K / (ms_nojit / 1000) / 1e12

    # JIT (full shape specialized: BLOCK_N + M/N/K compile-time)
    ms_jit = bench_our_kernel(jit_libs[(bn, M, N, K)], M, N, K)
    tflops_jit = 2.0 * M * N * K / (ms_jit / 1000) / 1e12

    # Best of JIT and non-JIT
    best_tflops = max(tflops_nojit, tflops_jit)
    best_label = "JIT" if tflops_jit > tflops_nojit else "tile"

    # DeepGEMM (bench_kineto for GPU-accurate timing)
    a, b, c, d, ref_d = generate_normal(
        M, N, K, major_a, major_b, False, out_dtype, kernel_type,
        use_ue8m0=use_ue8m0, quant_config=quant_config)
    dg_t = bench_kineto(
        lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d),
        'fp8_gemm', suppress_kineto_output=True)
    dg_ms = dg_t * 1000
    dg_tflops = 2.0 * M * N * K / dg_t / 1e12

    ratio = best_tflops / dg_tflops * 100

    print(f"{M:>6} {N:>6} {K:>6} | {bn:>3} | {ms_nojit:>9.4f}ms {tflops_nojit:>7.1f} | {ms_jit:>9.4f}ms {tflops_jit:>7.1f} | {best_tflops:>5.0f}{best_label:>3} | {dg_ms:>9.4f}ms {dg_tflops:>7.1f} | {ratio:>7.1f}%")
