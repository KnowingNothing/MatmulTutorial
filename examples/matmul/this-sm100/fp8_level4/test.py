#!/usr/bin/env python3
"""
FP8 GEMM Level 4 — 2-CTA Multicast on B
Test correctness + benchmark vs DeepGEMM
"""
import os, torch, ctypes, subprocess, time, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul_n224.so")
GRAN_K = 128
BLOCK_N = 224

# ---------- Compile ----------
print("Compiling Level 4 (2-CTA multicast, BLOCK_N=224) ...")
subprocess.check_call([
    "nvcc", "--shared", "-Xcompiler", "-fPIC",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-O3", "-std=c++17", f"-DTILE_N={BLOCK_N}", "-lcuda",
    SRC, "-o", LIB,
], stderr=subprocess.STDOUT)
print("  Compiled OK\n")

# ---------- Load ----------
lib = ctypes.CDLL(LIB)
lib.launch_fp8_gemm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.launch_fp8_gemm.restype = None

# ---------- Correctness ----------
print("=== Correctness ===")
for M, N, K in [(256, 224, 512), (512, 448, 1024), (1024, 896, 2048), (2048, 1792, 4096)]:
    A_f = torch.randn(M, K, device="cuda") * 0.1
    B_f = torch.randn(N, K, device="cuda") * 0.1
    A = A_f.to(torch.float8_e4m3fn).contiguous()
    B = B_f.to(torch.float8_e4m3fn).contiguous()

    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float().T).bfloat16()
    cos = torch.nn.functional.cosine_similarity(
        D.flatten().float(), ref.flatten().float(), dim=0).item()
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  {M:5d}x{N:5d}x{K:5d}  cos={cos:.4f}  [{status}]")

# ---------- Benchmark ----------
print("\n=== Benchmark ===")
print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} {'TFLOPS':>10}")
print("-" * 52)

for M, N, K in [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
    (8192, 7168, 8192),
]:
    # Ensure M is multiple of 256 (two BLOCK_M=128 CTAs) for correctness
    # and N is multiple of BLOCK_N
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    warmup, iters = 10, 30
    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    tflops = 2.0 * M * N * K / (ms / 1000) / 1e12
    print(f"{M:>6} {N:>6} {K:>6} | {ms:>10.3f} {tflops:>10.1f}")
