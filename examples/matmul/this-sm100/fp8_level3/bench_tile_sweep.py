#!/usr/bin/env python3
"""Sweep BLOCK_N values to find optimal tile size."""
import os, torch, ctypes, time, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
GRAN_K = 128

def compile_kernel(block_n):
    lib_path = os.path.join(SCRIPT_DIR, f"matmul_n{block_n}.so")
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a",
           "-O3", "-std=c++17", f"-DTILE_N={block_n}", "-lcuda",
           SRC, "-o", lib_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p]*5 + [ctypes.c_int]*3
    lib.launch_fp8_gemm.restype = None
    return lib

def create_data(M, N, K, bn):
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    return A, B, SFA, SFB, D

def bench(lib, M, N, K, bn, warmup=10, iters=30):
    A, B, SFA, SFB, D = create_data(M, N, K, bn)
    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    return 2.0 * M * N * K / (ms / 1000) / 1e12

test_sizes = [
    (4096, 7168, 4096),
    (8192, 7168, 8192),
]

tile_ns = [128, 160, 192, 224, 256]

print(f"{'BN':>4} | ", end="")
for M, N, K in test_sizes:
    print(f"  {M}x{N}x{K}", end="")
print()
print("-" * 60)

for bn in tile_ns:
    try:
        lib = compile_kernel(bn)
        print(f"{bn:>4} | ", end="")
        for M, N, K in test_sizes:
            tf = bench(lib, M, N, K, bn)
            print(f"  {tf:10.0f}", end="")
        print()
    except Exception as e:
        print(f"{bn:>4} | ERROR: {e}")
