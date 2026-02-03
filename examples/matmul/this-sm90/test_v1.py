#!/usr/bin/env python3
"""Test script for Level 1 WGMMA Basic kernel."""
import torch
import ctypes
import subprocess
import os

LEVEL = "level1"
KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(KERNEL_DIR, "matmul-v1.cu")
SO_PATH = f"/tmp/gemm_bf16_{LEVEL}.so"

# Unified test/benchmark size
TEST_SIZE = 8192
BENCH_SIZE = 8192

def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_90a,code=sm_90a", "-O3",
           KERNEL_PATH, "-o", SO_PATH]
    print(f"Compiling: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("OK")

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    return torch.nn.functional.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).item()

def test_correctness(M=TEST_SIZE, N=TEST_SIZE, K=TEST_SIZE, atol=1e-2, rtol=1e-2):
    lib = ctypes.CDLL(SO_PATH)
    lib.gemm_bf16_launch.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    lib.gemm_bf16_launch(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    
    C_ref = torch.matmul(A.float(), B.float())
    
    max_err = (C - C_ref).abs().max().item()
    mean_err = (C - C_ref).abs().mean().item()
    cos_sim = cosine_similarity(C, C_ref)
    passed = torch.allclose(C, C_ref, atol=atol, rtol=rtol)
    
    print(f"Correctness ({M}x{N}x{K}): max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")
    print(f"  allclose(atol={atol}, rtol={rtol})={'PASSED' if passed else 'FAILED'}")
    
    return passed, max_err, cos_sim

def benchmark(M=BENCH_SIZE, N=BENCH_SIZE, K=BENCH_SIZE, warmup=10, iters=100):
    lib = ctypes.CDLL(SO_PATH)
    lib.benchmark_kernel.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*5
    lib.benchmark_kernel.restype = ctypes.c_float
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    ms = lib.benchmark_kernel(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, warmup, iters)
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"Benchmark ({M}x{N}x{K}): {ms:.3f}ms, {tflops:.1f} TFLOPS")
    return ms, tflops

if __name__ == "__main__":
    compile_kernel()
    test_correctness()
    benchmark()
