#!/usr/bin/env python3
"""
Level 6: m64n256k16 + Cluster Multicast
BM=128, BN=256, BK=64 per CTA
Cluster 2x1x1: 2 CTAs share B (effective 256x256 tile)
"""
import subprocess, ctypes, torch, os

class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

BM, BN, BK = 128, 256, 64
CLUSTER_SIZE = 2
CLUSTER_M = BM * CLUSTER_SIZE  # 256

def compile():
    src = os.path.join(os.path.dirname(__file__), "matmul-v6.cu")
    out = "/tmp/gemm_bf16_level6.so"
    r = subprocess.run([
        "nvcc", "--shared", "-Xcompiler", "-fPIC", "-std=c++17",
        "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-lcuda",
        "--ptxas-options=-v", src, "-o", out
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("Compile Error:", r.stderr)
        raise RuntimeError("Compile failed")
    for line in r.stderr.split('\n'):
        if 'registers' in line.lower() or 'spill' in line.lower():
            print(line)
    return ctypes.CDLL(out)

def create_desc_a(tensor, inner, outer, stride):
    """A: box = (64, 64) for each 64x64 tile load"""
    libcuda = ctypes.CDLL("libcuda.so.1")
    tmap = CUtensorMap()
    dims = (ctypes.c_uint64 * 2)(inner, outer)
    strides = (ctypes.c_uint64 * 1)(stride * 2)
    box = (ctypes.c_uint32 * 2)(BK, 64)  # Load 64x64 tiles
    elem = (ctypes.c_uint32 * 2)(1, 1)
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(tmap), 9, 2,
        ctypes.c_void_p(tensor.data_ptr()), dims, strides, box, elem, 0, 3, 3, 0)
    return tmap

def create_desc_b(tensor, inner, outer, stride):
    """B: box = (64, 256) for full BK x BN tile"""
    libcuda = ctypes.CDLL("libcuda.so.1")
    tmap = CUtensorMap()
    dims = (ctypes.c_uint64 * 2)(inner, outer)
    strides = (ctypes.c_uint64 * 1)(stride * 2)
    box = (ctypes.c_uint32 * 2)(BK, BN)  # 64 x 256
    elem = (ctypes.c_uint32 * 2)(1, 1)
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(tmap), 9, 2,
        ctypes.c_void_p(tensor.data_ptr()), dims, strides, box, elem, 0, 3, 3, 0)
    return tmap

def align(s, b): return ((s + b - 1) // b) * b

def test(lib, size):
    M = align(size, CLUSTER_M)
    N = align(size, BN)
    K = align(size, BK)
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    dA = create_desc_a(A, K, M, A.stride(0))
    dB = create_desc_b(B, K, N, B.stride(0))
    
    lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), 
                          ctypes.c_void_p(C.data_ptr()), M, N, K)
    torch.cuda.synchronize()
    
    ref = torch.matmul(A.float(), B.t().float())
    cos = torch.nn.functional.cosine_similarity(
        C.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.99
    print(f"  {size}x{size}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")
    return ok

def bench(lib, size, iters=20):
    M = align(size, CLUSTER_M)
    N = align(size, BN)
    K = align(size, BK)
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    dA = create_desc_a(A, K, M, A.stride(0))
    dB = create_desc_b(B, K, N, B.stride(0))
    
    for _ in range(3):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), 
                              ctypes.c_void_p(C.data_ptr()), M, N, K)
    torch.cuda.synchronize()
    
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), 
                              ctypes.c_void_p(C.data_ptr()), M, N, K)
    e.record()
    torch.cuda.synchronize()
    
    ms = s.elapsed_time(e) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print(f"  {size}x{size}: {ms:.3f}ms, {tflops:.1f} TFLOPS")

if __name__ == "__main__":
    print("Level 7: m64n256k16 + Cluster Multicast")
    print(f"BM={BM}, BN={BN}, BK={BK}, Cluster={CLUSTER_SIZE}x1x1")
    
    lib = compile()
    print("Compiled OK\n")
    
    print("=== Correctness ===")
    if all(test(lib, s) for s in [512, 1024, 2048, 4096]):
        print("\n=== Benchmark ===")
        bench(lib, 4096)
        bench(lib, 8192)
