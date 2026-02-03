#!/usr/bin/env python3
"""Level 5: Cluster + TMA Multicast GEMM"""
import os, subprocess, ctypes, torch, gc

class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

BM, BN, BK, CS = 128, 128, 64, 2

def compile_kernel():
    src = os.path.join(os.path.dirname(__file__), "matmul-v5.cu")
    out = "/tmp/gemm_bf16_level5.so"
    print("Compiling...")
    subprocess.run(["nvcc", "--shared", "-Xcompiler", "-fPIC", "-std=c++17",
                    "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-lcuda",
                    src, "-o", out], check=True, capture_output=True)
    print("OK")
    return ctypes.CDLL(out)

def create_desc(t, inner, outer, stride):
    libcuda = ctypes.CDLL("libcuda.so.1")
    m = CUtensorMap()
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(m), 9, 2, ctypes.c_void_p(t.data_ptr()),
        (ctypes.c_uint64*2)(inner, outer), (ctypes.c_uint64*1)(stride*2),
        (ctypes.c_uint32*2)(64, 64), (ctypes.c_uint32*2)(1, 1), 0, 3, 3, 0)
    return m

def test(lib, size):
    M = ((size+BM*CS-1)//(BM*CS))*(BM*CS)
    N = ((size+BN-1)//BN)*BN
    K = ((size+BK-1)//BK)*BK
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    lib.gemm_bf16_launch(ctypes.byref(create_desc(A,K,M,A.stride(0))),
                         ctypes.byref(create_desc(B,K,N,B.stride(0))),
                         ctypes.c_void_p(C.data_ptr()), M, N, K)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        C.flatten().unsqueeze(0), torch.matmul(A.float(), B.t().float()).flatten().unsqueeze(0)).item()
    ok = cos > 0.99
    print(f"  {size}x{size}: cos={cos:.6f} {'PASSED' if ok else 'FAILED'}")
    del A, B, C; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return ok

def bench(lib, size, iters=20):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    M = ((size+BM*CS-1)//(BM*CS))*(BM*CS)
    N = ((size+BN-1)//BN)*BN
    K = ((size+BK-1)//BK)*BK
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    dA, dB = create_desc(A, K, M, A.stride(0)), create_desc(B, K, N, B.stride(0))
    # Warmup with sync after each
    for _ in range(3):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.c_void_p(C.data_ptr()), M, N, K)
        torch.cuda.synchronize()
    # Benchmark
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.c_void_p(C.data_ptr()), M, N, K)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {size}x{size}: {ms:.3f}ms, {2.0*M*N*K/(ms*1e9):.1f} TFLOPS")
    del A, B, C, dA, dB; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

if __name__ == "__main__":
    print("Level 6: Cluster + TMA Multicast GEMM")
    print(f"Block: {BM}x{BN}x{BK}, Cluster: {CS}x1x1")
    lib = compile_kernel()
    print("\n=== Correctness ===")
    if all(test(lib, s) for s in [512, 1024, 2048, 4096]):
        print("\n=== Benchmark ===")
        bench(lib, 4096, 20)
        bench(lib, 8192, 10)
