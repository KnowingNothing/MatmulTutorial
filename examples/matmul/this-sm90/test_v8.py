#!/usr/bin/env python3
"""
Level 8: TMA Store with BF16 Output
"""
import subprocess, ctypes, torch, os, sys

class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

BM, BN, BK = 128, 256, 64
libcuda = ctypes.CDLL("libcuda.so.1")
CLUSTER_SIZE = 2
CLUSTER_M = BM * CLUSTER_SIZE

def compile():
    import time
    src = os.path.join(os.path.dirname(__file__), "matmul-v8.cu")
    out = f"/tmp/gemm_bf16_level8_{int(time.time())}.so"
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
    lib = ctypes.CDLL(out)
    return lib

def create_desc(tensor, inner, outer, stride, box_x, box_y, swz=3):
    """Create TMA descriptor with explicit swizzle parameter"""
    tmap = CUtensorMap()
    dims = (ctypes.c_uint64 * 2)(inner, outer)
    strides = (ctypes.c_uint64 * 1)(stride * 2)  # bf16 = 2 bytes
    box = (ctypes.c_uint32 * 2)(box_x, box_y)
    elem = (ctypes.c_uint32 * 2)(1, 1)
    # Parameters: type=BF16(9), rank=2, global_addr, dims, strides, box, elem_strides,
    #             interleave=0, swizzle, l2_promotion=0, oob_fill=0
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(tmap), 9, 2,
        ctypes.c_void_p(tensor.data_ptr()), dims, strides, box, elem, 0, swz, 0, 0)
    return tmap

def align(s, b): return ((s + b - 1) // b) * b

def test(lib, size):
    """Test correctness for a given size"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    M = align(size, CLUSTER_M)
    N = align(size, BN)
    K = align(size, BK)
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous()
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").contiguous()
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    torch.cuda.synchronize()
    
    # Create TMA descriptors
    dA = create_desc(A, K, M, A.stride(0), BK, BM, swz=3)
    dB = create_desc(B, K, N, B.stride(0), BK, BN // 2, swz=3)
    dC = create_desc(C, N, M, C.stride(0), BN, 64, swz=0)
    
    # Run kernel
    lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                          ctypes.c_void_p(C.data_ptr()), M, N, K)
    torch.cuda.synchronize()
    
    # Check correctness
    ref = torch.matmul(A.float(), B.t().float())
    cos = torch.nn.functional.cosine_similarity(
        C.float().flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    print(f"  {size}x{size}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")
    
    del A, B, C, ref
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return ok

def bench(lib, size, iters=20):
    """Benchmark for a given size"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    M = align(size, CLUSTER_M)
    N = align(size, BN)
    K = align(size, BK)
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous()
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").contiguous()
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    torch.cuda.synchronize()
    
    # Create TMA descriptors
    dA = create_desc(A, K, M, A.stride(0), BK, BM, swz=3)
    dB = create_desc(B, K, N, B.stride(0), BK, BN // 2, swz=3)
    dC = create_desc(C, N, M, C.stride(0), BN, 64, swz=0)
    
    # Warmup
    for _ in range(3):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                              ctypes.c_void_p(C.data_ptr()), M, N, K)
    torch.cuda.synchronize()
    
    # Benchmark
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                              ctypes.c_void_p(C.data_ptr()), M, N, K)
    e.record()
    torch.cuda.synchronize()
    
    ms = s.elapsed_time(e) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print(f"  {size}x{size}: {ms:.3f}ms, {tflops:.1f} TFLOPS")
    
    del A, B, C
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Level 9: TMA Store with BF16 Output")
    print(f"BM={BM}, BN={BN}, BK={BK}, Cluster={CLUSTER_SIZE}x1x1")
    print("Improvement: TMA store, BF16 output\n")
    
    lib = compile()
    print("Compiled OK\n")
    
    print("=== Correctness ===")
    for size in [1024, 2048, 4096, 8192]:
        test(lib, size)
    
    print("\n=== Benchmark ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(lib, size)
