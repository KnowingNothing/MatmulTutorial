#!/usr/bin/env python3
"""
Level 11: DeepGEMM Style with Full Optimizations
Supports arbitrary M, N, K dimensions (not just aligned sizes)
"""
import subprocess, ctypes, torch, os, sys

class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

BM, BN, BK = 128, 256, 64  # BM=128 with 2 math warpgroups
libcuda = ctypes.CDLL("libcuda.so.1")
CLUSTER_SIZE = 2
CLUSTER_M = BM * CLUSTER_SIZE

# 128B swizzle for output
TMA_D_BLOCK_N = 64  # 128 bytes / 2 bytes per bf16 = 64 columns

def compile():
    import time
    src = os.path.join(os.path.dirname(__file__), "matmul-v10.cu")
    out = f"/tmp/gemm_{int(time.time())}.so"
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
    """Create TMA descriptor with oobFill=1 (zero fill for out-of-bounds)"""
    tmap = CUtensorMap()
    dims = (ctypes.c_uint64 * 2)(inner, outer)
    strides = (ctypes.c_uint64 * 1)(stride * 2)
    box = (ctypes.c_uint32 * 2)(box_x, box_y)
    elem = (ctypes.c_uint32 * 2)(1, 1)
    # oobFill=1 means zero-fill for out-of-bounds accesses
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(tmap), 9, 2,
        ctypes.c_void_p(tensor.data_ptr()), dims, strides, box, elem, 0, swz, 0, 1)
    return tmap

def create_desc_store(tensor, inner, outer, stride, box_x, box_y, swz=3):
    """Create TMA descriptor for store (no oobFill needed, but need proper bounds)"""
    tmap = CUtensorMap()
    dims = (ctypes.c_uint64 * 2)(inner, outer)
    strides = (ctypes.c_uint64 * 1)(stride * 2)
    box = (ctypes.c_uint32 * 2)(box_x, box_y)
    elem = (ctypes.c_uint32 * 2)(1, 1)
    libcuda.cuTensorMapEncodeTiled(ctypes.byref(tmap), 9, 2,
        ctypes.c_void_p(tensor.data_ptr()), dims, strides, box, elem, 0, swz, 0, 0)
    return tmap

def align_k(k): 
    """K must be aligned to BK for TMA swizzle"""
    return ((k + BK - 1) // BK) * BK

def align_n(n):
    """N stride must be aligned to 64 for TMA 128B swizzle on output"""
    return ((n + 63) // 64) * 64

def test(lib, M, N, K, label=""):
    """Test with arbitrary M, N dimensions (K aligned to BK, C stride aligned to 64)"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # K must be aligned for TMA 128B swizzle
    K_aligned = align_k(K)
    # N stride must be aligned to 64 for TMA 128B swizzle on output
    N_stride = align_n(N)
    
    # A: [M, K_aligned], only [:, :K] has real data
    # B: [N, K_aligned], only [:, :K] has real data  
    # C: [M, N_stride] stride, only [:, :N] is valid output
    A = torch.zeros(M, K_aligned, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(N, K_aligned, dtype=torch.bfloat16, device="cuda")
    A[:, :K] = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B[:, :K] = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # C uses N_stride for aligned stride, but we only care about [:, :N]
    C = torch.zeros(M, N_stride, dtype=torch.bfloat16, device="cuda")
    
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    torch.cuda.synchronize()
    
    # TMA descriptors: A, B use K_aligned; C uses N_stride for inner dim
    dA = create_desc(A, K_aligned, M, A.stride(0), BK, BM, swz=3)
    dB = create_desc(B, K_aligned, N, B.stride(0), BK, BN // 2, swz=3)
    dC = create_desc_store(C, N_stride, M, C.stride(0), TMA_D_BLOCK_N, BM, swz=3)
    
    # Pass actual M, N_stride and aligned K
    lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                          ctypes.c_void_p(C.data_ptr()), M, N_stride, K_aligned)
    torch.cuda.synchronize()
    
    # Compare - use only valid regions
    C_valid = C[:, :N]
    ref = torch.matmul(A[:, :K].float(), B[:, :K].t().float())
    
    cos = torch.nn.functional.cosine_similarity(
        C_valid.float().flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    
    shape_str = f"{M}x{N}x{K}" if label == "" else label
    print(f"  {shape_str}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")
    
    del A, B, C, ref
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return ok

def bench(lib, M, N, K, iters=20, label=""):
    """Benchmark with arbitrary M, N dimensions (K aligned to BK, C stride aligned to 64)"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # K must be aligned for TMA 128B swizzle
    K_aligned = align_k(K)
    # N stride must be aligned to 64 for TMA 128B swizzle on output
    N_stride = align_n(N)
    
    A = torch.randn(M, K_aligned, dtype=torch.bfloat16, device="cuda").contiguous()
    B = torch.randn(N, K_aligned, dtype=torch.bfloat16, device="cuda").contiguous()
    C = torch.zeros(M, N_stride, dtype=torch.bfloat16, device="cuda").contiguous()
    torch.cuda.synchronize()
    
    dA = create_desc(A, K_aligned, M, A.stride(0), BK, BM, swz=3)
    dB = create_desc(B, K_aligned, N, B.stride(0), BK, BN // 2, swz=3)
    dC = create_desc_store(C, N_stride, M, C.stride(0), TMA_D_BLOCK_N, BM, swz=3)
    
    for _ in range(3):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                              ctypes.c_void_p(C.data_ptr()), M, N_stride, K_aligned)
    torch.cuda.synchronize()
    
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        lib.gemm_bf16_launch(ctypes.byref(dA), ctypes.byref(dB), ctypes.byref(dC),
                              ctypes.c_void_p(C.data_ptr()), M, N_stride, K_aligned)
    e.record()
    torch.cuda.synchronize()
    
    ms = s.elapsed_time(e) / iters
    # Report TFLOPS based on actual M, N, K (not aligned)
    tflops = 2.0 * M * N * K / (ms * 1e9)
    
    shape_str = f"{M}x{N}x{K}" if label == "" else label
    print(f"  {shape_str}: {ms:.3f}ms, {tflops:.1f} TFLOPS")
    
    del A, B, C
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    lib = compile()
    print("Compiled OK\n")
    
    print("=== Correctness (aligned sizes) ===")
    for size in [1024, 2048, 4096, 8192]:
        test(lib, size, size, size, f"{size}x{size}")
    
    print("\n=== Correctness (unaligned M, N) ===")
    # Test various unaligned M, N dimensions (K aligned)
    test_cases = [
        (1000, 1000, 1024),    # Unaligned M, N; aligned K
        (2000, 3000, 1536),    # Asymmetric M, N
        (4097, 4097, 4096),    # Just over aligned M, N
        (7777, 7777, 7680),    # Odd M, N; K = 120 * 64
        (8000, 8192, 8192),    # Mixed M; aligned N, K
        (1234, 5678, 2048),    # Random M, N; aligned K
        (100, 200, 128),       # Small unaligned
        (500, 600, 256),       # Medium unaligned
    ]
    for M, N, K in test_cases:
        test(lib, M, N, K)
    
    print("\n=== Benchmark (aligned sizes) ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(lib, size, size, size, label=f"{size}x{size}")
    
    print("\n=== Benchmark (unaligned M, N) ===")
    for M, N, K in [(4097, 4097, 4096), (7777, 7777, 7680), (8000, 8192, 8192)]:
        bench(lib, M, N, K)
