#!/usr/bin/env python3
"""
Level 4: TMA Multicast GEMM Test

Architecture:
- Block size: BM=128, BN=256, BK=64
- Cluster size: 2x1x1 (2 CTAs share B matrix)
- TMA multicast: B matrix broadcast
- WGMMA: m64n256k16
"""
import torch
import ctypes
import subprocess
import os

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_void_p
cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64

class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]

CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 0x09
CU_TENSOR_MAP_INTERLEAVE_NONE = 0x00
CU_TENSOR_MAP_SWIZZLE_128B = 0x03
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 0x03
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0x00

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64


class TMAHelper:
    def __init__(self):
        self.libcuda = ctypes.CDLL("libcuda.so.1")
        self.libcuda.cuTensorMapEncodeTiled.argtypes = [
            ctypes.POINTER(CUtensorMap), ctypes.c_int, ctypes.c_uint32,
            CUdeviceptr, ctypes.POINTER(cuuint64_t), ctypes.POINTER(cuuint64_t),
            ctypes.POINTER(cuuint32_t), ctypes.POINTER(cuuint32_t),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.libcuda.cuTensorMapEncodeTiled.restype = CUresult
    
    def create_desc_a(self, tensor, K, M, stride):
        """Create TMA descriptor for A (M x K), K-contiguous"""
        tmap = CUtensorMap()
        elem_size = tensor.element_size()
        
        gmem_dims = (cuuint64_t * 2)(K, M)
        gmem_strides = (cuuint64_t * 1)(stride * elem_size)
        smem_dims = (cuuint32_t * 2)(64, 64)  # 64x64 TMA box
        elem_strides = (cuuint32_t * 2)(1, 1)
        
        err = self.libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tmap), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            CUdeviceptr(tensor.data_ptr()), gmem_dims, gmem_strides,
            smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
        if err != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled A failed: {err}")
        return tmap
    
    def create_desc_b(self, tensor, K, N, stride):
        """Create TMA descriptor for B_t (N x K), K-contiguous"""
        tmap = CUtensorMap()
        elem_size = tensor.element_size()
        
        gmem_dims = (cuuint64_t * 2)(K, N)
        gmem_strides = (cuuint64_t * 1)(stride * elem_size)
        smem_dims = (cuuint32_t * 2)(64, 64)  # 64x64 TMA box
        elem_strides = (cuuint32_t * 2)(1, 1)
        
        err = self.libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tmap), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            CUdeviceptr(tensor.data_ptr()), gmem_dims, gmem_strides,
            smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
        if err != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled B failed: {err}")
        return tmap


def compile_kernel():
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(kernel_dir, "matmul-v4.cu")
    so_path = "/tmp/gemm_bf16_level4.so"
    
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC", "-std=c++17",
           "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-lcuda",
           kernel_path, "-o", so_path]
    print("Compiling:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:", result.stderr)
        raise RuntimeError("nvcc failed")
    print("OK")
    
    lib = ctypes.CDLL(so_path)
    lib.gemm_bf16_launch.argtypes = [
        ctypes.POINTER(CUtensorMap), ctypes.POINTER(CUtensorMap),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.gemm_bf16_launch.restype = None
    
    lib.benchmark_kernel.argtypes = [
        ctypes.POINTER(CUtensorMap), ctypes.POINTER(CUtensorMap),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]
    lib.benchmark_kernel.restype = ctypes.c_float
    
    return lib


def cosine_similarity(x, y):
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    return torch.nn.functional.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).item()


def test_correctness(lib, M=1024, N=1024, K=1024):
    print(f"\n=== Correctness Test: {M}x{N}x{K} ===")
    
    # Ensure M and N are divisible by block sizes
    M = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    N = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    print(f"  Adjusted size: {M}x{N}x{K}")
    
    tma = TMAHelper()
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B_t = B.t().contiguous()  # N x K
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    desc_A = tma.create_desc_a(A, K, M, A.stride(0))
    desc_B = tma.create_desc_b(B_t, K, N, B_t.stride(0))
    
    lib.gemm_bf16_launch(
        ctypes.byref(desc_A), ctypes.byref(desc_B),
        ctypes.c_void_p(C.data_ptr()), M, N, K
    )
    torch.cuda.synchronize()
    
    C_ref = torch.matmul(A.float(), B.float())
    
    max_err = (C - C_ref).abs().max().item()
    mean_err = (C - C_ref).abs().mean().item()
    cos_sim = cosine_similarity(C, C_ref)
    passed = cos_sim > 0.999
    
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    
    if not passed:
        print(f"  Debug: C[0,0:4] = {C[0, 0:4].tolist()}")
        print(f"  Debug: C_ref[0,0:4] = {C_ref[0, 0:4].tolist()}")
    
    return passed, max_err, cos_sim


def benchmark(lib, M=4096, N=4096, K=4096, warmup=10, iters=100):
    # Adjust to be divisible
    M = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    N = ((N + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    
    print(f"\n=== Benchmark: {M}x{N}x{K} ===")
    
    tma = TMAHelper()
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B_t = B.t().contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    desc_A = tma.create_desc_a(A, K, M, A.stride(0))
    desc_B = tma.create_desc_b(B_t, K, N, B_t.stride(0))
    
    ms = lib.benchmark_kernel(
        ctypes.byref(desc_A), ctypes.byref(desc_B),
        ctypes.c_void_p(C.data_ptr()), M, N, K, warmup, iters
    )
    
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"  Time: {ms:.3f}ms, Throughput: {tflops:.1f} TFLOPS")
    
    return ms, tflops


def main():
    print("Level 5: TMA Multicast GEMM")
    print("=" * 50)
    print(f"Block: BM={BLOCK_M}, BN={BLOCK_N}, BK={BLOCK_K}")
    print("Cluster: 2x1x1 (TMA multicast B)")
    print("WGMMA: m64n256k16")
    
    lib = compile_kernel()
    
    # Correctness tests
    passed, _, _ = test_correctness(lib, M=512, N=512, K=512)
    if passed:
        test_correctness(lib, M=1024, N=1024, K=1024)
        test_correctness(lib, M=4096, N=4096, K=4096)
        
        # Benchmarks
        benchmark(lib, M=4096, N=4096, K=4096)
        benchmark(lib, M=8192, N=8192, K=8192)


if __name__ == "__main__":
    main()
