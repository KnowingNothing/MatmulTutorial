#!/usr/bin/env python3
"""
Level 3: Warp Specialization GEMM Test

Architecture:
- 256 threads = 2 warpgroups
- Warpgroup 0 (threads 0-127): Producer - TMA loads
- Warpgroup 1 (threads 128-255): Consumer - WGMMA compute
- Synchronization via full/empty barriers
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

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64


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
        self.libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        self.libcuda.cuGetErrorString.restype = CUresult
    
    def check_error(self, err, name):
        if err != 0:
            s = ctypes.c_char_p()
            self.libcuda.cuGetErrorString(err, ctypes.byref(s))
            raise RuntimeError(name + " failed: " + (s.value.decode() if s.value else str(err)))
    
    def create_k_major_desc(self, tensor, shape_k, shape_mn, outer_stride):
        elem_size = tensor.element_size()
        tmap = CUtensorMap()
        
        gmem_inner = shape_k
        gmem_outer = shape_mn
        smem_inner = 64
        smem_outer = BLOCK_M
        
        gmem_dims = (cuuint64_t * 2)(gmem_inner, gmem_outer)
        gmem_strides = (cuuint64_t * 1)(outer_stride * elem_size)
        smem_dims = (cuuint32_t * 2)(smem_inner, smem_outer)
        elem_strides = (cuuint32_t * 2)(1, 1)
        
        err = self.libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tmap), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            CUdeviceptr(tensor.data_ptr()), gmem_dims, gmem_strides,
            smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
        self.check_error(err, "cuTensorMapEncodeTiled")
        return tmap


def compile_kernel():
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(kernel_dir, "matmul-v3.cu")
    so_path = "/tmp/gemm_bf16_level3.so"
    
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
    
    lib.gemm_bf16_launch_cooperative.argtypes = [
        ctypes.POINTER(CUtensorMap), ctypes.POINTER(CUtensorMap),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.gemm_bf16_launch_cooperative.restype = None
    
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


def test_correctness(lib, M=1024, N=1024, K=1024, use_cooperative=False):
    print("\n=== Correctness Test: {}x{}x{} (cooperative={}) ===".format(M, N, K, use_cooperative))
    
    tma = TMAHelper()
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B_t = B.t().contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    desc_A = tma.create_k_major_desc(A, K, M, A.stride(0))
    desc_B = tma.create_k_major_desc(B_t, K, N, B_t.stride(0))
    
    launch_fn = lib.gemm_bf16_launch_cooperative if use_cooperative else lib.gemm_bf16_launch
    launch_fn(
        ctypes.byref(desc_A), ctypes.byref(desc_B),
        ctypes.c_void_p(C.data_ptr()), M, N, K
    )
    torch.cuda.synchronize()
    
    C_ref = torch.matmul(A.float(), B.float())
    
    max_err = (C - C_ref).abs().max().item()
    mean_err = (C - C_ref).abs().mean().item()
    cos_sim = cosine_similarity(C, C_ref)
    passed = cos_sim > 0.999
    
    print("  max_err={:.6f}, mean_err={:.6f}, cos_sim={:.6f}".format(max_err, mean_err, cos_sim))
    print("  Result: " + ("PASSED" if passed else "FAILED"))
    
    if not passed:
        print("  Debug: C[0,0:4] = {}".format(C[0, 0:4].tolist()))
        print("  Debug: C_ref[0,0:4] = {}".format(C_ref[0, 0:4].tolist()))
    
    return passed, max_err, cos_sim


def benchmark(lib, M=4096, N=4096, K=4096, warmup=10, iters=100):
    print("\n=== Benchmark: {}x{}x{} ===".format(M, N, K))
    
    tma = TMAHelper()
    
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    B_t = B.t().contiguous()
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')
    
    desc_A = tma.create_k_major_desc(A, K, M, A.stride(0))
    desc_B = tma.create_k_major_desc(B_t, K, N, B_t.stride(0))
    
    ms = lib.benchmark_kernel(
        ctypes.byref(desc_A), ctypes.byref(desc_B),
        ctypes.c_void_p(C.data_ptr()), M, N, K, warmup, iters
    )
    
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print("  Time: {:.3f}ms, Throughput: {:.1f} TFLOPS".format(ms, tflops))
    
    return ms, tflops


def main():
    print("Level 4: Warp Specialization GEMM")
    print("=" * 50)
    print("256 threads: Producer WG (TMA) + Consumer WG (WGMMA)")
    
    lib = compile_kernel()
    
    # Test regular launch
    test_correctness(lib, M=256, N=256, K=256)
    test_correctness(lib, M=1024, N=1024, K=1024)
    passed, _, _ = test_correctness(lib, M=4096, N=4096, K=4096)
    
    # Test cooperative launch
    test_correctness(lib, M=1024, N=1024, K=1024, use_cooperative=True)
    
    if passed:
        benchmark(lib, M=4096, N=4096, K=4096)
        benchmark(lib, M=8192, N=8192, K=8192)


if __name__ == "__main__":
    main()
