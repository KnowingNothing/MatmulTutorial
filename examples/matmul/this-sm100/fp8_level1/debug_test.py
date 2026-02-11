#!/usr/bin/env python3
"""Debug test for FP8 GEMM Level 1."""
import os, torch, ctypes, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul.so")

def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a",
           "-O3", "-std=c++17", "-lcuda", SRC, "-o", LIB]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_fp8_gemm.restype = None
    return lib

def run_test(lib, M, N, K):
    print(f"\n=== All-Ones Test: M={M}, N={N}, K={K} ===")
    A = torch.ones(M, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    B = torch.ones(N, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + 128 * 4 - 1) // (128 * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T).bfloat16()
    max_err = (D.float() - ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
    print(f"  Expected: {ref[0,0].item()}, Got: {D[0,0].item()}")
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")
    print(f"  D[0,:5]: {D[0,:5].tolist()}")

def run_random_test(lib, M, N, K):
    print(f"\n=== Random Test: M={M}, N={N}, K={K} ===")
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + 128 * 4 - 1) // (128 * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T).bfloat16()
    max_err = (D.float() - ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")
    print(f"  D[:3,:3]: {D[:3,:3].tolist()}")
    print(f"  Ref[:3,:3]: {ref[:3,:3].tolist()}")

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    # All-ones tests
    for M, N, K in [(128, 128, 128), (128, 128, 256), (128, 128, 512), (256, 256, 256)]:
        run_test(lib, M, N, K)

    # Random tests
    for M, N, K in [(128, 128, 128), (128, 128, 512), (256, 256, 256), (512, 512, 512)]:
        run_random_test(lib, M, N, K)
