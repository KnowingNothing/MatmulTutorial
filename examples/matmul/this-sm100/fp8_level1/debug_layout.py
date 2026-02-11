#!/usr/bin/env python3
"""Layout debug: identify which dimension has the permutation issue."""
import os, torch, ctypes, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul.so")

def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a",
           "-O3", "-std=c++17", "-lcuda", SRC, "-o", LIB]
    subprocess.check_call(cmd)

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_fp8_gemm.restype = None
    return lib

def run_gemm(lib, A, B, SFA, SFB, M, N, K):
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    return D

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    M, N, K = 128, 128, 128
    sf_k = 1
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()

    # Test 1: Varying M (A rows differ, B = ones)
    # A[m,:] = (m % 4 + 1) * 0.5 for all k
    # B = all 1.0
    # Expected: D[m,n] = A_val[m] * K
    print("=== Test 1: Varying M dimension ===")
    A_vals = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    for m in range(M):
        A_vals[m, :] = (m % 8 + 1) * 0.125  # 0.125, 0.25, 0.375, 0.5, ...
    A = A_vals.to(torch.float8_e4m3fn).contiguous()
    B = torch.ones(N, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()

    D = run_gemm(lib, A, B, SFA, SFB, M, N, K)
    ref = torch.matmul(A.float(), B.float().T).bfloat16()

    print(f"  D column 0 (first 16 M-rows): {D[:16, 0].tolist()}")
    print(f"  Ref column 0 (first 16 M-rows): {ref[:16, 0].tolist()}")
    print(f"  D row 0 (first 8 N-cols): {D[0, :8].tolist()}")
    print(f"  All cols same per row? max diff in row 0: {(D[0,:] - D[0,0]).abs().max().item()}")

    # Test 2: Varying N (A = ones, B rows differ)
    # B[n,:] = (n % 4 + 1) * 0.5 for all k
    # Expected: D[m,n] = B_val[n] * K
    print("\n=== Test 2: Varying N dimension ===")
    A = torch.ones(M, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    B_vals = torch.zeros(N, K, dtype=torch.float32, device="cuda")
    for n in range(N):
        B_vals[n, :] = (n % 8 + 1) * 0.125
    B = B_vals.to(torch.float8_e4m3fn).contiguous()

    D = run_gemm(lib, A, B, SFA, SFB, M, N, K)
    ref = torch.matmul(A.float(), B.float().T).bfloat16()

    print(f"  D row 0 (first 16 N-cols): {D[0, :16].tolist()}")
    print(f"  Ref row 0 (first 16 N-cols): {ref[0, :16].tolist()}")
    print(f"  D column 0 (first 8 M-rows): {D[:8, 0].tolist()}")
    print(f"  All rows same per col? max diff in col 0: {(D[:,0] - D[0,0]).abs().max().item()}")

    # Test 3: Identity-like — A[m,k] = 1 only when k == m
    # This makes D = B^T (picking specific columns)
    print("\n=== Test 3: A near-identity, B random ===")
    A_id = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    for m in range(min(M, K)):
        A_id[m, m] = 1.0
    A = A_id.to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()

    D = run_gemm(lib, A, B, SFA, SFB, M, N, K)
    ref = torch.matmul(A.float(), B.float().T).bfloat16()

    print(f"  D[0,:8]: {D[0,:8].tolist()}")
    print(f"  Ref[0,:8]: {ref[0,:8].tolist()}")
    print(f"  D[1,:8]: {D[1,:8].tolist()}")
    print(f"  Ref[1,:8]: {ref[1,:8].tolist()}")
    max_err = (D.float() - ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.float().flatten().unsqueeze(0), ref.float().flatten().unsqueeze(0)).item()
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")
