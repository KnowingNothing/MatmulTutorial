#!/usr/bin/env python3
"""Minimal UMMA test: manually swizzle data into SMEM, run UMMA, check result."""
import os, torch, ctypes, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "test_umma.cu")
LIB = os.path.join(SCRIPT_DIR, "test_umma.so")

def compile_kernel():
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC",
           "-gencode", "arch=compute_100a,code=sm_100a",
           "-O3", "-std=c++17", "-lcuda", SRC, "-o", LIB]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_test_umma.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_test_umma.restype = None
    return lib

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    M, N, K = 128, 128, 128

    # Test 1: All ones — D[m,n] = K = 128
    print("\n=== Test 1: All ones ===")
    A = torch.ones(M, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    B = torch.ones(N, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda").contiguous()
    lib.launch_test_umma(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T)
    print(f"  D[0,0]={D[0,0].item():.1f} (expect {ref[0,0].item():.1f})")
    print(f"  D[1,0]={D[1,0].item():.1f} (expect {ref[1,0].item():.1f})")
    max_err = (D - ref).abs().max().item()
    print(f"  Max error: {max_err:.4f}")

    # Test 2: Varying M - each row has a unique value
    print("\n=== Test 2: Varying M (A[m,:] = (m+1)*0.0078125, B = ones) ===")
    A_vals = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    for m in range(M):
        A_vals[m, :] = (m + 1) * 0.0078125  # unique per row
    A = A_vals.to(torch.float8_e4m3fn).contiguous()
    B = torch.ones(N, K, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).contiguous()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda").contiguous()
    lib.launch_test_umma(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T)
    # Print full column 0 for all 128 rows
    print(f"  D col 0, all 128 M-rows:")
    for start in range(0, 128, 16):
        d_vals = D[start:start+16, 0].tolist()
        r_vals = ref[start:start+16, 0].tolist()
        print(f"    [{start:3d}-{start+15:3d}] D={[f'{v:.2f}' for v in d_vals]}")
        print(f"    [{start:3d}-{start+15:3d}] R={[f'{v:.2f}' for v in r_vals]}")
    max_err = (D - ref).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")

    # Test 3: Random data
    print("\n=== Test 3: Random data ===")
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda").contiguous()
    lib.launch_test_umma(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T)
    print(f"  D[:3,:3]: {D[:3,:3].tolist()}")
    print(f"  Ref[:3,:3]: {ref[:3,:3].tolist()}")
    max_err = (D - ref).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")

    # Test 4: Identity A
    print("\n=== Test 4: Identity A, random B ===")
    A_id = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    for m in range(min(M, K)):
        A_id[m, m] = 1.0
    A = A_id.to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda").contiguous()
    lib.launch_test_umma(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ref = torch.matmul(A.float(), B.float().T)
    print(f"  D[0,:8]: {D[0,:8].tolist()}")
    print(f"  Ref[0,:8]: {ref[0,:8].tolist()}")
    print(f"  D[1,:8]: {D[1,:8].tolist()}")
    print(f"  Ref[1,:8]: {ref[1,:8].tolist()}")
    max_err = (D - ref).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(D.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  Max error: {max_err:.4f}, Cosine sim: {cos_sim:.6f}")
