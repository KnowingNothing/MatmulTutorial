import os, torch, ctypes, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(SCRIPT_DIR, "matmul_n224.so")
GRAN_K = 128

lib = ctypes.CDLL(LIB)
lib.launch_fp8_gemm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.launch_fp8_gemm.restype = None

def create_data(M, N, K):
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    return A, B, SFA, SFB, D

def bench(M, N, K, warmup=10, iters=30):
    A, B, SFA, SFB, D = create_data(M, N, K)
    for _ in range(warmup):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    tflops = 2.0 * M * N * K / (ms / 1000) / 1e12
    return ms, tflops

print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} {'TFLOPS':>10}")
print("-" * 52)
for M, N, K in [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
    (8192, 7168, 8192),
]:
    ms, tf = bench(M, N, K)
    print(f"{M:>6} {N:>6} {K:>6} | {ms:>10.3f} {tf:>10.1f}")
