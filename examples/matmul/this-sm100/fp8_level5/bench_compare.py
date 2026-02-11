#!/usr/bin/env python3
"""Side-by-side benchmark: Level 5 vs DeepGEMM on matching sizes."""
import os, sys, torch, ctypes, time

sys.path.insert(0, '/home/zhengsize/DeepGEMM/tests')
import deep_gemm
from deep_gemm.testing import bench_kineto
from generators import KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAN_K = 128
NUM_MULTICAST = 2

# Load pre-compiled Level 5 variants
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
libs = {}
for bn in [64, 96, 128, 192, 224]:
    lib_path = os.path.join(SCRIPT_DIR, f"matmul_n{bn}.so")
    if not os.path.exists(lib_path):
        import subprocess
        print(f"  Compiling BLOCK_N={bn}...", end=" ", flush=True)
        subprocess.check_call([
            "nvcc", "--shared", "-Xcompiler", "-fPIC",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-O3", "-std=c++17", f"-DTILE_N={bn}", "-lcuda",
            SRC, "-o", lib_path,
        ], stderr=subprocess.STDOUT)
        print("OK")
    lib = ctypes.CDLL(lib_path)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fp8_gemm.restype = None
    libs[bn] = lib

num_sms = torch.cuda.get_device_properties(0).multi_processor_count
print(f"GPU: {torch.cuda.get_device_name(0)}, SMs: {num_sms}\n")

BLOCK_N_CANDIDATES = sorted(libs.keys())

def select_best_block_n(M, N, K, block_m=128):
    best_bn = None
    best_waves = float('inf')
    best_last_util = 0
    for bn in BLOCK_N_CANDIDATES:
        num_m_blocks = (M + block_m - 1) // block_m
        num_n_blocks = (N + bn - 1) // bn
        total_tiles = num_m_blocks * num_n_blocks
        if num_m_blocks % NUM_MULTICAST != 0:
            continue
        waves = (total_tiles + num_sms - 1) // num_sms
        last_wave_blocks = total_tiles % num_sms
        last_util = last_wave_blocks if last_wave_blocks > 0 else num_sms
        if best_bn is None or waves < best_waves:
            best_bn, best_waves, best_last_util = bn, waves, last_util
        elif waves == best_waves:
            if last_util > best_last_util:
                best_bn, best_waves, best_last_util = bn, waves, last_util
            elif last_util == best_last_util and bn < best_bn:
                best_bn = bn
    return best_bn if best_bn else 224

shapes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
]

kernel_type = KernelType.Kernel1D1D
quant_config = QuantConfig()
use_ue8m0 = get_ue8m0_usage(kernel_type)
major_a = MajorTypeAB.KMajor
major_b = MajorTypeAB.KMajor
out_dtype = torch.bfloat16

print(f"{'M':>6} {'N':>6} {'K':>6} | {'BN':>4} {'Ours(ms)':>10} {'Ours TFLOPS':>12} | {'DG(ms)':>10} {'DG TFLOPS':>12} | {'Ratio':>6}")
print("-" * 95)

for M, N, K in shapes:
    # --- Our kernel ---
    bn = select_best_block_n(M, N, K)
    lib = libs[bn]
    A_ours = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B_ours = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA_ours = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB_ours = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D_ours = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()

    warmup, iters = 20, 100
    for _ in range(warmup):
        lib.launch_fp8_gemm(A_ours.data_ptr(), B_ours.data_ptr(), SFA_ours.data_ptr(), SFB_ours.data_ptr(), D_ours.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(A_ours.data_ptr(), B_ours.data_ptr(), SFA_ours.data_ptr(), SFB_ours.data_ptr(), D_ours.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    our_ms = (time.perf_counter() - start) / iters * 1000
    our_tflops = 2.0 * M * N * K / (our_ms / 1000) / 1e12

    # --- DeepGEMM ---
    a, b, c, d, ref_d = generate_normal(
        M, N, K, major_a, major_b, False, out_dtype, kernel_type,
        use_ue8m0=use_ue8m0, quant_config=quant_config)
    dg_t = bench_kineto(
        lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d),
        'fp8_gemm', suppress_kineto_output=True)
    dg_ms = dg_t * 1000
    dg_tflops = 2.0 * M * N * K / dg_t / 1e12

    ratio = our_tflops / dg_tflops * 100
    print(f"{M:>6} {N:>6} {K:>6} | {bn:>4} {our_ms:>10.4f} {our_tflops:>12.1f} | {dg_ms:>10.4f} {dg_tflops:>12.1f} | {ratio:>5.1f}%")
