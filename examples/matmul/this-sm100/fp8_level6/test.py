#!/usr/bin/env python3
"""
FP8 GEMM Level 6 — JIT Templates + Full DeepGEMM-style Heuristic

This script implements a JIT (Just-In-Time) compilation system that mirrors
DeepGEMM's approach:

1. HEURISTIC: For a given (M, N, K) problem size, select optimal BLOCK_N
   using DeepGEMM's algorithm:
   - Minimize number of SM waves
   - Maximize last-wave SM utilization
   - Tiebreak: prefer smaller BLOCK_N (less wasted compute)

2. JIT COMPILATION: Compile a kernel variant specialized for the exact
   (BLOCK_N, M, N, K) combination. When M/N/K are compile-time constants,
   the CUDA compiler can:
   - Compute K-loop trip count at compile time → optimal unroll decisions
   - Eliminate dead branches in the scheduler
   - Optimize register allocation with known tile counts

3. CACHING: Compiled .so files are cached in .cache/ directory.
   First call for a new shape pays compilation cost (~5-10s);
   subsequent calls are instant.

This is the key technique that gives DeepGEMM its performance edge over
statically compiled kernels.
"""
import os, sys, torch, ctypes, subprocess, time, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
GRAN_K = 128
NUM_MULTICAST = 2
BLOCK_M = 128
BLOCK_K = 128

# =============================================================================
# BLOCK_N Candidates — matching DeepGEMM's full set
#
# DeepGEMM uses: {16} ∪ {32, 64, 96, ..., 256}
# We exclude 256 because TMEM would exceed 512 columns.
# We also exclude very small values (16, 32) when multicast is required,
# because LOAD_BLOCK_N = BLOCK_N / 2 would be very small.
# =============================================================================

ALL_BLOCK_N_CANDIDATES = [16, 32, 64, 96, 128, 160, 192, 224]

# =============================================================================
# Legality Checks (matching DeepGEMM's is_block_size_legal)
# =============================================================================

def is_block_size_legal(block_n, block_m=128, k=None):
    """
    Check if a (block_m, block_n) configuration is legal.

    Checks performed (matching DeepGEMM):
      1. block_n must be divisible by 16 (layout constraint)
      2. For small K (≤256), restrict tile sizes to avoid epilogue bottleneck
      3. TMEM columns must not exceed 512
      4. LOAD_BLOCK_N must be ≥ 8 (minimum TMA transfer)
    """
    if block_n % 16 != 0:
        return False

    # For small K, avoid large tiles (reduces store/compute overlap)
    if k is not None and k <= 256:
        if block_n > 128 or block_m > 128:
            return False

    # TMEM limit: accum_cols + sf_cols ≤ 512
    sf_block_m = ((block_m + 127) // 128) * 128
    sf_block_n = ((block_n + 127) // 128) * 128
    tmem_cols = 2 * block_n + sf_block_m // 32 + sf_block_n // 32
    if tmem_cols > 512:
        return False

    # LOAD_BLOCK_N must be reasonable
    load_block_n = block_n // NUM_MULTICAST
    if load_block_n < 8:
        return False

    return True


def is_multicast_legal(M, block_m=128):
    """
    Check if 2-CTA multicast is legal for given M.

    Requirements (matching DeepGEMM):
      1. M ≥ 512 (enough work to benefit from multicast)
      2. num_m_blocks must be even (each cluster processes 2 M-blocks)
    """
    num_m_blocks = (M + block_m - 1) // block_m
    return M >= 512 and num_m_blocks % NUM_MULTICAST == 0


# =============================================================================
# Heuristic — DeepGEMM's get_best_config algorithm
# =============================================================================

def select_best_config(M, N, K, num_sms, block_m=128):
    """
    Select the best BLOCK_N for given problem dimensions.

    Algorithm (matching DeepGEMM):
      1. For each legal BLOCK_N candidate:
         - Compute total tiles = num_m_blocks × num_n_blocks
         - Compute waves = ceil(tiles / num_sms)
         - Compute last_wave_util = tiles % num_sms (or num_sms if exact)
      2. Select BLOCK_N that:
         a. Minimizes waves (fewer iterations = less overhead)
         b. Among equal-wave configs: maximizes last_wave_util
         c. Among equal-util configs: prefers smaller BLOCK_N (less waste)
         d. Exception: if both block_m and block_n differ, prefer larger
            block_n (better compute-to-memory ratio), but only if both
            ≤ their respective shape dimensions

    Returns: (best_block_n, num_sms_to_use)
    """
    candidates = [bn for bn in ALL_BLOCK_N_CANDIDATES
                  if is_block_size_legal(bn, block_m, K)]

    if not candidates:
        # Fallback
        return 128, num_sms

    best_bn = None
    best_waves = float('inf')
    best_last_util = 0

    for bn in candidates:
        num_m_blocks = (M + block_m - 1) // block_m
        num_n_blocks = (N + bn - 1) // bn
        total_tiles = num_m_blocks * num_n_blocks

        waves = (total_tiles + num_sms - 1) // num_sms
        last_wave_blocks = total_tiles % num_sms
        last_util = last_wave_blocks if last_wave_blocks > 0 else num_sms

        # Multicast requires even num_m_blocks
        if num_m_blocks % NUM_MULTICAST != 0:
            continue

        success = False
        if best_bn is None or waves < best_waves:
            success = True
        elif waves == best_waves:
            if last_util > best_last_util:
                success = True
            elif last_util == best_last_util:
                # Same block_m: prefer smaller block_n
                if bn < best_bn:
                    success = True

        if success:
            best_bn = bn
            best_waves = waves
            best_last_util = last_util

    if best_bn is None:
        best_bn = 128

    # Note: DeepGEMM's DG_JIT_MINIMIZE_NUM_SMS defaults to OFF.
    # Using all SMs gives better parallelism per wave, which typically
    # outweighs the benefit of perfect wave utilization with fewer SMs.
    return best_bn, num_sms


# =============================================================================
# JIT Compilation System
# =============================================================================

_kernel_cache = {}

def compile_kernel(block_n, shape_m=0, shape_n=0, shape_k=0, num_sms=0):
    """
    Compile a kernel variant with specific compile-time parameters.

    Parameters:
        block_n:  BLOCK_N tile size (required)
        shape_m:  Compile-time M dimension (0 = runtime)
        shape_n:  Compile-time N dimension (0 = runtime)
        shape_k:  Compile-time K dimension (0 = runtime)
        num_sms:  Number of SMs to use (0 = auto)

    Returns: ctypes.CDLL handle to the compiled shared library
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    name = f"gemm_n{block_n}"
    if shape_m > 0: name += f"_M{shape_m}"
    if shape_n > 0: name += f"_N{shape_n}"
    if shape_k > 0: name += f"_K{shape_k}"
    if num_sms > 0: name += f"_sms{num_sms}"

    lib_path = os.path.join(CACHE_DIR, f"{name}.so")

    if not os.path.exists(lib_path):
        cmd = [
            "nvcc", "--shared", "-Xcompiler", "-fPIC",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-O3", "-std=c++17",
            f"-DTILE_N={block_n}",
        ]
        if shape_m > 0: cmd.append(f"-DSHAPE_M={shape_m}")
        if shape_n > 0: cmd.append(f"-DSHAPE_N={shape_n}")
        if shape_k > 0: cmd.append(f"-DSHAPE_K={shape_k}")
        if num_sms > 0: cmd.append(f"-DNUM_SMS={num_sms}")

        cmd += ["-lcuda", SRC, "-o", lib_path]
        subprocess.check_call(cmd, stderr=subprocess.STDOUT)

    lib = ctypes.CDLL(lib_path)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.launch_fp8_gemm.restype = None
    return lib


def get_kernel(M, N, K, num_sms, jit=True):
    """
    Get the best compiled kernel for given dimensions.

    When jit=True, compiles a shape-specialized variant (like DeepGEMM).
    When jit=False, uses a generic BLOCK_N-only variant (like Level 5).

    Note: JIT can sometimes HURT performance for large shapes due to
    compiler register allocation changes. Use get_best_kernel() for
    adaptive selection that picks the faster of JIT vs non-JIT.

    Returns: (lib, block_n, actual_num_sms)
    """
    block_n, actual_sms = select_best_config(M, N, K, num_sms)

    if jit:
        key = (block_n, M, N, K, 0)
    else:
        key = (block_n, 0, 0, 0, 0)

    if key not in _kernel_cache:
        if jit:
            _kernel_cache[key] = compile_kernel(block_n, M, N, K)
        else:
            _kernel_cache[key] = compile_kernel(block_n)

    return _kernel_cache[key], block_n, actual_sms


def get_best_kernel(M, N, K, num_sms):
    """
    Adaptive kernel selection: compile both JIT and non-JIT variants,
    auto-tune by running both and picking the faster one.

    This handles the case where JIT shape specialization can sometimes
    cause performance regressions due to compiler behavior (e.g., the
    compiler may increase register pressure when loop bounds are known
    at compile time for large trip counts).

    Returns: (lib, block_n, label)
    """
    lib_nojit, bn, sms = get_kernel(M, N, K, num_sms, jit=False)
    lib_jit, _, _ = get_kernel(M, N, K, num_sms, jit=True)

    # Quick auto-tune: run each 5 times, pick faster
    ms_nojit = benchmark(lib_nojit, M, N, K, warmup=5, iters=5)
    ms_jit = benchmark(lib_jit, M, N, K, warmup=5, iters=5)

    if ms_jit < ms_nojit:
        return lib_jit, bn, "JIT"
    else:
        return lib_nojit, bn, "tile"


# =============================================================================
# Helper: Run GEMM
# =============================================================================

def make_tensors(M, N, K):
    """Create FP8 input tensors and BF16 output tensor."""
    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn).contiguous()
    sf_k = (K + GRAN_K * 4 - 1) // (GRAN_K * 4)
    SFA = torch.full((sf_k, M), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    SFB = torch.full((sf_k, N), 0x7F7F7F7F, dtype=torch.int32, device="cuda").contiguous()
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda").contiguous()
    return A, B, SFA, SFB, D


def run_gemm(lib, A, B, SFA, SFB, D, M, N, K):
    """Launch GEMM kernel."""
    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()


def benchmark(lib, M, N, K, warmup=20, iters=100):
    """Benchmark a kernel variant, returns time in ms.

    Launches all iterations without per-iteration sync to allow
    the CUDA runtime to pipeline kernel launches. Only syncs at
    the end, which amortizes synchronization overhead.
    """
    A, B, SFA, SFB, D = make_tensors(M, N, K)

    for _ in range(warmup):
        lib.launch_fp8_gemm(
            A.data_ptr(), B.data_ptr(),
            SFA.data_ptr(), SFB.data_ptr(),
            D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        lib.launch_fp8_gemm(
            A.data_ptr(), B.data_ptr(),
            SFA.data_ptr(), SFB.data_ptr(),
            D.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    return ms


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}, SMs: {num_sms}")

    # =================================================================
    # Phase 1: Correctness verification
    # =================================================================
    print("\n" + "="*70)
    print("Phase 1: Correctness Verification (JIT-compiled)")
    print("="*70)

    test_sizes = [
        (256,  256,  512),
        (512,  512,  1024),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    all_pass = True
    for M, N, K in test_sizes:
        lib, bn, sms = get_kernel(M, N, K, num_sms, jit=True)
        A, B, SFA, SFB, D = make_tensors(M, N, K)
        run_gemm(lib, A, B, SFA, SFB, D, M, N, K)
        ref = (A.float() @ B.float().T).bfloat16()
        cos = torch.nn.functional.cosine_similarity(
            D.flatten().float(), ref.flatten().float(), dim=0).item()
        status = "PASS" if cos > 0.99 else "FAIL"
        if cos <= 0.99:
            all_pass = False
        print(f"  {M:5d}x{N:5d}x{K:5d}  BN={bn:3d}  SMs={sms:3d}  cos={cos:.4f}  [{status}]")

    if not all_pass:
        print("\nSome tests FAILED! Aborting benchmark.")
        sys.exit(1)

    # =================================================================
    # Phase 2: Benchmark — Adaptive auto-tuning (best of JIT vs non-JIT)
    # =================================================================
    print("\n" + "="*70)
    print("Phase 2: Benchmark — JIT vs Non-JIT + Adaptive Selection")
    print("="*70)

    benchmark_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 7168, 4096),
        (4096, 7168, 7168),
        (8192, 8192, 8192),
    ]

    # Pre-compile all needed variants
    print("\nPre-compiling kernels...")
    for M, N, K in benchmark_sizes:
        get_kernel(M, N, K, num_sms, jit=False)
        get_kernel(M, N, K, num_sms, jit=True)
    print("Done.\n")

    print(f"{'M':>6} {'N':>6} {'K':>6} | {'BN':>3} | {'NonJIT(ms)':>11} {'TFLOPS':>7} | {'JIT(ms)':>11} {'TFLOPS':>7} | {'Best':>8}")
    print("-" * 85)

    for M, N, K in benchmark_sizes:
        lib_nojit, bn, _ = get_kernel(M, N, K, num_sms, jit=False)
        lib_jit, _, _ = get_kernel(M, N, K, num_sms, jit=True)

        ms_nojit = benchmark(lib_nojit, M, N, K)
        ms_jit = benchmark(lib_jit, M, N, K)

        tflops_nojit = 2.0 * M * N * K / (ms_nojit / 1000) / 1e12
        tflops_jit = 2.0 * M * N * K / (ms_jit / 1000) / 1e12

        best = "JIT" if tflops_jit > tflops_nojit else "tile"
        best_tflops = max(tflops_nojit, tflops_jit)

        print(f"{M:>6} {N:>6} {K:>6} | {bn:>3} | {ms_nojit:>10.4f}ms {tflops_nojit:>7.1f} | {ms_jit:>10.4f}ms {tflops_jit:>7.1f} | {best_tflops:>5.0f} {best:>3}")

    # =================================================================
    # Phase 3: Heuristic analysis — show what config is selected per size
    # =================================================================
    print("\n" + "="*70)
    print("Phase 3: Heuristic Analysis — Config Selection Details")
    print("="*70)

    print(f"\n{'M':>6} {'N':>6} {'K':>6} | {'BN':>3} {'SMs':>4} | {'Tiles':>6} {'Waves':>6} {'LastUtil':>9} | {'num_kb':>6} {'Stages':>7}")
    print("-" * 80)

    for M, N, K in benchmark_sizes:
        bn, sms = select_best_config(M, N, K, num_sms)
        num_m = (M + BLOCK_M - 1) // BLOCK_M
        num_n = (N + bn - 1) // bn
        tiles = num_m * num_n
        waves = (tiles + num_sms - 1) // num_sms
        last = tiles % num_sms
        last = last if last > 0 else num_sms
        num_kb = (K + BLOCK_K - 1) // BLOCK_K

        # Compute stages for this BLOCK_N
        load_bn = bn // 2
        smem_a = BLOCK_M * BLOCK_K
        smem_b = load_bn * BLOCK_K
        sf_bm = ((BLOCK_M + 127) // 128) * 128
        sf_bn = ((bn + 127) // 128) * 128
        smem_sfa = sf_bm * 4
        smem_sfb = sf_bn * 4
        smem_stage = smem_a + smem_b + smem_sfa + smem_sfb + 3 * 8

        swizzle_cd = 128 if (bn * 2) % 128 == 0 else (64 if (bn * 2) % 64 == 0 else 32)
        smem_cd = min(BLOCK_M, 128) * swizzle_cd * 2
        smem_fixed = smem_cd + 4 + 2 * 2 * 8
        stages = min(32, (232448 - smem_fixed) // smem_stage)

        print(f"{M:>6} {N:>6} {K:>6} | {bn:>3} {sms:>4} | {tiles:>6} {waves:>6} {last:>8}% | {num_kb:>6} {stages:>7}")
