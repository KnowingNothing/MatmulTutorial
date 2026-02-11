#!/usr/bin/env python3
"""Benchmark DeepGEMM FP8 on common GEMM shapes."""
import sys
sys.path.insert(0, '/home/zhengsize/DeepGEMM/tests')

import torch
import deep_gemm
from deep_gemm.testing import bench_kineto
from generators import (
    KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage, generate_normal
)

shapes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 7168, 7168),
    (8192, 8192, 8192),
]

kernel_type = KernelType.Kernel1D1D
quant_config = QuantConfig()  # legacy: gran_k_a=128, gran_k_b=128, fp8xfp8
use_ue8m0 = get_ue8m0_usage(kernel_type)
major_a = MajorTypeAB.KMajor
major_b = MajorTypeAB.KMajor
accumulate = False
out_dtype = torch.bfloat16

print(f"{'M':>6} {'N':>6} {'K':>6} | {'DG(us)':>10} {'DG TFLOPS':>10} | {'cuBLAS(us)':>12} {'cuBLAS TFLOPS':>14}")
print("-" * 88)

for M, N, K in shapes:
    a, b, c, d, ref_d = generate_normal(
        M, N, K, major_a, major_b, accumulate, out_dtype, kernel_type,
        use_ue8m0=use_ue8m0, quant_config=quant_config
    )

    try:
        t = bench_kineto(
            lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d),
            'fp8_gemm', suppress_kineto_output=True
        )
        flops = 2 * M * N * K
        tflops = flops / t / 1e12

        cublas_t, split_k_t = bench_kineto(
            lambda: deep_gemm.cublaslt_gemm_nt(a[0], b[0], d),
            ('nvjet', 'reduce'), suppress_kineto_output=True
        )
        cublas_total = cublas_t + split_k_t
        cublas_tflops = flops / cublas_total / 1e12 if cublas_total > 0 else 0

        print(f"{M:6d} {N:6d} {K:6d} | {t*1e6:10.1f} {tflops:10.0f} | {cublas_total*1e6:12.1f} {cublas_tflops:14.0f}")
    except Exception as e:
        print(f"{M:6d} {N:6d} {K:6d} | Error: {type(e).__name__}: {e}")
