import torch
import triton
import triton.language as tl
import argparse


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def complex_matmul_bf16_kernel(
    ar_ptr, ai_ptr, br_ptr, bi_ptr, cr_ptr, ci_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    
    offset_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offset_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offset_k = tl.arange(0, BLOCK_K)
    
    real_a_ptrs = ar_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    image_a_ptrs = ai_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    real_b_ptrs = br_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)
    image_b_ptrs = bi_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)
    
    real_real_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    image_image_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    real_image_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    image_real_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # real_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # image_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        real_a = tl.load(real_a_ptrs, mask=offset_k[None, :] < K - k * BLOCK_K, other=0.0)
        image_a = tl.load(image_a_ptrs, mask=offset_k[None, :] < K - k * BLOCK_K, other=0.0)
        real_b = tl.load(real_b_ptrs, mask=offset_k[:, None] < K - k * BLOCK_K, other=0.0)
        image_b = tl.load(image_b_ptrs, mask=offset_k[:, None] < K - k * BLOCK_K, other=0.0)
        real_real_accum += tl.dot(real_a, real_b)
        image_image_accum += tl.dot(image_a, image_b)
        real_image_accum += tl.dot(real_a, image_b)
        image_real_accum += tl.dot(image_a, real_b)
        # real_accum += tl.dot(real_a, real_b) - tl.dot(image_a, image_b)
        # image_accum += tl.dot(real_a, image_b) + tl.dot(image_a, real_b)
        
        real_a_ptrs += BLOCK_K * stride_ak
        image_a_ptrs += BLOCK_K * stride_ak
        real_b_ptrs += BLOCK_K * stride_bk
        image_b_ptrs += BLOCK_K * stride_bk

    real = (real_real_accum.to(tl.bfloat16) - image_image_accum.to(tl.bfloat16))
    image = (real_image_accum.to(tl.bfloat16) + image_real_accum.to(tl.bfloat16))
    # real = real_accum.to(tl.bfloat16)
    # image = image_accum.to(tl.bfloat16)
    
    offset_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    real_c_ptrs = cr_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn
    image_c_ptrs = ci_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(real_c_ptrs, real, mask=c_mask)
    tl.store(image_c_ptrs, image, mask=c_mask)
    

def complex_matmul_bf16(ar, ai, br, bi):
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    num_warps = 4
    num_stages = 3
    M, K = ar.shape
    K, N = br.shape
    cr = torch.empty([M, N], device=ar.device, dtype=ar.dtype)
    ci = torch.empty([M, N], device=ai.device, dtype=ai.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    complex_matmul_bf16_kernel[grid](
        ar, ai, br, bi, cr, ci,
        M, N, K,
        ar.stride(0), ar.stride(1), br.stride(0), br.stride(1), cr.stride(0), cr.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps, num_stages=num_stages
    )
    return (cr, ci)


def main(M, N, K):
    # torch.manual_seed(0)
    ar = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    ai = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    br = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    bi = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    cr, ci = complex_matmul_bf16(ar, ai, br, bi)
    tcr = torch.matmul(ar, br) - torch.matmul(ai, bi)
    tci = torch.matmul(ai, br) + torch.matmul(ar, bi)
    if torch.allclose(cr, tcr, atol=1e-2, rtol=0):
        if  torch.allclose(ci, tci, atol=1e-2, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch Image differ")
    else:
        print(cr - tcr)
        print("Mean=", torch.mean(cr - tcr))
        print("Max=", torch.max(cr - tcr))
        print("Min=", torch.min(cr - tcr))
        print("Ratio=", torch.max(torch.abs(cr - tcr) / torch.abs(tcr + 1e-5)))
        print("❌ Triton and Torch Real differ")
    
    def perf(func, args, iters=200):
        # warm-up
        outputs = func(*args)
        # Create CUDA events for measuring time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Record the start event
        start_event.record()
        for i in range(iters):
            outputs = func(*args)
        end_event.record()
        end_event.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / iters
    
    flops = M * N * K * 2 * 4 + M * N * 2
    print("Triton performance:", flops / perf(complex_matmul_bf16, (ar, ai, br, bi)) * 1e3 / 1e12, "TFLOPS")
    print("PyTorch performance:", flops / perf(lambda a, b, c, d: (torch.matmul(a, c) - torch.matmul(b, d), torch.matmul(a, d) + torch.matmul(b, c)), (ar, ai, br, bi)) * 1e3 / 1e12, "TFLOPS")
    
    
        
if __name__ == "__main__":
    M = 4096
    N = 4096
    K = 4096
    main(M, N, K)