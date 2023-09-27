import torch
import triton
import triton.language as tl

DTYPE = tl.float32
ACCUM_DTYPE = tl.float32
INF = float("inf")
NINF = float("-inf")


@triton.jit
def linear_attention_fwd_kernel(
    realQ, imageQ, realK, imageK, realV, imageV, realO, imageO,
    real_sm_scale, image_sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, model_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    rr_accum = tl.zeros([BLOCK_M, BLOCK_K], dtype=ACCUM_DTYPE)
    ii_accum = tl.zeros([BLOCK_M, BLOCK_K], dtype=ACCUM_DTYPE)
    ri_accum = tl.zeros([BLOCK_M, BLOCK_K], dtype=ACCUM_DTYPE)
    ir_accum = tl.zeros([BLOCK_M, BLOCK_K], dtype=ACCUM_DTYPE)
    
    batch_row_stride_q = stride_qh // stride_qm // stride_qk
    batch_head_seqlen = batch_size*num_heads*seq_len
    real_q_ptrs = tl.make_block_ptr(
        base=realQ,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_qm, stride_qk),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1,0)
    )
    image_q_ptrs = tl.make_block_ptr(
        base=imageQ,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_qm, stride_qk),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1,0)
    )
    real_k_ptrs = tl.make_block_ptr(
        base=realK,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_km, stride_kk),
        offsets=(batch_head_id * batch_row_stride_q, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1,0)
    )
    image_k_ptrs = tl.make_block_ptr(
        base=imageK,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_km, stride_kk),
        offsets=(batch_head_id * batch_row_stride_q, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1,0)
    )
    real_v_ptrs = tl.make_block_ptr(
        base=realV,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_vm, stride_vk),
        offsets=(batch_head_id * batch_row_stride_q, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1,0)
    )
    image_v_ptrs = tl.make_block_ptr(
        base=imageV,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_vm, stride_vk),
        offsets=(batch_head_id * batch_row_stride_q, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1,0)
    )
    real_o_ptrs = tl.make_block_ptr(
        base=realO,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_om, stride_ok),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1,0)
    )
    image_o_ptrs = tl.make_block_ptr(
        base=imageO,
        shape=(batch_head_seqlen, BLOCK_K),
        strides=(stride_om, stride_ok),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1,0)
    )
    
    real_q = tl.load(real_q_ptrs)
    image_q = tl.load(image_q_ptrs)
    
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        real_k = tl.trans(tl.load(real_k_ptrs, boundary_check=(0, 1)))
        image_k = tl.trans(tl.load(image_k_ptrs, boundary_check=(0, 1)))
        rr_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUM_DTYPE)
        ii_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUM_DTYPE)
        ri_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUM_DTYPE)
        ir_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUM_DTYPE)
        
        rr_qk += tl.dot(real_q, real_k)
        ii_qk += tl.dot(image_q, image_k)
        ri_qk += tl.dot(real_q, image_k)
        ir_qk += tl.dot(image_q, real_k)
        
        rr_qk *= real_sm_scale
        ii_qk *= real_sm_scale
        ir_qk *= image_sm_scale
        ri_qk *= image_sm_scale
        
        rr_qk = tl.where(offset_m[:, None] >= (start_n + offset_n[None, :]), rr_qk, 0)
        ii_qk = tl.where(offset_m[:, None] >= (start_n + offset_n[None, :]), ii_qk, 0)
        ri_qk = tl.where(offset_m[:, None] >= (start_n + offset_n[None, :]), ri_qk, 0)
        ir_qk = tl.where(offset_m[:, None] >= (start_n + offset_n[None, :]), ir_qk, 0)
        
        rr_qk = rr_qk.to(DTYPE)
        ii_qk = ii_qk.to(DTYPE)
        ri_qk = ri_qk.to(DTYPE)
        ir_qk = ir_qk.to(DTYPE)
        
        r_qk = rr_qk - ii_qk
        i_qk = ri_qk + ir_qk
        
        real_v = tl.load(real_v_ptrs, boundary_check=(0, 1))
        image_v = tl.load(image_v_ptrs, boundary_check=(0, 1))
        rr_accum += tl.dot(r_qk, real_v)
        ii_accum += tl.dot(i_qk, image_v)
        ri_accum += tl.dot(r_qk, image_v)
        ir_accum += tl.dot(i_qk, real_v)
        
        real_k_ptrs = tl.advance(real_k_ptrs, [BLOCK_N, 0])
        image_k_ptrs = tl.advance(image_k_ptrs, [BLOCK_N, 0])
        real_v_ptrs = tl.advance(real_v_ptrs, [BLOCK_N, 0])
        image_v_ptrs = tl.advance(image_v_ptrs, [BLOCK_N, 0])
        
    rr_accum = rr_accum.to(DTYPE)
    ii_accum = ii_accum.to(DTYPE)
    ri_accum = ri_accum.to(DTYPE)
    ir_accum = ir_accum.to(DTYPE)
    
    real = rr_accum - ii_accum
    image = ri_accum + ir_accum
    tl.store(real_o_ptrs, real, boundary_check=(0, 1))
    tl.store(image_o_ptrs, image, boundary_check=(0, 1))


def linear_attention_fwd(rq, iq, rk, ik, rv, iv, r_scale, i_scale):
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    batch_size, num_heads, seq_len, model_k = rq.shape
    assert BLOCK_K >= model_k
    # do some check for shape later...
    ro = torch.empty_like(rq)
    io = torch.empty_like(iq)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    num_warps = 4
    num_stages = 2
    linear_attention_fwd_kernel[grid](
        rq, iq, rk, ik, rv, iv, ro, io,
        r_scale, i_scale,
        rq.stride(0), rq.stride(1), rq.stride(2), rq.stride(3),
        rk.stride(0), rk.stride(1), rk.stride(2), rk.stride(3),
        rv.stride(0), rv.stride(1), rv.stride(2), rv.stride(3),
        ro.stride(0), ro.stride(1), ro.stride(2), ro.stride(3),
        batch_size, num_heads, seq_len, model_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages
    )
    return ro, io


def main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale):
    torch.manual_seed(0)
    rq = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    iq = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    rk = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    ik = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    rv = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    iv = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=torch.float32)
    ro, io = linear_attention_fwd(rq, iq, rk, ik, rv, iv, r_scale, i_scale)
    
    # reference impl
    def torch_impl(rq, iq, rk, ik, rv, iv, r_scale, i_scale):
        torch.backends.cuda.matmul.allow_tf32 = False
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        rp = torch.matmul(rq, rk.transpose(2, 3)) * r_scale - torch.matmul(iq, ik.transpose(2, 3)) * r_scale
        ip = torch.matmul(rq, ik.transpose(2, 3)) * i_scale + torch.matmul(iq, rk.transpose(2, 3)) * i_scale
        rp[:, :, mask == 0] = 0
        ip[:, :, mask == 0] = 0
        tro = torch.matmul(rp, rv) - torch.matmul(ip, iv)
        tio = torch.matmul(rp, iv) + torch.matmul(ip, rv)
        return tro, tio
    
    tro, tio = torch_impl(rq, iq, rk, ik, rv, iv, r_scale, i_scale)
    
    if torch.allclose(ro, tro, atol=1, rtol=1e-1):
        if  torch.allclose(io, tio, atol=1, rtol=1e-1):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch Image differ")
    else:
        print(ro)
        print(tro)
        print((ro - tro).abs().max())
        print((ro - tro).abs().max()/tro.abs().mean())
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
    
    print("Triton average latency:", perf(linear_attention_fwd, (rq, iq, rk, ik, rv, iv, r_scale, i_scale)), "ms")
    print("PyTorch average latency:", perf(torch_impl, (rq, iq, rk, ik, rv, iv, r_scale, i_scale)), "ms")
    
    
batch_size = 4
num_heads = 128
seq_len = 1024
model_k = 32
r_scale = 0.2
i_scale = 0.2
main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale)
    
    