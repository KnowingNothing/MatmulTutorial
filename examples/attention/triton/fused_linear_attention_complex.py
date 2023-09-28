import torch
import triton
import triton.language as tl
import numpy as np

DTYPE = tl.float32
ACCUM_DTYPE = tl.float32
TORCH_DTYPE = torch.float32
INF = float("inf")
NINF = float("-inf")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print(torch.__version__)
print(triton.__version__)


@triton.jit
def linear_attention_fwd_kernel(
    realQ, imageQ, realK, imageK, realV, imageV, realO, imageO, realP, imageP,
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
    real_p_ptrs = tl.make_block_ptr(
        base=realP,
        shape=(batch_head_seqlen, seq_len),
        strides=(seq_len, 1),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1,0)
    )
    image_p_ptrs = tl.make_block_ptr(
        base=imageP,
        shape=(batch_head_seqlen, seq_len),
        strides=(seq_len, 1),
        offsets=(batch_head_id * batch_row_stride_q + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
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
        
        rr_qk += tl.dot(real_q, real_k, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ii_qk += tl.dot(image_q, image_k, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ri_qk += tl.dot(real_q, image_k, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ir_qk += tl.dot(image_q, real_k, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
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
        tl.store(real_p_ptrs, r_qk, boundary_check=(0, 1))
        tl.store(image_p_ptrs, i_qk, boundary_check=(0, 1))
        
        real_v = tl.load(real_v_ptrs, boundary_check=(0, 1))
        image_v = tl.load(image_v_ptrs, boundary_check=(0, 1))
        rr_accum += tl.dot(r_qk, real_v, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ii_accum += tl.dot(i_qk, image_v, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ri_accum += tl.dot(r_qk, image_v, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        ir_accum += tl.dot(i_qk, real_v, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
        real_k_ptrs = tl.advance(real_k_ptrs, [BLOCK_N, 0])
        image_k_ptrs = tl.advance(image_k_ptrs, [BLOCK_N, 0])
        real_v_ptrs = tl.advance(real_v_ptrs, [BLOCK_N, 0])
        image_v_ptrs = tl.advance(image_v_ptrs, [BLOCK_N, 0])
        
        real_p_ptrs = tl.advance(real_p_ptrs, [0, BLOCK_N])
        image_p_ptrs = tl.advance(image_p_ptrs, [0, BLOCK_N])
        
    rr_accum = rr_accum.to(DTYPE)
    ii_accum = ii_accum.to(DTYPE)
    ri_accum = ri_accum.to(DTYPE)
    ir_accum = ir_accum.to(DTYPE)
    
    real = rr_accum - ii_accum
    image = ri_accum + ir_accum
    tl.store(real_o_ptrs, real, boundary_check=(0, 1))
    tl.store(image_o_ptrs, image, boundary_check=(0, 1))


def linear_attention_fwd(rq, iq, rk, ik, rv, iv, r_scale, i_scale):
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = BLOCK_N
    batch_size, num_heads, seq_len, model_k = rq.shape
    assert BLOCK_K >= model_k
    # do some check for shape later...
    ro = torch.empty_like(rq)
    io = torch.empty_like(iq)
    rp = torch.empty((batch_size, num_heads, seq_len, seq_len), device=rq.device, dtype=rq.dtype)
    ip = torch.empty((batch_size, num_heads, seq_len, seq_len), device=iq.device, dtype=iq.dtype)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    num_warps = 4
    num_stages = 2
    linear_attention_fwd_kernel[grid](
        rq, iq, rk, ik, rv, iv, ro, io, rp, ip,
        r_scale, i_scale,
        rq.stride(0), rq.stride(1), rq.stride(2), rq.stride(3),
        rk.stride(0), rk.stride(1), rk.stride(2), rk.stride(3),
        rv.stride(0), rv.stride(1), rv.stride(2), rv.stride(3),
        ro.stride(0), ro.stride(1), ro.stride(2), ro.stride(3),
        batch_size, num_heads, seq_len, model_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages
    )
    return rp, ip, ro, io


def main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale):
    # torch.manual_seed(20)
    rq = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    iq = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    rk = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    ik = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    rv = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    iv = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    
    # rq = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    # iq = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    # rk = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    # ik = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    # rv = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    # iv = torch.randn((batch_size, num_heads, seq_len, model_k), device="cuda", dtype=TORCH_DTYPE)
    rp, ip, ro, io = linear_attention_fwd(rq, iq, rk, ik, rv, iv, r_scale, i_scale)
    
    # print(list(linear_attention_fwd_kernel.cache[0].values())[0].asm['ptx'])
    
    # reference impl
    def torch_impl(rq, iq, rk, ik, rv, iv, r_scale, i_scale):
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        rp = torch.matmul(rq, rk.transpose(2, 3)) * r_scale - torch.matmul(iq, ik.transpose(2, 3)) * r_scale
        ip = torch.matmul(rq, ik.transpose(2, 3)) * i_scale + torch.matmul(iq, rk.transpose(2, 3)) * i_scale
        rp[:, :, mask == 0] = 0
        ip[:, :, mask == 0] = 0
        tro = torch.matmul(rp, rv) - torch.matmul(ip, iv)
        tio = torch.matmul(rp, iv) + torch.matmul(ip, rv)
        return rp, ip, tro, tio
    
    trp, tip, tro, tio = torch_impl(rq, iq, rk, ik, rv, iv, r_scale, i_scale)
    
    if torch.allclose(trp, rp, atol=1e-2, rtol=1e-2):
        if  torch.allclose(tip, ip, atol=1e-2, rtol=1e-2):
            print("✅ Triton and Torch P match")
        else:
            print((ip - tip).abs().max())
            print((ip - tip).abs().max()/tip.abs().mean())
            print("❌ Triton and Torch P Image differ")
    else:
        print((rp - trp).abs().max())
        print((rp - trp).abs().max()/trp.abs().mean())
        print("❌ Triton and Torch P Real differ")
        
    triton.testing.assert_close(rp, trp, atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(ip, tip, atol=1e-2, rtol=1e-2)
    
    if torch.allclose(ro, tro, atol=1e-2, rtol=1e-2):
        if  torch.allclose(io, tio, atol=1e-2, rtol=1e-2):
            print("✅ Triton and Torch match")
        else:
            print((io - tio).abs().max())
            print((io - tio).abs().max()/tio.abs().mean())
            print("❌ Triton and Torch Image differ")
    else:
        print((ro - tro).abs().max())
        print((ro - tro).abs().max()/tro.abs().mean())
        print("❌ Triton and Torch Real differ")
        
    triton.testing.assert_close(ro, tro, atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(io, tio, atol=1e-2, rtol=1e-2)
    
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
    
    
batch_size = 1
num_heads = 48
seq_len = 1024
model_k = 64
r_scale = 1.0
i_scale = 1.0
main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale)
