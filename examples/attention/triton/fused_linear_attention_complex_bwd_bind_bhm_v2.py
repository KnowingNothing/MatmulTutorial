import torch
import triton
import triton.language as tl
import numpy as np
import math

DTYPE = tl.float32
ACCUM_DTYPE = tl.float32
TORCH_DTYPE = torch.float32
assert DTYPE == ACCUM_DTYPE, "backward requires fp32 for elements and accumulators"
INF = float("inf")
NINF = float("-inf")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print(torch.__version__)
print(triton.__version__)


@triton.jit
def linear_attention_bwd_kernel(
    realQ, imageQ, realK, imageK, realV, imageV, realgO, imagegO, # realP, imageP,
    realgQ, imagegQ, realgK, imagegK, realgV, imagegV, # realgP, imagegP,
    real_sm_scale, image_sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, model_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    bh_id = tl.program_id(1)
    bid = bh_id // num_heads
    hid = bh_id % num_heads
    
    stride_2d_row = stride_qb // stride_qm // stride_qk
    stride_2d_col = stride_qh // stride_qm // stride_qk
    
    start_n = tl.program_id(0)
    start_m = start_n * BLOCK_N // BLOCK_M
    
    # tl.arange(0, BLOCK_M) = tl.arange(0, BLOCK_M)
    # tl.arange(0, BLOCK_N) = tl.arange(0, BLOCK_N)
    # tl.arange(0, BLOCK_K) = tl.arange(0, BLOCK_K)
    
    # rq_ptrs = tl.make_block_ptr(
    #     base=realQ,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    rq_ptrs = realQ + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # iq_ptrs = tl.make_block_ptr(
    #     base=imageQ,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    iq_ptrs = imageQ + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_rk_ptrs = tl.make_block_ptr(
    #     base=realK,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_rk_ptrs = realK + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_ik_ptrs = tl.make_block_ptr(
    #     base=imageK,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_ik_ptrs = imageK + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_rv_ptrs = tl.make_block_ptr(
    #     base=realV,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_rv_ptrs = realV + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_iv_ptrs = tl.make_block_ptr(
    #     base=imageV,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_iv_ptrs = imageV + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # rgo_ptrs = tl.make_block_ptr(
    #     base=realgO,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    rgo_ptrs = realgO + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # igo_ptrs = tl.make_block_ptr(
    #     base=imagegO,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    igo_ptrs = imagegO + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # rgq_ptrs = tl.make_block_ptr(
    #     base=realgQ,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    
    # igq_ptrs = tl.make_block_ptr(
    #     base=imagegQ,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_K],
    #     order=(1,0)
    # )
    
    rgq_ptrs = realgQ + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    igq_ptrs = imagegQ + (bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_rgk_ptrs = tl.make_block_ptr(
    #     base=realgK,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_rgk_ptrs = realgK + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_igk_ptrs = tl.make_block_ptr(
    #     base=imagegK,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_igk_ptrs = imagegK + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_rgv_ptrs = tl.make_block_ptr(
    #     base=realgV,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_rgv_ptrs = realgV + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # n_igv_ptrs = tl.make_block_ptr(
    #     base=imagegV,
    #     shape=[batch_size*num_heads*seq_len, model_k],
    #     strides=[stride_qm, stride_qk],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N, 0],
    #     block_shape=[BLOCK_N, BLOCK_K],
    #     order=(1,0)
    # )
    n_igv_ptrs = imagegV + (bid * stride_2d_row + hid * stride_2d_col + start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]) * stride_qm + tl.arange(0, BLOCK_K)[None, :]
    
    # rp_ptrs = tl.make_block_ptr(
    #     base=realP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, start_n * BLOCK_N],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )
    
    # ip_ptrs = tl.make_block_ptr(
    #     base=imageP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col + start_m * BLOCK_M, start_n * BLOCK_N],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )

    # gv accum
    pgo_rr = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    pgo_ii = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    pgo_ri = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    pgo_ir = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    # gk accum
    gpq_rr = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    gpq_ii = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    gpq_ri = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    gpq_ir = tl.zeros([BLOCK_N, BLOCK_K], dtype=ACCUM_DTYPE)
    
    for m in range(start_m * BLOCK_M, seq_len, BLOCK_M):          
        rq = tl.load(rq_ptrs, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
        iq = tl.load(iq_ptrs, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
        rk = tl.load(n_rk_ptrs, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
        ik = tl.load(n_ik_ptrs, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
        rk_T = tl.trans(rk)
        ik_T = tl.trans(ik)
        qk_rr = tl.dot(rq, rk_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        qk_ii = tl.dot(iq, -ik_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        qk_ri = tl.dot(rq, -ik_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        qk_ir = tl.dot(iq, rk_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            
        qk_rr = qk_rr * real_sm_scale
        qk_ii = qk_ii * real_sm_scale
        qk_ri = qk_ri * image_sm_scale
        qk_ir = qk_ir * image_sm_scale
        
        qk_rr = qk_rr.to(DTYPE)
        qk_ii = qk_ii.to(DTYPE)
        qk_ri = qk_ri.to(DTYPE)
        qk_ir = qk_ir.to(DTYPE)
            
        offset_m = m + tl.arange(0, BLOCK_M)
        offset_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        qk_rr = tl.where(offset_m[:, None] >= offset_n[None, :], qk_rr, 0.0)
        qk_ii = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ii, 0.0)
        qk_ri = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ri, 0.0)
        qk_ir = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ir, 0.0)
        
        rp = qk_rr - qk_ii
        ip = qk_ri + qk_ir
        
        scale = tl.full([BLOCK_M, BLOCK_N], model_k, DTYPE)
        scale = tl.sqrt(scale)
        scale = scale * (1 + start_n * BLOCK_N + tl.arange(0, BLOCK_N))
        rp = rp / scale
        ip = ip / scale
        
        # tl.store(rp_ptrs, rp)
        # tl.store(ip_ptrs, ip)
        
        rp_T = tl.trans(rp)
        ip_N_T = tl.trans(ip)
        
        # compute gv
        rgo = tl.load(rgo_ptrs, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
        igo = tl.load(igo_ptrs, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))

        pgo_rr += tl.dot(rp_T, rgo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        pgo_ii += tl.dot(ip_N_T, -igo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        pgo_ri += tl.dot(rp_T, igo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        pgo_ir += tl.dot(-ip_N_T, rgo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
        # compute gp
        rv_T = tl.trans(tl.load(n_rv_ptrs, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k))) # boundary_check=(0, 1))
        iv_N_T = tl.trans(tl.load(n_iv_ptrs, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k))) # boundary_check=(0, 1))
        gov_rr = tl.dot(rgo, rv_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gov_ii = tl.dot(-igo, iv_N_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gov_ri = tl.dot(rgo, -iv_N_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gov_ir = tl.dot(igo, rv_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
        gov_rr = gov_rr * real_sm_scale
        gov_ii = gov_ii * real_sm_scale
        gov_ri = gov_ri * image_sm_scale
        gov_ir = gov_ir * image_sm_scale
        
        gov_rr = tl.where(offset_m[:, None] >= offset_n[None, :], gov_rr, 0.0)
        gov_ii = tl.where(offset_m[:, None] >= offset_n[None, :], gov_ii, 0.0)
        gov_ri = tl.where(offset_m[:, None] >= offset_n[None, :], gov_ri, 0.0)
        gov_ir = tl.where(offset_m[:, None] >= offset_n[None, :], gov_ir, 0.0)
        
        rgp = (gov_rr.to(DTYPE) - gov_ii.to(DTYPE))
        igp = (gov_ri.to(DTYPE) + gov_ir.to(DTYPE))
        
        rgp = rgp / scale
        igp = igp / scale
        
        # compute gq
        # rgq = tl.load(rgq_ptrs, boundary_check=(0, 1))
        # igq = tl.load(igq_ptrs, boundary_check=(0, 1))
        
        gpk_rr = tl.dot(rgp, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpk_ii = tl.dot(igp, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpk_ri = tl.dot(rgp, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpk_ir = tl.dot(igp, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
        rgq = (gpk_rr.to(DTYPE) - gpk_ii.to(DTYPE))
        igq = (gpk_ri.to(DTYPE) + gpk_ir.to(DTYPE))
        
        tl.atomic_add(rgq_ptrs, rgq, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1)))
        tl.atomic_add(igq_ptrs, igq, mask=(m + tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1)))
        
        # tl.store(rgq_ptrs, rgq, boundary_check=(0, 1))
        # tl.store(igq_ptrs, igq, boundary_check=(0, 1))
        
        # compute gk
        rgp_T = tl.trans(rgp)
        igp_N_T = tl.trans(igp)
        # rq = tl.load(rq_ptrs, boundary_check=(0, 1))
        # iq = tl.load(rq_ptrs, boundary_check=(0, 1))
        
        gpq_rr += tl.dot(rgp_T, rq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpq_ii += tl.dot(-igp_N_T, iq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpq_ri += tl.dot(rgp_T, iq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        gpq_ir += tl.dot(-igp_N_T, rq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
        
        # rq_ptrs = tl.advance(rq_ptrs, [BLOCK_M, 0])
        rq_ptrs = rq_ptrs + BLOCK_M * model_k
        # iq_ptrs = tl.advance(iq_ptrs, [BLOCK_M, 0])
        iq_ptrs = iq_ptrs + BLOCK_M * model_k
        # rgq_ptrs = tl.advance(rgq_ptrs, [BLOCK_M, 0])
        # igq_ptrs = tl.advance(igq_ptrs, [BLOCK_M, 0])
        rgq_ptrs = rgq_ptrs + BLOCK_M * stride_qm
        igq_ptrs = igq_ptrs + BLOCK_M * stride_qm
        # rgo_ptrs = tl.advance(rgo_ptrs, [BLOCK_M, 0])
        rgo_ptrs = rgo_ptrs + BLOCK_M * model_k
        # igo_ptrs = tl.advance(igo_ptrs, [BLOCK_M, 0])
        igo_ptrs = igo_ptrs + BLOCK_M * model_k
        
        # rp_ptrs = tl.advance(rp_ptrs, [BLOCK_M, 0])
        # ip_ptrs = tl.advance(ip_ptrs, [BLOCK_M, 0])
        
    rgv = (pgo_rr.to(DTYPE) - pgo_ii.to(DTYPE))
    igv = (pgo_ri.to(DTYPE) + pgo_ir.to(DTYPE))
    tl.store(n_rgv_ptrs, rgv, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
    tl.store(n_igv_ptrs, igv, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
    
    rgk = (gpq_rr.to(DTYPE) - gpq_ii.to(DTYPE))
    igk = (gpq_ri.to(DTYPE) + gpq_ir.to(DTYPE))
    tl.store(n_rgk_ptrs, rgk, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
    tl.store(n_igk_ptrs, igk, mask=(start_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < model_k)) # boundary_check=(0, 1))
            


def linear_attention_bwd(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale):
    batch_size, num_heads, seq_len, model_k = rq.shape
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 64
    # do some check for shape later...
    rgq = torch.zeros_like(rq)
    igq = torch.zeros_like(iq)
    rgk = torch.zeros_like(rk)
    igk = torch.zeros_like(ik)
    rgv = torch.zeros_like(rv)
    igv = torch.zeros_like(iv)
    # rp = torch.empty((batch_size, num_heads, seq_len, seq_len), device=rq.device, dtype=rq.dtype)
    # ip = torch.empty((batch_size, num_heads, seq_len, seq_len), device=iq.device, dtype=iq.dtype)
    # rgp = torch.empty((batch_size, num_heads, seq_len, seq_len), device=rq.device, dtype=rq.dtype)
    # igp = torch.empty((batch_size, num_heads, seq_len, seq_len), device=iq.device, dtype=iq.dtype)
    grid = (triton.cdiv(seq_len, BLOCK_N), batch_size * num_heads, 1)
    num_warps = 8
    num_stages = 1
    linear_attention_bwd_kernel[grid](
        rq, iq, rk, ik, rv, iv, rgo, igo, # rp, ip,
        rgq, igq, rgk, igk, rgv, igv, # rgp, igp,
        r_scale, i_scale,
        rq.stride(0), rq.stride(1), rq.stride(2), rq.stride(3),
        rk.stride(0), rk.stride(1), rk.stride(2), rk.stride(3),
        rv.stride(0), rv.stride(1), rv.stride(2), rv.stride(3),
        rgo.stride(0), rgo.stride(1), rgo.stride(2), rgo.stride(3),
        batch_size, num_heads, seq_len, model_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages
    )
    return rgq, igq, rgk, igk, rgv, igv


def main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale):
    # torch.manual_seed(20)
    rq = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    iq = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    rk = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    ik = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    rv = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    iv = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    
    rgo = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    igo = torch.tensor(np.random.uniform(-1, 1, (batch_size, num_heads, seq_len, model_k)), device="cuda", dtype=TORCH_DTYPE)
    
    rgq, igq, rgk, igk, rgv, igv = linear_attention_bwd(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale)
    
    # print(list(linear_attention_fwd_kernel.cache[0].values())[0].asm['ptx'])
    
    # reference impl
    def torch_impl(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale):
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        rp = torch.matmul(rq, rk.transpose(2, 3)) * r_scale - torch.matmul(iq, -ik.transpose(2, 3)) * r_scale
        ip = torch.matmul(rq, -ik.transpose(2, 3)) * i_scale + torch.matmul(iq, rk.transpose(2, 3)) * i_scale
        rp[:, :, mask == 0] = 0
        ip[:, :, mask == 0] = 0
        row_sum = math.sqrt(model_k) * torch.arange(1, seq_len + 1, device="cuda").unsqueeze(0).unsqueeze(0)
        rp /= row_sum
        ip /= row_sum
        rgv = torch.matmul(rp.transpose(-2, -1), rgo) - torch.matmul(ip.transpose(-2, -1), -igo)
        igv = torch.matmul(rp.transpose(-2, -1), igo) + torch.matmul(-ip.transpose(-2, -1), rgo)
        rgp = torch.matmul(rgo, rv.transpose(-2, -1)) * r_scale - torch.matmul(igo, -iv.transpose(-2, -1)) * r_scale
        igp = torch.matmul(rgo, -iv.transpose(-2, -1)) * i_scale + torch.matmul(igo, rv.transpose(-2, -1)) * i_scale
        rgp[:, :, mask == 0] = 0
        igp[:, :, mask == 0] = 0
        rgp /= row_sum
        igp /= row_sum
        rgq = torch.matmul(rgp, rk) - torch.matmul(igp, ik)
        igq = torch.matmul(rgp, ik) + torch.matmul(igp, rk)
        rgk = torch.matmul(rgp.transpose(-2, -1), rq) - torch.matmul(-igp.transpose(-2, -1), iq)
        igk = torch.matmul(-igp.transpose(-2, -1), rq) + torch.matmul(rgp.transpose(-2, -1), iq)
        return rgq, igq, rgk, igk, rgv, igv
    
    trgq, tigq, trgk, tigk, trgv, tigv = torch_impl(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale)
    
    def torch_c64_fwd_impl(q, k, v, r_scale, i_scale):
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        p = torch.matmul(q, k.conj().transpose(-2, -1))
        p.real = p.real * r_scale
        p.imag = p.imag * i_scale
        p[:, :, mask == 0] = 0
        row_sum = math.sqrt(model_k) * torch.arange(1, seq_len + 1, device="cuda").unsqueeze(0).unsqueeze(0)
        p.real /= row_sum
        p.imag /= row_sum
        o = torch.matmul(p, v)
        return o
    
    q = torch.view_as_complex(torch.stack([rq, iq], dim=-1)).requires_grad_()
    k = torch.view_as_complex(torch.stack([rk, ik], dim=-1)).requires_grad_()
    v = torch.view_as_complex(torch.stack([rv, iv], dim=-1)).requires_grad_()
    go = torch.view_as_complex(torch.stack([rgo, igo], dim=-1))
    o = torch_c64_fwd_impl(q, k, v, r_scale, i_scale)
    
    def torch_c64_bwd_impl(q, k, v, o, go):
        o.backward(go, retain_graph=True)
        return
    
    torch_c64_bwd_impl(q, k, v, o, go)
    crgq, cigq, crgk, cigk, crgv, cigv = q.grad.real.clone(), q.grad.imag.clone(), k.grad.real.clone(), k.grad.imag.clone(), v.grad.real.clone(), v.grad.imag.clone()
    
    
    for tensor, torch_tensor, name in zip([rgv, igv, rgk, igk, rgq, igq], [trgv, tigv, trgk, tigk, trgq, tigq],
                                          ["Real GradV", "Image GradV",  "Real GradK", "Image GradK", "Real GradQ", "Image GradQ"]):
        if  torch.allclose(tensor, torch_tensor, atol=1e-3, rtol=1e-3):
            print(f"✅ Triton and Torch {name} match")
        else:
            # print(tensor)
            # print(torch_tensor)
            print((tensor - torch_tensor).abs().max())
            print((tensor - torch_tensor).abs().max()/torch_tensor.abs().mean())
            print(f"❌ Triton and Torch {name} differ")
            
        # triton.testing.assert_close(tensor, torch_tensor, atol=1e-2, rtol=1e-2)
        
    for tensor, torch_tensor, name in zip([trgv, tigv, trgk, tigk, trgq, tigq], [crgv, cigv, crgk, cigk, crgq, cigq],
                                          ["Real GradV", "Image GradV",  "Real GradK", "Image GradK", "Real GradQ", "Image GradQ"]):
        if  torch.allclose(tensor, torch_tensor, atol=1e-3, rtol=1e-3):
            print(f"✅ Torch and Torch C64 {name} match")
        else:
            # print(tensor)
            # print(torch_tensor)
            print((tensor - torch_tensor).abs().max())
            print((tensor - torch_tensor).abs().max()/torch_tensor.abs().mean())
            print(f"❌ Torch and Torch C64 {name} differ")
            
        # triton.testing.assert_close(tensor, torch_tensor, atol=1e-2, rtol=1e-2)
    
    def perf(func, args, iters=1):
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
    
    print("Triton average latency:", perf(linear_attention_bwd, (rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale)), "ms")
    print("PyTorch average latency:", perf(torch_impl, (rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale)), "ms")
    print("PyTorch C64 average latency:", perf(torch_c64_bwd_impl, (q, k, v, o, go)), "ms")
    

if __name__ == "__main__":
    batch_size = 1
    num_heads = 32
    seq_len = 19
    model_k = 64
    r_scale = 1.0
    i_scale = 1.0
    main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale)
