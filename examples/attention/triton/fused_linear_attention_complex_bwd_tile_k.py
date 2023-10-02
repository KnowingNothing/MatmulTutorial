import torch
import triton
import triton.language as tl
import numpy as np

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
    bh_id = tl.program_id(0)
    bid = bh_id // num_heads
    hid = bh_id % num_heads
    
    stride_2d_row = stride_qb // stride_qm // stride_qk
    stride_2d_col = stride_qh // stride_qm // stride_qk
    
    m_rq_ptrs = tl.make_block_ptr(
        base=realQ,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    m_iq_ptrs = tl.make_block_ptr(
        base=imageQ,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    rk_ptrs = tl.make_block_ptr(
        base=realK,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    ik_ptrs = tl.make_block_ptr(
        base=imageK,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    rv_ptrs = tl.make_block_ptr(
        base=realV,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    iv_ptrs = tl.make_block_ptr(
        base=imageV,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    m_rgo_ptrs = tl.make_block_ptr(
        base=realgO,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    m_igo_ptrs = tl.make_block_ptr(
        base=imagegO,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    m_rgq_ptrs = tl.make_block_ptr(
        base=realgQ,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    m_igq_ptrs = tl.make_block_ptr(
        base=imagegQ,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_M, BLOCK_K],
        order=(1,0)
    )
    
    rgk_ptrs = tl.make_block_ptr(
        base=realgK,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    igk_ptrs = tl.make_block_ptr(
        base=imagegK,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    rgv_ptrs = tl.make_block_ptr(
        base=realgV,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    igv_ptrs = tl.make_block_ptr(
        base=imagegV,
        shape=[batch_size*num_heads*seq_len, model_k],
        strides=[stride_qm, stride_qk],
        offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
        block_shape=[BLOCK_N, BLOCK_K],
        order=(1,0)
    )
    
    # rp_ptrs = tl.make_block_ptr(
    #     base=realP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )
    
    # ip_ptrs = tl.make_block_ptr(
    #     base=imageP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )
    
    # rgp_ptrs = tl.make_block_ptr(
    #     base=realgP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )
    
    # igp_ptrs = tl.make_block_ptr(
    #     base=imagegP,
    #     shape=[batch_size*num_heads*seq_len, seq_len],
    #     strides=[seq_len, 1],
    #     offsets=[bid * stride_2d_row + hid * stride_2d_col, 0],
    #     block_shape=[BLOCK_M, BLOCK_N],
    #     order=(1,0)
    # )
    
    # m_rq_ptrs = rq_ptrs
    # m_iq_ptrs = iq_ptrs
    # m_rgq_ptrs = rgq_ptrs
    # m_igq_ptrs = igq_ptrs
    # m_rgo_ptrs = rgo_ptrs
    # m_igo_ptrs = igo_ptrs
    
    for n in range(0, seq_len, BLOCK_N):
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
        for m in range(0, seq_len, BLOCK_M):            
            rq = tl.load(m_rq_ptrs, boundary_check=(0, 1))
            iq = tl.load(m_iq_ptrs, boundary_check=(0, 1))
            rk = tl.trans(tl.load(rk_ptrs, boundary_check=(0, 1)))
            ik = tl.trans(tl.load(ik_ptrs, boundary_check=(0, 1)))
            qk_rr = tl.dot(rq, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            qk_ii = tl.dot(iq, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            qk_ri = tl.dot(rq, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            qk_ir = tl.dot(iq, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
                
            qk_rr = qk_rr * real_sm_scale
            qk_ii = qk_ii * real_sm_scale
            qk_ri = qk_ri * image_sm_scale
            qk_ir = qk_ir * image_sm_scale
            
            qk_rr = qk_rr.to(DTYPE)
            qk_ii = qk_ii.to(DTYPE)
            qk_ri = qk_ri.to(DTYPE)
            qk_ir = qk_ir.to(DTYPE)
                
            offset_m = m + tl.arange(0, BLOCK_M)
            offset_n = n + tl.arange(0, BLOCK_N)
            qk_rr = tl.where(offset_m[:, None] >= offset_n[None, :], qk_rr, 0.0)
            qk_ii = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ii, 0.0)
            qk_ri = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ri, 0.0)
            qk_ir = tl.where(offset_m[:, None] >= offset_n[None, :], qk_ir, 0.0)
            
            rp = qk_rr - qk_ii
            ip = qk_ri + qk_ir
            
            # tl.store(rp_ptrs, rp)
            # tl.store(ip_ptrs, ip)
            
            rp_T = tl.trans(rp)
            ip_N_T = -tl.trans(ip)
            
            # compute gv
            rgo = tl.load(m_rgo_ptrs, boundary_check=(0, 1))
            igo = tl.load(m_igo_ptrs, boundary_check=(0, 1))

            pgo_rr += tl.dot(rp_T, rgo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            pgo_ii += tl.dot(ip_N_T, igo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            pgo_ri += tl.dot(rp_T, igo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            pgo_ir += tl.dot(ip_N_T, rgo, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            
            # compute gp
            rv_T = tl.trans(tl.load(rv_ptrs, boundary_check=[0, 1]))
            iv_N_T = -tl.trans(tl.load(iv_ptrs, boundary_check=[0, 1]))
            gov_rr = tl.dot(rgo, rv_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gov_ii = tl.dot(igo, iv_N_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gov_ri = tl.dot(rgo, iv_N_T, allow_tf32=False, out_dtype=ACCUM_DTYPE)
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
            
            # tl.store(rgp_ptrs, rgp)
            # tl.store(igp_ptrs, igp)
            
            # compute gq
            rgq = tl.load(m_rgq_ptrs, boundary_check=(0, 1))
            igq = tl.load(m_igq_ptrs, boundary_check=(0, 1))
            
            rk = tl.load(rk_ptrs, boundary_check=(0, 1))
            ik = tl.load(ik_ptrs, boundary_check=(0, 1))
            
            gpk_rr = tl.dot(rgp, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpk_ii = tl.dot(igp, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpk_ri = tl.dot(rgp, ik, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpk_ir = tl.dot(igp, rk, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            
            rgq += (gpk_rr.to(DTYPE) - gpk_ii.to(DTYPE))
            igq += (gpk_ri.to(DTYPE) + gpk_ir.to(DTYPE))
            
            tl.store(m_rgq_ptrs, rgq, boundary_check=(0, 1))
            tl.store(m_igq_ptrs, igq, boundary_check=(0, 1))
            
            # compute gk
            rgp_T = tl.trans(rgp)
            igp_N_T = -tl.trans(igp)
            # rq = tl.load(m_rq_ptrs, boundary_check=(0, 1))
            # iq = tl.load(m_rq_ptrs, boundary_check=(0, 1))
            
            gpq_rr += tl.dot(rgp_T, rq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpq_ii += tl.dot(igp_N_T, iq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpq_ri += tl.dot(rgp_T, iq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            gpq_ir += tl.dot(igp_N_T, rq, allow_tf32=False, out_dtype=ACCUM_DTYPE)
            
            
            m_rq_ptrs = tl.advance(m_rq_ptrs, [BLOCK_M, 0])
            m_iq_ptrs = tl.advance(m_iq_ptrs, [BLOCK_M, 0])
            m_rgq_ptrs = tl.advance(m_rgq_ptrs, [BLOCK_M, 0])
            m_igq_ptrs = tl.advance(m_igq_ptrs, [BLOCK_M, 0])
            m_rgo_ptrs = tl.advance(m_rgo_ptrs, [BLOCK_M, 0])
            m_igo_ptrs = tl.advance(m_igo_ptrs, [BLOCK_M, 0])
            
            # rp_ptrs = tl.advance(rp_ptrs, [BLOCK_M, 0])
            # ip_ptrs = tl.advance(ip_ptrs, [BLOCK_M, 0])
            # rgp_ptrs = tl.advance(rgp_ptrs, [BLOCK_M, 0])
            # igp_ptrs = tl.advance(igp_ptrs, [BLOCK_M, 0])
            
        rgv = (pgo_rr.to(DTYPE) - pgo_ii.to(DTYPE))
        igv = (pgo_ri.to(DTYPE) + pgo_ir.to(DTYPE))
        tl.store(rgv_ptrs, rgv, boundary_check=(0, 1))
        tl.store(igv_ptrs, igv, boundary_check=(0, 1))
        
        rgk = (gpq_rr.to(DTYPE) - gpq_ii.to(DTYPE))
        igk = (gpq_ri.to(DTYPE) + gpq_ir.to(DTYPE))
        tl.store(rgk_ptrs, rgk, boundary_check=(0, 1))
        tl.store(igk_ptrs, igk, boundary_check=(0, 1))
        
        rk_ptrs = tl.advance(rk_ptrs, [BLOCK_N, 0])
        ik_ptrs = tl.advance(ik_ptrs, [BLOCK_N, 0])
        rv_ptrs = tl.advance(rv_ptrs, [BLOCK_N, 0])
        iv_ptrs = tl.advance(iv_ptrs, [BLOCK_N, 0])
        rgk_ptrs = tl.advance(rgk_ptrs, [BLOCK_N, 0])
        igk_ptrs = tl.advance(igk_ptrs, [BLOCK_N, 0])
        rgv_ptrs = tl.advance(rgv_ptrs, [BLOCK_N, 0])
        igv_ptrs = tl.advance(igv_ptrs, [BLOCK_N, 0])
        
        m_rq_ptrs = tl.advance(m_rq_ptrs, [-seq_len, 0])
        m_iq_ptrs = tl.advance(m_iq_ptrs, [-seq_len, 0])
        m_rgq_ptrs = tl.advance(m_rgq_ptrs, [-seq_len, 0])
        m_igq_ptrs = tl.advance(m_igq_ptrs, [-seq_len, 0])
        m_rgo_ptrs = tl.advance(m_rgo_ptrs, [-seq_len, 0])
        m_igo_ptrs = tl.advance(m_igo_ptrs, [-seq_len, 0])

        # rp_ptrs = tl.advance(rp_ptrs, [-seq_len, BLOCK_N])
        # ip_ptrs = tl.advance(ip_ptrs, [-seq_len, BLOCK_N])
        # rgp_ptrs = tl.advance(rgp_ptrs, [-seq_len, BLOCK_N])
        # igp_ptrs = tl.advance(igp_ptrs, [-seq_len, BLOCK_N])
            


def linear_attention_bwd(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale):
    batch_size, num_heads, seq_len, model_k = rq.shape
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 128
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
    grid = (batch_size * num_heads, 1)
    num_warps = 4
    num_stages = 2
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
    return rgq, igq, rgk, igk, rgv, igv # , rp, ip, rgp, igp


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
        rp = torch.matmul(rq, rk.transpose(2, 3)) * r_scale - torch.matmul(iq, ik.transpose(2, 3)) * r_scale
        ip = torch.matmul(rq, ik.transpose(2, 3)) * i_scale + torch.matmul(iq, rk.transpose(2, 3)) * i_scale
        rp[:, :, mask == 0] = 0
        ip[:, :, mask == 0] = 0
        rgv = torch.matmul(rp.transpose(-2, -1), rgo) - torch.matmul(-ip.transpose(-2, -1), igo)
        igv = torch.matmul(rp.transpose(-2, -1), igo) + torch.matmul(-ip.transpose(-2, -1), rgo)
        rgp = torch.matmul(rgo, rv.transpose(-2, -1)) * r_scale - torch.matmul(igo, -iv.transpose(-2, -1)) * r_scale
        igp = torch.matmul(rgo, -iv.transpose(-2, -1)) * i_scale + torch.matmul(igo, rv.transpose(-2, -1)) * i_scale
        rgp[:, :, mask == 0] = 0
        igp[:, :, mask == 0] = 0
        rgq = torch.matmul(rgp, rk) - torch.matmul(igp, ik)
        igq = torch.matmul(rgp, ik) + torch.matmul(igp, rk)
        rgk = torch.matmul(rgp.transpose(-2, -1), rq) - torch.matmul(-igp.transpose(-2, -1), iq)
        igk = torch.matmul(-igp.transpose(-2, -1), rq) + torch.matmul(rgp.transpose(-2, -1), iq)
        return rgq, igq, rgk, igk, rgv, igv #, rp, ip, rgp, igp
    
    trgq, tigq, trgk, tigk, trgv, tigv = torch_impl(rq, iq, rk, ik, rv, iv, rgo, igo, r_scale, i_scale)
    
    # if torch.allclose(trp, rp, atol=1e-2, rtol=1e-2):
    #     if  torch.allclose(tip, ip, atol=1e-2, rtol=1e-2):
    #         print("✅ Triton and Torch P match")
    #     else:
    #         print((ip - tip).abs().max())
    #         print((ip - tip).abs().max()/tip.abs().mean())
    #         print("❌ Triton and Torch P Image differ")
    # else:
    #     print((rp - trp).abs().max())
    #     print((rp - trp).abs().max()/trp.abs().mean())
    #     print("❌ Triton and Torch P Real differ")
        
    # triton.testing.assert_close(rp, trp, atol=1e-2, rtol=1e-2)
    # triton.testing.assert_close(ip, tip, atol=1e-2, rtol=1e-2)
    
    for tensor, torch_tensor, name in zip([rgv, igv, rgk, igk, rgq, igq], [trgv, tigv, trgk, tigk, trgq, tigq],
                                          ["Real GradV", "Image GradV", "Real GradK", "Image GradK", "Real GradQ", "Image GradQ"]):
        if  torch.allclose(tensor, torch_tensor, atol=1e-2, rtol=1e-2):
            print(f"✅ Triton and Torch {name} match")
        else:
            print(tensor)
            print(torch_tensor)
            print((tensor - torch_tensor).abs().max())
            print((tensor - torch_tensor).abs().max()/torch_tensor.abs().mean())
            print(f"❌ Triton and Torch {name} differ")
            
        triton.testing.assert_close(tensor, torch_tensor, atol=1e-2, rtol=1e-2)
    
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
    
    
batch_size = 1
num_heads = 32
seq_len = 1024
model_k = 128
r_scale = 1.0
i_scale = 1.0
main(batch_size, num_heads, seq_len, model_k, r_scale, i_scale)
