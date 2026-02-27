#!/usr/bin/env python3
"""Test and benchmark Level6 varlen flash attention vs FA3."""
import math
import os
import sys
import subprocess
import ctypes
import random
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CU_PATH = os.path.join(SCRIPT_DIR, "fa3.cu")
SO_PATH = os.path.join(SCRIPT_DIR, "fa3.so")

# Test shapes: (batch, seqlen, nheads, nkv_heads, headdim)
# Includes both uniform and random variable-length cases
TEST_SHAPES = [
    # Uniform length (all batches same seqlen)
    {"batch": 1, "seqlen": 256, "nheads": 4, "nkv_heads": 1, "headdim": 128},
    {"batch": 2, "seqlen": 512, "nheads": 8, "nkv_heads": 2, "headdim": 128},
    {"batch": 4, "seqlen": 1024, "nheads": 16, "nkv_heads": 4, "headdim": 128},
    {"batch": 8, "seqlen": 2048, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "seqlen": 4096, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "seqlen": 4096, "nheads": 64, "nkv_heads": 8, "headdim": 128},
    # Random variable length (seqlens is list of per-batch lengths)
    {"seqlens": [64, 128, 256], "nheads": 4, "nkv_heads": 2, "headdim": 128},
    {"seqlens": [100, 200, 50, 300], "nheads": 8, "nkv_heads": 8, "headdim": 128},
    # Extreme variance (random lengths in range)
    {"batch": 16, "min_len": 10, "max_len": 10000, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 32, "min_len": 50, "max_len": 5000, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 64, "min_len": 100, "max_len": 2000, "nheads": 32, "nkv_heads": 8, "headdim": 128},
    {"batch": 8, "min_len": 10, "max_len": 10000, "nheads": 32, "nkv_heads": 8, "headdim": 128},
]


def ensure_built():
    """Compile if needed."""
    need_build = not os.path.isfile(SO_PATH)
    if not need_build and os.path.isfile(CU_PATH):
        need_build = os.path.getmtime(SO_PATH) < os.path.getmtime(CU_PATH)
    if not need_build:
        return
    cmd = ["nvcc", "--shared", "-Xcompiler", "-fPIC", "-O3", "-std=c++17", "--use_fast_math",
           "-gencode", "arch=compute_90a,code=sm_90a", "-DNDEBUG", CU_PATH, "-o", SO_PATH]
    print("Building", SO_PATH, "...")
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        sys.exit(r.returncode)


def load_lib():
    ensure_built()
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so', mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(SO_PATH)
    lib.flash_attn_fwd_varlen.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_bool, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.flash_attn_fwd_varlen.restype = None
    return lib


def get_fa3_func():
    """Import official FA3 varlen function if available."""
    try:
        p = "/path/to/flash-attention/hopper"
        if p not in sys.path:
            sys.path.insert(0, p)
        from flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func
    except Exception:
        return None


def create_test_data(shape, device):
    """Create Q,K,V,cu_seqlens from shape spec. Returns (q,k,v,cu_q,cu_k,batch,max_q,max_k,seq_lens)."""
    dtype = torch.bfloat16
    nheads, nkv_heads, headdim = shape["nheads"], shape["nkv_heads"], shape["headdim"]
    
    if "seqlens" in shape:
        # Explicit per-batch lengths
        seq_lens = shape["seqlens"]
        batch = len(seq_lens)
    elif "min_len" in shape:
        # Random lengths in range
        random.seed(42)
        seq_lens = [random.randint(shape["min_len"], shape["max_len"]) for _ in range(shape["batch"])]
        batch = shape["batch"]
    else:
        # Uniform length
        seq_lens = [shape["seqlen"]] * shape["batch"]
        batch = shape["batch"]
    
    total = sum(seq_lens)
    max_len = max(seq_lens)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
    
    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total, nkv_heads, headdim, dtype=dtype, device=device)
    v = torch.randn(total, nkv_heads, headdim, dtype=dtype, device=device)
    
    return q, k, v, cu, cu, batch, max_len, max_len, seq_lens


def pytorch_ref_varlen(q, k, v, cu_q, cu_k, causal=True):
    """Per-sequence PyTorch reference with GQA support."""
    batch = cu_q.size(0) - 1
    out_list = []
    for i in range(batch):
        sq, eq = cu_q[i].item(), cu_q[i + 1].item()
        sk, ek = cu_k[i].item(), cu_k[i + 1].item()
        q_i = q[sq:eq].unsqueeze(0).float()
        k_i = k[sk:ek].unsqueeze(0).float()
        v_i = v[sk:ek].unsqueeze(0).float()
        h, nkv = q_i.size(2), k_i.size(2)
        if h != nkv:
            k_i = k_i.repeat_interleave(h // nkv, dim=2)
            v_i = v_i.repeat_interleave(h // nkv, dim=2)
        scale = 1.0 / math.sqrt(q_i.size(-1))
        scores = torch.einsum("bqhd,bkhd->bhqk", q_i, k_i) * scale
        if causal:
            mask = torch.triu(torch.ones(eq - sq, ek - sk, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o_i = torch.einsum("bhqk,bkhd->bqhd", attn, v_i).to(q.dtype)
        out_list.append(o_i.squeeze(0))
    return torch.cat(out_list, dim=0)


def run_ours(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k):
    """Run our kernel."""
    total_q = q.size(0)
    nheads, nkv_heads, headdim = q.size(1), k.size(1), q.size(2)
    out = torch.zeros_like(q)
    lse = torch.empty(nheads * total_q, dtype=torch.float32, device=q.device)
    sem = torch.zeros(1, dtype=torch.int32, device=q.device)
    scale = 1.0 / math.sqrt(headdim)
    stream = torch.cuda.current_stream().cuda_stream
    lib.flash_attn_fwd_varlen(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        cu_q.data_ptr(), cu_k.data_ptr(),
        total_q, total_q, max_q, max_k, batch, nheads, nkv_heads, headdim,
        scale, True, stream, lse.data_ptr(), sem.data_ptr(), None, None, None,
    )
    torch.cuda.synchronize()
    return out


def run_fa3(fa3_func, q, k, v, cu_q, cu_k, max_q, max_k):
    """Run FA3."""
    out = fa3_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    return out


def compute_flops(seq_lens, nheads, headdim, causal=True):
    """Compute FLOPs for attention."""
    total = 0
    for s in seq_lens:
        tokens = s * (s + 1) // 2 if causal else s * s
        total += 4 * tokens * headdim * nheads
    return total


def shape_str(shape):
    """Format shape for display."""
    if "seqlens" in shape:
        return "seqlens=%s h=%d/%d" % (shape["seqlens"], shape["nheads"], shape["nkv_heads"])
    elif "min_len" in shape:
        return "b=%d len=%d-%d h=%d/%d" % (shape["batch"], shape["min_len"], shape["max_len"], shape["nheads"], shape["nkv_heads"])
    else:
        return "b=%d s=%d h=%d/%d" % (shape["batch"], shape["seqlen"], shape["nheads"], shape["nkv_heads"])


def test_correctness():
    """Test correctness for all shapes."""
    print("=" * 70)
    print("Correctness Test")
    print("=" * 70)
    
    lib = load_lib()
    fa3_func = get_fa3_func()
    device = torch.device("cuda")
    all_pass = True
    
    for shape in TEST_SHAPES:
        torch.manual_seed(42)
        q, k, v, cu_q, cu_k, batch, max_q, max_k, seq_lens = create_test_data(shape, device)
        
        # Our output
        out_ours = run_ours(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k)
        
        # Reference (FA3 if available, else PyTorch)
        if fa3_func:
            ref = run_fa3(fa3_func, q, k, v, cu_q, cu_k, max_q, max_k)
            ref_name = "FA3"
        else:
            ref = pytorch_ref_varlen(q, k, v, cu_q, cu_k, causal=True)
            ref_name = "PyTorch"
        
        max_diff = (out_ours.float() - ref.float()).abs().max().item()
        ok = max_diff < 0.02
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print("  %-45s vs %-7s diff=%.6f %s" % (shape_str(shape), ref_name, max_diff, status))
    
    print()
    return all_pass


def benchmark_all(warmup=50, iters=30):
    """Benchmark all shapes."""
    print("=" * 70)
    print("Performance Benchmark (warmup=%d, iters=%d)" % (warmup, iters))
    print("=" * 70)
    print("  %-45s %10s %10s %8s" % ("Shape", "FA3", "Ours", "Ratio"))
    print("  " + "-" * 75)
    
    lib = load_lib()
    fa3_func = get_fa3_func()
    device = torch.device("cuda")
    
    for shape in TEST_SHAPES:
        torch.manual_seed(42)
        q, k, v, cu_q, cu_k, batch, max_q, max_k, seq_lens = create_test_data(shape, device)
        nheads, headdim = shape["nheads"], shape["headdim"]
        flops = compute_flops(seq_lens, nheads, headdim, causal=True)
        
        # Benchmark FA3
        fa3_tflops = 0.0
        if fa3_func:
            for _ in range(warmup):
                run_fa3(fa3_func, q, k, v, cu_q, cu_k, max_q, max_k)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                run_fa3(fa3_func, q, k, v, cu_q, cu_k, max_q, max_k)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            fa3_tflops = flops / (ms * 1e-3) / 1e12
        
        # Benchmark ours
        for _ in range(warmup):
            run_ours(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_ours(lib, q, k, v, cu_q, cu_k, batch, max_q, max_k)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        our_tflops = flops / (ms * 1e-3) / 1e12
        
        ratio = our_tflops / fa3_tflops if fa3_tflops > 0 else 0
        fa3_str = "%.1f" % fa3_tflops if fa3_tflops > 0 else "N/A"
        print("  %-45s %10s %10.1f %7.2fx" % (shape_str(shape), fa3_str, our_tflops, ratio))
    
    print()


if __name__ == "__main__":
    if not test_correctness():
        print("CORRECTNESS FAILED!")
        sys.exit(1)
    benchmark_all()
