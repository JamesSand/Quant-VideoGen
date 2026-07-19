#!/usr/bin/env python3
"""M3 speed duel: Budget-PCA real encode/decode vs QVG triton-nstages-kmeans,
same dumped chunks, CUDA-event medians. Usage: bench_speed.py [n_chunks=3]
"""
import os, sys, json
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, "repro/0720/kernel")
sys.path.insert(0, ".")
from bp_quant import bp_encode, bp_encode_fast, bp_decode, bp_encode_packed256, bp_decode_packed256, bp_bytes
from bp_triton import triton_decode, triton_decode_packed256
from quant_videogen.functions import triton_prq_quantize_tensor, triton_prq_dequantize_tensor

NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DEV = "cuda"

CFG = {
    "lc": dict(iters=100, ours=dict(r=4, grid="asym", block=128, axis="channel")),
    "sf": dict(iters=2,   ours=dict(r=4, grid="asym", block=128, axis="channel")),
    "hy": dict(iters=2,   ours=None),  # packed256 path
}

def timed(fn, reps, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]

def rel_l2(a, b):
    return float((a.float()-b.float()).norm() / b.float().norm())

report = {}
for model in ("lc", "sf", "hy"):
    rows = []
    for ci in range(NCH):
        f = f"repro/0720/chunks/{model}/chunk_{ci:03d}.pt"
        if not os.path.exists(f): continue
        d = torch.load(f, map_location=DEV)
        k, v = d["k"].to(DEV), d["v"].to(DEV)
        iters = CFG[model]["iters"]
        qvg_reps = 10 if iters >= 100 else 30
        # ---- QVG encode (k and v, as the pipeline does) ----
        def qvg_enc():
            triton_prq_quantize_tensor(k, num_stages=1, num_clusters=256, block_size=64, max_iters=iters, quantize_fn=lambda t: 2)
            triton_prq_quantize_tensor(v, num_stages=1, num_clusters=256, block_size=64, max_iters=iters, quantize_fn=lambda t: 2)
        t_qvg_enc = timed(qvg_enc, qvg_reps, warmup=3)
        qk = triton_prq_quantize_tensor(k, num_stages=1, num_clusters=256, block_size=64, max_iters=iters, quantize_fn=lambda t: 2)
        qv = triton_prq_quantize_tensor(v, num_stages=1, num_clusters=256, block_size=64, max_iters=iters, quantize_fn=lambda t: 2)
        def qvg_dec():
            triton_prq_dequantize_tensor(qk, block_size=64, num_bits=2)
            triton_prq_dequantize_tensor(qv, block_size=64, num_bits=2)
        t_qvg_dec = timed(qvg_dec, 30)
        dk = triton_prq_dequantize_tensor(qk, block_size=64, num_bits=2)
        q_rel = rel_l2(dk, k)
        # ---- ours ----
        if model == "hy":
            k1, k2 = k[..., :128].contiguous(), k[..., 128:].contiguous()
            v1, v2 = v[..., :128].contiguous(), v[..., 128:].contiguous()
            def enc_k(): return {"halves": (bp_encode_fast(k1, r=9, grid="asym", block=64, axis="channel"),
                                            bp_encode_fast(k2, r=0, grid="ternary", block=64, axis="channel"))}
            def enc_v(): return {"halves": (bp_encode_fast(v1, r=9, grid="asym", block=128, axis="token"),
                                            bp_encode_fast(v2, r=0, grid="asym", block=128, axis="token"))}
            def ours_enc(): enc_k(); enc_v()
            ek, ev = enc_k(), enc_v()
            def ours_dec(): triton_decode_packed256(ek); triton_decode_packed256(ev)
            dec_k = triton_decode_packed256(ek)
            nbytes = bp_bytes(ek) + bp_bytes(ev)
        else:
            cfg = CFG[model]["ours"]
            def ours_enc(): bp_encode_fast(k, **cfg); bp_encode_fast(v, **cfg)
            ek, ev = bp_encode_fast(k, **cfg), bp_encode_fast(v, **cfg)
            def ours_dec(): triton_decode(ek); triton_decode(ev)
            dec_k = triton_decode(ek)
            nbytes = bp_bytes(ek) + bp_bytes(ev)
        t_our_enc = timed(ours_enc, 30)
        t_our_dec = timed(ours_dec, 30)
        o_rel = rel_l2(dec_k, k)
        bpe = nbytes*8/(k.numel()+v.numel())
        rows.append(dict(chunk=ci, qvg_enc_ms=round(t_qvg_enc,2), our_enc_ms=round(t_our_enc,2),
                         speedup_enc=round(t_qvg_enc/t_our_enc,1),
                         qvg_dec_ms=round(t_qvg_dec,2), our_dec_ms=round(t_our_dec,2),
                         qvg_relL2=round(q_rel,4), our_relL2=round(o_rel,4), our_bpe=round(bpe,4)))
        print(model, rows[-1], flush=True)
        del qk, qv, ek, ev, dk, dec_k
        torch.cuda.empty_cache()
    report[model] = rows
json.dump(report, open("repro/0720/kernel/bench_report.json", "w"), indent=1)
print("saved repro/0720/kernel/bench_report.json")
