"""Encode/decode kernel speed on the SAME real KV chunk:
QVG (S=1,B=64,iters=100 & streaming iters=2), QVG-Pro (S=4,B=16,iters=100),
PCA-KV N4 (mean + top-4 self-cov PCA + coef 2bit + asym 2bit residual B=128, torch).
Data: real LongCat K chunk [1,32,29640,128] bf16 from results/kvplot/lc_kv.pt.
"""
import time
import torch

from quant_videogen.functions import triton_prq_quantize_tensor, triton_prq_dequantize_tensor

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

d = torch.load("results/kvplot/lc_kv.pt", map_location="cpu", weights_only=False)
x = d["selected"][24]["k"].to(dev, torch.bfloat16).contiguous()
if x.shape[2] < 29640:  # capture was capped at 12k tokens; tile to true LC chunk size
    x = torch.cat([x, x, x], dim=2)[:, :, :29640].contiguous()
print("tensor:", tuple(x.shape), x.dtype)

def bench(fn, n=3, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n

def qvg_fn(stages, block, iters):
    def f():
        return triton_prq_quantize_tensor(x, num_stages=stages, num_clusters=256,
                                          block_size=block, max_iters=iters,
                                          quantize_fn=lambda t: 2)
    return f

def n4_encode():
    X = x[0].float()  # [H,S,D]
    mu = X.mean(1, keepdim=True); Xc = X - mu
    cov = torch.einsum("hsd,hse->hde", Xc, Xc) / X.shape[1]
    _, vecs = torch.linalg.eigh(cov)
    Vr = vecs[:, :, -4:]
    c = torch.einsum("hsd,hdr->hsr", Xc, Vr)
    mn, mx = c.amin(-1, keepdim=True), c.amax(-1, keepdim=True)
    sc = ((mx-mn)/3).clamp_min(1e-8)
    ch = torch.clamp(torch.round((c-mn)/sc), 0, 3)*sc+mn
    res = Xc - torch.einsum("hsr,hdr->hsd", ch, Vr)
    rb = res.unsqueeze(-2)
    rmn, rmx = rb.amin(-1, keepdim=True), rb.amax(-1, keepdim=True)
    rsc = ((rmx-rmn)/3).clamp_min(1e-8)
    return torch.clamp(torch.round((rb-rmn)/rsc), 0, 3)

rows = []
rows.append(("QVG (S=1,B=64,iters=100, LC 配置)", bench(qvg_fn(1, 64, 100))))
rows.append(("QVG 流式 (S=1,B=64,iters=2, SF/HY 配置)", bench(qvg_fn(1, 64, 2))))
rows.append(("QVG-Pro (S=4,B=16,iters=100)", bench(qvg_fn(4, 16, 100))))
rows.append(("PCA-KV N4 encode (torch, fp32)", bench(n4_encode)))

packed = triton_prq_quantize_tensor(x, num_stages=1, num_clusters=256, block_size=64,
                                    max_iters=100, quantize_fn=lambda t: 2)
rows.append(("QVG decode (fused triton)", bench(lambda: triton_prq_dequantize_tensor(packed, 64, 2))))

# N4 decode: coef GEMM + residual dequant (naive torch, unpacked)
Xf = x[0].float()
mu = Xf.mean(1, keepdim=True); Xc = Xf - mu
cov = torch.einsum("hsd,hse->hde", Xc, Xc) / Xf.shape[1]
_, vecs = torch.linalg.eigh(cov); Vr = vecs[:, :, -4:]
c = torch.einsum("hsd,hdr->hsr", Xc, Vr)
H_, S_, D_ = Xf.shape
q = torch.randint(0, 4, (H_, S_, D_), device=dev, dtype=torch.uint8)
rsc = torch.rand(H_, S_, 1, device=dev); rmn = torch.rand(H_, S_, 1, device=dev)
def n4_decode():
    low = torch.einsum("hsr,hdr->hsd", c, Vr)
    return mu + low + (q.float()*rsc + rmn)
rows.append(("PCA-KV N4 decode (torch, 未打包)", bench(n4_decode)))

print()
for name, t in rows:
    print(f"{name:44s} {t*1000:9.1f} ms/层张量   全模型48层xKV: {t*96:6.2f} s")
