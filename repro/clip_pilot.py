"""Pilot: does percentile clipping help INT2/INT4 residual quantization?

Tensor-level A/B through the repo's OWN pipeline (kmeans smoothing + residual
quant), clip vs no-clip, on two data regimes. Also measures the storage cost
of the clip path's dense outlier-residual tensor.
"""

import torch

from quant_videogen.functions import (
    prq_quantize_tensor,
    triton_prq_quantize_tensor,
    triton_prq_dequantize_tensor,
)
from quant_videogen.sim.quant.lowbit_quantize import (
    blockwise_int2_quantize_triton,
    blockwise_int4_quantize_triton,
)


def relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


torch.manual_seed(0)
dev = "cuda"
B, H, S, D = 1, 8, 4096, 128

t = torch.linspace(0, 6.28, S, device=dev)[:, None]
struct = torch.sin(t * torch.randn(1, D, device=dev) * 3.0) * 2.0
smooth = (struct[None, None] + 0.3 * torch.randn(B, H, S, D, device=dev)).to(torch.bfloat16)

heavy = smooth.clone().float()
idx = torch.rand_like(heavy) < 0.001          # 0.1% random spike outliers
heavy[idx] *= 15.0
heavy = heavy.to(torch.bfloat16)

for name, x in [("smooth(video-KV-like)", smooth), ("heavy-outlier(0.1% x15)", heavy)]:
    print(f"===== {name} =====")
    for bits, fn in [(2, blockwise_int2_quantize_triton), (4, blockwise_int4_quantize_triton)]:
        base = prq_quantize_tensor(x, 1, 256, 100,
                                   quantize_fn=lambda r: fn(r.contiguous(), block_size=64))
        e0 = relerr(base, x)
        row = f"INT{bits}: no-clip={e0:.4f}"
        for pct in (90.0, 92.0, 94.0, 96.0, 98.0, 99.0, 100.0):
            clip = prq_quantize_tensor(x, 1, 256, 100,
                                       quantize_fn=lambda r, p=pct: fn(r.contiguous(), block_size=64,
                                                                        use_percentile_clipping=True, percentile=p),
                                       )
            row += f"  clip@{pct}={relerr(clip, x):.4f}"
        print(row)

    # real (triton) path with its shipped clip wiring, incl. storage accounting
    packed = triton_prq_quantize_tensor(smooth if name.startswith("smooth") else heavy,
                                        1, 256, 64, max_iters=100,
                                        quantize_fn=lambda t_: 2,
                                        use_percentile_clipping=True, percentile=99.0)
    rec = triton_prq_dequantize_tensor(packed, 64, 2, output_dtype=torch.bfloat16)
    e_clip_real = relerr(rec, x)
    extra = packed.get("residual")
    extra_mb = extra.numel() * extra.element_size() / 1e6 if extra is not None else 0.0
    dense_kv_mb = x.numel() * 2 / 1e6
    print(f"triton INT2 clip@99: rel_err={e_clip_real:.4f}; clip 残差张量额外存储 = {extra_mb:.1f} MB "
          f"(bf16 原 KV 为 {dense_kv_mb:.1f} MB → 若稠密存储,压缩率被拉回 ~{dense_kv_mb/max(extra_mb,1e-9):.1f}x 以下)")
