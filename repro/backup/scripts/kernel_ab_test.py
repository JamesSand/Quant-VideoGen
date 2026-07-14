"""A/B correctness test: triton real-quant path vs pure-torch sim path.

Same input tensor, same algorithm (1-stage k-means smoothing + blockwise
residual quant, K=256, B=64), INT2 and INT4. If the triton path's
reconstruction error is much larger than the sim path's, the triton
kernels are numerically broken in this environment.
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

torch.manual_seed(0)
B, H, S, D = 1, 8, 4096, 128

# KV-like structured data: shared smooth structure across tokens + noise
t = torch.linspace(0, 6.28, S, device="cuda")[:, None]
struct = torch.sin(t * torch.randn(1, D, device="cuda") * 3.0) * 2.0
x = (struct[None, None] + 0.3 * torch.randn(B, H, S, D, device="cuda")).to(
    torch.bfloat16
)


def relerr(recon, ref):
    recon, ref = recon.float(), ref.float()
    return ((recon - ref).norm() / ref.norm()).item()


print(f"input: {tuple(x.shape)} bf16, |x|_rms={x.float().pow(2).mean().sqrt():.4f}")

for bits, blockfn in [(2, blockwise_int2_quantize_triton), (4, blockwise_int4_quantize_triton)]:
    # --- sim path (fake quant, torch gather/subtract + triton blockwise quantizer)
    recon_sim = prq_quantize_tensor(
        x, num_stages=1, codebook_size=256, kmeans_max_iters=100,
        quantize_fn=lambda r: blockfn(r.contiguous(), block_size=64),
    )
    e_sim = relerr(recon_sim, x)

    # --- triton real path (packed int2/int4 + fp8 scales, kernels under test)
    packed = triton_prq_quantize_tensor(
        x, num_stages=1, num_clusters=256, block_size=64, max_iters=100,
        quantize_fn=lambda t_, b=bits: b,
    )
    recon_tri = triton_prq_dequantize_tensor(packed, 64, bits, output_dtype=torch.bfloat16)
    e_tri = relerr(recon_tri, x)

    # --- naive RTN (no smoothing) for context
    e_rtn = relerr(blockfn(x.float().contiguous(), block_size=64), x)

    print(f"INT{bits}: rel_err sim={e_sim:.4f}  triton={e_tri:.4f}  "
          f"rtn(no-smoothing)={e_rtn:.4f}  ratio triton/sim={e_tri/max(e_sim,1e-9):.2f}")
