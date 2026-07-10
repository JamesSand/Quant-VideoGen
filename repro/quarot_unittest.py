"""Unit tests for the QuaRot port: rotation identity, outlier redistribution
benefit, and error levels vs plain RTN on KV-like data."""

import torch

from quarot_quant import blockwise_rtn, hadamard, quarot_fake_quant


def relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


torch.manual_seed(0)
dev = "cuda"

# 1) Hadamard orthonormality / round-trip identity
H = hadamard(128, dev)
tf32 = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
eye_err = (H @ H.T - torch.eye(128, device=dev)).abs().max().item()
x = torch.randn(4, 8, 1024, 128, device=dev)
rt_err = relerr(x @ H @ H.T, x); torch.backends.cuda.matmul.allow_tf32 = tf32
print(f"H@H^T max dev from I: {eye_err:.2e}   rotate round-trip rel err: {rt_err:.2e}")
assert eye_err < 1e-5 and rt_err < 1e-6

# 2) Outlier redistribution: KV-like data with a few large channels
base = torch.randn(1, 8, 4096, 128, device=dev)
outlier = base.clone()
outlier[..., [3, 40, 77]] *= 12.0  # heavy per-channel outliers, typical of KV
for bits in (4, 2):
    e_plain = relerr(blockwise_rtn(outlier.float(), bits, 16, sym=False), outlier)
    e_rot = relerr(quarot_fake_quant(outlier, bits, rotate=True), outlier)
    print(f"INT{bits} outlier data: plain-RTN rel_err={e_plain:.4f}  QuaRot rel_err={e_rot:.4f}  gain={e_plain/e_rot:.2f}x")

# 3) Error levels on smooth structured data (like real KV)
t = torch.linspace(0, 6.28, 4096, device=dev)[:, None]
struct = torch.sin(t * torch.randn(1, 128, device=dev) * 3.0) * 2.0
kvlike = (struct[None, None] + 0.3 * torch.randn(1, 8, 4096, 128, device=dev)).to(torch.bfloat16)
for bits in (4, 2):
    e_plain = relerr(blockwise_rtn(kvlike.float(), bits, 16, sym=False), kvlike)
    e_rot = relerr(quarot_fake_quant(kvlike, bits, rotate=True), kvlike)
    print(f"INT{bits} structured data: plain-RTN rel_err={e_plain:.4f}  QuaRot rel_err={e_rot:.4f}")

print("ALL_QUAROT_UNIT_TESTS_PASSED")
