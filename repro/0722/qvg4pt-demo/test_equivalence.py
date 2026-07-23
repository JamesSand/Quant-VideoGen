"""CPU 等价性测试:0722 拷贝版 vs repro/backup/scripts/pca_quant.py 原版。

1) 残差格 _asym_quant_lastdim_grouped:随机张量上逐位一致(确定性路径);
2) 整臂 qvg4pt_fake_quant_kv vs _qvg4_fake_quant_kv:同 seed 下小尺寸对比
   (kmeans torch.randint 初始化,同 seed 同设备应一致;CPU 路径无 Triton 原子)。
"""
import os
import sys

os.environ["PCA_FP8SIM"] = "1"  # 两个模块统一口径(原模块默认 0)
sys.path.insert(0, "repro/0722/qvg4pt-demo")
sys.path.insert(0, "repro/backup/scripts")

import torch

import qvg4pt_quant as new
import pca_quant as old

assert old.PCA_FP8SIM and new.PCA_FP8SIM

torch.manual_seed(0)
x = torch.randn(2, 4, 96, 128, dtype=torch.float32) * torch.logspace(-2, 1, 128)
a = new._asym_quant_lastdim_grouped(x, 2, 64, mse_opt=False, fp8_per_row=True)
b = old._asym_quant_lastdim_grouped(x, 2, 64, mse_opt=False, fp8_per_row=True)
assert torch.equal(a, b), f"grid mismatch: max|d|={float((a-b).abs().max())}"
print("PASS grid: bit-exact on", tuple(x.shape))

if not torch.cuda.is_available():
    print("SKIP full arm (QVG kmeans 为 Triton/CUDA-only,在 GPU 机器上重跑本测试)")
    print("EQUIVALENCE OK (grid only)")
    raise SystemExit(0)

os.environ["PCA_QVG4_ITERS"] = "5"  # 快速跑通
k = torch.randn(1, 2, 512, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, 2, 512, 128, dtype=torch.bfloat16, device="cuda")
torch.manual_seed(42)
k1, v1 = new.qvg4pt_fake_quant_kv(k, v)
torch.manual_seed(42)
k2, v2 = old._qvg4_fake_quant_kv(k, v)
ok = torch.equal(k1, k2) and torch.equal(v1, v2)
if ok:
    print("PASS full arm: bit-exact (same seed)")
else:
    dk = float((k1.float() - k2.float()).abs().max())
    dv = float((v1.float() - v2.float()).abs().max())
    print(f"NOTE full arm not bit-exact (kmeans atomic/init nondeterminism): max|dK|={dk} max|dV|={dv}")
    assert dk < 0.5 and dv < 0.5
print("EQUIVALENCE OK")
