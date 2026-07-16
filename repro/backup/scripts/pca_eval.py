"""Evaluate PCA auto-research arms: frame-93 PSNR + exact avg-BPE accounting."""
import sys
import imageio.v3 as iio
import numpy as np

N = 29640  # LC chunk tokens
def bpe_side(r, res_block, asym, coeff_bits=2):
    b = 2.0 + 8.0/res_block + (8.0/res_block if asym else 0.0)
    if r > 0:
        b += (r*coeff_bits + 8)/128.0 + (1+r)*16.0/N
    else:
        b += 16.0/N
    return b

def avg_bpe(r, res_block, asym, v_pca):
    k = bpe_side(r, res_block, asym)
    v = bpe_side(r if v_pca else 0, res_block, asym)
    return (k+v)/2

ref = iio.imread('results/longcat/bf16/1-0/segment_1.mp4', index=93, plugin='pyav').astype(np.float64)/255
QVG_PSNR, QVG_BPE = 28.88, 2.326
print(f"{'arm':10s} {'BPE':>6s} {'PSNR':>7s}  verdict (target: PSNR>28.88 & BPE<2.326)")
for tag, (r, rb, asym, vp) in dict(
    pca_n1=(8,128,True,False), pca_n2=(8,128,True,True),
    pca_n3=(6,128,True,False), pca_n4=(4,128,True,True)).items():
    try:
        gen = iio.imread(f'results/pcastudy/{tag}/1-0/segment_1.mp4', index=93, plugin='pyav').astype(np.float64)/255
        psnr = 10*np.log10(1/np.mean((ref-gen)**2))
        b = avg_bpe(r, rb, asym, vp)
        win = "WIN" if (psnr > QVG_PSNR and b < QVG_BPE) else ("quality-only" if psnr > QVG_PSNR else "under")
        print(f"{tag:10s} {b:6.3f} {psnr:7.3f}  {win}")
    except Exception as e:
        print(f"{tag:10s} pending/err: {e}")
