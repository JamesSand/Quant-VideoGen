"""VBench imaging_quality (MUSIQ-SPAQ) replicated verbatim from
VBench/vbench/imaging_quality.py (mode='longer'), computed per frame so any
prefix-window average is available. Usage: vbench_iq.py <video> [<video>...]
Prints per-video prefix means at 350/700/1050/1400 frames (x100, paper scale).
"""
import sys
import imageio.v3 as iio
import numpy as np
import torch
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ

CKPT = "/home/zhizhousha/.cache/vbench_musiq/musiq_spaq_ckpt-358bb6af.pth"
WINDOWS = [350, 700, 1050, 1400]
dev = "cuda"
model = MUSIQ(pretrained_model_path=CKPT).to(dev)
model.training = False

for path in sys.argv[1:]:
    scores = []
    for frame in iio.imiter(path, plugin="pyav"):
        img = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).unsqueeze(0).float()
        _, _, h, w = img.size()
        if max(h, w) > 512:  # VBench 'longer' mode, antialias=False
            scale = 512.0 / max(h, w)
            img = transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(img)
        img = img / 255.0
        with torch.no_grad():
            scores.append(float(model(img.to(dev))))
    s = np.array(scores)
    outs = [f"{w}f={s[:w].mean():.2f}" if len(s) >= w else f"{w}f=--" for w in WINDOWS]
    print(f"{path}  n={len(s)}  " + "  ".join(outs) + f"  full={s.mean():.2f}")
