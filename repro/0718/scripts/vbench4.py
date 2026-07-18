#!/usr/bin/env python3
"""VBench 4-dim scorer: Background Consistency (CLIP B/32), Subject Consistency
(DINO ViT-B/16), Aesthetic Quality (CLIP L/14 + LAION linear head), Imaging
Quality (MUSIQ-SPAQ, same as vbench_iq.py). VBench formulas:
  consistency = mean_i>=1 0.5*(cos(f_i,f_0)+cos(f_i,f_{i-1}))  (x100)
  aesthetic   = mean_frames(head(clip_feat))/10                 (x100)
  imaging     = mean_frames(MUSIQ)/100                          (x100)
Usage: vbench4.py <video.mp4> [...]; optional --max-frames N (SF: 700).
Caches per-video into repro/0718/npz/vbench4.json.
"""
import os, sys, json
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch_rw"))
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
CACHE = "repro/0718/npz/vbench4.json"
dev = "cuda"

args = [a for a in sys.argv[1:]]
max_frames = 0
if "--max-frames" in args:
    i = args.index("--max-frames"); max_frames = int(args[i+1]); del args[i:i+2]

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [p for p in args if f"{p}::{max_frames}" not in cache]

if todo:
    import open_clip
    clip_b, _, prep_b = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_l, _, prep_l = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_b = clip_b.to(dev).eval(); clip_l = clip_l.to(dev).eval()
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", verbose=False, trust_repo=True).to(dev).eval()
    from pyiqa.archs.musiq_arch import MUSIQ
    musiq = MUSIQ(pretrained_model_path=os.path.expanduser(
        "~/.cache/vbench_musiq/musiq_spaq_ckpt-358bb6af.pth")).to(dev).eval()
    aes_head = torch.nn.Linear(768, 1)
    sd = torch.load(os.path.expanduser("~/.cache/vbench_extra/aesthetic_l14_linear.pth"), map_location="cpu")
    # improved-aesthetic-predictor is an MLP; accept both MLP and linear formats
    if any(k.startswith("layers") for k in sd):
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
                    torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
                    torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                    torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
            def forward(self, x): return self.layers(x)
        aes_head = MLP()
    aes_head.load_state_dict(sd); aes_head = aes_head.to(dev).eval()

    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    DMEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    DSTD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def frames_tensor(path):
        fs = []
        for i, f in enumerate(iio.imiter(path, plugin="pyav")):
            if max_frames and i >= max_frames: break
            fs.append(torch.from_numpy(np.asarray(f)).permute(2, 0, 1))
        return torch.stack(fs).float() / 255  # [N,3,H,W] cpu

    def consistency(feats):
        f = F.normalize(feats, dim=-1)
        sim = 0.5 * ((f[1:] @ f[0]) + (f[1:] * f[:-1]).sum(-1))
        return float(sim.clamp(0, 1).mean()) * 100

    @torch.no_grad()
    def score(path):
        X = frames_tensor(path)
        N = X.shape[0]
        bc_f, sc_f, aq_f, iq_f = [], [], [], []
        B = 32
        for i in range(0, N, B):
            x = X[i:i+B].to(dev)
            x224 = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            bc_f.append(clip_b.encode_image((x224 - MEAN) / STD).float())
            sc_f.append(dino((x224 - DMEAN) / DSTD).float())
            lf = clip_l.encode_image((x224 - MEAN) / STD).float()
            aq_f.append(aes_head(F.normalize(lf, dim=-1)).squeeze(-1))
            h, w = x.shape[-2:]
            if max(h, w) > 512:  # VBench 'longer' mode, antialias=False (== vbench_iq.py)
                sc = 512.0 / max(h, w)
                xi = F.interpolate(x, size=(int(h*sc), int(w*sc)), mode="bilinear", align_corners=False)
            else:
                xi = x
            iq_f.append(musiq(xi))
        bc = consistency(torch.cat(bc_f)); sc = consistency(torch.cat(sc_f))
        aq = float(torch.cat(aq_f).mean()) * 10.0     # /10 * 100
        iq = float(torch.cat(iq_f).mean())            # MUSIQ-SPAQ is 0-100 already
        return {"BC": round(bc, 2), "SC": round(sc, 2), "AQ": round(aq, 2), "IQ": round(iq, 2), "n": N}

    for p in todo:
        try:
            cache[f"{p}::{max_frames}"] = score(p)
            json.dump(cache, open(CACHE, "w"), indent=0)
            print(p, cache[f"{p}::{max_frames}"], flush=True)
        except Exception as e:
            print(f"FAIL {p}: {e}", flush=True)

for p in args:
    k = f"{p}::{max_frames}"
    if k in cache: print(p, cache[k])
