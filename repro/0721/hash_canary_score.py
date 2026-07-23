#!/usr/bin/env python3
"""Score one paired sample per model before launching MP100."""
import glob
import json
import os
import subprocess

import imageio.v3 as iio
import lpips
import numpy as np
import torch
import torch.nn.functional as F

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")


def one(pattern):
    values = glob.glob(pattern)
    if not values:
        raise FileNotFoundError(pattern)
    return values[0]


PATHS = {
    "lc": {
        "ref": one("results/multiprompt/lc/bf16_rep0/p1/*/segment_1.mp4"),
        "qvg": one("results/multiprompt/lc/qvg_rep0/p1/*/segment_1.mp4"),
        "hash": one("results/multiprompt/lc/pcahash_rep0/p1/*/segment_1.mp4"),
    },
    "sf": {
        "qvg": one("results/multiprompt/sf/qvg_rep0_f180/p1/*.mp4"),
        "hash": one("results/multiprompt/sf/pcahash_rep0_f180/p1/*.mp4"),
    },
    "hy": {
        "ref": "results/multiprompt/hy/bf16_s0/0-0.mp4",
        "qvg": "results/multiprompt/hy/qvg_s0/0-0.mp4",
        "hash": "results/multiprompt/hy/pcahash_s0/0-0.mp4",
    },
}


def ssim_batch(a, b):
    c1, c2 = 0.01**2, 0.03**2
    m1, m2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    v1 = F.avg_pool2d(a * a, 11, 1, 5) - m1**2
    v2 = F.avg_pool2d(b * b, 11, 1, 5) - m2**2
    v12 = F.avg_pool2d(a * b, 11, 1, 5) - m1 * m2
    return (
        ((2 * m1 * m2 + c1) * (2 * v12 + c2))
        / ((m1**2 + m2**2 + c1) * (v1 + v2 + c2))
    ).mean((1, 2, 3))


lp = lpips.LPIPS(net="vgg").cuda().eval()


def ref_metrics(ref_path, test_path):
    values = []
    batch_a, batch_b = [], []

    def flush():
        if not batch_a:
            return
        a = torch.stack(batch_a).cuda(non_blocking=True)
        b = torch.stack(batch_b).cuda(non_blocking=True)
        with torch.no_grad():
            mse = (a - b).square().mean((1, 2, 3))
            pval = 10 * torch.log10(1 / mse)
            sval = ssim_batch(a, b)
            lval = lp(a, b).reshape(-1)
        values.extend(
            zip(
                pval.cpu().tolist(),
                sval.cpu().tolist(),
                lval.cpu().tolist(),
            )
        )
        batch_a.clear()
        batch_b.clear()

    for ref_frame, test_frame in zip(
        iio.imiter(ref_path, plugin="pyav"),
        iio.imiter(test_path, plugin="pyav"),
    ):
        batch_a.append(
            torch.from_numpy(np.asarray(ref_frame)).float().permute(2, 0, 1) / 255
        )
        batch_b.append(
            torch.from_numpy(np.asarray(test_frame)).float().permute(2, 0, 1) / 255
        )
        if len(batch_a) == 8:
            flush()
    flush()
    return np.asarray(values)


def ref_metric_at(ref_path, test_path, wanted):
    for idx, (ref_frame, test_frame) in enumerate(
        zip(
            iio.imiter(ref_path, plugin="pyav"),
            iio.imiter(test_path, plugin="pyav"),
        )
    ):
        if idx != wanted:
            continue
        a = torch.from_numpy(np.asarray(ref_frame)).float().permute(2, 0, 1)[None].cuda() / 255
        b = torch.from_numpy(np.asarray(test_frame)).float().permute(2, 0, 1)[None].cuda() / 255
        with torch.no_grad():
            mse = (a - b).square().mean()
            return np.asarray(
                [
                    float(10 * torch.log10(1 / mse)),
                    float(ssim_batch(a, b)[0]),
                    float(lp(a, b).reshape(-1)[0]),
                ]
            )
    raise IndexError(wanted)


ref_cache_path = "repro/0721/hash_canary_ref.json"
ref_cache = (
    json.load(open(ref_cache_path)) if os.path.exists(ref_cache_path) else {}
)
for arm in ("qvg", "hash"):
    key = f"lc_{arm}"
    if key not in ref_cache:
        ref_cache[key] = ref_metric_at(
            PATHS["lc"]["ref"], PATHS["lc"][arm], 93
        ).tolist()
        json.dump(ref_cache, open(ref_cache_path, "w"), indent=2)
if "hy_qvg" not in ref_cache:
    cached = np.load("repro/0718/npz/hy_s0_qvg.npz")
    ref_cache["hy_qvg"] = [
        float(cached["psnr"][13:].mean()),
        float(cached["ssim"][13:].mean()),
        float(cached["lpips"][13:].mean()),
    ]
    json.dump(ref_cache, open(ref_cache_path, "w"), indent=2)
if "hy_hash" not in ref_cache:
    ref_cache["hy_hash"] = ref_metrics(
        PATHS["hy"]["ref"], PATHS["hy"]["hash"]
    )[13:].mean(0).tolist()
    json.dump(ref_cache, open(ref_cache_path, "w"), indent=2)

result = {"paths": PATHS, "metrics": {}}
for model in ("lc", "hy"):
    result["metrics"][model] = {}
    for arm in ("qvg", "hash"):
        selected = ref_cache[f"{model}_{arm}"]
        result["metrics"][model][arm] = {
            "PSNR": float(selected[0]),
            "SSIM": float(selected[1]),
            "LPIPS": float(selected[2]),
        }

cache_path = "repro/0721/hash_canary_vbench.json"
env = dict(os.environ, VBENCH4_CACHE=cache_path)
subprocess.run(
    [".venv/bin/python", "repro/0718/scripts/vbench4.py"]
    + [PATHS[m][a] for m in ("lc", "hy") for a in ("qvg", "hash")],
    check=True,
    env=env,
)
subprocess.run(
    [".venv/bin/python", "repro/0718/scripts/vbench4.py"]
    + [PATHS["sf"][a] for a in ("qvg", "hash")]
    + ["--max-frames", "700"],
    check=True,
    env=env,
)
with open(cache_path) as f:
    vb = json.load(f)
for model in ("lc", "sf", "hy"):
    result["metrics"].setdefault(model, {})
    max_frames = 700 if model == "sf" else 0
    for arm in ("qvg", "hash"):
        result["metrics"][model].setdefault(arm, {})
        result["metrics"][model][arm].update(
            vb[f"{PATHS[model][arm]}::{max_frames}"]
        )

deltas = {}
passed = True
for model, arms in result["metrics"].items():
    deltas[model] = {}
    for metric in arms["hash"]:
        if metric == "n":
            continue
        delta = arms["hash"][metric] - arms["qvg"][metric]
        deltas[model][metric] = delta
        if metric == "PSNR":
            passed &= delta >= -0.10
        elif metric == "SSIM":
            passed &= delta >= -0.002
        elif metric == "LPIPS":
            passed &= delta <= 0.003
        elif metric == "BC":
            passed &= delta >= -0.20
        else:
            passed &= delta >= -0.30
result["deltas_hash_minus_qvg"] = deltas
result["pass"] = bool(passed)
with open("repro/0721/hash_canary_score.json", "w") as f:
    json.dump(result, f, indent=2)

lines = [
    "# PCA-Grid Hash paired canary",
    "",
    "| model | metric | QVG | hash | Δ(hash-QVG) |",
    "|---|---|---:|---:|---:|",
]
for model, arms in result["metrics"].items():
    for metric, value in arms["hash"].items():
        if metric == "n":
            continue
        lines.append(
            f"| {model} | {metric} | {arms['qvg'][metric]:.4f} | "
            f"{value:.4f} | {value-arms['qvg'][metric]:+.4f} |"
        )
lines += ["", f"**Canary：{'PASS' if passed else 'FAIL'}**", ""]
with open("repro/0721/hash-canary-report.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
raise SystemExit(0 if passed else 1)
