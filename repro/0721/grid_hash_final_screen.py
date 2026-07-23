#!/usr/bin/env python3
"""Validate the speed-selected per-model hash configurations on all chunks."""
import json
import os
import sys

import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path[:0] = ["repro/0721", "repro/0720/kernel", "."]

from pca_grid_hash import grid_hash_encode_kv_fast, grid_hash_kv_bytes
from pca_grid_hash_triton import triton_grid_hash_decode


CFG = {
    "lc": dict(iters=5, refine=1, shared_labels=None),
    "sf": dict(iters=1, refine=0, shared_labels="v"),
    "hy": dict(iters=1, refine=0, shared_labels="v_rope"),
}
QVG_BPE = {"lc": 2.4639, "sf": 2.4063, "hy": 3.3199}


def metric(y, x):
    sse = float((y.float() - x.float()).square().sum())
    signal = float(x.float().square().sum())
    return {"sse": sse, "signal": signal, "rel_l2": (sse / signal) ** 0.5}


with open("repro/0721/grid_hash_screen.json") as f:
    reference = json.load(f)
qvg = {
    (r["model"], r["chunk"], r["kind"]): r["qvg"]
    for r in reference["rows"]
}

rows = []
for model, cfg in CFG.items():
    for chunk in range(8):
        data = torch.load(
            f"repro/0720/chunks/{model}/chunk_{chunk:03d}.pt",
            map_location="cuda",
        )
        state = grid_hash_encode_kv_fast(data["k"], data["v"], **cfg)
        pair_bytes = grid_hash_kv_bytes(state)
        cache_bpe = pair_bytes * 8 / (
            data["k"].numel() + data["v"].numel()
        )
        for kind in ("k", "v"):
            result = metric(triton_grid_hash_decode(state[kind]), data[kind])
            rows.append(
                {
                    "model": model,
                    "chunk": chunk,
                    "kind": kind,
                    "hash": result,
                    "qvg": qvg[(model, chunk, kind)],
                    "cache_bpe": cache_bpe,
                }
            )
            print(
                model,
                chunk,
                kind,
                f"qvg={qvg[(model, chunk, kind)]['rel_l2']:.6f}",
                f"hash={result['rel_l2']:.6f}",
                f"bpe={cache_bpe:.4f}",
                flush=True,
            )
        del data, state
        torch.cuda.empty_cache()

summary = {}
passed = True
for model in CFG:
    model_rows = [r for r in rows if r["model"] == model]
    sm = {"config": CFG[model]}
    for kind in ("k", "v"):
        rr = [r for r in model_rows if r["kind"] == kind]
        hsse = sum(r["hash"]["sse"] for r in rr)
        qsse = sum(r["qvg"]["sse"] for r in rr)
        sig = sum(r["hash"]["signal"] for r in rr)
        sm[kind] = {
            "qvg_rel_l2": (qsse / sig) ** 0.5,
            "hash_rel_l2": (hsse / sig) ** 0.5,
            "mse_ratio": hsse / qsse,
            "worst_chunk_mse_ratio": max(
                r["hash"]["sse"] / r["qvg"]["sse"] for r in rr
            ),
        }
    sm["cache_bpe"] = model_rows[0]["cache_bpe"]
    sm["bpe_pass"] = sm["cache_bpe"] <= QVG_BPE[model]
    sm["mse_pass"] = all(
        sm[kind]["mse_ratio"] <= 1
        and sm[kind]["worst_chunk_mse_ratio"] <= 1.5
        for kind in ("k", "v")
    )
    sm["pass"] = sm["bpe_pass"] and sm["mse_pass"]
    passed &= sm["pass"]
    summary[model] = sm

result = {"rows": rows, "summary": summary, "pass": passed}
with open("repro/0721/grid_hash_final_screen.json", "w") as f:
    json.dump(result, f, indent=2)

lines = [
    "# Speed-selected PCA-Grid Hash：最终 chunk 闸门",
    "",
    "| model | config | K relL2 QVG→hash | V relL2 QVG→hash | "
    "worst MSE ratio K/V | BPE hash≤QVG | 判定 |",
    "|---|---|---:|---:|---:|---:|---|",
]
for model, sm in summary.items():
    lines.append(
        f"| {model} | `{sm['config']}` | "
        f"{sm['k']['qvg_rel_l2']:.5f}→**{sm['k']['hash_rel_l2']:.5f}** | "
        f"{sm['v']['qvg_rel_l2']:.5f}→**{sm['v']['hash_rel_l2']:.5f}** | "
        f"{sm['k']['worst_chunk_mse_ratio']:.3f}/"
        f"{sm['v']['worst_chunk_mse_ratio']:.3f} | "
        f"{sm['cache_bpe']:.4f}≤{QVG_BPE[model]:.4f} | "
        f"{'PASS' if sm['pass'] else 'FAIL'} |"
    )
lines += ["", f"**最终 G1+G2：{'PASS' if passed else 'FAIL'}**", ""]
with open("repro/0721/grid_hash-final-screen.md", "w") as f:
    f.write("\n".join(lines))
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if passed else 1)
