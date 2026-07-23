#!/usr/bin/env python3
"""Run the preregistered chunk-level MSE and BPE screen."""
import json
import os
import sys

import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path[:0] = ["repro/0721", "repro/0720/kernel", "."]

from bp_quant import bp_bytes, bp_encode_fast
from bp_triton import triton_decode, triton_decode_packed256
from pca_grid_hash import grid_hash_bytes, grid_hash_decode, grid_hash_encode
from quant_videogen.functions import (
    triton_prq_dequantize_tensor,
    triton_prq_quantize_tensor,
)


QVG_BPE = {"lc": 2.4639, "sf": 2.4063, "hy": 3.3199}
ITERS = {"lc": 100, "sf": 2, "hy": 2}


def metrics(y, x):
    diff = y.float() - x.float()
    hsse = diff.square().sum((0, 2, 3))
    hsig = x.float().square().sum((0, 2, 3))
    sse, sig = float(hsse.sum()), float(hsig.sum())
    return {
        "sse": sse,
        "signal": sig,
        "rel_l2": (sse / sig) ** 0.5,
        "head_rel_l2": (hsse / hsig.clamp_min(1e-30)).sqrt().cpu().tolist(),
    }


def current_bp(model, kind, x):
    if model != "hy":
        state = bp_encode_fast(
            x, r=4, grid="asym", block=128, axis="channel"
        )
        return triton_decode(state), bp_bytes(state)
    x1, x2 = x[..., :128].contiguous(), x[..., 128:].contiguous()
    if kind == "k":
        d1 = bp_encode_fast(
            x1, r=9, grid="asym", block=64, axis="channel"
        )
        d2 = bp_encode_fast(
            x2, r=0, grid="ternary", block=64, axis="channel"
        )
    else:
        d1 = bp_encode_fast(
            x1, r=9, grid="asym", block=128, axis="token"
        )
        d2 = bp_encode_fast(
            x2, r=0, grid="asym", block=128, axis="token"
        )
    state = {"halves": (d1, d2)}
    return triton_decode_packed256(state), bp_bytes(d1) + bp_bytes(d2)


def run():
    rows = []
    for model in ("lc", "sf", "hy"):
        for chunk in range(8):
            path = f"repro/0720/chunks/{model}/chunk_{chunk:03d}.pt"
            data = torch.load(path, map_location="cuda")
            for kind in ("k", "v"):
                x = data[kind]
                torch.manual_seed(720_000 + chunk + (kind == "v"))
                qvg = triton_prq_quantize_tensor(
                    x,
                    num_stages=1,
                    num_clusters=256,
                    block_size=64,
                    max_iters=ITERS[model],
                    quantize_fn=lambda _: 2,
                )
                qvg_y = triton_prq_dequantize_tensor(qvg, 64, 2)

                state = grid_hash_encode(
                    x,
                    r=4,
                    binning="gaussian",
                    block=64,
                    iters=5,
                    refine=1,
                )
                hash_y = grid_hash_decode(state)
                bp_y, bp_nbytes = current_bp(model, kind, x)

                row = {
                    "model": model,
                    "chunk": chunk,
                    "kind": kind,
                    "numel": x.numel(),
                    "qvg": metrics(qvg_y, x),
                    "hash": metrics(hash_y, x),
                    "bp": metrics(bp_y, x),
                    "hash_bpe": grid_hash_bytes(state) * 8 / x.numel(),
                    "bp_bpe": bp_nbytes * 8 / x.numel(),
                }
                rows.append(row)
                print(
                    model,
                    chunk,
                    kind,
                    f"qvg={row['qvg']['rel_l2']:.6f}",
                    f"hash={row['hash']['rel_l2']:.6f}",
                    f"bp={row['bp']['rel_l2']:.6f}",
                    f"bpe={row['hash_bpe']:.4f}",
                    flush=True,
                )
                del qvg, qvg_y, state, hash_y, bp_y
            del data
            torch.cuda.empty_cache()

    summary = {}
    all_pass = True
    for model in ("lc", "sf", "hy"):
        sm = {"k": {}, "v": {}}
        model_rows = [r for r in rows if r["model"] == model]
        for kind in ("k", "v"):
            rr = [r for r in model_rows if r["kind"] == kind]
            q_sse = sum(r["qvg"]["sse"] for r in rr)
            h_sse = sum(r["hash"]["sse"] for r in rr)
            signal = sum(r["qvg"]["signal"] for r in rr)
            worst = max(r["hash"]["sse"] / r["qvg"]["sse"] for r in rr)
            sm[kind] = {
                "qvg_rel_l2": (q_sse / signal) ** 0.5,
                "hash_rel_l2": (h_sse / signal) ** 0.5,
                "mse_ratio": h_sse / q_sse,
                "worst_chunk_mse_ratio": worst,
            }
        cache_bpe = sum(r["hash_bpe"] for r in model_rows) / len(model_rows)
        sm["hash_cache_bpe"] = cache_bpe
        sm["qvg_cache_bpe"] = QVG_BPE[model]
        sm["bpe_pass"] = cache_bpe <= QVG_BPE[model]
        sm["mse_pass"] = all(
            sm[kind]["mse_ratio"] <= 1.0
            and sm[kind]["worst_chunk_mse_ratio"] <= 1.5
            for kind in ("k", "v")
        )
        sm["pass"] = sm["bpe_pass"] and sm["mse_pass"]
        all_pass &= sm["pass"]
        summary[model] = sm

    result = {"config": {"r": 4, "block": 64, "refine": 1}, "rows": rows,
              "summary": summary, "pass": all_pass}
    with open("repro/0721/grid_hash_screen.json", "w") as f:
        json.dump(result, f, indent=2)

    lines = [
        "# PCA-Grid Hash chunk 筛选",
        "",
        "| model | K relL2 QVG→hash | V relL2 QVG→hash | "
        "worst chunk MSE ratio K/V | BPE hash≤QVG | 判定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, sm in summary.items():
        lines.append(
            f"| {model} | {sm['k']['qvg_rel_l2']:.5f}→"
            f"**{sm['k']['hash_rel_l2']:.5f}** | "
            f"{sm['v']['qvg_rel_l2']:.5f}→**{sm['v']['hash_rel_l2']:.5f}** | "
            f"{sm['k']['worst_chunk_mse_ratio']:.3f}/"
            f"{sm['v']['worst_chunk_mse_ratio']:.3f} | "
            f"{sm['hash_cache_bpe']:.4f}≤{sm['qvg_cache_bpe']:.4f} | "
            f"{'PASS' if sm['pass'] else 'FAIL'} |"
        )
    lines += ["", f"**G1+G2 总判定：{'PASS' if all_pass else 'FAIL'}**", ""]
    with open("repro/0721/grid_hash-screen.md", "w") as f:
        f.write("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)
    print("PASS" if all_pass else "FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(run())
