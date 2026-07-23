#!/usr/bin/env python3
"""Validate the bounded factor-only fallback on the preregistered chunks."""
import json
import math
import os
import sys

import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path[:0] = ["repro/0721", "repro/0720/kernel", "."]

from bp_quant import bp_bytes, bp_decode, bp_encode_fast
from factor_grid_triton import triton_factor_decode
from quant_videogen.functions import (
    triton_prq_dequantize_tensor,
    triton_prq_quantize_tensor,
)


MODELS = ("sf", "hy")
CFG = dict(r=4, grid="asym", block=128, axis="channel", iters=2)
QVG_BPE = {"sf": 2.4063, "hy": 3.3199}


def timed(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median": samples[len(samples) // 2],
        "p10": float(np.quantile(samples, 0.1)),
        "p90": float(np.quantile(samples, 0.9)),
        "samples": samples,
    }


def metric(y, x):
    sse = float((y.float() - x.float()).square().sum())
    signal = float(x.float().square().sum())
    return {"sse": sse, "signal": signal, "rel_l2": math.sqrt(sse / signal)}


with open("repro/0721/grid_hash_screen.json") as f:
    reference = json.load(f)
qvg_metric = {
    (r["model"], r["chunk"], r["kind"]): r["qvg"]
    for r in reference["rows"]
}

rows = []
for model in MODELS:
    for chunk in range(8):
        data = torch.load(
            f"repro/0720/chunks/{model}/chunk_{chunk:03d}.pt",
            map_location="cuda",
        )
        k, v = data["k"], data["v"]
        states = {kind: bp_encode_fast(data[kind], **CFG) for kind in ("k", "v")}
        decoded = {
            kind: triton_factor_decode(states[kind], bs=64, bd=64)
            for kind in ("k", "v")
        }

        parity = {}
        for kind in ("k", "v"):
            ref = bp_decode(states[kind])
            diff = (decoded[kind].float() - ref.float()).abs()
            parity[kind] = {
                "max_abs": float(diff.max()),
                "mismatch_fraction": float((diff != 0).float().mean()),
            }

        def factor_encode():
            bp_encode_fast(k, **CFG)
            bp_encode_fast(v, **CFG)

        def factor_decode():
            triton_factor_decode(states["k"], bs=64, bd=64)
            triton_factor_decode(states["v"], bs=64, bd=64)

        # Rebuild QVG states on this exact GPU/input so latency remains paired.
        torch.manual_seed(720_000 + chunk)
        qk = triton_prq_quantize_tensor(
            k, 1, 256, 64, max_iters=2, quantize_fn=lambda _: 2
        )
        qv = triton_prq_quantize_tensor(
            v, 1, 256, 64, max_iters=2, quantize_fn=lambda _: 2
        )

        def qvg_encode():
            triton_prq_quantize_tensor(
                k, 1, 256, 64, max_iters=2, quantize_fn=lambda _: 2
            )
            triton_prq_quantize_tensor(
                v, 1, 256, 64, max_iters=2, quantize_fn=lambda _: 2
            )

        def qvg_decode():
            triton_prq_dequantize_tensor(qk, 64, 2)
            triton_prq_dequantize_tensor(qv, 64, 2)

        factor_enc = timed(factor_encode)
        factor_dec = timed(factor_decode, reps=50)
        qvg_enc = timed(qvg_encode)
        qvg_dec = timed(qvg_decode, reps=50)
        qvg_total = qvg_enc["median"] + qvg_dec["median"]
        factor_total = factor_enc["median"] + factor_dec["median"]
        bpe = (
            sum(bp_bytes(state) for state in states.values())
            * 8
            / (k.numel() + v.numel())
        )

        row = {
            "model": model,
            "chunk": chunk,
            "bpe": bpe,
            "factor_encode": factor_enc,
            "factor_decode": factor_dec,
            "qvg_encode": qvg_enc,
            "qvg_decode": qvg_dec,
            "qvg_total_ms": qvg_total,
            "factor_total_ms": factor_total,
            "speedup": qvg_total / factor_total,
            "parity": parity,
        }
        for kind in ("k", "v"):
            row[kind] = {
                "factor": metric(decoded[kind], data[kind]),
                "qvg": qvg_metric[(model, chunk, kind)],
            }
        rows.append(row)
        print(
            model,
            chunk,
            f"bpe={bpe:.4f}",
            f"speedup={row['speedup']:.3f}",
            f"k={row['k']['factor']['rel_l2']:.5f}",
            f"v={row['v']['factor']['rel_l2']:.5f}",
            flush=True,
        )
        del data, states, decoded, qk, qv
        torch.cuda.empty_cache()

summary = {}
all_pass = True
for model in MODELS:
    rr = [r for r in rows if r["model"] == model]
    speedups = np.asarray([r["speedup"] for r in rr])
    rng = np.random.default_rng(721)
    log_speedups = np.log(speedups)
    boots = np.asarray(
        [
            math.exp(float(rng.choice(log_speedups, len(rr)).mean()))
            for _ in range(10_000)
        ]
    )
    sm = {
        "config": CFG,
        "bpe": rr[0]["bpe"],
        "bpe_pass": all(r["bpe"] <= QVG_BPE[model] for r in rr),
        "geomean_speedup": math.exp(float(log_speedups.mean())),
        "bootstrap_95_lower": float(np.quantile(boots, 0.025)),
        "all_chunks_faster": bool(np.all(speedups > 1)),
        "max_decode_parity_abs": max(
            r["parity"][kind]["max_abs"] for r in rr for kind in ("k", "v")
        ),
    }
    for kind in ("k", "v"):
        fsse = sum(r[kind]["factor"]["sse"] for r in rr)
        qsse = sum(r[kind]["qvg"]["sse"] for r in rr)
        signal = sum(r[kind]["factor"]["signal"] for r in rr)
        sm[kind] = {
            "factor_rel_l2": math.sqrt(fsse / signal),
            "qvg_rel_l2": math.sqrt(qsse / signal),
            "mse_ratio": fsse / qsse,
            "worst_chunk_mse_ratio": max(
                r[kind]["factor"]["sse"] / r[kind]["qvg"]["sse"] for r in rr
            ),
        }
    sm["mse_pass"] = all(
        sm[kind]["mse_ratio"] <= 1
        and sm[kind]["worst_chunk_mse_ratio"] <= 1.5
        for kind in ("k", "v")
    )
    sm["speed_pass"] = (
        sm["all_chunks_faster"] and sm["bootstrap_95_lower"] > 1
    )
    sm["pass"] = sm["bpe_pass"] and sm["mse_pass"] and sm["speed_pass"]
    all_pass &= sm["pass"]
    summary[model] = sm

result = {"rows": rows, "summary": summary, "pass": all_pass}
with open("repro/0721/factor_grid_validate.json", "w") as f:
    json.dump(result, f, indent=2)

lines = [
    "# Factor-only product-grid fallback",
    "",
    "| model | K relL2 QVG→factor | V relL2 QVG→factor | worst MSE K/V | "
    "BPE factor≤QVG | speedup (95% lower) | result |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for model, sm in summary.items():
    lines.append(
        f"| {model} | {sm['k']['qvg_rel_l2']:.5f}→{sm['k']['factor_rel_l2']:.5f} | "
        f"{sm['v']['qvg_rel_l2']:.5f}→{sm['v']['factor_rel_l2']:.5f} | "
        f"{sm['k']['worst_chunk_mse_ratio']:.3f}/{sm['v']['worst_chunk_mse_ratio']:.3f} | "
        f"{sm['bpe']:.4f}≤{QVG_BPE[model]:.4f} | "
        f"{sm['geomean_speedup']:.3f}× ({sm['bootstrap_95_lower']:.3f}×) | "
        f"{'PASS' if sm['pass'] else 'FAIL'} |"
    )
lines += ["", f"**Fallback G1–G3: {'PASS' if all_pass else 'FAIL'}**", ""]
with open("repro/0721/factor-grid-report.md", "w") as f:
    f.write("\n".join(lines))
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if all_pass else 1)
