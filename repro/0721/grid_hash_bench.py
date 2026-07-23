#!/usr/bin/env python3
"""Isolated QVG/hash latency runs and paired G3 report."""
import json
import math
import os
import sys

import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path[:0] = ["repro/0721", "repro/0720/kernel", "."]

from pca_grid_hash import grid_hash_encode_kv_fast
from pca_grid_hash_triton import triton_grid_hash_decode
from quant_videogen.functions import (
    triton_prq_dequantize_tensor,
    triton_prq_quantize_tensor,
)


CFG = {
    "lc": dict(iters=5, refine=1, shared_labels=None),
    "sf": dict(iters=1, refine=0, shared_labels="v"),
    "hy": dict(iters=1, refine=0, shared_labels="v_rope"),
}
QVG_ITERS = {"lc": 100, "sf": 2, "hy": 2}


def timed(fn, reps, warmup=5):
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


def run_qvg():
    rows = []
    for model in ("lc", "sf", "hy"):
        reps = 10 if model == "lc" else 30
        for chunk in range(8):
            data = torch.load(
                f"repro/0720/chunks/{model}/chunk_{chunk:03d}.pt",
                map_location="cuda",
            )
            k, v = data["k"], data["v"]
            qk = triton_prq_quantize_tensor(
                k, 1, 256, 64, max_iters=QVG_ITERS[model],
                quantize_fn=lambda _: 2,
            )
            qv = triton_prq_quantize_tensor(
                v, 1, 256, 64, max_iters=QVG_ITERS[model],
                quantize_fn=lambda _: 2,
            )

            def encode():
                triton_prq_quantize_tensor(
                    k, 1, 256, 64, max_iters=QVG_ITERS[model],
                    quantize_fn=lambda _: 2,
                )
                triton_prq_quantize_tensor(
                    v, 1, 256, 64, max_iters=QVG_ITERS[model],
                    quantize_fn=lambda _: 2,
                )

            def decode():
                triton_prq_dequantize_tensor(qk, 64, 2)
                triton_prq_dequantize_tensor(qv, 64, 2)

            enc, dec = timed(encode, reps, 3), timed(decode, 30, 5)
            row = {"model": model, "chunk": chunk, "encode": enc, "decode": dec}
            rows.append(row)
            print(
                model, chunk, f"enc={enc['median']:.4f}",
                f"dec={dec['median']:.4f}", flush=True,
            )
            del data, qk, qv
            torch.cuda.empty_cache()
    with open("repro/0721/grid_hash_bench_qvg.json", "w") as f:
        json.dump(rows, f, indent=2)


def run_hash():
    rows = []
    for model in ("lc", "sf", "hy"):
        for chunk in range(8):
            data = torch.load(
                f"repro/0720/chunks/{model}/chunk_{chunk:03d}.pt",
                map_location="cuda",
            )
            k, v = data["k"], data["v"]
            state = grid_hash_encode_kv_fast(k, v, **CFG[model])

            def encode():
                grid_hash_encode_kv_fast(k, v, **CFG[model])

            def decode():
                triton_grid_hash_decode(state["k"])
                triton_grid_hash_decode(state["v"])

            enc, dec = timed(encode, 30, 5), timed(decode, 50, 5)
            row = {"model": model, "chunk": chunk, "encode": enc, "decode": dec}
            rows.append(row)
            print(
                model, chunk, f"enc={enc['median']:.4f}",
                f"dec={dec['median']:.4f}", flush=True,
            )
            del data, state
            torch.cuda.empty_cache()
    with open("repro/0721/grid_hash_bench_hash.json", "w") as f:
        json.dump(rows, f, indent=2)


def report():
    with open("repro/0721/grid_hash_bench_qvg.json") as f:
        qrows = json.load(f)
    with open("repro/0721/grid_hash_bench_hash.json") as f:
        hrows = json.load(f)
    qmap = {(r["model"], r["chunk"]): r for r in qrows}
    result = {}
    passed = True
    for model in ("lc", "sf", "hy"):
        speedups, rows = [], []
        for h in (r for r in hrows if r["model"] == model):
            q = qmap[(model, h["chunk"])]
            qt = q["encode"]["median"] + q["decode"]["median"]
            ht = h["encode"]["median"] + h["decode"]["median"]
            speedups.append(qt / ht)
            rows.append(
                {
                    "chunk": h["chunk"],
                    "qvg_total_ms": qt,
                    "hash_total_ms": ht,
                    "speedup": qt / ht,
                }
            )
        rng = np.random.default_rng(720)
        logs = np.log(np.array(speedups))
        boots = np.array(
            [
                math.exp(float(rng.choice(logs, len(logs)).mean()))
                for _ in range(10000)
            ]
        )
        lower = float(np.quantile(boots, 0.025))
        model_pass = all(v > 1 for v in speedups) and lower > 1
        passed &= model_pass
        result[model] = {
            "rows": rows,
            "geomean_speedup": math.exp(float(logs.mean())),
            "bootstrap_95_lower": lower,
            "pass": model_pass,
        }

    with open("repro/0721/grid_hash_bench_report.json", "w") as f:
        json.dump({"models": result, "pass": passed}, f, indent=2)
    lines = [
        "# PCA-Grid Hash 总 kernel 时延",
        "",
        "| model | QVG total ms mean | hash total ms mean | "
        "geomean speedup | bootstrap 95% lower | all chunks |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, value in result.items():
        qmean = np.mean([r["qvg_total_ms"] for r in value["rows"]])
        hmean = np.mean([r["hash_total_ms"] for r in value["rows"]])
        lines.append(
            f"| {model} | {qmean:.4f} | {hmean:.4f} | "
            f"{value['geomean_speedup']:.3f}× | "
            f"{value['bootstrap_95_lower']:.3f}× | "
            f"{'PASS' if value['pass'] else 'FAIL'} |"
        )
    lines += ["", f"**G3 总判定：{'PASS' if passed else 'FAIL'}**", ""]
    with open("repro/0721/grid_hash-kernel-report.md", "w") as f:
        f.write("\n".join(lines))
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "qvg":
        run_qvg()
    elif mode == "hash":
        run_hash()
    elif mode == "report":
        raise SystemExit(report())
    else:
        raise SystemExit(f"unknown mode: {mode}")
