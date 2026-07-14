"""Joint protocol search: find an evaluation rule that reproduces Table 1.

For each Table-1 block, sweep candidate protocols and score how well ONE
protocol simultaneously reproduces every method's PSNR (and optionally
SSIM/LPIPS) in that block.

Protocol families:
  W-mean : window [a,b), per-frame-PSNR mean (metric.py aggregation)
  W-mse  : window [a,b), PSNR of window-mean MSE
  P-pct  : percentile p of the per-frame PSNR distribution (whole video)

Usage: python repro/protocol_search.py [--max-end N]
"""

import argparse
import itertools
import json

import numpy as np

D = "repro/backup/protosearch"

BLOCKS = {
    "LC_INT2": {
        "methods": {
            "rtn": ("lc_rtn_int2_b16", 20.872, 0.719, 0.203),
            "quarot": ("lc_quarot_int2_asym_b16", 21.573, 0.759, 0.171),
            "qvg": ("lc_qvg_int2_released", 28.716, 0.909, 0.065),
            "qvgpro": ("lc_qvgpro_int2", 30.376, 0.935, 0.048),
        },
        "quarot_variants": ["lc_quarot_int2_asym_b16", "lc_quarot_int2_sym_b16", "lc_quarot_int2_asym_b128"],
        "qvg_variants": ["lc_qvg_int2_released", "lc_qvg_int2_rngiso"],
    },
    "LC_INT4": {
        "methods": {
            "rtn": ("lc_rtn_int4_b16", 32.984, 0.940, 0.045),
            "quarot": ("lc_quarot_int4_asym_b16", 33.744, 0.960, 0.033),
            "qvg": ("lc_qvg_int4_released", 37.141, 0.978, 0.024),
        },
        "quarot_variants": ["lc_quarot_int4_asym_b16", "lc_quarot_int4_sym_b16", "lc_quarot_int4_asym_b128"],
        "qvg_variants": ["lc_qvg_int4_released", "lc_qvg_int4_rngiso"],
    },
    "HY_INT2": {
        "methods": {
            "rtn": ("hy_rtn_int2", 24.199, 0.696, 0.229),
            "quarot": ("hy_quarot_int2", 25.207, 0.738, 0.205),
            "qvg": ("hy_qvg_int2", 29.174, 0.882, 0.094),
        },
        "quarot_variants": ["hy_quarot_int2"],
        "qvg_variants": ["hy_qvg_int2"],
    },
    "HY_INT4": {
        "methods": {
            "rtn": ("hy_rtn_int4", 33.634, 0.948, 0.056),
            "quarot": ("hy_quarot_int4", 33.997, 0.951, 0.053),
            "qvg": ("hy_qvg_int4", 34.454, 0.954, 0.051),
        },
        "quarot_variants": ["hy_quarot_int4"],
        "qvg_variants": ["hy_qvg_int4"],
    },
}

ARR = {}


def arr(name):
    if name not in ARR:
        ARR[name] = np.load(f"{D}/{name}.npz")
    return ARR[name]


def wmean(a, s, e):
    x = a[s:e]
    x = x[np.isfinite(x)]
    return x.mean() if len(x) else np.nan


def wmse(a, s, e):
    mse = 10 ** (-a[s:e][np.isfinite(a[s:e])] / 10)
    return 10 * np.log10(1 / mse.mean()) if len(mse) else np.nan


def fit_block(block, agg, max_end, use_ssim_lpips=False):
    """Return best (a,b,err_max,per-method deltas) under one shared window."""
    methods = block["methods"]
    names = list(methods)
    best = None
    lens = [len(arr(methods[m][0])["psnr"]) for m in names]
    N = min(min(lens), max_end)
    for a, b in itertools.combinations(range(N + 1), 2):
        errs = []
        for m in names:
            f, tp, ts, tl = methods[m]
            d = arr(f)
            pv = (wmean if agg == "mean" else wmse)(d["psnr"], a, b)
            e = abs(pv - tp)
            if use_ssim_lpips:
                sv = np.nanmean(d["ssim"][a:b])
                lv = np.nanmean(d["lpips"][a:b])
                e = max(e, abs(sv - ts) / 0.02, abs(lv - tl) / 0.02)
            errs.append(e)
        worst = max(errs)
        if best is None or worst < best[2]:
            best = (a, b, worst, {m: round((wmean if agg == "mean" else wmse)(arr(methods[m][0])["psnr"], a, b) - methods[m][1], 3) for m in names})
    return best


def pct_fit(block):
    """Find per-method percentile matching its target; consistent percentile?"""
    out = {}
    for m, (f, tp, _, _) in block["methods"].items():
        pv = arr(f)["psnr"]
        pv = pv[np.isfinite(pv)]
        pcts = np.arange(1, 100)
        vals = np.percentile(pv, pcts)
        i = int(np.argmin(np.abs(vals - tp)))
        out[m] = (int(pcts[i]), round(float(vals[i] - tp), 3))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-end", type=int, default=113)
    args = ap.parse_args()

    report = {}
    for bname, block in BLOCKS.items():
        me = args.max_end if bname.startswith("LC") else 189
        res = {}
        for agg in ("mean", "mse"):
            a, b, err, deltas = fit_block(block, agg, me)
            res[f"joint_{agg}"] = {"window": [a, b], "worst_abs_err_dB": round(err, 3), "deltas": deltas}
            a, b, err, deltas = fit_block(block, agg, me, use_ssim_lpips=True)
            res[f"joint_{agg}_3metric"] = {"window": [a, b], "worst_norm_err": round(err, 3), "psnr_deltas": deltas}
        res["percentiles"] = pct_fit(block)
        # QuaRot variant substitution under PSNR-mean
        subs = {}
        for qv in block["quarot_variants"]:
            blk = {"methods": dict(block["methods"])}
            blk["methods"]["quarot"] = (qv,) + block["methods"]["quarot"][1:]
            a, b, err, deltas = fit_block(blk, "mean", me)
            subs[qv] = {"window": [a, b], "worst": round(err, 3), "deltas": deltas}
        res["quarot_substitution"] = subs
        report[bname] = res

    print(json.dumps(report, indent=1))
    json.dump(report, open(f"{D}/search_report.json", "w"), indent=1)
