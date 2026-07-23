"""qvg4pt 一帧最小 demo:单文件跑通 QVG-4 点位算法,LongCat 出一帧 PNG。

int2 四点位的本质 = affine(asymmetric)量化:每 64 元素一块,只存
【块最小值 mn + 步长 s】两个 fp8,外加每元素 2-bit 码 q ∈ {0,1,2,3}:

    s  = (max - mn) / 3                # 2 bit -> 3 段
    q  = clamp(round((x - mn) / s), 0, 3)
    x^ = q * s + mn                    # 四个格点 {mn, mn+s, mn+2s, max}

数值与 repro/backup/scripts/pca_quant.py 逐行一致(test_equivalence.py 已证
逐位相等);kmeans 减法直接用 QVG 发布库,一个字不动。

跑法(1 张 GPU,~5 分钟;去噪步数 50->8 只影响画质,不影响量化路径):
    bash repro/0722/qvg4pt-demo/run_one_frame.sh [prompt_idx]
产出:repro/0722/qvg4pt-demo/demo_out/oneframe_f93.png(首个生成帧)
"""
import os
import re
import runpy
import sys

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
os.chdir(REPO)
sys.path.insert(0, REPO)                          # quant_videogen(QVG kmeans)
sys.path.insert(0, f"{REPO}/experiments/LongCat")  # longcat_video

import torch

# ---------- 1. affine int2 四点位(存 mn + s;fp8 元数据) ----------

def _fp8(t, per_row=False):
    """scale/mn 的 fp8(E4M3)存储模拟:除以归一因子 -> fp8 往返 -> 乘回。"""
    if per_row:
        f = t.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    else:
        f = t.abs().amax().clamp_min(1e-12)
    return (t / f).to(torch.float8_e4m3fn).to(t.dtype) * f


def affine_int2_b64(x):
    """x: [B,H,S,D] 残差。沿 D 每 64 个一块(token 轴 B64)。
    每块实际"存"的就是 mn(zero-point)和 s(scale)两个 fp8 + 2-bit 码。"""
    S = x.shape
    xg = x.reshape(*S[:-1], S[-1] // 64, 64)      # [B,H,S,2,64]
    mn = xg.amin(dim=-1, keepdim=True)            # 块最小值 = 格子起点
    mx = xg.amax(dim=-1, keepdim=True)
    s = ((mx - mn) / 3).clamp_min(1e-8)           # 步长:min->max 均分 3 段
    s = _fp8(s, per_row=True).clamp_min(1e-8)     # 元数据过 fp8(合法口径)
    mn = _fp8(mn, per_row=True)
    q = torch.clamp(torch.round((xg - mn) / s), 0, 3)   # 码字 {0,1,2,3} 全用满
    return (q * s + mn).reshape(S)                # 反量化 = q*s + mn


# ---------- 2. qvg4pt = QVG 原装 kmeans 减法(不动)+ 上面的 affine 格 ----------

def qvg4pt_fake_quant_kv(k, v):
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    outs = []
    for x in (k, v):
        B, H, S, D = x.shape
        X = x.float().view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=100)
        g = torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))
        res = (X - g).view(B, H, S, D)            # 残差 = X - 最近质心
        outs.append((g.view(B, H, S, D) + affine_int2_b64(res)).to(x.dtype))
    return outs[0], outs[1]


# ---------- 3. 挂载:劫持 KV 压缩入口 + 把去噪步数压到 8 ----------

import quant_videogen.compress as _compress

_orig = _compress.compress_kv_cache


def _patched(k, v, quant_type, quant_config, quantize_fn):
    if re.fullmatch(r"naive-int(\d+)", quant_type) is None:
        return _orig(k, v, quant_type, quant_config, quantize_fn)
    print(f"[one_frame] qvg4pt on k{tuple(k.shape)}", flush=True)
    return qvg4pt_fake_quant_kv(k, v)


_compress.compress_kv_cache = _patched

import longcat_video.pipeline_longcat_video as _plv

_orig_gen = _plv.LongCatVideoPipeline.generate_vc


def _fast_gen(self, *a, **kw):
    kw["num_inference_steps"] = 8   # demo 提速;量化发生在去噪前,路径不变
    return _orig_gen(self, *a, **kw)


_plv.LongCatVideoPipeline.generate_vc = _fast_gen

# ---------- 4. 跑 LongCat 单 segment 续写(参数与正式协议一致) ----------

P = os.environ.get("ONEFRAME_P", "1")
OUTD = "repro/0722/qvg4pt-demo/demo_out/oneframe"
sys.argv = ["run_long_t2v.py",
            "--checkpoint_dir=ckpts/LongCat-Video", "--workload", "480p_long_gen",
            "--init_video_path", f"results/multiprompt/mp100/lc/base/{P}-0.mp4",
            "--num_segments", "1", "--num_cond_frames", "73",
            "--seed", "0", "--prompt_source", "text_to_video_from_file",
            "--prompt", "repro/0718/prompts100/selected.txt", "--prompt_idx", P,
            "--output_dir", OUTD,
            "--quant_type", "naive-int2", "--quant_block_size", "64"]
runpy.run_path("experiments/LongCat/run_long_t2v.py", run_name="__main__")

# ---------- 5. 抽第 93 帧(首个生成帧)存 PNG ----------

import glob

import imageio.v3 as iio

vid = glob.glob(f"{OUTD}/{P}-0/segment_1.mp4")[0]
for i, fr in enumerate(iio.imiter(vid, plugin="pyav")):
    if i == 93:
        png = "repro/0722/qvg4pt-demo/demo_out/oneframe_f93.png"
        iio.imwrite(png, fr)
        print(f"ONE FRAME OK: {png}", flush=True)
        break
