"""Steps 3-6: k-means clusterability, kmeans+INT2 pipeline, frequency-split
evidence, per-channel stats; save repro/ropestudy_data.npz."""
import os, math, json
import numpy as np
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
SCRATCH = "/tmp/claude-0/-home-zhizhousha-workspace-video-project/4fb6e620-a325-4b3c-8f86-3fcd80917455/scratchpad"

from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
from quant_videogen.functions import prq_quantize_tensor
from quant_videogen.sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

dev = "cuda"
blob = torch.load(os.path.join(SCRATCH, "ropestudy_layer15_sub.pt"),
                  map_location="cpu", weights_only=False)
pre = blob["pre_bf16"].squeeze(0).permute(1, 0, 2).contiguous().float()   # [H=12, S, D=128]
post = blob["post_bf16"].squeeze(0).permute(1, 0, 2).contiguous().float()
H, S, D = pre.shape
print("shape (H,S,D):", pre.shape)

results = {"rel_norm_f64": blob["rel_norm_f64"], "rel_norm_bf16": blob["rel_norm_bf16"]}

def rel_l2(res, x, dims=None):
    if dims is not None:
        res, x = res[..., dims], x[..., dims]
    return (res.norm() / x.norm()).item()

# ---------------- Step 3: k-means clusterability ----------------
def run_kmeans(x):
    xg = x.to(dev)
    torch.manual_seed(0)  # identical init draw for pre and post
    labels, centroids, sizes, n_iters = batch_kmeans_Euclid(
        xg, n_clusters=256, max_iters=100)
    gathered = torch.gather(centroids, 1,
                            labels.long().unsqueeze(-1).expand(-1, -1, D))
    res = (xg - gathered).cpu()
    return res, n_iters, labels.cpu(), centroids.cpu()

res_pre, it_pre, lab_pre, cen_pre = run_kmeans(pre)
res_post, it_post, lab_post, cen_post = run_kmeans(post)
results["kmeans_relL2_pre"] = rel_l2(res_pre, pre)
results["kmeans_relL2_post"] = rel_l2(res_post, post)
results["kmeans_iters_pre"], results["kmeans_iters_post"] = it_pre, it_post
per_head_pre = (res_pre.norm(dim=(1, 2)) / pre.norm(dim=(1, 2))).numpy()
per_head_post = (res_post.norm(dim=(1, 2)) / post.norm(dim=(1, 2))).numpy()
print(f"kmeans SAS rel-L2  pre={results['kmeans_relL2_pre']:.4f} (iters {it_pre})  "
      f"post={results['kmeans_relL2_post']:.4f} (iters {it_post})")

# ---------------- Step 4: 1-stage kmeans(256) + INT2(block64) pipeline ----------------
def run_pipeline(x):
    xg = x.unsqueeze(0).to(dev)  # [B=1, H, S, D]
    torch.manual_seed(0)
    q = prq_quantize_tensor(
        xg, num_stages=1, codebook_size=256, kmeans_max_iters=100,
        quantize_fn=lambda r: blockwise_int2_quantize_triton(r, block_size=64))
    out = rel_l2((q - xg).cpu(), x)
    del xg, q
    torch.cuda.empty_cache()
    return out

results["int2_relL2_pre"] = run_pipeline(pre)
results["int2_relL2_post"] = run_pipeline(post)
print(f"kmeans+INT2 rel-L2  pre={results['int2_relL2_pre']:.4f}  "
      f"post={results['int2_relL2_post']:.4f}")

# ---------------- Step 5: frequency-split evidence ----------------
# theta spectrum, exactly as rope_params(1024, dim): theta_j = 10000^{-2j/dim}
d = D
dim_t, dim_s = d - 4 * (d // 6), 2 * (d // 6)          # 44, 42
theta_t = 10000.0 ** (-np.arange(0, dim_t, 2) / dim_t)  # 22 pairs
theta_h = 10000.0 ** (-np.arange(0, dim_s, 2) / dim_s)  # 21 pairs
theta_w = theta_h.copy()                                # 21 pairs
T_t, T_h, T_w = 180.0, 30.0, 52.0                       # grid extents (f,h,w)
theta = np.concatenate([theta_t, theta_h, theta_w])     # per-pair, order = freq cat order
theta_T = np.concatenate([theta_t * T_t, theta_h * T_h, theta_w * T_w])  # 64 pairs
axis = np.array([0] * 22 + [1] * 21 + [2] * 21)         # 0=t,1=h,2=w
disp_pair = theta_T >= math.pi / 2
print(f"dispersed pairs: {disp_pair.sum()}/64 "
      f"(t:{disp_pair[:22].sum()}/22 h:{disp_pair[22:43].sum()}/21 w:{disp_pair[43:].sum()}/21)")
# pair p occupies head dims (2p, 2p+1)  [view_as_complex on reshape(...,-1,2)]
disp_dims = torch.from_numpy(np.repeat(disp_pair, 2))
stab_dims = ~disp_dims
results["n_disp_dims"] = int(disp_dims.sum())

for name, dims in [("disp", disp_dims), ("stab", stab_dims)]:
    results[f"kmeans_relL2_pre_{name}"] = rel_l2(res_pre, pre, dims)
    results[f"kmeans_relL2_post_{name}"] = rel_l2(res_post, post, dims)
print("restricted kmeans rel-L2:",
      {k: round(v, 4) for k, v in results.items() if "_disp" in k or "_stab" in k})

# per-channel std (over tokens), per (head, dim)
std_pre = pre.std(dim=1).numpy()    # [12, 128]
std_post = post.std(dim=1).numpy()
ratio = std_post / std_pre
n_gt2 = int(((ratio > 2.0) | (ratio < 0.5)).sum())
n_up2, n_dn2 = int((ratio > 2.0).sum()), int((ratio < 0.5).sum())
results["n_channels_std_change_gt2x"] = n_gt2
results["n_channels_std_up_gt2x"] = n_up2
results["n_channels_std_down_gt2x"] = n_dn2
print(f"channels with >2x std change: {n_gt2}/1536 (up {n_up2}, down {n_dn2})")
print(f"std ratio range: [{ratio.min():.3f}, {ratio.max():.3f}], median {np.median(ratio):.3f}")

# per-pair dispersion metrics
ratio_pair = ratio.reshape(H, 64, 2).mean(axis=(0, 2))        # mean std ratio per pair
res_pre_p = res_pre.pow(2).sum(dim=(0, 1)).reshape(64, 2).sum(1)
res_post_p = res_post.pow(2).sum(dim=(0, 1)).reshape(64, 2).sum(1)
sig_pre_p = pre.pow(2).sum(dim=(0, 1)).reshape(64, 2).sum(1)
sig_post_p = post.pow(2).sum(dim=(0, 1)).reshape(64, 2).sum(1)
relL2_pair_pre = (res_pre_p / sig_pre_p).sqrt().numpy()       # per-pair kmeans rel-L2
relL2_pair_post = (res_post_p / sig_post_p).sqrt().numpy()

# per-pair mean |cos similarity across token pairs|-free simple dispersion proxy:
# ratio of post/pre residual energy per pair
res_ratio_pair = (res_post_p / res_pre_p.clamp_min(1e-30)).sqrt().numpy()

# ---------------- Step 6: save npz ----------------
np.savez(os.path.join(REPO, "repro/ropestudy_data.npz"),
         std_pre=std_pre, std_post=std_post, std_ratio=ratio,
         theta=theta, theta_T=theta_T, axis=axis,
         disp_pair_mask=disp_pair,
         std_ratio_per_pair=ratio_pair,
         kmeans_relL2_per_pair_pre=relL2_pair_pre,
         kmeans_relL2_per_pair_post=relL2_pair_post,
         kmeans_res_ratio_per_pair=res_ratio_pair,
         per_head_kmeans_relL2_pre=per_head_pre,
         per_head_kmeans_relL2_post=per_head_post,
         kmeans_relL2_pre=results["kmeans_relL2_pre"],
         kmeans_relL2_post=results["kmeans_relL2_post"],
         int2_relL2_pre=results["int2_relL2_pre"],
         int2_relL2_post=results["int2_relL2_post"],
         kmeans_relL2_pre_disp=results["kmeans_relL2_pre_disp"],
         kmeans_relL2_post_disp=results["kmeans_relL2_post_disp"],
         kmeans_relL2_pre_stab=results["kmeans_relL2_pre_stab"],
         kmeans_relL2_post_stab=results["kmeans_relL2_post_stab"],
         sel_frames=np.array(blob["sel_frames"]),
         layer=np.array(blob["layer"]),
         rel_norm_f64=np.array(blob["rel_norm_f64"]),
         rel_norm_bf16=np.array(blob["rel_norm_bf16"]))
print("saved", os.path.join(REPO, "repro/ropestudy_data.npz"))
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                  for k, v in results.items()}, indent=1))
