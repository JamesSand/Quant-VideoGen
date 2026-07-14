"""Step 1+2: extract layer-15 pre-RoPE K, reconstruct post-RoPE with the model's
own machinery, sanity-check norms, subsample every 8th latent frame, save temporaries.

Grid: f=180 latent frames, h=30, w=52 -> 1560 tokens/frame, 280800 total.
Cache layout BSHD: chunks of [1, 1560, 12, 128] bf16.
"""
import os, sys, math
import torch

REPO = "/home/zhizhousha/workspace/video-project/Quant-VideoGen"
SCRATCH = "/tmp/claude-0/-home-zhizhousha-workspace-video-project/4fb6e620-a325-4b3c-8f86-3fcd80917455/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "experiments/Self-Forcing"))

from wan.modules.model import rope_params
from wan.modules.causal_model import causal_rope_apply, causal_rope_apply_long_input

torch.manual_seed(0)

LAYER = 15
H_GRID, W_GRID = 30, 52
FSL = H_GRID * W_GRID  # 1560
N_FRAMES = 180
FRAME_STRIDE = 8
D = 128

# ---- model's own freqs construction (CausalWanModel.__init__, d = dim//num_heads = 128)
d = D
freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),   # temporal: dim 44 -> 22 pairs
    rope_params(1024, 2 * (d // 6)),       # spatial h: dim 42 -> 21 pairs
    rope_params(1024, 2 * (d // 6)),       # spatial w: dim 42 -> 21 pairs
], dim=1)
print("freqs:", freqs.shape, freqs.dtype)

# ---- load cache (mmap)
dump = torch.load(os.path.join(REPO, "results/ropestudy/kv_cache_frames180.pt"),
                  map_location="cpu", mmap=True, weights_only=False)
kcache = dump["kv_cache"][LAYER]["k"]
assert kcache.layout == "BSHD" and kcache.frame_seq_length == FSL
assert all(int(s) == 1 for s in kcache.chunk_state)

sel_frames = list(range(0, N_FRAMES, FRAME_STRIDE))  # 23 frames
print("selected frames:", sel_frames)

pre_chunks = [kcache.chunks[t].clone() for t in sel_frames]  # each [1,1560,12,128] bf16
pre = torch.cat(pre_chunks, dim=1)  # [1, 23*1560, 12, 128] bf16
print("pre:", pre.shape, pre.dtype)

# ---- sanity: per-frame causal_rope_apply(start_frame=t) == causal_rope_apply_long_input
# The model path (attn_kv_cache_prerope) calls causal_rope_apply_long_input(k_all,
# grid_sizes, freqs) with grid f = frames-per-chunk; chunk_start_frame = chunk_idx*f,
# so temporal freq index == global latent frame index. Verify with f=3 on frames 0..5.
test_in = torch.cat([kcache.chunks[t].clone() for t in range(6)], dim=1)
gs3 = torch.tensor([[3, H_GRID, W_GRID]])
out_long = causal_rope_apply_long_input(test_in, gs3, freqs)
gs1 = torch.tensor([[1, H_GRID, W_GRID]])
out_pf = torch.cat([
    causal_rope_apply(test_in[:, t*FSL:(t+1)*FSL], gs1, freqs, start_frame=t)
    for t in range(6)], dim=1)
eq = (out_long == out_pf).all().item()
print("long_input(f=3) == per-frame(start_frame=t):", eq,
      "| max abs diff:", (out_long.float()-out_pf.float()).abs().max().item())
assert eq
del test_in, out_long, out_pf

# ---- apply RoPE per selected frame at its true global frame index
post_bf16_parts, post_f64_parts = [], []
for i, t in enumerate(sel_frames):
    xf = pre[:, i*FSL:(i+1)*FSL]                     # bf16 [1,1560,12,128]
    post_bf16_parts.append(causal_rope_apply(xf, gs1, freqs, start_frame=t))
    post_f64_parts.append(causal_rope_apply(xf.double(), gs1, freqs, start_frame=t))
post_bf16 = torch.cat(post_bf16_parts, dim=1)
post_f64 = torch.cat(post_f64_parts, dim=1)
print("post:", post_bf16.shape, post_bf16.dtype)

# ---- norm preservation check (per token, per head)
pre_norm = pre.double().norm(dim=-1)               # [1, S, 12]
post64_norm = post_f64.norm(dim=-1)
post_bf16_norm = post_bf16.double().norm(dim=-1)
rel64 = ((post64_norm - pre_norm).abs() / pre_norm).max().item()
relbf = ((post_bf16_norm - pre_norm).abs() / pre_norm).max().item()
print(f"max rel norm change, exact rotation (f64): {rel64:.3e}")
print(f"max rel norm change, after bf16 cast (model-faithful): {relbf:.3e}")
assert rel64 < 1e-3

# ---- save temporaries for GPU stage
torch.save({
    "pre_bf16": pre, "post_bf16": post_bf16,
    "sel_frames": sel_frames, "layer": LAYER,
    "rel_norm_f64": rel64, "rel_norm_bf16": relbf,
}, os.path.join(SCRATCH, "ropestudy_layer15_sub.pt"))
print("saved", os.path.join(SCRATCH, "ropestudy_layer15_sub.pt"),
      os.path.getsize(os.path.join(SCRATCH, "ropestudy_layer15_sub.pt"))/1e6, "MB")
