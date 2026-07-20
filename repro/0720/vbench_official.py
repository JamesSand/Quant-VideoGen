#!/usr/bin/env python3
"""E2-B: score videos with the OFFICIAL VBench package (separate venv) on the
same 4 dims as our vbench4 reimpl. Custom-input mode, per-arm output JSON.
Run with: /home/zhizhousha/workspace/video-project/.venv-vbench/bin/python vbench_official.py <videos_dir> <tag>
(videos_dir = flat dir of .mp4; results -> repro/0720/e2b/<tag>/)
"""
import os, sys, json, torch

VID_DIR, TAG = sys.argv[1], sys.argv[2]
OUT = f"repro/0720/e2b/{TAG}"
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch_rw"))

from vbench import VBench
import vbench as vbench_pkg
full_info = os.path.join(os.path.dirname(vbench_pkg.__file__), "VBench_full_info.json")

DIMS = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality"]
device = torch.device("cuda")
vb = VBench(device, full_info, OUT)
for dim in DIMS:
    vb.evaluate(videos_path=VID_DIR, name=f"{TAG}_{dim}", dimension_list=[dim], mode="custom_input")
print("OFFICIAL VBENCH DONE", TAG)
# summarize
summary = {}
for dim in DIMS:
    f = f"{OUT}/{TAG}_{dim}_eval_results.json"
    if os.path.exists(f):
        d = json.load(open(f))
        summary[dim] = d[dim][0] * 100
print(json.dumps(summary, indent=1))
json.dump(summary, open(f"{OUT}/summary.json", "w"))
