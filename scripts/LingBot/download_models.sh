#!/bin/bash

# Download LingBot-World-Base (Cam): includes VAE, T5 encoder,
# low/high-noise models, and other shared assets.
hf download robbyant/lingbot-world-base-cam --local-dir ckpts/LingBot/lingbot-world-base-cam

# Download LingBot-World-Fast: the causal/streaming DiT used by generate_fast.py,
# placed as a sub-folder inside the base-cam checkpoint dir.
hf download robbyant/lingbot-world-fast --local-dir ckpts/LingBot/lingbot-world-base-cam/lingbot_world_fast
