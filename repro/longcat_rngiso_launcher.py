"""RNG-isolated launcher for LongCat run_long_t2v.py.

As released, the QVG run's k-means centroid init (quant_videogen/kmeans/
kmeans_euclid.py:54, torch.randint on the default CUDA RNG) advances the
global RNG stream that prepare_latents also draws from, so from segment 2
onward the QVG run samples different latent noise than the BF16 baseline and
PSNR measures quantization error PLUS content drift. The paper's Table-1
numbers (esp. INT4 = 37.141 dB) are only reachable with noise-aligned runs.

This launcher redirects any torch.randint call that does not already pass a
generator to a dedicated, fixed-seed generator. The default RNG stream is
left untouched, so the QVG run consumes it exactly like the BF16 run.
Repo-wide audit: kmeans_euclid.py:54 is the only runtime randint in the
generation+quant path, so this is equivalent to seeding k-means separately.
"""

import runpy
import torch

_orig_randint = torch.randint
_gens = {}


def _iso_randint(*args, **kwargs):
    if kwargs.get("generator") is None:
        device = torch.device(kwargs.get("device") or "cpu")
        gen = _gens.get(device.type)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(20260710)
            _gens[device.type] = gen
        kwargs["generator"] = gen
    return _orig_randint(*args, **kwargs)


torch.randint = _iso_randint

runpy.run_path("experiments/LongCat/run_long_t2v.py", run_name="__main__")
