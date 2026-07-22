# Report 0721 — Budget-PCA: Method · Results · Kernel · Why (Simple English)

> Plain-English version of [report-0721.md](report-0721.md). All numbers are
> real, byte-audited measurements. Figures and detail docs live in
> [../0720/](../0720/README.md).

---

## 1. Method: how Budget-PCA works

### 1.1 One shared skeleton (no dictionary, no k-means, no calibration, no iteration)

For each (head, chunk) KV tensor $`X \in \mathbb{R}^{S\times D}`$ (S tokens,
D dims per head), we do four steps:

```math
\mu = \tfrac{1}{S}\textstyle\sum_{s} x_s,
\qquad
X_c = X - \mathbf{1}\mu^\top
\qquad\text{(subtract the mean; this absorbs channel offsets)}
```

```math
C = \tfrac{1}{S} X_c^\top X_c,
\qquad
V_r = [\,v_1,\dots,v_r\,],\quad C v_i = \lambda_i v_i,\ \ \lambda_1 \ge \dots \ge \lambda_D
\qquad\text{(one eigendecomposition of THIS chunk; take the top-r eigenvectors)}
```

```math
\hat{C}_{\mathrm{coef}} = Q_2^{\mathrm{tok}}\!\left(X_c V_r\right),
\qquad
R = X_c - \hat{C}_{\mathrm{coef}} V_r^\top
\qquad\text{(residual is computed AFTER coefficient quantization)}
```

```math
\hat R = Q_2^{\mathrm{ch},B}(R),
\qquad
\hat X = \mathbf{1}\mu^\top + \hat{C}_{\mathrm{coef}} V_r^\top + \hat R
\qquad\text{(decode = one thin GEMM + elementwise adds)}
```

- $`Q_2^{\mathrm{tok}}`$: 2-bit asymmetric grid, one fp8 scale/zero pair per
  token (used for the r projection coefficients).
- $`Q_2^{\mathrm{ch},B}`$: 2-bit asymmetric grid **along the channel axis** —
  transpose, cut each channel into blocks of B tokens, one fp8 scale/zero pair
  per block. Same metadata cost as the token axis; only the direction of the
  scales changes (this is the KIVI insight).
- Key property: encoder and decoder use the same
  $`\hat{C}_{\mathrm{coef}} V_r^\top`$, so the low-rank branch has **zero**
  reconstruction error. All error lives in one place: the residual.

### 1.2 The three models (only static config differs)

| | LongCat | Self-Forcing | HY-WorldPlay |
|---|---|---|---|
| What is quantized | pre-RoPE K + V | pre-RoPE K + V | **post-transform packed [rope‖prope], 256-dim** |
| When | once per segment window (S=29640) | when a chunk ages out (S=37440) | when a chunk ages out (S=7040) |
| Rank | K=V: r=4 | K=V: r=4 | **half-split ranks K=9:0, V=9:0** (all rank to the rope half) |
| Residual grid | asym B128, channel axis for K and V | same as LC (+ a storage transpose fix) | K: channel-axis asym **B64**; K-prope: **3-level symmetric** B64; V: asym B128 |

HY: split $`X = [X_{\mathrm{ro}} \| X_{\mathrm{pr}}]`$ into two 128-dim halves
and run the same skeleton on each. Ranks are 9:0 because the prope half has a
lot of energy but no value for generation (see §4).

### 1.3 BPE (bits per element; budget = 2.326)

General formula (every term is "bits per stored element"):

```math
\mathrm{BPE}
= \underbrace{2}_{\text{residual codes}}
+ \underbrace{\tfrac{16}{B}}_{\text{residual fp8 s/z}}
+ \underbrace{\tfrac{2\tilde r}{D}}_{\text{coef codes}}
+ \underbrace{\tfrac{16}{D}}_{\text{coef fp8 s/z}}
+ \underbrace{\tfrac{8}{S}}_{\text{per-channel norm factor (int8 exponent)}}
+ \underbrace{\tfrac{16(r{+}1)D}{SD}}_{\mu,\,V_r\ \text{(bf16, amortized)}}
```

($`\tilde r`$ = coefficient packing width, r rounded up to a multiple of 4.
The 3-level grid stores only a scale, so its s/z term is halved.)

Checked against **byte-level audits of the real kernel**
([bpe-audit.md](../0720/bpe-audit.md)):

| Model | residual codes | residual s/z | coef codes | coef s/z | norm factor | amortized | **measured total** |
|---|---|---|---|---|---|---|---|
| LC (r=4, B=128) | 2.004 | 0.125 | 0.0625 | 0.125 | 0.0003 | 0.0027 | **2.3195** ✓ |
| SF (r=4, B=128) | 2.003 | 0.125 | 0.0625 | 0.125 | 0.0002 | 0.0021 | **2.3185** ✓ |
| HY-K (9:0, B64 + 3-level B64) | 2.000 | 0.1875 | 0.0938 | 0.0625 | 0.0011 | 0.0125 | 2.3574 |
| HY-V (9:0, B128) | 2.000 | 0.125 | 0.0938 | 0.0625 | 0.0000 | 0.0125 | 2.2938 |
| **HY cache (K+V, the verdict number)** | | | | | | | **2.3256** ✓ |

For comparison, **QVG's own code, byte-audited the same way: LC 2.464 /
SF 2.406 / HY 3.320** (it stores centroids in fp32). Its paper claims a
nominal 2.326, which its released code cannot reach. Accounting rule:
baselines are counted exactly as their code stores things — we never
"improve" a baseline's storage for it.

---

## 2. Benchmark: the MP100 table

MovieGen, 100 random prompts (seed 42). Protocols: LC = frame 93 (first
generated frame), HY = mean over frames[13:], SF = 700-frame window
(VBench only). Best quantized method per column in bold.
(Source: [mp100-table.md](../0720/mp100-table.md))

### LongCat-Video-13B (100 prompts × f93)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | BC↑ | IQ↑ | SC↑ | AQ↑ | BPE↓ |
|---|---|---|---|---|---|---|---|---|
| BF16 KV (reference) | — | — | — | 95.54 | 70.18 | 92.67 | 56.52 | 16 |
| RTN | 23.56 | 0.7848 | 0.1678 | 95.46 | 70.05 | 92.56 | 56.46 | 2.25 |
| KIVI | 30.55 | 0.9228 | 0.0640 | **95.56** | 70.16 | **92.69** | 56.54 | 2.32 |
| QuaRot | 26.69 | 0.8675 | 0.0974 | 95.54 | 70.13 | 92.68 | **56.55** | 2.25 |
| QVG | 28.20 | 0.8987 | 0.0798 | 95.53 | 70.16 | 92.69 | 56.53 | 2.464ᵇ |
| **Budget-PCA** | **31.68** | **0.9370** | **0.0547** | 95.54 | **70.19** | 92.69 | 56.54 | **2.3195** |

### HY-WorldPlay-8B (10 seeds × frames[13:])

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | BC↑ | IQ↑ | SC↑ | AQ↑ | BPE↓ |
|---|---|---|---|---|---|---|---|---|
| BF16 KV (reference) | — | — | — | 97.23 | 78.07 | 95.44 | 64.03 | 16 |
| RTN | 17.32 | 0.4286 | 0.3954 | 95.90 | 76.92 | 94.26 | **63.02** | 2.25 |
| KIVI | 17.13 | 0.4239 | 0.3978 | **96.12** | 77.18 | 94.42 | 61.60 | 2.25 |
| QuaRot | 17.21 | 0.4159 | 0.4153 | 95.43 | 77.56 | 92.74 | 60.10 | 2.25 |
| QVG | 17.45 | 0.4204 | 0.3886 | 95.61 | **78.50** | 93.82 | 62.66 | 3.320ᵇ |
| **Budget-PCA** | **18.77** | **0.5016** | **0.3286** | 95.82 | 78.18 | **94.68** | 62.17 | **2.3256** |

### Self-Forcing-Wan (100 prompts × 700-frame window, VBench)

| Method | BC↑ | IQ↑ | SC↑ | AQ↑ | BPE↓ |
|---|---|---|---|---|---|
| BF16 KV (reference) | 92.93 | 66.74 | 88.11 | 53.09 | 16 |
| RTN | 90.40 | 58.60 | 82.66 | 50.57 | 2.25 |
| KIVI | 93.07 | 65.73 | 88.43 | 53.02 | 2.32 |
| QuaRot | 90.90 | 59.90 | 83.69 | 51.25 | 2.25 |
| QVG | 92.50 | 65.91 | 87.09 | 52.68 | 2.406ᵇ |
| **Budget-PCA** | **93.21** | **66.65** | **88.69** | **53.27** | **2.3185** |

### Verdict vs QVG (paired two-sided sign tests, per column)

**18 columns = 11 significant wins (p<0.05) + 7 statistical ties + 0 losses.**
LC: +3.48 dB / +0.038 SSIM / −0.025 LPIPS (89-91 of 100 prompts, p<0.001).
SF: all four VBench columns win (p≤0.007). HY: PSNR/SSIM/LPIPS/SC win on
10/10 seeds (p=0.002).

> Notes. Our baselines are strong, honest implementations (the paper's own
> baselines were shown to be weak). AQ absolute values and the HY PSNR window
> cannot be compared directly with the paper — see
> [paper-diff-plan.md](../0720/paper-diff-plan.md).
> ᵇ QVG BPE is measured byte-by-byte from its released code (fp32 centroids).

---

## 3. Kernel and speed

### 3.1 Implementation

- **encode** ([bp_quant.py](../0720/kernel/bp_quant.py)): one whole-graph
  `torch.compile` — mean → bf16 covariance GEMM → 5 rounds of subspace
  iteration with Cholesky orthonormalization (replaces slow batched eigh) →
  2-bit coef quant → blockwise 2-bit residual quant → bit packing.
- **decode** ([bp_triton.py](../0720/kernel/bp_triton.py)): one fused Triton
  kernel (unpack → fp8 dequant → +μ → r FMAs for the low-rank part → bf16
  write). Numerically equal to the reference within **1 ulp**
  (max|Δ| = 0.0625 for LC/SF, 0.03125 for HY; bf16 summation order).
- fp8 metadata pitfalls we fixed along the way: raw fp8 scales saturate
  (>448); one global factor underflows small channels (−3.6 dB end-to-end);
  final design = **one power-of-two factor per channel, stored as an int8
  exponent** (exact division, ~0.0003 BPE, fully counted).

### 3.2 Same-input speed duel (real pipeline chunks, H100, CUDA-event medians)

We follow QVG's official configs: **k-means iterations = 100 on LC, 2 on
SF/HY** (as published).

| Model (chunk shape) | QVG iters | QVG encode | **our encode** | speedup | enc+dec total |
|---|---|---|---|---|---|
| LC [32,29640,128] | **100 (official)** | 176.7 ms | **5.4 ms** | **32.5×** | **27.3×** |
| SF [12,37440,128] | 2 | 3.1 ms | **3.0 ms** | 1.0-1.1× | 0.95-1.10× (tie) |
| HY [24,7040,256] | 2 | 3.1 ms | **2.3 ms** | **1.4×** | **1.16×** |

**Why SF/HY are ties but LC is 32×**: QVG's encode cost = iterations ×
distance pass; ours is iteration-free and constant. At iters=2 both sides sit
on the same memory-bound floor (read/write X, quantize, pack). Side check: LC
at (non-official) iters=2 also drops to ~3.1 ms, and we are still 1.2× faster.
And cutting iterations does not rescue QVG's quality (§4, H4).

Same-table quality (speed is not bought with quality): relL2 LC
0.0940→**0.0805**, SF 0.0885→**0.0830**, HY 0.2090→**0.1856**. Honest limit:
our decode alone is 2-3× slower (one extra low-rank add, 0.3-0.6 ms); the
combined encode+decode is ≥ tie.

---

## 4. Why the method works

Pre-registered hypotheses, all re-checked on 8 real dump chunks × 3 models.
Main report: [why-budget-pca-wins.md](../0720/why-budget-pca-wins.md);
refuted claims and errata: [why-refuted-and-errata.md](../0720/why-refuted-and-errata.md).

### 4.1 Core mechanism: residual-grid efficiency (attribution corrected 0721)

Define grid efficiency $`\eta = \lVert X-\hat X\rVert^2 / \lVert R\rVert^2`$
(final error energy ÷ residual energy; lower is better; recovery = $`1-\eta`$).
8-chunk means [min-max]:

| Model | Method | residual energy | final error energy | $`\eta`$ | recovery |
|---|---|---|---|---|---|
| LC | QVG | 8.4% [0.4-20.0] | 4.38% [0.19-10.4] | 0.52 | ~48% |
| LC | **ours** | 11.9% [0.6-27.6] | **2.69% [0.15-6.2]** | **0.24** | **~76%** |
| LC | ours with 3-level gridᵈ | 11.9% (unchanged) | 6.03% | 0.51 | ~49% |
| SF | QVG | 7.6% | 3.96% | 0.52 | ~48% |
| SF | **ours** | 12.0% | **2.58%** | **0.24** | **~76%** |
| SF | ours with 3-level gridᵈ | 12.0% (unchanged) | 5.94% | 0.50 | ~50% |
| HY | QVG | 16.7% | 8.13% | 0.46 | ~54% |
| HY | **ours** | 27.4% | 9.26%ᶜ | **0.30** | **~70%** |

How to read it (LC): we subtract **less** on purpose (residual 11.9% > 8.4%),
yet the final error is smaller (2.69% < 4.38%) — the whole gap comes from the
grid. ᶜ HY's mean is pulled up by 3 deep layers; per-chunk we still win 5/8,
and end-to-end HY we win (18.77 > 17.45). ᵈ Counterfactual: swap our grid to
QVG's 3-level grid — η jumps to its ~0.50 ceiling and we **lose to QVG**
(6.0% > 4.4%); end-to-end the same swap drops us from ~31.7 to **23.97 dB**,
below QVG's 27.61.

![fig3](../0720/why/fig3_residual_efficiency.png)

The key fact (found after a user challenge, `why/grid_cross.py`): **QVG's
released int2 quantizer is a 3-level symmetric absmax grid** — it uses only 3
of the 4 codewords that 2 bits pay for. Its 46-49% recovery is that grid's
ceiling on ANY input (our residual also gets 48% on it). K-means residual is
NOT unrecoverable noise: a proper 4-level channel-axis grid recovers 73.6% of
it. Attribution at equal metadata budget (0.125 bits/elem): **levels +
asymmetry ≈ +20pp (main factor), channel axis ≈ +6-10pp, residual structure ≈
+4pp**. Both directions verified end-to-end: ① our method with the 3-level
grid falls to 23.97 (below QVG); ② QVG with a 4-level asym B64 grid rises to
**32.29** (final, with fp8 metadata properly simulated; the first run's 32.85
had fp32 scales and is void) — but at BPE **2.589** (11% over budget) and 33×
slower encode. Its dictionary is too expensive to afford a good grid within the
budget; our cheap subtraction is exactly what pays for one.

Supporting fact: the KV token cloud is low-rank ("pancake"), layer-dependent
(top-4 eigenvalue energy, 8-layer means: LC 55% / SF 56% / HY 42%; earliest
layers are flattest). This sets the ceiling of what cheap subtraction can do —
it is not why we beat k-means.

![fig1](../0720/why/fig1_spectra.png)

### 4.2 Supporting verdicts

**H2 Channel hijack (confirmed).** K-means uses Euclidean distance, which is
dominated by large channels; small channels never get encoded. QVG's
error-to-signal on the 16 smallest-variance channels is **1.8-2.4×** ours (LC,
8/8 layers; HY 1.4-2.4×; SF weakest at 1.1-1.8×, matching its small 7× channel
spread). Large channels are equal for both — the damage lands exactly on the
hijacked side.

> **Plain-English "error/signal" (bank-account analogy)**: quantization is
> depositing money and withdrawing it with some loss. What matters is not
> "how much was lost" but "**lost ÷ what you had**". Two accounts each lose 50
> cents: the $100 account loses 0.5% (fine), the $1 account loses 50% (half
> its savings). 128 channels are 128 accounts, some rich (large values), some
> poor (small values but still useful). Measured: QVG loses ~0.4% on rich
> channels and ~15% on poor ones — because k-means picks templates by summing
> distances over all channels, and the rich channels shout loudest. We give
> every channel its own scale, so poor channels lose only ~6%. That is the
> 2.4× in the table: **QVG is not bad at storing overall — it specifically
> hurts poor channels.**

![fig4](../0720/why/fig4_channel_error.png)

**H3 Curse of dimensionality (confirmed).** A continuous token cloud cannot be
covered by 256 discrete points; ×16 more centroids buys only +0.3 to +8.6pp of
removed energy (rate-distortion: VQ needs $`K\approx 2^{nR}`$ to match
transform coding).

![fig5](../0720/why/fig5_pc_plane.png)

**H4 "Just tune k-means harder" is ruled out.** Sweeping iterations changes
nothing:

| k-means iters | LC f93 PSNR (same 10 prompts) |
|---|---|
| 2 | 27.13 |
| 10 | 27.72 |
| 100 (official) | 27.61 |
| **Budget-PCA** | **31.68** |

**KIVI triangulation.** Channel mechanism is worth +2.35 dB (QVG 28.20 → KIVI
30.55); the subtraction framework adds +1.13 dB (KIVI → us 31.68). KIVI fails
on packed/post-transform data (HY 17.13 < QVG 17.45) while our framework still
works (18.77). Placement matters a lot: the same full-strength KIVI grid moved
to post-RoPE loses **7.5 dB** (30.22 → 22.74, kivipost arm) — once RoPE
scatters the channel structure, the channel axis is nearly worthless.

**Energy ≠ value** (HY 9:0 rank flip). The prope half carries 66% of the
energy and is even lower-rank (top-9: 81.8% vs 76.2%), so spectra say "give it
the rank" — but end-to-end, giving ALL rank to the rope half wins. Budget must
follow what downstream attention actually reads, not the spectrum.

![fig6](../0720/why/fig6_hy_halves.png)

### 4.3 Four reusable design rules

1. Spend budget on the grid first, subtraction second (use all 4 codewords,
   asymmetric, aligned to the right axis — worth ~30pp of recovery; extra
   subtraction and residual structure are small change).
2. Turn on the channel axis when channel heterogeneity >10× (biggest payoff on
   pre-RoPE data).
3. Direction-level error shaping is poison in closed loop; channel-level
   amplitude adaptation is safe.
4. If a no-reference IQ metric scores a quantized method ≥ the lossless
   reference, you are in artifact-preference territory — the end-to-end gate
   is the only judge.

---

## 5. External audit round 2 (all five criticisms were true; fixed and re-measured)

An external engineering review raised five issues. Each one was verified as
**true** and fixed. This was the third honesty round of the project.

| # | Criticism | Verified | Fix and re-measurement |
|---|---|---|---|
| ① | HY real BPE 2.3295, over budget; audit missed normalization metadata | True: per-channel fp32 factors were not in `bp_bytes()` | Factors → one power-of-two per channel, int8 exponent (exact); quality unchanged (HY even slightly better); re-audit: **LC 2.3195 / SF 2.3185 / HY 2.3256, all ≤ 2.326 ✓** |
| ② | QVG's real HY BPE ≈ 2.738, not 2.326 | True and worse: its code stores fp32 centroids — LC 2.464 / SF 2.406 / **HY 3.320** | Final tables now show measured values; both sides are always byte-audited |
| ③ | Triton kernel is not bit-identical to the reference | True: max\|Δ\| = 0.0625 / 0.03125 (1 ulp, bf16 summation order) | Claim downgraded to "numerically equal within 1 ulp" |
| ④ | From-scratch scoring script had wrong arm names / HY paths | True and worse: old names would silently score stale contaminated outputs | Arm names fixed (rtnfp8/kivifp8); HY path falls back correctly |
| ⑤ | Why-analysis scripts missed `PCA_FP8SIM=1` | True | Re-ran all 8 chunks with fp8: every number moved only in the 3rd decimal; all conclusions unchanged |

Net effect: after all fixes our claims stand (quality, mechanism, speed,
budget compliance), and ② actually widened our compression advantage
(2.32 vs 2.41-3.32 at equal accounting).

**Lessons**: audit key lists must be diffed against the encoder's full output;
"bit-identical"-class claims may only be guarded by asserts, never by memory
of one run.

---

## Late-0721 additions (after this report's baseline)

- **qvg-nom arm** (user-requested; NOT the paper's "QVG-Pro"): the strongest
  *nominally legal* single-stage k-means variant — BF16 centroids + 4-level
  asym token-axis B128, nominal BPE 2.3257 (0.0003 under the line), fp8
  metadata properly simulated. **Result: LC 32.80 dB — it beats us by 1.1 dB
  on LC quality.** We disclose this openly. Its costs: 33× slower encode per
  quantization event, no legal configuration on HY (nominal 2.738), and a
  razor-thin budget margin. Our position: we sell the best
  quality-speed-budget-coverage combination, not single-point quality.
  Details: [sell-budget-pca.md](sell-budget-pca.md).
- Full-prompt campaign (all 1003 MovieGen prompts, 6 arms, LC+SF) is running;
  results will land in `mpfull-table.md` with a sha256 manifest of every
  generated video.
