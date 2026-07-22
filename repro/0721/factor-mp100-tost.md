# Factor-only fallback: paired MP100 TOST

90% paired t interval; every effect is oriented so positive is better.

| model | role | metric | n | mean | 90% CI | TOST margin | result |
|---|---|---|---:|---:|---:|---:|---|
| lc | primary | PSNR | 100 | +3.4755 | [+2.9610, +3.9900] | ±0.1000 | SUPERIOR |
| lc | primary | SSIM | 100 | +0.0383 | [+0.0287, +0.0478] | ±0.0020 | SUPERIOR |
| lc | primary | LPIPS | 100 | +0.0251 | [+0.0198, +0.0304] | ±0.0030 | SUPERIOR |
| lc | guardrail | BC | 100 | +0.0106 | [-0.0050, +0.0262] | ±0.2000 | EQUIV |
| lc | guardrail | IQ | 100 | +0.0208 | [-0.0105, +0.0521] | ±0.3000 | EQUIV |
| lc | guardrail | SC | 100 | -0.0047 | [-0.0183, +0.0089] | ±0.3000 | EQUIV |
| lc | guardrail | AQ | 100 | +0.0076 | [-0.0131, +0.0283] | ±0.3000 | EQUIV |
| sf | primary | BC | 100 | +0.7095 | [+0.5503, +0.8687] | ±0.2000 | SUPERIOR |
| sf | primary | IQ | 100 | +0.7409 | [+0.2921, +1.1897] | ±0.3000 | SUPERIOR |
| sf | primary | SC | 100 | +1.5955 | [+1.2892, +1.9018] | ±0.3000 | SUPERIOR |
| sf | primary | AQ | 100 | +0.5977 | [+0.4220, +0.7734] | ±0.3000 | SUPERIOR |
| hy | primary | PSNR | 10 | +1.3248 | [+1.0912, +1.5583] | ±0.1000 | SUPERIOR |
| hy | primary | SSIM | 10 | +0.0813 | [+0.0691, +0.0935] | ±0.0020 | SUPERIOR |
| hy | primary | LPIPS | 10 | +0.0600 | [+0.0513, +0.0688] | ±0.0030 | SUPERIOR |
| hy | guardrail | BC | 10 | +0.2110 | [-0.1949, +0.6169] | ±0.2000 | FAIL |
| hy | guardrail | IQ | 10 | -0.3150 | [-0.6191, -0.0109] | ±0.3000 | FAIL |
| hy | guardrail | SC | 10 | +0.8590 | [+0.6532, +1.0648] | ±0.3000 | SUPERIOR |
| hy | guardrail | AQ | 10 | -0.4840 | [-1.1518, +0.1838] | ±0.3000 | FAIL |

**G4 fallback: FAIL**
