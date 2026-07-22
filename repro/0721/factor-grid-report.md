# Factor-only product-grid fallback

| model | K relL2 QVG→factor | V relL2 QVG→factor | worst MSE K/V | BPE factor≤QVG | speedup (95% lower) | result |
|---|---:|---:|---:|---:|---:|---|
| sf | 0.20352→0.16652 | 0.36995→0.32304 | 0.977/1.118 | 2.3185≤2.4063 | 1.067× (1.063×) | PASS |
| hy | 0.28390→0.21651 | 0.44692→0.35076 | 0.636/0.672 | 2.2313≤3.3199 | 1.426× (1.420×) | PASS |

**Fallback G1–G3: PASS**
