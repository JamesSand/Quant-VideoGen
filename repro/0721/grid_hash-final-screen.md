# Speed-selected PCA-Grid Hash：最终 chunk 闸门

| model | config | K relL2 QVG→hash | V relL2 QVG→hash | worst MSE ratio K/V | BPE hash≤QVG | 判定 |
|---|---|---:|---:|---:|---:|---|
| lc | `{'iters': 5, 'refine': 1, 'shared_labels': None}` | 0.21135→**0.14807** | 0.40349→**0.29152** | 0.703/0.955 | 2.3864≤2.4639 | PASS |
| sf | `{'iters': 1, 'refine': 0, 'shared_labels': 'v'}` | 0.20352→**0.15411** | 0.36995→**0.28242** | 0.966/0.794 | 2.3364≤2.4063 | PASS |
| hy | `{'iters': 1, 'refine': 0, 'shared_labels': 'v_rope'}` | 0.28390→**0.20221** | 0.44692→**0.30669** | 0.544/0.500 | 2.5588≤3.3199 | PASS |

**最终 G1+G2：PASS**
