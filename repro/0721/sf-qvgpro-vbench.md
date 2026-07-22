# SF qvg-nom 判决:VBench 四轴(700 帧窗,mp100 同协议)

| Arm | n | BC | IQ | SC | AQ |
|---|---|---|---|---|---|
| bf16 | 100 | 92.93 | 66.74 | 88.11 | 53.09 |
| qvg | 100 | 92.50 | 65.91 | 87.09 | 52.68 |
| pcaa128kaxvaxfp8 | 100 | 93.21 | 66.65 | 88.69 | 53.27 |
| qvgprot | 100 | 93.35 | 67.14 | 88.88 | 53.47 |
| qvgproc | 100 | 93.21 | 66.73 | 88.63 | 53.24 |

## 配对符号检验(vs ours=pcaa128kaxvaxfp8(终版臂),同 prompt 双向)

| 对比 | 轴 | win/tie/loss(对方视角) | p(双侧) |
|---|---|---|---|
| qvgprot vs ours | BC | 58/2/40 | 0.0854 |
| qvgprot vs ours | IQ | 55/0/45 | 0.3682 |
| qvgprot vs ours | SC | 52/0/48 | 0.7644 |
| qvgprot vs ours | AQ | 55/2/43 | 0.2664 |
| qvgproc vs ours | BC | 53/1/46 | 0.5467 |
| qvgproc vs ours | IQ | 48/0/52 | 0.7644 |
| qvgproc vs ours | SC | 52/0/48 | 0.7644 |
| qvgproc vs ours | AQ | 48/1/51 | 0.8408 |
