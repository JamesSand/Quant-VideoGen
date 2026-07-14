# Reference code

- `modeling_qwen3.py` — 复制自本地 transformers 4.51.3（`transformers/models/qwen3/`），未改动。
  用途：QK-Norm 讨论的证据（`Qwen3Attention.q_norm/k_norm`，第 195-196 行：per-head RMSNorm）——
  Qwen3 带 QK-Norm 却仍被 OScaR 观察到 TNI（附录 D），证伪"QK-Norm 阻止 TNI"的假设。
