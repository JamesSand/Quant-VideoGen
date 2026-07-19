"""Triton fused decode (K3) for Budget-PCA packed buffers.

Channel-axis kernel: packed [BH, D, Sp/4] -> out [BH, S, D] bf16, fusing
unpack -> dequant(fp8 scale/zp) -> + mu -> + c_hat @ basis^T (r-dot per elem).
Token-axis kernel: packed [BH, S, Dp/4] -> out [BH, S, D] bf16, same fusion.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dq_ch(pack_ptr, sc_ptr, zp_ptr, mu_ptr, chat_ptr, basis_ptr, out_ptr,
           S, D, SP4, NBLK, G: tl.constexpr, R: tl.constexpr, TERN: tl.constexpr,
           BS: tl.constexpr, BD: tl.constexpr):
    pb = tl.program_id(0)
    ps = tl.program_id(1)
    pd = tl.program_id(2)
    offs_s = ps * BS + tl.arange(0, BS)
    offs_d = pd * BD + tl.arange(0, BD)
    ms = offs_s < S
    md = offs_d < D
    # unpack 2-bit codes from [D, SP4] bytes
    byte_i = offs_s // 4
    shift = (offs_s % 4) * 2
    p = tl.load(pack_ptr + pb * D * SP4 + offs_d[:, None] * SP4 + byte_i[None, :],
                mask=md[:, None] & ms[None, :], other=0)
    code = ((p >> shift[None, :]) & 3).to(tl.float32)
    blk = offs_s // G
    sc = tl.load(sc_ptr + pb * D * NBLK + offs_d[:, None] * NBLK + blk[None, :],
                 mask=md[:, None] & ms[None, :], other=1.0).to(tl.float32)
    if TERN:
        res = (code - 1.0) * sc
    else:
        zp = tl.load(zp_ptr + pb * D * NBLK + offs_d[:, None] * NBLK + blk[None, :],
                     mask=md[:, None] & ms[None, :], other=0.0).to(tl.float32)
        res = code * sc + zp
    mu = tl.load(mu_ptr + pb * D + offs_d, mask=md, other=0.0).to(tl.float32)
    acc = res + mu[:, None]
    if R > 0:
        for r in tl.static_range(R):
            ch = tl.load(chat_ptr + pb * S * R + offs_s * R + r, mask=ms, other=0.0).to(tl.float32)
            bs = tl.load(basis_ptr + pb * D * R + offs_d * R + r, mask=md, other=0.0).to(tl.float32)
            acc += bs[:, None] * ch[None, :]
    # write transposed: out[bh, s, d]
    tl.store(out_ptr + pb * S * D + offs_s[None, :] * D + offs_d[:, None],
             acc.to(tl.bfloat16), mask=md[:, None] & ms[None, :])


@triton.jit
def _dq_tok(pack_ptr, sc_ptr, zp_ptr, mu_ptr, chat_ptr, basis_ptr, out_ptr,
            S, D, DP4, NBLK, G: tl.constexpr, R: tl.constexpr, TERN: tl.constexpr,
            BS: tl.constexpr, BD: tl.constexpr):
    pb = tl.program_id(0)
    ps = tl.program_id(1)
    pd = tl.program_id(2)
    offs_s = ps * BS + tl.arange(0, BS)
    offs_d = pd * BD + tl.arange(0, BD)
    ms = offs_s < S
    md = offs_d < D
    byte_i = offs_d // 4
    shift = (offs_d % 4) * 2
    p = tl.load(pack_ptr + pb * S * DP4 + offs_s[:, None] * DP4 + byte_i[None, :],
                mask=ms[:, None] & md[None, :], other=0)
    code = ((p >> shift[None, :]) & 3).to(tl.float32)
    blk = offs_d // G
    sc = tl.load(sc_ptr + pb * S * NBLK + offs_s[:, None] * NBLK + blk[None, :],
                 mask=ms[:, None] & md[None, :], other=1.0).to(tl.float32)
    if TERN:
        res = (code - 1.0) * sc
    else:
        zp = tl.load(zp_ptr + pb * S * NBLK + offs_s[:, None] * NBLK + blk[None, :],
                     mask=ms[:, None] & md[None, :], other=0.0).to(tl.float32)
        res = code * sc + zp
    mu = tl.load(mu_ptr + pb * D + offs_d, mask=md, other=0.0).to(tl.float32)
    acc = res + mu[None, :]
    if R > 0:
        for r in tl.static_range(R):
            ch = tl.load(chat_ptr + pb * S * R + offs_s * R + r, mask=ms, other=0.0).to(tl.float32)
            bs = tl.load(basis_ptr + pb * D * R + offs_d * R + r, mask=md, other=0.0).to(tl.float32)
            acc += ch[:, None] * bs[None, :]
    tl.store(out_ptr + pb * S * D + offs_s[:, None] * D + offs_d[None, :],
             acc.to(tl.bfloat16), mask=ms[:, None] & md[None, :])


@torch.no_grad()
def triton_decode(d, c_hat=None):
    """d: bp_encode dict (single tensor, not packed256). Returns [B,H,S,D] bf16."""
    B, H, S, D = d["shape"]
    BH = B * H
    g = d["res_g"]
    tern = d["grid"] == "ternary"
    r = d["r"]
    ax_ch = d["axis"] == "channel"
    L = S if ax_ch else D
    Lp = L + d.get("res_pad", 0)
    if r > 0 and c_hat is None:
        from bp_quant import _unpack2, _fp8_load
        cq = _unpack2(d["coef_pack"].reshape(BH, S, -1), r).float()
        c_hat = cq * _fp8_load(d["coef_scale"]).unsqueeze(-1) + _fp8_load(d["coef_zp"]).unsqueeze(-1)
    out = torch.empty(BH, S, D, device=d["res_pack"].device, dtype=torch.bfloat16)
    pack = d["res_pack"].reshape(BH, D if ax_ch else S, Lp // 4)
    sc = d["res_scale"].reshape(BH, D if ax_ch else S, Lp // g)
    zp = d["res_zp"].reshape(BH, D if ax_ch else S, Lp // g) if d["res_zp"] is not None else sc
    mu = d["mu"].reshape(BH, D).contiguous()
    basis = d["basis"].reshape(BH, D, r).contiguous() if r > 0 else mu  # dummy ptr ok when R==0
    chat = c_hat.reshape(BH, S, r).to(torch.float32).contiguous() if r > 0 else mu
    BS, BD = 64, 64
    grid = (BH, triton.cdiv(S, BS), triton.cdiv(D, BD))
    kern = _dq_ch if ax_ch else _dq_tok
    kern[grid](pack, sc, zp, mu, chat, basis, out,
               S, D, Lp // 4, Lp // g, g, r, tern, BS, BD)
    return out.reshape(B, H, S, D)


@torch.no_grad()
def triton_decode_packed256(d):
    return torch.cat([triton_decode(d["halves"][0]), triton_decode(d["halves"][1])], dim=-1)
