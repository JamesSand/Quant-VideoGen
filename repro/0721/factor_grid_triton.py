"""Fully fused decoder for the factor-only product-grid fallback."""
import torch
import triton
import triton.language as tl


@triton.jit
def _factor_decode_ch_kernel(
    pack_ptr,
    scale_ptr,
    zp_ptr,
    res_exp_ptr,
    mu_ptr,
    coef_pack_ptr,
    coef_scale_ptr,
    coef_zp_ptr,
    basis_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    SP4: tl.constexpr,
    NBLK: tl.constexpr,
    CPB: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    FCS: tl.constexpr,
    FCZ: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    bh = tl.program_id(0)
    ps = tl.program_id(1)
    pd = tl.program_id(2)
    os = ps * BS + tl.arange(0, BS)
    od = pd * BD + tl.arange(0, BD)
    ms, md = os < S, od < D
    mask = md[:, None] & ms[None, :]

    byte_index = os // 4
    shift = (os % 4) * 2
    packed = tl.load(
        pack_ptr + bh * D * SP4 + od[:, None] * SP4 + byte_index[None, :],
        mask=mask,
        other=0,
    )
    code = ((packed >> shift[None, :]) & 3).to(tl.float32)
    block_index = os // G
    meta_offset = bh * D * NBLK + od[:, None] * NBLK + block_index[None, :]
    scale = tl.load(scale_ptr + meta_offset, mask=mask, other=1.0).to(tl.float32)
    zp = tl.load(zp_ptr + meta_offset, mask=mask, other=0.0).to(tl.float32)
    exponent = tl.load(
        res_exp_ptr + bh * D + od, mask=md, other=0
    ).to(tl.float32)
    factor = tl.exp2(exponent)
    acc = (code * scale + zp) * factor[:, None]
    mu = tl.load(mu_ptr + bh * D + od, mask=md, other=0.0).to(tl.float32)
    acc += mu[:, None]

    cscale = tl.load(
        coef_scale_ptr + bh * S + os, mask=ms, other=1.0
    ).to(tl.float32) * FCS
    czp = tl.load(
        coef_zp_ptr + bh * S + os, mask=ms, other=0.0
    ).to(tl.float32) * FCZ
    for r in tl.static_range(R):
        cpacked = tl.load(
            coef_pack_ptr + (bh * S + os) * CPB + r // 4,
            mask=ms,
            other=0,
        )
        ccode = ((cpacked >> (2 * (r % 4))) & 3).to(tl.float32)
        coef = ccode * cscale + czp
        basis = tl.load(
            basis_ptr + (bh * D + od) * R + r,
            mask=md,
            other=0.0,
        ).to(tl.float32)
        acc += basis[:, None] * coef[None, :]

    out_offset = bh * S * D + os[None, :] * D + od[:, None]
    tl.store(out_ptr + out_offset, acc.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def triton_factor_decode(state, bs=16, bd=128):
    b, h, s, d = state["shape"]
    bh = b * h
    r = state["r"]
    if state["axis"] != "channel" or state["grid"] != "asym":
        raise ValueError("factor fallback kernel supports channel-axis asym only")
    if "f_exp" not in state:
        raise ValueError("audited int8 residual exponent is required")
    lp = s + state["res_pad"]
    g = state["res_g"]
    cpb = state["coef_pack"].shape[-1]
    out = torch.empty(bh, s, d, device=state["res_pack"].device, dtype=torch.bfloat16)
    grid = (bh, triton.cdiv(s, bs), triton.cdiv(d, bd))
    _factor_decode_ch_kernel[grid](
        state["res_pack"].reshape(bh, d, lp // 4),
        state["res_scale"].reshape(bh, d, lp // g),
        state["res_zp"].reshape(bh, d, lp // g),
        state["f_exp"].reshape(bh, d),
        state["mu"].reshape(bh, d),
        state["coef_pack"].reshape(bh, s, cpb),
        state["coef_scale"].reshape(bh, s),
        state["coef_zp"].reshape(bh, s),
        state["basis"].reshape(bh, d, r),
        out,
        s,
        d,
        lp // 4,
        lp // g,
        cpb,
        g,
        r,
        state["f_cs"],
        state["f_cz"],
        bs,
        bd,
    )
    return out.reshape(b, h, s, d)
