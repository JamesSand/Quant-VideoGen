"""Fused PCA-grid hash decoder: table gather + INT2 residual."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grid_hash_decode_kernel(
    labels_ptr,
    table_ptr,
    table_exp_ptr,
    pack_ptr,
    scale_ptr,
    zp_ptr,
    res_exp_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    SP4: tl.constexpr,
    NBLK: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    bh = tl.program_id(0)
    ps = tl.program_id(1)
    pd = tl.program_id(2)
    os = ps * BS + tl.arange(0, BS)
    od = pd * BD + tl.arange(0, BD)
    ms, md = os < S, od < D
    mask = ms[:, None] & md[None, :]

    labels = tl.load(labels_ptr + bh * S + os, mask=ms, other=0).to(tl.int32)
    texp = tl.load(
        table_exp_ptr + bh * D + od, mask=md, other=0
    ).to(tl.float32)
    table_factor = tl.exp2(texp)
    table_offset = bh * K * D + labels[:, None] * D + od[None, :]
    pred = tl.load(table_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    pred *= table_factor[None, :]

    byte_index = os // 4
    shift = (os % 4) * 2
    packed = tl.load(
        pack_ptr + bh * D * SP4 + od[None, :] * SP4 + byte_index[:, None],
        mask=mask,
        other=0,
    )
    code = ((packed >> shift[:, None]) & 3).to(tl.float32)
    block_index = os // G
    meta_offset = (
        bh * D * NBLK + od[None, :] * NBLK + block_index[:, None]
    )
    scale = tl.load(scale_ptr + meta_offset, mask=mask, other=1.0)
    zp = tl.load(zp_ptr + meta_offset, mask=mask, other=0.0)
    rexp = tl.load(
        res_exp_ptr + bh * D + od, mask=md, other=0
    ).to(tl.float32)
    residual_factor = tl.exp2(rexp)
    residual = (
        code * scale.to(tl.float32) + zp.to(tl.float32)
    ) * residual_factor[None, :]

    out_offset = bh * S * D + os[:, None] * D + od[None, :]
    tl.store(out_ptr + out_offset, (pred + residual).to(tl.bfloat16), mask=mask)


@torch.no_grad()
def triton_grid_hash_decode(state, bs=16, bd=128):
    b, h, s, d = state["shape"]
    bh = b * h
    lp = s + state["res_pad"]
    g = state["res_g"]
    out = torch.empty(bh, s, d, device=state["labels"].device, dtype=torch.bfloat16)
    grid = (bh, triton.cdiv(s, bs), triton.cdiv(d, bd))
    _grid_hash_decode_kernel[grid](
        state["labels"].reshape(bh, s),
        state["table"].reshape(bh, 256, d),
        state["table_exp"].reshape(bh, d),
        state["res_pack"].reshape(bh, d, lp // 4),
        state["res_scale"].reshape(bh, d, lp // g),
        state["res_zp"].reshape(bh, d, lp // g),
        state["res_exp"].reshape(bh, d),
        out,
        s,
        d,
        lp // 4,
        lp // g,
        256,
        g,
        bs,
        bd,
    )
    return out.reshape(b, h, s, d)
