# FlashVSR+ https://github.com/OpenImagingLab/FlashVSR_Plus (Apache-2.0): Wan2.1 1.3B video super-resolution
import math

import comfy_kitchen
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.wan.model import WanModel, WanSelfAttention, sinusoidal_embedding_1d
from comfy.ldm.wan.vae import RMS_norm
import comfy.ldm.common_dit

# 4 latent frames per stream chunk, attention windows of 2x8x8 tokens (128)
CHUNK_PATCH = (4, 16, 16)
LOCAL_MAX_WINDOWS = 64
GLOBAL_MASK_BYTES = 1 << 29


def causal_conv(conv, x, cache):
    if cache is None:
        x = F.pad(x, (1, 1, 1, 1, 2, 0), mode="replicate")
    else:
        x = F.pad(torch.cat([cache, x], dim=2), (1, 1, 1, 1, 0, 0), mode="replicate")
    return conv(x)


class LQProjection(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, hidden_dim2=3072, out_dim=1536, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = operations.Conv3d(in_dim, hidden_dim, (4, 3, 3), stride=(2, 1, 1), device=device, dtype=dtype)
        self.norm1 = RMS_norm(hidden_dim, images=False)
        self.conv2 = operations.Conv3d(hidden_dim, hidden_dim2, (4, 3, 3), stride=(2, 1, 1), device=device, dtype=dtype)
        self.norm2 = RMS_norm(hidden_dim2, images=False)
        self.linear = operations.Linear(hidden_dim2, out_dim, device=device, dtype=dtype)

    def forward(self, video):
        # latent i sees pixel frames 4i+1..4i+4, the first chunk only primes the conv caches
        video = torch.cat([video[:, :, :1].expand(-1, -1, 3, -1, -1), video], dim=2)
        b, c, f, h, w = video.shape
        cache1 = cache2 = None
        out = []
        for i in range(f // 4):
            x = video[:, :, i * 4:(i + 1) * 4].reshape(b, c, 4, h // 16, 16, w // 16, 16)
            x = x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 256, 4, h // 16, w // 16)
            next_cache = x[:, :, -2:]
            x = F.silu(self.norm1(causal_conv(self.conv1, x, cache1)))
            cache1 = next_cache
            next_cache = x[:, :, -2:]
            if i > 0:
                x = F.silu(self.norm2(causal_conv(self.conv2, x, cache2)))
                out.append(self.linear(x.flatten(2).transpose(1, 2)))
            cache2 = next_cache
        return torch.cat(out, dim=1)


def to_windows(x, grid):
    b, _, c = x.shape
    t, h, w = grid
    x = x.view(b, t // 2, 2, h // 8, 8, w // 8, 8, c).permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(b, t // 2, h // 8, w // 8, 128, c)


def from_windows(x):
    b, nt, nh, nw, _, c = x.shape
    x = x.view(b, nt, nh, nw, 2, 8, 8, c).permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(b, nt * 2 * nh * 8 * nw * 8, c)


def stream_block_mask(qw, kw, spatial_grid, local_range, topk):
    # === from models/attention_backends/sparse.py build_stream_sparse_layout (FlashVSR_Plus, Apache-2.0) ===
    b, n, query_blocks, _, d = qw.shape
    key_blocks = kw.shape[2]
    rows, cols = spatial_grid
    device = qw.device
    q_spatial = torch.arange(query_blocks, device=device) % (rows * cols)
    k_spatial = torch.arange(key_blocks, device=device) % (rows * cols)
    q_row, q_col = q_spatial // cols, q_spatial % cols
    k_row, k_col = k_spatial // cols, k_spatial % cols
    row_start = q_row - local_range // 2
    col_start = q_col - local_range // 2
    candidates = ((k_row[None] >= row_start[:, None]) & (k_row[None] <= row_start[:, None] + local_range - 1)
                  & (k_col[None] >= col_start[:, None]) & (k_col[None] <= col_start[:, None] + local_range - 1))

    scores = torch.einsum("bhqd,bhkd->bhqk", qw.mean(dim=3, dtype=torch.float32), kw.mean(dim=3, dtype=torch.float32)) / math.sqrt(d)
    scores.masked_fill_(~candidates, -torch.inf)
    scores.sub_(torch.logsumexp(scores, dim=-1, keepdim=True)).exp_()
    # one top-k per chunk with a strict threshold, as trained
    grouped = scores.masked_fill(~candidates, -torch.inf).view(b, n, -1)
    count = min(grouped.shape[-1] - 1, topk)
    threshold = grouped.topk(count + 1, dim=-1).values[..., -1:]
    mask = (grouped > threshold).view(b, n, query_blocks, key_blocks) & candidates
    first_candidate = candidates.to(torch.int32).argmax(dim=-1, keepdim=True).expand(b, n, -1, -1)
    return torch.where(mask.any(dim=-1, keepdim=True), mask, torch.zeros_like(mask).scatter_(-1, first_candidate, True))


def local_neighbourhood(nt, nh, nw, device):
    # window ids and validity of each chunk position's 3x3 neighbourhood in the previous and current windows
    offsets = torch.arange(-1, 2, device=device)
    tw = torch.arange(0, nt, 2, device=device)[:, None] + offsets
    rows = torch.arange(nh, device=device)[:, None] + offsets
    cols = torch.arange(nw, device=device)[:, None] + offsets
    valid = (tw >= 0)[:, None, None, :, None, None] & ((rows >= 0) & (rows < nh))[None, :, None, None, :, None] & ((cols >= 0) & (cols < nw))[None, None, :, None, None, :]
    ids = (tw * nh * nw)[:, None, None, :, None, None] + (rows * nw)[None, :, None, None, :, None] + cols[None, None, :, None, None, :]
    return ids.reshape(nt // 2, nh, nw, 27), valid.reshape(nt // 2, nh, nw, 27)


def half_window_lists(ids, counts):
    # 128-token windows as pairs of the kernel's 64-token blocks
    ids = torch.stack([ids * 2, ids * 2 + 1], dim=-1).flatten(-2).repeat_interleave(2, dim=-2)
    return ids, counts.repeat_interleave(2, dim=-1) * 2


class FlashVSRSelfAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, local_window, eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, eps=eps, operation_settings=operation_settings)
        self.local_window = local_window

    def forward(self, x, freqs, transformer_options={}):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs).view(b, s, n * d)
        k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs).view(b, s, n * d)
        v = self.v(x)
        grid = transformer_options["grid_sizes"]
        fused = comfy_kitchen.sol_attn_is_available(x.device, block_lists=True)
        attention = self.local_attention if self.local_window else self.global_attention
        return self.o(attention(q, k, v, grid, fused, transformer_options))

    def global_attention(self, q, k, v, grid, fused, transformer_options):
        # chunk plus 3 previous windows, 11x11 window neighbourhood, trained top-k
        n, d = self.num_heads, self.head_dim
        qw, kw, vw = (to_windows(a, grid) for a in (q, k, v))
        b, nt, nh, nw, _, c = qw.shape
        spatial = nh * nw
        topk = int(spatial * spatial * 2.5) - 1
        if fused:
            max_windows = min(5 * spatial, 5 * 11 * 11)
            kv_indices = torch.empty((b, n, nt * spatial * 2, 2 * max_windows), dtype=torch.int16, device=q.device)
            kv_counts = torch.empty((b, n, nt * spatial * 2), dtype=torch.int32, device=q.device)
        else:
            out = torch.empty_like(qw)
        for w1 in range(2, nt + 1, 2):
            w0 = max(0, w1 - 5)
            qc = qw[:, w1 - 2:w1].reshape(b, 2 * spatial, 128, n, d).permute(0, 3, 1, 2, 4)
            kc, vc = (a[:, w0:w1].reshape(b, (w1 - w0) * spatial, 128, n, d).permute(0, 3, 1, 2, 4) for a in (kw, vw))
            block_mask = stream_block_mask(qc, kc, (nh, nw), 11, topk)
            if fused:
                selected = min(max_windows, block_mask.shape[-1])
                ids = torch.argsort((~block_mask).to(torch.uint8), dim=-1, stable=True)[..., :selected] + w0 * spatial
                rows = slice((w1 - 2) * spatial * 2, w1 * spatial * 2)
                kv_indices[:, :, rows, :2 * selected], kv_counts[:, :, rows] = half_window_lists(ids, block_mask.sum(dim=-1, dtype=torch.int32))
                continue
            kc, vc = (a.reshape(b, n, -1, d) for a in (kc, vc))
            out_c = out[:, w1 - 2:w1].view(b, 2 * spatial, 128, c)
            slab = max(1, GLOBAL_MASK_BYTES // (n * 128 * kc.shape[2]))
            for q0 in range(0, 2 * spatial, slab):
                q1 = min(q0 + slab, 2 * spatial)
                mask = block_mask[:, :, q0:q1, :, None, None].expand(-1, -1, -1, -1, 128, 128).transpose(3, 4).reshape(b, n, (q1 - q0) * 128, -1)
                out_c[:, q0:q1] = optimized_attention(qc[:, :, q0:q1].reshape(b, n, -1, d), kc, vc, heads=n, mask=mask, skip_reshape=True, transformer_options=transformer_options).view(b, q1 - q0, 128, c)
        if fused:
            out = comfy_kitchen.sol_attn(*(a.view(b, -1, n, d) for a in (qw, kw, vw)), kv_indices=kv_indices, kv_counts=kv_counts).view(qw.shape)
        return from_windows(out)

    def local_attention(self, q, k, v, grid, fused, transformer_options):
        # chunk plus previous window, 3x3 window neighbourhood (the trained top-k keeps all of it from 512x512 up)
        n, d = self.num_heads, self.head_dim
        qw = to_windows(q, grid)
        b, nt, nh, nw, _, c = qw.shape
        nc, spatial = nt // 2, nh * nw
        ids, valid = local_neighbourhood(nt, nh, nw, q.device)
        if fused:
            ids = torch.gather(ids, -1, torch.argsort((~valid).to(torch.uint8), dim=-1, stable=True))
            ids = ids.view(nc, 1, spatial, 27).expand(-1, 2, -1, -1).reshape(nt * spatial, 27)
            counts = valid.sum(dim=-1, dtype=torch.int32).view(nc, 1, spatial).expand(-1, 2, -1).reshape(nt * spatial)
            kv_indices, kv_counts = half_window_lists(ids, counts)
            kw, vw = (to_windows(a, grid) for a in (k, v))
            out = comfy_kitchen.sol_attn(*(a.view(b, -1, n, d) for a in (qw, kw, vw)), kv_indices=kv_indices.expand(b, n, -1, -1), kv_counts=kv_counts.expand(b, n, -1))
            return from_windows(out.view(qw.shape))

        qw = qw.view(b, nc, 2, nh, nw, 128, c).permute(0, 1, 3, 4, 2, 5, 6).reshape(b, nc, nh, nw, 256, c)
        kw, vw = (F.pad(to_windows(a, grid), (0, 0, 0, 0, 1, 1, 1, 1, 1, 0)).unfold(1, 3, 2).unfold(2, 3, 1).unfold(3, 3, 1) for a in (k, v))
        valid = valid.view(nc, nh, nw, 27, 1).expand(-1, -1, -1, -1, 128).reshape(nc, nh, nw, 1, 27 * 128)
        out = torch.empty_like(qw)
        rows_per_slab = max(1, LOCAL_MAX_WINDOWS // nw)
        for ci in range(nc):
            for r0 in range(0, nh, rows_per_slab):
                r1 = min(r0 + rows_per_slab, nh)
                n_win = (r1 - r0) * nw
                ks, vs = (a[:, ci, r0:r1].permute(0, 1, 2, 5, 6, 7, 3, 4).reshape(b * n_win, 27 * 128, c) for a in (kw, vw))
                mask = valid[ci, r0:r1].reshape(1, n_win, 1, 27 * 128).expand(b, -1, -1, -1).reshape(b * n_win, 1, 27 * 128)
                x = optimized_attention(qw[:, ci, r0:r1].reshape(b * n_win, 256, c), ks, vs, heads=n, mask=mask, transformer_options=transformer_options)
                out[:, ci, r0:r1] = x.view(b, r1 - r0, nw, 256, c)
                del ks, vs, x

        out = out.view(b, nc, nh, nw, 2, 2, 8, 8, c).permute(0, 1, 4, 5, 2, 6, 3, 7, 8)
        return out.reshape(b, nt * 2 * nh * 8 * nw * 8, c)


class FlashVSRWanModel(WanModel):
    def __init__(self, model_type="flashvsr", local_attn_layers=(), operations=None, device=None, dtype=None, **kwargs):
        super().__init__(model_type="t2v", operations=operations, device=device, dtype=dtype, **kwargs)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        for i, block in enumerate(self.blocks):
            block.self_attn = FlashVSRSelfAttention(self.dim, self.num_heads, i in local_attn_layers, eps=self.eps, operation_settings=operation_settings)
        self.lq_proj_in = LQProjection(out_dim=self.dim, device=device, dtype=dtype, operations=operations)
        self.register_buffer("positive_conditioning", torch.empty((self.text_len, self.text_dim), device=device, dtype=dtype))

    def forward_orig(self, x, t, context, freqs=None, lq_pixels=None, transformer_options={}, **kwargs):
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2) + self.lq_proj_in(lq_pixels)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context = self.text_embedding(context)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, transformer_options=transformer_options)

        x = self.head(x, e)
        return self.unpatchify(x, grid_sizes)

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, lq_pixels=None, **kwargs):
        if lq_pixels is None:
            raise ValueError("FlashVSR needs the low-quality video, connect the model through the FlashVSR Conditioning node.")
        t, h, w = x.shape[2:]
        x = comfy.ldm.common_dit.pad_to_patch_size(x, CHUNK_PATCH, padding_mode="replicate")
        tp, hp, wp = x.shape[2:]
        lq_pixels = lq_pixels[:, :, :tp * 4 + 1, :hp * 8, :wp * 8]
        lq_pixels = F.pad(lq_pixels, (0, wp * 8 - lq_pixels.shape[-1], 0, hp * 8 - lq_pixels.shape[-2], 0, tp * 4 + 1 - lq_pixels.shape[-3]), mode="replicate")
        freqs = self.rope_encode(tp, hp, wp, device=x.device, dtype=x.dtype, transformer_options=transformer_options)
        return self.forward_orig(x, timestep, context, freqs=freqs, lq_pixels=lq_pixels, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]
