# SAM3 video tracker: memory encoder, memory attention, SAM mask decoder/prompt encoder.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.sam3.backbone import rope_2d, PositionEmbeddingSine
from comfy.ops import cast_to_input
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.cascade.common import LayerNorm2d_op
from comfy.ldm.sam3.sam import MLP, PositionEmbeddingRandom

NO_OBJ_SCORE = -1024.0


def fill_holes_in_mask_scores(mask, max_area=0):
    """Remove small foreground sprinkles and fill small background holes via morphological ops."""
    if max_area <= 0:
        return mask
    k = max(3, int(math.sqrt(max_area)) | 1)  # odd kernel
    p = k // 2
    def _pool(x):
        """max_pool2d with replicate padding to preserve edges."""
        return F.max_pool2d(F.pad(x, [p]*4, mode='replicate'), k, stride=1)
    fg = (mask > 0).float()
    # Opening (erode→dilate): removes small foreground sprinkles
    opened = _pool(-_pool(-fg))
    mask = torch.where((fg > 0.5) & (opened < 0.5), torch.tensor(-0.1, device=mask.device, dtype=mask.dtype), mask)
    # Closing (dilate→erode): fills small background holes
    fg = (mask > 0).float()
    closed = -_pool(-_pool(fg))
    mask = torch.where((fg < 0.5) & (closed > 0.5), torch.tensor(0.1, device=mask.device, dtype=mask.dtype), mask)
    return mask


def apply_rope_memory(q, k, freqs, num_heads, num_k_exclude_rope=0):
    """Apply 2D axial RoPE to memory attention using flux rope format.

    Args:
        q: [B, Nq, C] projected queries (current frame features)
        k: [B, Nk, C] projected keys (memory tokens)
        freqs: [1, Nq, dim//2, 2, 2] flux-format rotation matrices for one frame
        num_heads: number of attention heads
        num_k_exclude_rope: number of trailing k tokens to skip RoPE (object pointers)
    """
    B, Nq, C = q.shape
    head_dim = C // num_heads

    # freqs shape: [1, 1, Nq, dim//2, 2, 2] (heads broadcast dim already included)
    q_h = q.view(B, Nq, num_heads, head_dim).transpose(1, 2)
    q_h = apply_rope1(q_h, freqs)
    q = q_h.transpose(1, 2).reshape(B, Nq, C)

    # Apply RoPE to k (excluding last num_k_exclude_rope tokens)
    Nk = k.shape[1]
    num_k_rope = Nk - num_k_exclude_rope
    if num_k_rope > 0:
        # Repeat freqs for multiple frames of spatial memory
        Nf = freqs.shape[2]  # spatial positions in one frame
        if num_k_rope > Nf:
            r = (num_k_rope + Nf - 1) // Nf
            pe_k = freqs.repeat(1, 1, r, 1, 1, 1)[:, :, :num_k_rope]
        else:
            pe_k = freqs[:, :, :num_k_rope]

        k_h = k[:, :num_k_rope].view(B, num_k_rope, num_heads, head_dim).transpose(1, 2)
        k_h = apply_rope1(k_h, pe_k)
        k = k.clone()
        k[:, :num_k_rope] = k_h.transpose(1, 2).reshape(B, num_k_rope, C)

    return q, k


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """1D sinusoidal positional encoding for temporal positions."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


# Split attention with configurable input dims (for asymmetric cross-attention)
class SplitAttn(nn.Module):
    def __init__(self, embed_dim, num_heads=1, kv_dim=None, internal_dim=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        kv_dim = kv_dim or embed_dim
        internal_dim = internal_dim or embed_dim
        self.q_proj = operations.Linear(embed_dim, internal_dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(internal_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, q, k=None, v=None, rope=None, num_k_exclude_rope=0):
        if k is None:
            k = q
        if v is None:
            v = k
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, num_k_exclude_rope)
        out = optimized_attention(q, k, v, self.num_heads)
        return self.out_proj(out)


class MemoryAttnLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = SplitAttn(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image = SplitAttn(d_model, num_heads, kv_dim=kv_dim, device=device, dtype=dtype, operations=operations)
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, src, memory, memory_pos=None, rope=None, num_k_exclude_rope=0):
        # Pre-norm self-attention (with RoPE)
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, rope=rope)
        src = src + src2
        # Pre-norm cross-attention to memory (with RoPE + additive pos on keys)
        src2 = self.norm2(src)
        mem_k = memory + memory_pos if memory_pos is not None else memory
        src2 = self.cross_attn_image(src2, mem_k, memory, rope=rope, num_k_exclude_rope=num_k_exclude_rope)
        src = src + src2
        # Pre-norm FFN
        src2 = self.norm3(src)
        src2 = self.linear2(F.relu(self.linear1(src2)))
        src = src + src2
        return src


class MemoryAttnEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttnLayer(d_model, num_heads, kv_dim, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, src, memory, src_pos=None, memory_pos=None, num_k_exclude_rope=0):
        # pos_enc_at_input: add 0.1 * position encoding
        if src_pos is not None:
            src = src + 0.1 * src_pos

        # Compute RoPE frequencies for current frame spatial grid (flux format)
        B, N, C = src.shape
        hw = int(math.sqrt(N))
        rope = rope_2d(hw, hw, C).to(device=src.device)

        for layer in self.layers:
            src = layer(src, memory, memory_pos=memory_pos, rope=rope, num_k_exclude_rope=num_k_exclude_rope)
        return self.norm(src)


class MemoryTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, kv_dim=64, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = MemoryAttnEncoder(d_model, num_heads, kv_dim, dim_ff, num_layers, device=device, dtype=dtype, operations=operations)


class TwoWayAttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, mlp_dim=2048, downsample_rate=2, device=None, dtype=None, operations=None):
        super().__init__()
        internal_dim = d_model // downsample_rate

        self.self_attn = SplitAttn(d_model, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_token_to_image = SplitAttn(d_model, num_heads, internal_dim=internal_dim, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image_to_token = SplitAttn(d_model, num_heads, internal_dim=internal_dim, device=device, dtype=dtype, operations=operations)

        self.mlp = nn.Sequential()
        self.mlp.lin1 = operations.Linear(d_model, mlp_dim, device=device, dtype=dtype)
        self.mlp.lin2 = operations.Linear(mlp_dim, d_model, device=device, dtype=dtype)

        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm4 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, queries, keys, query_pe, key_pe):
        # Self-attention on queries
        q = queries + query_pe
        attn_out = self.self_attn(q)
        queries = self.norm1(queries + attn_out)

        # Cross-attention: tokens -> image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = self.norm2(queries + attn_out)

        # MLP
        mlp_out = self.mlp.lin2(F.relu(self.mlp.lin1(queries)))
        queries = self.norm3(queries + mlp_out)

        # Cross-attention: image -> tokens
        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(q, k, queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, depth=2, mlp_dim=2048, downsample_rate=2,  device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttnBlock(d_model, num_heads, mlp_dim, downsample_rate, device=device, dtype=dtype, operations=operations)
            for _ in range(depth)
        ])
        self.final_attn_token_to_image = SplitAttn(
            d_model, num_heads, internal_dim=d_model // downsample_rate, device=device, dtype=dtype, operations=operations)
        self.norm_final_attn = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, image_embedding, image_pe, point_embedding):
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys


class SAMMaskDecoder(nn.Module):
    def __init__(self, d_model=256, num_multimask_outputs=3, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(d_model, num_heads=8, depth=2, mlp_dim=2048, device=device, dtype=dtype, operations=operations)

        self.iou_token = operations.Embedding(1, d_model, device=device, dtype=dtype)
        self.mask_tokens = operations.Embedding(self.num_mask_tokens, d_model, device=device, dtype=dtype)
        self.obj_score_token = operations.Embedding(1, d_model, device=device, dtype=dtype)

        # Output upscaling: d_model -> d_model//4 -> d_model//8 at 4x resolution
        LN2d = LayerNorm2d_op(operations)
        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2, device=device, dtype=dtype), LN2d(d_model // 4, device=device, dtype=dtype), nn.GELU(),
            operations.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2, device=device, dtype=dtype), nn.GELU(),
        )

        # High-res feature integration
        self.conv_s0 = operations.Conv2d(d_model, d_model // 8, kernel_size=1, device=device, dtype=dtype)
        self.conv_s1 = operations.Conv2d(d_model, d_model // 4, kernel_size=1, device=device, dtype=dtype)

        # Per-mask hypernetwork MLPs
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3, device=device, dtype=dtype, operations=operations)
            for _ in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_tokens, 3, device=device, dtype=dtype, operations=operations)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3, device=device, dtype=dtype, operations=operations)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                high_res_features=None, multimask_output=False, return_all=False):
        B = sparse_prompt_embeddings.shape[0]
        tokens = torch.cat([cast_to_input(self.iou_token.weight, image_embeddings),
                            cast_to_input(self.mask_tokens.weight, image_embeddings),
                            cast_to_input(self.obj_score_token.weight, image_embeddings)], dim=0)
        tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos_src.flatten(2).permute(0, 2, 1)

        hs, src_out = self.transformer(src_flat, pos_flat, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        obj_score_token_out = hs[:, 1 + self.num_mask_tokens, :]

        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        # Upscale in two stages, inserting high-res features at matching channel dims
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        upscaled = act1(ln1(dc1(src_out)))
        if high_res_features is not None:
            upscaled = upscaled + F.interpolate(self.conv_s1(high_res_features[1]), size=upscaled.shape[-2:], mode="bilinear", align_corners=False)
            upscaled = act2(dc2(upscaled))
            upscaled = upscaled + F.interpolate(self.conv_s0(high_res_features[0]), size=upscaled.shape[-2:], mode="bilinear", align_corners=False)
        else:
            upscaled = act2(dc2(upscaled))

        hyper_in = torch.stack([
            mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)
        ], dim=1)

        masks = (hyper_in @ upscaled.flatten(2)).view(B, self.num_mask_tokens, upscaled.shape[2], upscaled.shape[3])
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)

        if multimask_output:
            out_masks = masks[:, 1:]
            out_iou = iou_pred[:, 1:]
            out_tokens = mask_tokens_out[:, 1:]
        else:
            out_masks = masks[:, 0:1]
            out_iou = iou_pred[:, 0:1]
            out_tokens = mask_tokens_out[:, 0:1]

        if return_all:
            return out_masks, out_iou, out_tokens, object_score_logits
        return out_masks, out_iou


class SAMPromptEncoder(nn.Module):
    def __init__(self, d_model=256, image_embedding_size=(72, 72), input_image_size=(1008, 1008), device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = d_model
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)
        self.point_embeddings = nn.ModuleList([
            operations.Embedding(1, d_model, device=device, dtype=dtype) for _ in range(4)
        ])
        self.not_a_point_embed = operations.Embedding(1, d_model, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.mask_downscaling = nn.Sequential(
            operations.Conv2d(1, 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(4, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(4, 16, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(16, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(16, d_model, kernel_size=1, device=device, dtype=dtype),
        )
        self.no_mask_embed = operations.Embedding(1, d_model, device=device, dtype=dtype)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size)

    def forward(self, points=None, boxes=None, masks=None):
        ref = points[0] if points is not None else boxes if boxes is not None else masks
        B = 1
        sparse = torch.empty((B, 0, self.embed_dim), device=ref.device, dtype=ref.dtype)

        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            pe = self.pe_layer.forward_with_coords(coords + 0.5, self.input_image_size)
            pe[labels == 0] += cast_to_input(self.point_embeddings[0].weight, ref)
            pe[labels == 1] += cast_to_input(self.point_embeddings[1].weight, ref)
            invalid = (labels == -1)
            pe[invalid] = 0.0
            pe[invalid] += cast_to_input(self.not_a_point_embed.weight, ref)
            sparse = torch.cat([sparse.expand(B, -1, -1), pe], dim=1)

        if boxes is not None:
            B = boxes.shape[0]
            corners = self.pe_layer.forward_with_coords((boxes.reshape(-1, 2, 2) + 0.5), self.input_image_size)
            corners[:, 0] += cast_to_input(self.point_embeddings[2].weight, ref)
            corners[:, 1] += cast_to_input(self.point_embeddings[3].weight, ref)
            sparse = torch.cat([sparse.expand(B, -1, -1), corners], dim=1)

        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = cast_to_input(self.no_mask_embed.weight, ref).reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1])

        return sparse, dense


class CXBlock(nn.Module):
    def __init__(self, dim=256, kernel_size=7, device=None, dtype=None, operations=None):
        super().__init__()
        self.dwconv = operations.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.pwconv1 = operations.Linear(dim, 4 * dim, device=device, dtype=dtype)
        self.pwconv2 = operations.Linear(4 * dim, dim, device=device, dtype=dtype)
        self.gamma = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(F.gelu(self.pwconv1(self.norm(x))))
        x.mul_(cast_to_input(self.gamma, x))
        return residual + x.permute(0, 3, 1, 2)


class MaskDownSampler(nn.Module):
    def __init__(self, out_dim=256, device=None, dtype=None, operations=None):
        super().__init__()
        LN2d = LayerNorm2d_op(operations)
        self.encoder = nn.Sequential(operations.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(4, device=device, dtype=dtype), nn.GELU(), operations.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(16, device=device, dtype=dtype), nn.GELU(), operations.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(64, device=device, dtype=dtype), nn.GELU(), operations.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(out_dim, device=device, dtype=dtype), nn.GELU(), operations.Conv2d(out_dim, out_dim, kernel_size=1, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.encoder(x)


class Fuser(nn.Module):
    def __init__(self, dim=256, num_layers=2, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.Sequential(*[CXBlock(dim, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


class MemoryBackbone(nn.Module):
    def __init__(self, d_model=256, out_dim=64, device=None, dtype=None, operations=None):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(d_model, device=device, dtype=dtype, operations=operations)
        self.pix_feat_proj = operations.Conv2d(d_model, d_model, kernel_size=1, device=device, dtype=dtype)
        self.fuser = Fuser(d_model, num_layers=2, device=device, dtype=dtype, operations=operations)
        self.out_proj = operations.Conv2d(d_model, out_dim, kernel_size=1, device=device, dtype=dtype)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=out_dim, normalize=True)

    def forward(self, image_features, mask_for_mem, skip_mask_sigmoid=False):
        """Encode image features + mask into compact memory.

        Args:
            image_features: [B, C, H, W] pixel features from backbone
            mask_for_mem: [B, 1, H_mask, W_mask] mask (pre-processed or raw logits)
            skip_mask_sigmoid: if True, mask_for_mem is already processed (no sigmoid needed)
        Returns:
            dict with 'vision_features' [B, out_dim, H, W] and 'vision_pos_enc' [list of pos tensors]
        """
        if not skip_mask_sigmoid:
            mask_for_mem = mask_for_mem.sigmoid()
        mask_features = self.mask_downsampler(mask_for_mem)
        if mask_features.shape[-2:] != image_features.shape[-2:]:
            mask_features = F.interpolate(mask_features, size=image_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.pix_feat_proj(image_features) + mask_features
        features = self.fuser(features)
        features = self.out_proj(features)
        pos = self.position_encoding(features).to(dtype=features.dtype)
        return {"vision_features": features, "vision_pos_enc": [pos]}


# --- SAM3.1 Multiplex components ---

class DecoupledMemoryAttnLayer(nn.Module):
    """Decoupled cross-attention layer for SAM3.1: fuses image and memory projections."""

    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        # Self-attention projections (flat, not nested)
        self.self_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.self_attn_out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # Cross-attention projections
        self.cross_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_v_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cross_attn_out_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # Image cross-attention (q/k only, fused with cross_attn)
        self.image_cross_attn_q_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.image_cross_attn_k_proj = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        # FFN
        self.linear1 = operations.Linear(d_model, dim_ff, device=device, dtype=dtype)
        self.linear2 = operations.Linear(dim_ff, d_model, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, image, tgt, memory_image, memory, query_pos=None, memory_image_pos=None,
                rope=None, num_k_exclude_rope=0):
        # Self-attention with RoPE
        tgt2 = self.norm1(tgt)
        q = self.self_attn_q_proj(tgt2)
        k = self.self_attn_k_proj(tgt2)
        v = self.self_attn_v_proj(tgt2)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, 0)
        tgt = tgt + self.self_attn_out_proj(optimized_attention(q, k, v, self.num_heads))

        # Decoupled cross-attention: fuse image and memory projections
        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if memory_image_pos is not None:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, num_k_exclude_rope)
        tgt = tgt + self.cross_attn_out_proj(optimized_attention(q, k, v, self.num_heads))

        # FFN
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.linear2(F.relu(self.linear1(tgt2)))
        return image, tgt


class DecoupledMemoryEncoder(nn.Module):
    """Memory attention encoder for SAM3.1 with decoupled cross-attention."""

    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoupledMemoryAttnLayer(d_model, num_heads, dim_ff, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)
        ])
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, src, memory, memory_pos=None, src_pos=None, num_k_exclude_rope=0):
        """
        Args:
            src: [B, HW, C] current frame features
            memory: [B, N_mem, C] concatenated spatial memory + obj pointers
            memory_pos: [B, N_mem, C] position encoding for memory
            src_pos: [B, HW, C] position encoding for current frame
            num_k_exclude_rope: number of trailing memory tokens to skip RoPE (obj pointers)
        """
        image = src  # constant residual
        output = src
        if src_pos is not None:
            output = output + 0.1 * src_pos

        B, N, C = src.shape
        hw = int(math.sqrt(N))
        rope = rope_2d(hw, hw, C).to(device=src.device)

        # Split memory into spatial (memory_image) and all (memory with pos)
        num_spatial = memory.shape[1] - num_k_exclude_rope
        memory_image = memory[:, :num_spatial]
        memory_image_pos = memory_pos[:, :num_spatial] if memory_pos is not None else None
        # Pad memory_image to same length as memory (zeros for obj pointer tokens)
        if num_k_exclude_rope > 0:
            pad = torch.zeros(B, num_k_exclude_rope, C, device=memory.device, dtype=memory.dtype)
            memory_image = torch.cat([memory_image, pad], dim=1)
            if memory_image_pos is not None:
                # Use temporal pos from memory_pos for pointer tokens
                ptr_pos = memory_pos[:, num_spatial:]
                memory_image_pos = torch.cat([memory_image_pos, ptr_pos], dim=1)

        # Add position encoding to memory (memory = raw features + pos)
        if memory_pos is not None:
            memory = memory + memory_pos

        for layer in self.layers:
            image, output = layer(image, output, memory_image, memory,
                                  query_pos=src_pos, memory_image_pos=memory_image_pos,
                                  rope=rope, num_k_exclude_rope=num_k_exclude_rope)

        return self.norm(output + image)


class DecoupledMemoryTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=1, dim_ff=2048, num_layers=4, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = DecoupledMemoryEncoder(d_model, num_heads, dim_ff, num_layers,
                                              device=device, dtype=dtype, operations=operations)


class MultiplexMaskDownSampler(nn.Module):
    """Wider mask downsampler for SAM3.1 multiplex (32 input channels, 4x wider intermediate)."""

    def __init__(self, in_chans=32, out_dim=256, device=None, dtype=None, operations=None):
        super().__init__()
        LN2d = LayerNorm2d_op(operations)
        self.encoder = nn.Sequential(
            operations.Conv2d(in_chans, 16, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(16, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(64, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(256, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            LN2d(1024, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(1024, out_dim, kernel_size=1, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.encoder(x)


class MultiplexMemoryBackbone(nn.Module):
    """Memory backbone for SAM3.1 multiplex: mem_dim=256 (no out_proj compression)."""

    def __init__(self, d_model=256, num_multiplex=16, device=None, dtype=None, operations=None):
        super().__init__()
        in_chans = num_multiplex * 2  # 32 channels for 16 multiplex masks
        self.mask_downsampler = MultiplexMaskDownSampler(in_chans, d_model, device=device, dtype=dtype, operations=operations)
        self.pix_feat_proj = operations.Conv2d(d_model, d_model, kernel_size=1, device=device, dtype=dtype)
        self.fuser = Fuser(d_model, num_layers=2, device=device, dtype=dtype, operations=operations)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model, normalize=True)

    def forward(self, image_features, mask_for_mem, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            mask_for_mem = mask_for_mem.sigmoid()
        mask_features = self.mask_downsampler(mask_for_mem)
        if mask_features.shape[-2:] != image_features.shape[-2:]:
            mask_features = F.interpolate(mask_features, size=image_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.pix_feat_proj(image_features) + mask_features
        features = self.fuser(features)
        pos = self.position_encoding(features).to(dtype=features.dtype)
        return {"vision_features": features, "vision_pos_enc": [pos]}


class MultiplexMaskDecoder(nn.Module):
    """SAM mask decoder for SAM3.1 multiplex: predicts masks for num_multiplex objects simultaneously.

    Uses multimask_outputs_only=True: num_mask_output_per_object = num_multimask_outputs (no +1).
    Hypernetwork MLPs are shared across multiplex objects.
    Token order: [obj_score_token(M), iou_token(M), mask_tokens(M*T)].
    """

    def __init__(self, d_model=256, num_multiplex=16, num_multimask_outputs=3, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_multiplex = num_multiplex
        self.num_mask_output_per_object = num_multimask_outputs  # 3 (multimask_outputs_only)
        total_mask_tokens = num_multiplex * self.num_mask_output_per_object  # 48

        self.transformer = TwoWayTransformer(d_model, num_heads=8, depth=2, mlp_dim=2048, device=device, dtype=dtype, operations=operations)

        self.obj_score_token = operations.Embedding(num_multiplex, d_model, device=device, dtype=dtype)
        self.iou_token = operations.Embedding(num_multiplex, d_model, device=device, dtype=dtype)
        self.mask_tokens = operations.Embedding(total_mask_tokens, d_model, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(d_model // 4, device=device, dtype=dtype), nn.GELU(),
            operations.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2, device=device, dtype=dtype), nn.GELU(),
        )
        self.conv_s0 = operations.Conv2d(d_model, d_model // 8, kernel_size=1, device=device, dtype=dtype)
        self.conv_s1 = operations.Conv2d(d_model, d_model // 4, kernel_size=1, device=device, dtype=dtype)

        # Shared across all multiplex objects (one per mask output)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3, device=device, dtype=dtype, operations=operations)
            for _ in range(self.num_mask_output_per_object)
        ])
        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_output_per_object, 3, device=device, dtype=dtype, operations=operations)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3, device=device, dtype=dtype, operations=operations)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                high_res_features=None, multimask_output=False, return_all=False):
        B = sparse_prompt_embeddings.shape[0]
        M = self.num_multiplex
        T = self.num_mask_output_per_object

        # Token order: [obj_score(M), iou(M), mask(M*T)]
        tokens = torch.cat([
            cast_to_input(self.obj_score_token.weight, image_embeddings),
            cast_to_input(self.iou_token.weight, image_embeddings),
            cast_to_input(self.mask_tokens.weight, image_embeddings),
        ], dim=0)
        tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        hs, src_out = self.transformer(src.flatten(2).permute(0, 2, 1), pos_src.flatten(2).permute(0, 2, 1), tokens)

        # Parse output tokens
        obj_score_token_out = hs[:, :M]
        iou_token_out = hs[:, M:2 * M]
        mask_tokens_out = hs[:, 2 * M:2 * M + M * T]

        # Upscale features
        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        upscaled = act1(ln1(dc1(src_out)))
        if high_res_features is not None:
            upscaled = upscaled + F.interpolate(self.conv_s1(high_res_features[1]), size=upscaled.shape[-2:], mode="bilinear", align_corners=False)
            upscaled = act2(dc2(upscaled))
            upscaled = upscaled + F.interpolate(self.conv_s0(high_res_features[0]), size=upscaled.shape[-2:], mode="bilinear", align_corners=False)
        else:
            upscaled = act2(dc2(upscaled))

        # Reshape mask tokens to [B, M, T, C] and apply shared hypernetwork MLPs per mask output index
        mask_tokens_2d = mask_tokens_out.view(B, M, T, -1)
        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_2d[:, :, i, :])  # [B, M, C//8]
            for i in range(T)
        ], dim=2)  # [B, M, T, C//8]

        # Generate masks: [B, M*T, H*W] -> [B, M, T, H, W]
        masks = torch.bmm(hyper_in.flatten(1, 2), upscaled.flatten(2)).view(b, M, T, upscaled.shape[2], upscaled.shape[3])

        # IoU and object scores
        iou_pred = self.iou_prediction_head(iou_token_out).view(b, M, T)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)  # [B, M, 1]

        # multimask_outputs_only: always output all T masks (no singlemask token)
        sam_tokens_out = mask_tokens_2d[:, :, 0:1]  # [B, M, 1, C]

        if return_all:
            return masks, iou_pred, sam_tokens_out, object_score_logits
        return masks, iou_pred


class SAM3Tracker(nn.Module):
    def __init__(self, d_model=256, mem_dim=64, num_maskmem=7, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()

        # Memory attention transformer
        self.transformer = MemoryTransformer(d_model, num_heads=1, kv_dim=mem_dim, dim_ff=2048, num_layers=4,
                                             device=device, dtype=dtype, operations=operations)
        # SAM components
        self.sam_mask_decoder = SAMMaskDecoder(d_model, device=device, dtype=dtype, operations=operations)
        self.sam_prompt_encoder = SAMPromptEncoder(d_model, device=device, dtype=dtype, operations=operations)

        # Memory backbone
        self.maskmem_backbone = MemoryBackbone(d_model, mem_dim, device=device, dtype=dtype, operations=operations)

        # Standalone parameters
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim, device=device, dtype=dtype))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))  # checkpoint key, unused in forward
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, mem_dim, device=device, dtype=dtype))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, d_model, device=device, dtype=dtype))

        # Object pointer projection
        self.obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)
        self.obj_ptr_tpos_proj = operations.Linear(d_model, mem_dim, device=device, dtype=dtype)

        # Mask downsample: Conv2d stride 4 to reduce GT mask to SAM logit scale
        self.mask_downsample = operations.Conv2d(1, 1, kernel_size=4, stride=4, device=device, dtype=dtype)

        # Config
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.num_maskmem = num_maskmem
        self.image_size = 1008
        self.backbone_stride = 14
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0

    # Tracking methods
    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None):
        """Temporal position encoding for object pointers."""
        pos_enc = torch.tensor(rel_pos_list, dtype=torch.float32, device=device) / max((max_abs_pos or 2) - 1, 1)
        pos_enc = get_1d_sine_pe(pos_enc, dim=self.d_model).to(self.obj_ptr_tpos_proj.weight.dtype)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        """Forward SAM prompt encoder + mask decoder. Returns all outputs for tracking."""
        B = backbone_features.shape[0]
        device = backbone_features.device

        # Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # Handle mask prompts
        if mask_inputs is not None:
            prompt_size = (self.sam_prompt_encoder.image_embedding_size[0] * 4,
                           self.sam_prompt_encoder.image_embedding_size[1] * 4)
            if mask_inputs.shape[-2:] != prompt_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs, size=prompt_size,
                    mode="bilinear", align_corners=False, antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse, dense = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            masks=sam_mask_prompt,
        )
        feat_dtype = backbone_features.dtype # pe computations use fp32 internally
        sparse, dense = sparse.to(feat_dtype), dense.to(feat_dtype)
        image_pe = self.sam_prompt_encoder.get_dense_pe().to(feat_dtype)

        low_res_multimasks, ious, sam_output_tokens, object_score_logits = \
            self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                return_all=True,
            )

        is_obj_appearing = object_score_logits > 0

        # Zero out masks when object not appearing
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_multimasks.dtype))
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        alpha = is_obj_appearing.to(obj_ptr.dtype)
        obj_ptr = torch.lerp(cast_to_input(self.no_obj_ptr, obj_ptr), obj_ptr, alpha)

        return low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """Use binary mask input directly as output (for conditioning frames)."""
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.to(backbone_features.dtype)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks, size=(self.image_size // self.backbone_stride * 4,) * 2, mode="bilinear", align_corners=False, antialias=True)
        # Get object pointer from SAM decoder using mask as input
        _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1) > 0.0, dim=1)[..., None]
        alpha = is_obj_appearing.to(obj_ptr.dtype)
        object_score_logits = out_scale * alpha + out_bias
        obj_ptr = torch.lerp(cast_to_input(self.no_obj_ptr, obj_ptr), obj_ptr, alpha)

        return low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames):
        """Fuse current frame features with memory from previous frames."""
        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(0, 2, 1).view(B, C, H, W)

        if is_init_cond_frame:
            # First conditioning frame: no memory yet, add no_mem_embed
            pix_feat = current_vision_feats[-1] + cast_to_input(self.no_mem_embed, current_vision_feats[-1])
            return pix_feat.view(B, H, W, C).permute(0, 3, 1, 2)

        # Collect spatial memory and position encodings
        to_cat_memory = []  # [B, HW, mem_dim]
        to_cat_memory_pos = []  # [B, HW, mem_dim]

        # Add conditioning frames (t_pos=0 for temporal encoding)
        cond_outputs = output_dict["cond_frame_outputs"]
        for t, out in cond_outputs.items():
            feats = out["maskmem_features"].to(device)  # [B, mem_dim, H, W]
            to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))  # [B, HW, mem_dim]

            maskmem_enc = out["maskmem_pos_enc"][-1].to(device)  # [B, mem_dim, H, W]
            maskmem_enc = maskmem_enc.flatten(2).permute(0, 2, 1)
            # Add temporal encoding for conditioning frame (t=0)
            maskmem_enc = maskmem_enc + cast_to_input(self.maskmem_tpos_enc[self.num_maskmem - 1], maskmem_enc)
            to_cat_memory_pos.append(maskmem_enc)

        # Add recent non-conditioning frames
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                prev_frame_idx = frame_idx - 1
            else:
                prev_frame_idx = frame_idx - t_rel

            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                continue

            feats = out["maskmem_features"]
            if feats is None:
                continue
            feats = feats.to(device)
            to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))

            maskmem_enc = out["maskmem_pos_enc"][-1].to(device)
            maskmem_enc = maskmem_enc.flatten(2).permute(0, 2, 1)
            maskmem_enc = maskmem_enc + cast_to_input(self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1], maskmem_enc)
            to_cat_memory_pos.append(maskmem_enc)

        # Collect object pointers from past frames
        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = []

        # Object pointers from conditioning frames
        for t, out in cond_outputs.items():
            if t <= frame_idx:
                pos_and_ptrs.append(((frame_idx - t), out["obj_ptr"]))

        # Object pointers from recent non-conditioning frames
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = output_dict["non_cond_frame_outputs"].get(t, None)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        num_obj_ptr_tokens = 0
        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=1)  # [B, N, C=256]

            # Temporal position encoding for pointers
            obj_pos = self._get_tpos_enc(
                list(pos_list), max_abs_pos=max_obj_ptrs, device=device
            )  # [N, mem_dim=64]
            obj_pos = obj_pos.unsqueeze(0).expand(B, -1, -1)  # [B, N, 64]

            # Split each 256-dim pointer into 4 x 64-dim tokens
            if self.mem_dim < C:
                N = obj_ptrs.shape[1]
                obj_ptrs = obj_ptrs.view(B, N, C // self.mem_dim, self.mem_dim)  # [B, N, 4, 64]
                obj_ptrs = obj_ptrs.reshape(B, N * (C // self.mem_dim), self.mem_dim)  # [B, N*4, 64]
                obj_pos = obj_pos.unsqueeze(2).expand(-1, -1, C // self.mem_dim, -1)
                obj_pos = obj_pos.reshape(B, N * (C // self.mem_dim), self.mem_dim)  # [B, N*4, 64]

            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[1]

        if len(to_cat_memory) == 0:
            # No memory available yet, add no_mem_embed
            pix_feat = current_vision_feats[-1] + cast_to_input(self.no_mem_embed, current_vision_feats[-1])
            return pix_feat.view(B, H, W, C).permute(0, 3, 1, 2)

        # Concatenate all memory and position encodings [B, total_mem, mem_dim=64]
        memory = torch.cat(to_cat_memory, dim=1)
        memory_pos = torch.cat(to_cat_memory_pos, dim=1)

        # Run memory attention encoder
        pix_feat = current_vision_feats[-1]  # [B, HW, C]
        src_pos = current_vision_pos_embeds[-1]  # [B, HW, C]

        pix_feat_with_mem = self.transformer.encoder(
            src=pix_feat, # current frame features [B, HW, C=256]
            memory=memory, # past memory [B, total_mem, mem_dim=64]
            src_pos=src_pos, # spatial pos for current frame [B, HW, C=256]
            memory_pos=memory_pos, # spatial + temporal pos for memory [B, total_mem, mem_dim=64]
            num_k_exclude_rope=num_obj_ptr_tokens,
        )
        return pix_feat_with_mem.view(B, H, W, C).permute(0, 3, 1, 2)

    def _encode_new_memory(self, pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False):
        """Encode predicted mask into memory features."""
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        mask_for_mem.mul_(self.sigmoid_scale_for_mem_enc).add_(self.sigmoid_bias_for_mem_enc)

        maskmem_out = self.maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # Add no_obj_embed for occluded objects
        alpha = (object_score_logits > 0).to(maskmem_features.dtype)[..., None, None]
        no_obj = cast_to_input(self.no_obj_embed_spatial, maskmem_features)[..., None, None].expand_as(maskmem_features)
        return maskmem_features + (1 - alpha) * no_obj, maskmem_pos_enc

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, mask_inputs, output_dict,
                   num_frames, point_inputs=None):
        """Track one frame: fuse with memory, predict mask, encode memory."""
        current_out = {}

        # High-res features for SAM head [stride-8, stride-4]
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.view(x.shape[0], feat_sizes[i][0], feat_sizes[i][1], -1).permute(0, 3, 1, 2)
                for i, x in enumerate(current_vision_feats[:-1])
            ]
        else:
            high_res_features = None

        # Top-level feature for memory
        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]

        if mask_inputs is not None:
            # Conditioning frame: use mask directly
            pix_feat = current_vision_feats[-1].view(B, H, W, C).permute(0, 3, 1, 2)
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            # Track frame: fuse with memory, then SAM decoder
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                high_res_features=high_res_features,
                multimask_output=False,
            )

        (low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs

        # Clean low-res masks: remove sprinkles and fill holes
        # Remove corner artifacts (up to ~200px at 288x288) while preserving main object
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, max_area=200)
        # Re-derive high-res from cleaned low-res
        high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        # Encode memory
        if self.num_maskmem > 0:
            pix_feat = current_vision_feats[-1].view(B, H, W, C).permute(0, 3, 1, 2)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=pix_feat,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _precompute_backbone(self, backbone_fn, images):
        """Pre-compute and cache backbone features for all frames."""
        cached = []
        for frame_idx in range(images.shape[0]):
            sam2_features, sam2_positions = backbone_fn(images[frame_idx:frame_idx + 1])
            backbone_fpn = sam2_features[:-1]
            vision_pos_enc = sam2_positions[:-1]
            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in backbone_fpn]
            vision_feats = [x.flatten(2).permute(0, 2, 1) for x in backbone_fpn]
            vision_pos = [x.flatten(2).permute(0, 2, 1) for x in vision_pos_enc]
            cached.append((vision_feats, vision_pos, feat_sizes))
        return cached

    def _track_single_object(self, cached_features, images, initial_mask, pbar=None):
        """Track one object using pre-computed backbone features."""
        N = len(cached_features)
        device, dt = images.device, images.dtype
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        all_masks = []

        for frame_idx in range(N):
            vision_feats, vision_pos, feat_sizes = cached_features[frame_idx]
            mask_input = None
            if frame_idx == 0:
                mask_input = F.interpolate(initial_mask.to(device=device, dtype=dt),
                    size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                mask_input = (mask_input > 0.5).to(dt)

            current_out = self.track_step(
                frame_idx=frame_idx, is_init_cond_frame=(frame_idx == 0),
                current_vision_feats=vision_feats, current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes, mask_inputs=mask_input, output_dict=output_dict, num_frames=N)

            if frame_idx == 0:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            all_masks.append(current_out["pred_masks_high_res"])
            if pbar is not None:
                pbar.update(1)

        return torch.cat(all_masks, dim=0)  # [N, 1, H, W]

    def track_video(self, backbone_fn, images, initial_masks, pbar=None):
        """Track one or more objects across video frames.

        Args:
            backbone_fn: callable that returns (sam2_features, sam2_positions) for a frame
            images: [N, 3, 1008, 1008] video frames
            initial_masks: [N_obj, 1, H, W] binary masks for first frame (one per object)
            pbar: optional progress bar

        Returns:
            [N, N_obj, image_size, image_size] predicted mask logits per frame per object
        """
        cached = self._precompute_backbone(backbone_fn, images)

        N_obj = initial_masks.shape[0]
        per_object = []
        for obj_idx in range(N_obj):
            obj_masks = self._track_single_object(
                cached, images, initial_masks[obj_idx:obj_idx + 1], pbar=pbar)
            per_object.append(obj_masks)

        return torch.cat(per_object, dim=1)  # [N, N_obj, H, W]


class SAM31Tracker(nn.Module):
    """SAM3.1 multiplex tracker: decoupled memory attention, dual decoder, 16-object multiplex."""

    def __init__(self, d_model=256, mem_dim=256, num_maskmem=7, num_multiplex=16, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.num_maskmem = num_maskmem
        self.num_multiplex = num_multiplex
        self.image_size = 1008
        self.backbone_stride = 14
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0

        # Memory attention (decoupled cross-attention)
        self.transformer = DecoupledMemoryTransformer(d_model, num_heads=1, dim_ff=2048, num_layers=4,
                                                      device=device, dtype=dtype, operations=operations)

        # Propagation decoder (multiplex: 16 objects, multimask_outputs_only)
        self.sam_mask_decoder = MultiplexMaskDecoder(d_model, num_multiplex, num_multimask_outputs=3,
                                                     device=device, dtype=dtype, operations=operations)
        # Interactive decoder (single object, same as SAM3)
        self.interactive_sam_mask_decoder = SAMMaskDecoder(d_model, num_multimask_outputs=3,
                                                           device=device, dtype=dtype, operations=operations)
        self.interactive_sam_prompt_encoder = SAMPromptEncoder(d_model, device=device, dtype=dtype, operations=operations)

        # Memory backbone (mem_dim=256, no out_proj compression)
        self.maskmem_backbone = MultiplexMemoryBackbone(d_model, num_multiplex, device=device, dtype=dtype, operations=operations)

        # Standalone parameters
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, mem_dim, device=device, dtype=dtype))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(num_multiplex, mem_dim, device=device, dtype=dtype))
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))

        # Object pointer projection
        self.obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)
        self.obj_ptr_tpos_proj = operations.Linear(d_model, mem_dim, device=device, dtype=dtype)
        self.no_obj_ptr_linear = operations.Linear(d_model, d_model, device=device, dtype=dtype)
        self.interactive_obj_ptr_proj = MLP(d_model, d_model, d_model, 3, device=device, dtype=dtype, operations=operations)

        # Interactive mask downsample
        self.interactive_mask_downsample = operations.Conv2d(1, 1, kernel_size=4, stride=4, device=device, dtype=dtype)

        # Multiplex validity embeddings
        self.output_valid_embed = nn.Parameter(torch.zeros(num_multiplex, d_model, device=device, dtype=dtype))
        self.output_invalid_embed = nn.Parameter(torch.zeros(num_multiplex, d_model, device=device, dtype=dtype))

        # Position encoding for image (used by multiplex decoder)
        self.image_pe_layer = PositionEmbeddingRandom(d_model // 2)

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None):
        pos_enc = torch.tensor(rel_pos_list, dtype=torch.float32, device=device) / max((max_abs_pos or 2) - 1, 1)
        pos_enc = get_1d_sine_pe(pos_enc, dim=self.d_model).to(self.obj_ptr_tpos_proj.weight.dtype)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None,
                           high_res_features=None, multimask_output=False):
        """Forward interactive SAM decoder (single object, for interactive segmentation)."""
        B = backbone_features.shape[0]
        device = backbone_features.device

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            prompt_size = (self.interactive_sam_prompt_encoder.image_embedding_size[0] * 4,
                           self.interactive_sam_prompt_encoder.image_embedding_size[1] * 4)
            if mask_inputs.shape[-2:] != prompt_size:
                sam_mask_prompt = F.interpolate(mask_inputs, size=prompt_size, mode="bilinear", align_corners=False, antialias=True)
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse, dense = self.interactive_sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels), masks=sam_mask_prompt,
        )
        feat_dtype = backbone_features.dtype
        sparse, dense = sparse.to(feat_dtype), dense.to(feat_dtype)
        image_pe = self.interactive_sam_prompt_encoder.get_dense_pe().to(feat_dtype)

        low_res_multimasks, ious, sam_output_tokens, object_score_logits = \
            self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features, image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                high_res_features=high_res_features, multimask_output=multimask_output, return_all=True,
            )

        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks,
                                          torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_multimasks.dtype))
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.interactive_obj_ptr_proj(sam_output_token)
        no_obj = self.no_obj_ptr_linear(sam_output_token)
        alpha = is_obj_appearing.to(obj_ptr.dtype)
        obj_ptr = torch.lerp(no_obj, obj_ptr, alpha)

        return low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.to(backbone_features.dtype)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks,
            size=(self.image_size // self.backbone_stride * 4,) * 2,
            mode="bilinear", align_corners=False, antialias=True)
        _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1) > 0.0, dim=1)[..., None]
        alpha = is_obj_appearing.to(obj_ptr.dtype)
        object_score_logits = out_scale * alpha + out_bias
        return low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats,
                                              current_vision_pos_embeds, feat_sizes, output_dict, num_frames):
        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(0, 2, 1).view(B, C, H, W)

        if is_init_cond_frame:
            pix_feat = current_vision_feats[-1] + cast_to_input(self.interactivity_no_mem_embed, current_vision_feats[-1])
            return pix_feat.view(B, H, W, C).permute(0, 3, 1, 2)

        to_cat_memory = []
        to_cat_memory_pos = []

        cond_outputs = output_dict["cond_frame_outputs"]
        for t, out in cond_outputs.items():
            feats = out["maskmem_features"].to(device)
            to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))
            maskmem_enc = out["maskmem_pos_enc"][-1].to(device).flatten(2).permute(0, 2, 1)
            maskmem_enc = maskmem_enc + cast_to_input(self.maskmem_tpos_enc[self.num_maskmem - 1], maskmem_enc)
            to_cat_memory_pos.append(maskmem_enc)

        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            prev_frame_idx = frame_idx - 1 if t_rel == 1 else frame_idx - t_rel
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None or out["maskmem_features"] is None:
                continue
            feats = out["maskmem_features"].to(device)
            to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))
            maskmem_enc = out["maskmem_pos_enc"][-1].to(device).flatten(2).permute(0, 2, 1)
            maskmem_enc = maskmem_enc + cast_to_input(self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1], maskmem_enc)
            to_cat_memory_pos.append(maskmem_enc)

        # Object pointers (no sub-token splitting since mem_dim=d_model=256)
        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = []
        for t, out in cond_outputs.items():
            if t <= frame_idx:
                pos_and_ptrs.append(((frame_idx - t), out["obj_ptr"]))
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = output_dict["non_cond_frame_outputs"].get(t, None)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        num_obj_ptr_tokens = 0
        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=1)  # [B, N, C]
            obj_pos = self._get_tpos_enc(list(pos_list), max_abs_pos=max_obj_ptrs, device=device)
            obj_pos = obj_pos.unsqueeze(0).expand(B, -1, -1)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[1]

        if len(to_cat_memory) == 0:
            pix_feat = current_vision_feats[-1] + cast_to_input(self.interactivity_no_mem_embed, current_vision_feats[-1])
            return pix_feat.view(B, H, W, C).permute(0, 3, 1, 2)

        memory = torch.cat(to_cat_memory, dim=1)
        memory_pos = torch.cat(to_cat_memory_pos, dim=1)

        pix_feat_with_mem = self.transformer.encoder(
            src=current_vision_feats[-1],
            memory=memory,
            memory_pos=memory_pos,
            src_pos=current_vision_pos_embeds[-1],
            num_k_exclude_rope=num_obj_ptr_tokens,
        )
        return pix_feat_with_mem.view(B, H, W, C).permute(0, 3, 1, 2)

    def _encode_new_memory(self, pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False):
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem.mul_(self.sigmoid_scale_for_mem_enc).add_(self.sigmoid_bias_for_mem_enc)

        # Pad single-channel mask to 32 channels (multiplex backbone expects num_multiplex*2 channels)
        if mask_for_mem.shape[1] < self.num_multiplex * 2:
            pad = torch.zeros(mask_for_mem.shape[0], self.num_multiplex * 2 - mask_for_mem.shape[1],
                              *mask_for_mem.shape[2:], device=mask_for_mem.device, dtype=mask_for_mem.dtype)
            mask_for_mem = torch.cat([mask_for_mem, pad], dim=1)

        maskmem_out = self.maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        alpha = (object_score_logits > 0).to(maskmem_features.dtype)[..., None, None]
        no_obj = cast_to_input(self.no_obj_embed_spatial[0], maskmem_features)[..., None, None].expand_as(maskmem_features)
        return maskmem_features + (1 - alpha) * no_obj, maskmem_pos_enc

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                   feat_sizes, mask_inputs, output_dict, num_frames, point_inputs=None):
        current_out = {}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.view(x.shape[0], feat_sizes[i][0], feat_sizes[i][1], -1).permute(0, 3, 1, 2)
                for i, x in enumerate(current_vision_feats[:-1])
            ]
        else:
            high_res_features = None

        B = current_vision_feats[-1].shape[0]
        C = self.d_model
        H, W = feat_sizes[-1]

        if mask_inputs is not None:
            pix_feat = current_vision_feats[-1].view(B, H, W, C).permute(0, 3, 1, 2)
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes, output_dict=output_dict, num_frames=num_frames,
            )
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem, point_inputs=point_inputs,
                high_res_features=high_res_features, multimask_output=False,
            )

        (low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, max_area=200)
        high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        if self.num_maskmem > 0:
            pix_feat = current_vision_feats[-1].view(B, H, W, C).permute(0, 3, 1, 2)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=pix_feat, pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _precompute_backbone(self, backbone_fn, images):
        """Pre-compute and cache backbone features for all frames."""
        cached = []
        for frame_idx in range(images.shape[0]):
            tracker_features, tracker_positions = backbone_fn(images[frame_idx:frame_idx + 1])
            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in tracker_features]
            vision_feats = [x.flatten(2).permute(0, 2, 1) for x in tracker_features]
            vision_pos = [x.flatten(2).permute(0, 2, 1) for x in tracker_positions]
            cached.append((vision_feats, vision_pos, feat_sizes))
        return cached

    def _track_single_object(self, cached_features, images, initial_mask, pbar=None):
        """Track one object using pre-computed backbone features."""
        N = len(cached_features)
        device, dt = images.device, images.dtype
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        all_masks = []

        for frame_idx in range(N):
            vision_feats, vision_pos, feat_sizes = cached_features[frame_idx]
            mask_input = None
            if frame_idx == 0:
                mask_input = F.interpolate(initial_mask.to(device=device, dtype=dt),
                    size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                mask_input = (mask_input > 0.5).to(dt)

            current_out = self.track_step(
                frame_idx=frame_idx, is_init_cond_frame=(frame_idx == 0),
                current_vision_feats=vision_feats, current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes, mask_inputs=mask_input, output_dict=output_dict, num_frames=N)

            if frame_idx == 0:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            all_masks.append(current_out["pred_masks_high_res"])
            if pbar is not None:
                pbar.update(1)

        return torch.cat(all_masks, dim=0)  # [N, 1, H, W]

    def track_video(self, backbone_fn, images, initial_masks, pbar=None):
        """Track one or more objects across video frames.

        Args:
            backbone_fn: callable that returns (tracker_features, tracker_positions) for a frame
            images: [N, 3, 1008, 1008] video frames
            initial_masks: [N_obj, 1, H, W] binary masks for first frame (one per object)
            pbar: optional progress bar

        Returns:
            [N, N_obj, image_size, image_size] predicted mask logits per frame per object
        """
        cached = self._precompute_backbone(backbone_fn, images)

        N_obj = initial_masks.shape[0]
        per_object = []
        for obj_idx in range(N_obj):
            obj_masks = self._track_single_object(
                cached, images, initial_masks[obj_idx:obj_idx + 1], pbar=pbar)
            per_object.append(obj_masks)

        return torch.cat(per_object, dim=1)  # [N, N_obj, H, W]
