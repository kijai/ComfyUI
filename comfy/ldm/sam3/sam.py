# SAM components shared by detector and tracker

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.cascade.common import LayerNorm2d_op
from comfy.ops import cast_to_input


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False, device=None, dtype=None, operations=None):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([operations.Linear(dims[i], dims[i + 1], device=device, dtype=dtype) for i in range(num_layers)])
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class SAMAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, kv_in_dim=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = embedding_dim // downsample_rate
        kv_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.q_proj = operations.Linear(embedding_dim, internal_dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(kv_dim, internal_dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(internal_dim, embedding_dim, device=device, dtype=dtype)

    def forward(self, q, k, v):
        return self.out_proj(optimized_attention(self.q_proj(q), self.k_proj(k), self.v_proj(v), self.num_heads))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, attention_downsample_rate=2, skip_first_layer_pe=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = SAMAttention(embedding_dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.cross_attn_token_to_image = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.cross_attn_image_to_token = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.mlp = nn.Sequential(operations.Linear(embedding_dim, mlp_dim, device=device, dtype=dtype), nn.ReLU(), operations.Linear(mlp_dim, embedding_dim, device=device, dtype=dtype))
        self.norm1 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)
        self.norm4 = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.norm1(queries + self.self_attn(queries, queries, queries))
        else:
            q = queries + query_pe
            queries = self.norm1(queries + self.self_attn(q, q, queries))
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048, attention_downsample_rate=2, device=None, dtype=None, operations=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, attention_downsample_rate,
                                 skip_first_layer_pe=(i == 0), device=device, dtype=dtype, operations=operations)
            for i in range(depth)
        ])
        self.final_attn_token_to_image = SAMAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate, device=device, dtype=dtype, operations=operations)
        self.norm_final = operations.LayerNorm(embedding_dim, device=device, dtype=dtype)

    def forward(self, image_embedding, image_pe, point_embedding):
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, queries, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """Fourier feature positional encoding with random gaussian projection."""
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", (scale or 1.0) * torch.randn(2, num_pos_feats))

    def _encode(self, normalized_coords):
        """Map normalized [0,1] coordinates to fourier features via random projection. Computes in fp32."""
        orig_dtype = normalized_coords.dtype
        proj_matrix = self.positional_encoding_gaussian_matrix.to(device=normalized_coords.device, dtype=torch.float32)
        projected = (2 * normalized_coords.float() - 1) @ proj_matrix
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(orig_dtype)

    def forward(self, size):
        h, w = size
        dev = self.positional_encoding_gaussian_matrix.device
        ones = torch.ones((h, w), device=dev, dtype=torch.float32)
        norm_xy = torch.stack([(ones.cumsum(1) - 0.5) / w, (ones.cumsum(0) - 0.5) / h], dim=-1)
        return self._encode(norm_xy).permute(2, 0, 1).unsqueeze(0)

    def forward_with_coords(self, pixel_coords, image_size):
        norm = pixel_coords.clone()
        norm[:, :, 0] /= image_size[1]
        norm[:, :, 1] /= image_size[0]
        return self._encode(norm)


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=(72, 72), input_image_size=(1008, 1008), mask_in_chans=16, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList([operations.Embedding(1, embed_dim, device=device, dtype=dtype) for _ in range(4)])
        self.not_a_point_embed = operations.Embedding(1, embed_dim, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.mask_downscaling = nn.Sequential(
            operations.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans // 4, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(mask_in_chans, device=device, dtype=dtype), nn.GELU(),
            operations.Conv2d(mask_in_chans, embed_dim, kernel_size=1, device=device, dtype=dtype),
        )
        self.no_mask_embed = operations.Embedding(1, embed_dim, device=device, dtype=dtype)

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
            if boxes is None:
                pe = torch.cat([pe, torch.zeros(B, 1, self.embed_dim, device=ref.device, dtype=ref.dtype)], dim=1)
                labels = torch.cat([labels, -torch.ones(B, 1, device=labels.device, dtype=labels.dtype)], dim=1)
            pe[labels == 0] += cast_to_input(self.point_embeddings[0].weight, ref)
            pe[labels == 1] += cast_to_input(self.point_embeddings[1].weight, ref)
            invalid = labels == -1
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
            dense = cast_to_input(self.no_mask_embed.weight, ref).reshape(1, -1, 1, 1).expand(B, -1, *self.image_embedding_size)

        return sparse, dense


class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256,
                 use_high_res_features=False, pred_obj_scores=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, num_heads=8, mlp_dim=2048, device=device, dtype=dtype, operations=operations)
        self.iou_token = operations.Embedding(1, transformer_dim, device=device, dtype=dtype)
        self.mask_tokens = operations.Embedding(self.num_mask_tokens, transformer_dim, device=device, dtype=dtype)
        self.pred_obj_scores = pred_obj_scores
        if pred_obj_scores:
            self.obj_score_token = operations.Embedding(1, transformer_dim, device=device, dtype=dtype)

        LN2d = LayerNorm2d_op(operations)
        self.output_upscaling = nn.Sequential(
            operations.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2, device=device, dtype=dtype),
            LN2d(transformer_dim // 4, device=device, dtype=dtype), nn.GELU(),
            operations.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2, device=device, dtype=dtype), nn.GELU(),
        )

        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = operations.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, device=device, dtype=dtype)
            self.conv_s1 = operations.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, device=device, dtype=dtype)

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3, device=device, dtype=dtype, operations=operations)
            for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, device=device, dtype=dtype, operations=operations)
        if pred_obj_scores:
            self.pred_obj_score_head = operations.Linear(transformer_dim, 1, device=device, dtype=dtype)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True, high_res_features=None):
        masks, iou_pred = self._predict_masks(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, high_res_features)
        if multimask_output:
            return masks[:, 1:], iou_pred[:, 1:]
        return masks[:, 0:1], iou_pred[:, 0:1]

    def _predict_masks(self, img_feat, img_pe, sparse_prompts, dense_prompts, high_res_features=None):
        B = sparse_prompts.shape[0]

        special_tokens = [cast_to_input(self.iou_token.weight, sparse_prompts), cast_to_input(self.mask_tokens.weight, sparse_prompts)]
        if self.pred_obj_scores:
            special_tokens.append(cast_to_input(self.obj_score_token.weight, sparse_prompts))
        all_tokens = torch.cat([torch.cat(special_tokens, dim=0).unsqueeze(0).expand(B, -1, -1), sparse_prompts], dim=1)

        # Add dense prompts to image features and flatten for transformer
        feat = img_feat.expand(B, -1, -1, -1) + dense_prompts if img_feat.shape[0] != B else img_feat + dense_prompts
        b, c, h, w = feat.shape
        token_out, feat_out = self.transformer(feat.flatten(2).permute(0, 2, 1), img_pe.expand(B, -1, -1, -1).flatten(2).permute(0, 2, 1), all_tokens)

        # Upscale image features and optionally add high-res skip connections
        feat_2d = feat_out.permute(0, 2, 1).view(b, c, h, w)
        upsampled = self.output_upscaling(feat_2d)
        if self.use_high_res_features and high_res_features is not None:
            upsampled = upsampled + F.interpolate(self.conv_s1(high_res_features[1]), size=upsampled.shape[-2:], mode="bilinear", align_corners=False)
            upsampled = upsampled + F.interpolate(self.conv_s0(high_res_features[0]), size=upsampled.shape[-2:], mode="bilinear", align_corners=False)

        # Per-mask prediction via hypernetwork dot product
        mask_out = token_out[:, 1:1 + self.num_mask_tokens]
        per_mask_embed = torch.stack([mlp(mask_out[:, i]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], dim=1)
        masks = (per_mask_embed @ upsampled.flatten(2)).view(B, self.num_mask_tokens, upsampled.shape[2], upsampled.shape[3])
        return masks, self.iou_prediction_head(token_out[:, 0])
