# SAM components shared by detector and tracker

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention


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
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        return self.out_proj(optimized_attention(q, k, v, self.num_heads))


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
            queries = self.norm1(self.self_attn(queries, queries, queries))
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
            queries, keys = layer(queries, keys, point_embedding, image_pe)
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
        projected = 2 * math.pi * (2 * normalized_coords.float() - 1) @ proj_matrix
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(orig_dtype)

    def forward(self, size, device=None):
        h, w = size
        dev = device if device is not None else self.positional_encoding_gaussian_matrix.device
        ones = torch.ones((h, w), device=dev, dtype=torch.float32)
        norm_xy = torch.stack([(ones.cumsum(1) - 0.5) / w, (ones.cumsum(0) - 0.5) / h], dim=-1)
        return self._encode(norm_xy).permute(2, 0, 1).unsqueeze(0)

    def forward_with_coords(self, pixel_coords, image_size):
        norm = pixel_coords.clone()
        norm[:, :, 0] /= image_size[1]
        norm[:, :, 1] /= image_size[0]
        return self._encode(norm)
