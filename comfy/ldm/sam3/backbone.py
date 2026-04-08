# SAM3 ViTDet backbone + FPN neck + position encoding.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import EmbedND
from comfy.ops import cast_to_input


def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def rope_2d(end_x: int, end_y: int, dim: int, theta: float = 10000.0, scale_pos: float = 1.0):
    """Generate 2D axial RoPE using flux EmbedND. Returns [1, 1, HW, dim//2, 2, 2]."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    ids = torch.stack([(t % end_x) * scale_pos,
                       torch.div(t, end_x, rounding_mode="floor") * scale_pos], dim=-1)
    return EmbedND(dim=dim, theta=theta, axes_dim=[dim // 2, dim // 2])(ids.unsqueeze(0))


# ViTDet Attention
class Attention(nn.Module):
    """Multi-head attention with fused QKV projection"""

    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rope=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        # Fused QKV: single linear matching checkpoint's "attn.qkv"
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # 3x [B, heads, N, dim]
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)
        return self.proj(optimized_attention(q, k, v, self.num_heads, skip_reshape=True))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, device=None, dtype=None, operations=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = operations.Linear(dim, hidden, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = operations.Linear(hidden, dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, window_size=0, use_rope=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.window_size = window_size
        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.attn = Attention(dim, num_heads, qkv_bias, use_rope, device=device, dtype=dtype, operations=operations)
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.mlp = MLP(dim, mlp_ratio, device=device, dtype=dtype, operations=operations)

    def forward(self, x, freqs_cis=None):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(x.shape[0], self.window_size * self.window_size, -1)
            x = self.attn(x, freqs_cis=freqs_cis)  # windowed blocks also use RoPE
            x = x.view(-1, self.window_size, self.window_size, x.shape[-1])
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.attn(x, freqs_cis=freqs_cis)
            x = x.view(B, H, W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=1024, device=None, dtype=None, operations=None):
        super().__init__()
        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.proj(x)


class ViTDet(nn.Module):
    def __init__(self, img_size=1008, patch_size=14, embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625, qkv_bias=True, window_size=24,
                 global_att_blocks=(7, 15, 23, 31), use_rope=True, pretrain_img_size=336, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_att_blocks = set(global_att_blocks)

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim, device=device, dtype=dtype, operations=operations)

        # pos_embed: [1, 577, 1024] = 576 patches + 1 cls token (pretrain 336/14=24, 24*24+1=577)
        num_patches = (pretrain_img_size // patch_size) ** 2 + 1  # +1 for cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim, device=device, dtype=dtype))

        self.ln_pre = operations.LayerNorm(embed_dim, device=device, dtype=dtype)

        grid_size = img_size // patch_size
        pretrain_grid = pretrain_img_size // patch_size

        self.blocks = nn.ModuleList()
        for i in range(depth):
            is_global = i in self.global_att_blocks
            self.blocks.append(Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias,
                window_size=0 if is_global else window_size,
                use_rope=use_rope,
                device=device, dtype=dtype, operations=operations,
            ))

        if use_rope:
            # Global attention blocks: RoPE for full grid with interpolation scaling
            rope_scale = pretrain_grid / grid_size
            self.register_buffer("freqs_cis", rope_2d(grid_size, grid_size, embed_dim // num_heads, scale_pos=rope_scale), persistent=False)
            # Windowed attention blocks: RoPE for window size (pretrain_grid == window_size, so scale=1)
            self.register_buffer("freqs_cis_window", rope_2d(window_size, window_size, embed_dim // num_heads), persistent=False)
        else:
            self.freqs_cis = None

    def _get_pos_embed(self, num_tokens):
        pos = self.pos_embed
        if pos.shape[1] == num_tokens:
            return pos
        # Strip cls token (first), tile spatial positions, re-add
        # SAM3 uses tile_abs_pos=True
        cls_pos = pos[:, :1]
        spatial_pos = pos[:, 1:]
        old_size = int(math.sqrt(spatial_pos.shape[1]))
        new_size = int(math.sqrt(num_tokens - 1)) if num_tokens > 1 else old_size
        spatial_2d = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        # Tile: repeat the pretrain grid to fill the runtime grid
        tiles_h = new_size // old_size + 1
        tiles_w = new_size // old_size + 1
        tiled = spatial_2d.tile([1, 1, tiles_h, tiles_w])[:, :, :new_size, :new_size]
        tiled = tiled.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return torch.cat([cls_pos, tiled], dim=1)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, H', W')
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)  # (B, N, C)

        # Add position embedding (skip cls token portion, use spatial only)
        pos = cast_to_input(self._get_pos_embed(Hp * Wp + 1), x)
        x = x + pos[:, 1:Hp * Wp + 1]

        x = x.view(B, Hp, Wp, C)
        x = self.ln_pre(x)

        freqs_cis_global = getattr(self, 'freqs_cis', None)
        freqs_cis_win = getattr(self, 'freqs_cis_window', None)
        if freqs_cis_global is not None:
            freqs_cis_global = cast_to_input(freqs_cis_global, x)
        if freqs_cis_win is not None:
            freqs_cis_win = cast_to_input(freqs_cis_win, x)

        for block in self.blocks:
            # Windowed blocks use window-sized RoPE, global blocks use full-grid RoPE
            fc = freqs_cis_win if block.window_size > 0 else freqs_cis_global
            x = block(x, freqs_cis=fc)

        return x.permute(0, 3, 1, 2)  # (B, C, H', W')


class FPNScaleConv(nn.Module):
    def __init__(self, in_dim, out_dim, scale, device=None, dtype=None, operations=None):
        super().__init__()
        if scale == 4.0:
            self.dconv_2x2_0 = operations.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
            self.dconv_2x2_1 = operations.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2, device=device, dtype=dtype)
            proj_in = in_dim // 4
        elif scale == 2.0:
            self.dconv_2x2 = operations.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
            proj_in = in_dim // 2
        elif scale == 1.0:
            proj_in = in_dim
        elif scale == 0.5:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            proj_in = in_dim
        self.scale = scale
        self.conv_1x1 = operations.Conv2d(proj_in, out_dim, kernel_size=1, device=device, dtype=dtype)
        self.conv_3x3 = operations.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, device=device, dtype=dtype)

    def forward(self, x):
        if self.scale == 4.0:
            x = F.gelu(self.dconv_2x2_0(x))
            x = self.dconv_2x2_1(x)
        elif self.scale == 2.0:
            x = self.dconv_2x2(x)
        elif self.scale == 0.5:
            x = self.pool(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal position encoding (DETR-style) with result caching."""
    def __init__(self, num_pos_feats=256, temperature=10000.0, normalize=True, scale=None):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.half_dim = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi
        self._cache = {}

    def _sincos(self, vals):
        """Encode 1D values to interleaved sin/cos features."""
        freqs = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=vals.device) // 2) / self.half_dim)
        raw = vals[..., None] * self.scale / freqs
        return torch.stack((raw[..., 0::2].sin(), raw[..., 1::2].cos()), dim=-1).flatten(-2)

    def forward(self, x):
        B, C, H, W = x.shape
        key = (H, W, x.device)
        if key not in self._cache:
            gy = torch.arange(H, dtype=torch.float32, device=x.device)
            gx = torch.arange(W, dtype=torch.float32, device=x.device)
            if self.normalize:
                gy, gx = gy / (H - 1 + 1e-6), gx / (W - 1 + 1e-6)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            self._cache[key] = torch.cat((self._sincos(yy), self._sincos(xx)), dim=-1).permute(2, 0, 1).unsqueeze(0)
        return self._cache[key].expand(B, -1, -1, -1)


class SAM3VisionBackbone(nn.Module):
    def __init__(self, embed_dim=1024, d_model=256, multiplex=False, device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.trunk = ViTDet(embed_dim=embed_dim, device=device, dtype=dtype, operations=operations, **kwargs)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model, normalize=True)
        self.multiplex = multiplex

        fpn_args = dict(device=device, dtype=dtype, operations=operations)
        if multiplex:
            # SAM3.1: 3 FPN necks, 3 levels each (4x, 2x, 1x)
            scales = [4.0, 2.0, 1.0]
            self.convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.propagation_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.interactive_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
        else:
            # SAM3: 2 FPN necks, 4 levels each (4x, 2x, 1x, 0.5x)
            scales = [4.0, 2.0, 1.0, 0.5]
            self.convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])
            self.sam2_convs = nn.ModuleList([FPNScaleConv(embed_dim, d_model, s, **fpn_args) for s in scales])

    def forward(self, images, need_tracker=False, tracker_mode=None):
        backbone_out = self.trunk(images)
        features = [conv(backbone_out) for conv in self.convs]
        positions = [self.position_encoding(f).to(dtype=f.dtype) for f in features]

        if self.multiplex:
            if tracker_mode == "propagation":
                tracker_convs = self.propagation_convs
            elif tracker_mode == "interactive":
                tracker_convs = self.interactive_convs
            else:
                return features, positions, None, None
        elif need_tracker:
            tracker_convs = self.sam2_convs
        else:
            return features, positions, None, None

        tracker_features = [conv(backbone_out) for conv in tracker_convs]
        tracker_positions = [self.position_encoding(f).to(dtype=f.dtype) for f in tracker_features]
        return features, positions, tracker_features, tracker_positions
