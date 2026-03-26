# Adapted from: https://github.com/SandAI-org/DaVinci-MagiHuman
# Copyright 2026 SandAI. Licensed under Apache 2.0.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from comfy.ldm.modules.attention import optimized_attention
import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import comfy.ops


# ──────────────────────── Activation functions ────────────────────────

def swiglu7(x, alpha=1.702, limit=7.0):
    out_dtype = x.dtype
    x = x.float()
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha=1.702, limit=7.0):
    out_dtype = x.dtype
    x = x.float()
    x = x.clamp(max=limit)
    return (x * torch.sigmoid(alpha * x)).to(out_dtype)


# ──────────────────────── RoPE helpers ────────────────────────

def freq_bands(num_bands, temperature=10000.0, step=1, device=None):
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).float() / num_bands
    return 1.0 / (temperature ** exp)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)")
    sin = repeat(sin, "... d -> ... 1 (2 d)")
    return torch.cat([
        x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin,
        x[..., ro_dim:]
    ], dim=-1)


# ──────────────────────── Coordinate generation ────────────────────────

def get_coords(shape, ref_feat_shape, offset_thw=(0, 0, 0), device=None, dtype=torch.float32):
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw

    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    time_grid, height_grid, width_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")
    coords_flat = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)

    meta = torch.tensor([ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype)
    meta_expanded = meta.expand(coords_flat.size(0), -1)
    return torch.cat([coords_flat, meta_expanded], dim=-1)


# ──────────────────────── Modality Dispatcher ────────────────────────

MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2


class ModalityDispatcher:
    def __init__(self, modality_mapping, num_modalities):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        permuted = modality_mapping[self.permute_mapping]
        self.group_size = torch.bincount(permuted, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu = [int(x) for x in self.group_size.cpu().tolist()]

    def dispatch(self, x):
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups):
        return torch.cat(groups, dim=0)

    @staticmethod
    def permute(x, mapping):
        return x[mapping]

    @staticmethod
    def inv_permute(x, mapping):
        return x[mapping]


# ──────────────────────── ElementWise Fourier Embed ────────────────────────

class ElementWiseFourierEmbed(nn.Module):
    def __init__(self, dim, temperature=10000.0, device=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        bands = freq_bands(dim // 8, temperature=temperature, step=1, device=device).to(dtype)
        self.register_buffer("bands", bands)

    def forward(self, coords):
        coords_xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        refs = coords[:, 6:9]

        scales = (refs - 1) / (sizes - 1)
        scales[(refs == 1) & (sizes == 1)] = 1

        centers = (sizes - 1) / 2
        centers[:, 0] = 0
        coords_xyz = coords_xyz - centers

        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        sin_proj = proj.sin()
        cos_proj = proj.cos()
        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)


# ──────────────────────── MultiModality RMSNorm ────────────────────────

class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with per-modality weights. For multi-expert layers, weight has
    shape (dim * num_modality,) and is chunked per modality during forward.
    Uses separate ops.Linear-style sub-norms when split by process_unet_state_dict."""

    def __init__(self, dim, eps=1e-6, num_modality=1, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        # Single weight parameter matching checkpoint layout
        self.weight = nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.float32))

    def _rms(self, x):
        t = x.float()
        return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x, modality_dispatcher=None):
        # Keep weight in fp32 for precision — the original model uses fp32 norms
        weight = comfy.model_management.cast_to(self.weight, dtype=torch.float32, device=x.device)
        t = self._rms(x)  # always fp32
        if self.num_modality > 1 and modality_dispatcher is not None:
            weight_chunked = weight.chunk(self.num_modality, dim=0)
            t_list = modality_dispatcher.dispatch(t)
            for i in range(self.num_modality):
                t_list[i] = t_list[i] * (weight_chunked[i] + 1)
            return modality_dispatcher.undispatch(*t_list)
        return t * (weight + 1)


# ──────────────────────── MoE-aware Linear ────────────────────────

class MagiMoELinear(nn.Module):
    """Linear with multi-expert support using proper comfy ops.
    Single expert: uses operations.Linear directly.
    Multi expert: uses nn.ModuleList of operations.Linear, one per expert.
    State dict remapping handled by process_unet_state_dict in supported_models.
    """
    def __init__(self, in_features, out_features, num_experts=1, bias=False, operation_settings={}):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        ops = operation_settings.get("operations", comfy.ops.disable_weight_init)
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        if num_experts > 1:
            self.experts = nn.ModuleList([
                ops.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
                for _ in range(num_experts)
            ])
        else:
            self.linear = ops.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x, modality_dispatcher=None):
        if self.num_experts > 1 and modality_dispatcher is not None:
            parts = modality_dispatcher.dispatch(x)
            for i in range(self.num_experts):
                parts[i] = self.experts[i](parts[i])
            return modality_dispatcher.undispatch(*parts)
        return self.linear(x)


# ──────────────────────── Attention ────────────────────────

class MagiAttention(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim,
                 num_modality=1, enable_attn_gating=True, num_layers=40,
                 operation_settings={}):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.num_modality = num_modality
        self.enable_attn_gating = enable_attn_gating

        self.gating_size = num_heads_q if enable_attn_gating else 0
        self.q_size = num_heads_q * head_dim
        self.kv_size = num_heads_kv * head_dim

        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality=num_modality)

        qkv_out = self.q_size + self.kv_size * 2 + self.gating_size
        self.linear_qkv = MagiMoELinear(hidden_size, qkv_out, num_experts=num_modality,
                                        bias=False, operation_settings=operation_settings)
        self.linear_proj = MagiMoELinear(self.q_size, hidden_size, num_experts=num_modality,
                                         bias=False, operation_settings=operation_settings)
        self.q_norm = MultiModalityRMSNorm(head_dim, num_modality=num_modality)
        self.k_norm = MultiModalityRMSNorm(head_dim, num_modality=num_modality)

    def forward(self, hidden_states, rope, modality_dispatcher, transformer_options={}):
        normed = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        qkv = self.linear_qkv(normed, modality_dispatcher=modality_dispatcher).to(torch.float32)

        q, k, v, g = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=-1)
        q = q.view(-1, self.num_heads_q, self.head_dim)
        k = k.view(-1, self.num_heads_kv, self.head_dim)
        v = v.view(-1, self.num_heads_kv, self.head_dim)
        if self.enable_attn_gating:
            g = g.view(q.shape[0], self.num_heads_q, -1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, modality_dispatcher.inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, modality_dispatcher.inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, modality_dispatcher.inv_permute_mapping).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        # Cast to bf16 and use native GQA - matching original's flash attention
        # which takes q(40 heads) and k,v(8 heads) without repeating.
        q = q.to(torch.bfloat16).transpose(1, 2)  # (1, heads_q, L, dim)
        k = k.to(torch.bfloat16).transpose(1, 2)  # (1, heads_kv, L, dim)
        v = v.to(torch.bfloat16).transpose(1, 2)  # (1, heads_kv, L, dim)

        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        out = out.transpose(1, 2).squeeze(0)  # (L, heads_q, dim)
        out = out.squeeze(0)

        out = ModalityDispatcher.permute(out, modality_dispatcher.permute_mapping)

        if self.enable_attn_gating:
            g = ModalityDispatcher.inv_permute(g.squeeze(-1), modality_dispatcher.inv_permute_mapping)
            g = ModalityDispatcher.permute(g, modality_dispatcher.permute_mapping)
            out = out.view(-1, self.num_heads_q, self.head_dim)
            out = out * torch.sigmoid(g).unsqueeze(-1)

        out = out.reshape(-1, self.num_heads_q * self.head_dim).to(torch.bfloat16)
        out = self.linear_proj(out, modality_dispatcher=modality_dispatcher)
        return out.float()


# ──────────────────────── MLP ────────────────────────

class MagiMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_type="swiglu7",
                 num_modality=1, gated_act=True, operation_settings={}):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality=num_modality)

        up_size = intermediate_size * 2 if gated_act else intermediate_size
        self.up_gate_proj = MagiMoELinear(hidden_size, up_size, num_experts=num_modality,
                                          bias=False, operation_settings=operation_settings)
        self.down_proj = MagiMoELinear(intermediate_size, hidden_size, num_experts=num_modality,
                                       bias=False, operation_settings=operation_settings)
        self.activation = swiglu7 if activation_type == "swiglu7" else gelu7

    def forward(self, x, modality_dispatcher):
        normed = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        up = self.up_gate_proj(normed, modality_dispatcher=modality_dispatcher).to(torch.float32)
        activated = self.activation(up).to(torch.bfloat16)
        return self.down_proj(activated, modality_dispatcher=modality_dispatcher).to(torch.float32)


# ──────────────────────── Transformer Layer ────────────────────────

class MagiTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim,
                 layer_idx, mm_layers, gelu7_layers, post_norm_layers,
                 enable_attn_gating=True, num_layers=40, operation_settings={}):
        super().__init__()
        num_modality = 3 if layer_idx in mm_layers else 1
        self.has_post_norm = layer_idx in post_norm_layers

        self.attention = MagiAttention(
            hidden_size=hidden_size, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
            head_dim=head_dim, num_modality=num_modality, enable_attn_gating=enable_attn_gating,
            num_layers=num_layers, operation_settings=operation_settings,
        )

        if layer_idx in gelu7_layers:
            activation_type = "gelu7"
            gated_act = False
            intermediate_size = hidden_size * 4
        else:
            activation_type = "swiglu7"
            gated_act = True
            intermediate_size = int(hidden_size * 4 * 2 / 3) // 4 * 4

        self.mlp = MagiMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            activation_type=activation_type, num_modality=num_modality,
            gated_act=gated_act, operation_settings=operation_settings,
        )

        if self.has_post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(hidden_size, num_modality=num_modality)

    def forward(self, x, rope, modality_dispatcher, transformer_options={}):
        # Accumulate residuals in fp32 to match original's precision
        x = x.float()
        attn_out = self.attention(x, rope, modality_dispatcher, transformer_options)
        if self.has_post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        x = x + attn_out.float()

        mlp_out = self.mlp(x, modality_dispatcher)
        if self.has_post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        x = x + mlp_out.float()
        return x


# ──────────────────────── Adapter (Input Embeddings) ────────────────────────

class MagiAdapter(nn.Module):
    def __init__(self, hidden_size, num_heads_q, video_in_channels, audio_in_channels,
                 text_in_channels, head_dim, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.video_embedder = operations.Linear(video_in_channels, hidden_size, bias=True, device=device, dtype=dtype)
        self.text_embedder = operations.Linear(text_in_channels, hidden_size, bias=True, device=device, dtype=dtype)
        self.audio_embedder = operations.Linear(audio_in_channels, hidden_size, bias=True, device=device, dtype=dtype)
        self.rope = ElementWiseFourierEmbed(head_dim, device=device, dtype=torch.float32)

    def forward(self, x, coords_mapping, video_mask, audio_mask, text_mask):
        rope = self.rope(coords_mapping)
        hidden = torch.zeros(x.shape[0], self.video_embedder.out_features, device=x.device, dtype=x.dtype)
        if text_mask.any():
            hidden[text_mask] = self.text_embedder(x[text_mask, :self.text_embedder.in_features])
        if audio_mask.any():
            hidden[audio_mask] = self.audio_embedder(x[audio_mask, :self.audio_embedder.in_features])
        if video_mask.any():
            hidden[video_mask] = self.video_embedder(x[video_mask, :self.video_embedder.in_features])
        return hidden, rope


# ──────────────────────── Top-level Model ────────────────────────

class MagiModel(nn.Module):
    def __init__(self,
                 hidden_size=5120,
                 num_layers=40,
                 head_dim=128,
                 num_query_groups=8,
                 video_in_channels=192,
                 audio_in_channels=64,
                 text_in_channels=3584,
                 mm_layers=None,
                 gelu7_layers=None,
                 post_norm_layers=None,
                 enable_attn_gating=True,
                 patch_size=None,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels

        if mm_layers is None:
            mm_layers = [0, 1, 2, 3, 36, 37, 38, 39]
        if gelu7_layers is None:
            gelu7_layers = [0, 1, 2, 3]
        if post_norm_layers is None:
            post_norm_layers = []
        if patch_size is None:
            patch_size = (1, 2, 2)
        if operations is None:
            operations = comfy.ops.disable_weight_init

        self.mm_layers = mm_layers
        self.gelu7_layers = gelu7_layers
        self.post_norm_layers = post_norm_layers
        self.patch_size = patch_size

        num_heads_q = hidden_size // head_dim
        num_heads_kv = num_query_groups

        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        self.adapter = MagiAdapter(
            hidden_size=hidden_size, num_heads_q=num_heads_q,
            video_in_channels=video_in_channels, audio_in_channels=audio_in_channels,
            text_in_channels=text_in_channels, head_dim=head_dim,
            device=device, dtype=dtype, operations=operations,
        )

        self.block = nn.Module()
        self.block.layers = nn.ModuleList([
            MagiTransformerLayer(
                hidden_size=hidden_size, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                head_dim=head_dim, layer_idx=i, mm_layers=mm_layers, gelu7_layers=gelu7_layers,
                post_norm_layers=post_norm_layers, enable_attn_gating=enable_attn_gating,
                num_layers=num_layers, operation_settings=operation_settings,
            )
            for i in range(num_layers)
        ])

        self.final_norm_video = MultiModalityRMSNorm(hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(hidden_size)
        self.final_linear_video = operations.Linear(hidden_size, video_in_channels, bias=False, device=device, dtype=dtype)
        self.final_linear_audio = operations.Linear(hidden_size, audio_in_channels, bias=False, device=device, dtype=dtype)



    def _patchify(self, x):
        """Match UnfoldNd ordering: per-token values are (C, pT, pH, pW) - channel outermost."""
        pt, ph, pw = self.patch_size
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, T', H', W', C, pt, ph, pw)
        x = x.reshape(B, -1, C * pt * ph * pw)
        return x

    def _unpatchify(self, x, T, H, W):
        """Depack: C-innermost ordering (pT, pH, pW, C) per token, matching original depack rearrange."""
        pt, ph, pw = self.patch_size
        C = self.video_in_channels // (pt * ph * pw)
        T_p, H_p, W_p = T // pt, H // ph, W // pw
        B = x.shape[0]
        x = x.reshape(B, T_p, H_p, W_p, pt, ph, pw, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, T', pt, H', ph, W', pw)
        x = x.reshape(B, C, T, H, W)
        return x

    def _build_sequence(self, video_tokens, audio_tokens, text_tokens, device, dtype):
        n_video = video_tokens.shape[0]
        n_audio = audio_tokens.shape[0]
        n_text = text_tokens.shape[0]
        max_channel = max(video_tokens.shape[-1], audio_tokens.shape[-1], text_tokens.shape[-1])

        padded = []
        for t in [video_tokens, audio_tokens, text_tokens]:
            if t.shape[-1] < max_channel:
                t = F.pad(t, (0, max_channel - t.shape[-1]))
            padded.append(t)
        token_sequence = torch.cat(padded, dim=0)

        modality_mapping = torch.cat([
            torch.full((n_video,), MODALITY_VIDEO, dtype=torch.int64, device=device),
            torch.full((n_audio,), MODALITY_AUDIO, dtype=torch.int64, device=device),
            torch.full((n_text,), MODALITY_TEXT, dtype=torch.int64, device=device),
        ], dim=0)

        return token_sequence, modality_mapping

    def _build_coords(self, T, H, W, n_audio, n_text, device, dtype):
        pt, ph, pw = self.patch_size
        t_p, h_p, w_p = T // pt, H // ph, W // pw

        video_coords = get_coords(
            shape=(t_p, h_p, w_p), ref_feat_shape=(t_p, h_p, w_p),
            device=device, dtype=dtype,
        )
        audio_coords = get_coords(
            shape=(n_audio, 1, 1), ref_feat_shape=(t_p, 1, 1),
            device=device, dtype=dtype,
        )
        text_coords = get_coords(
            shape=(n_text, 1, 1), ref_feat_shape=(1, 1, 1),
            offset_thw=(-n_text, 0, 0), device=device, dtype=dtype,
        )

        return torch.cat([video_coords, audio_coords, text_coords], dim=0)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        # Handle list input from _apply_model's unpack_latents
        if isinstance(x, list):
            video_latent = None
            audio_latent = None
            for component in x:
                if component.ndim == 5 and component.shape[1] == self.video_in_channels // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2]):
                    video_latent = component
                elif component.ndim == 3 and component.shape[2] == self.audio_in_channels:
                    audio_latent = component
            if video_latent is None:
                video_latent = x[0]  # fallback
            has_audio = audio_latent is not None
        else:
            video_latent = x
            audio_latent = None
            has_audio = False

        B, C, T, H, W = video_latent.shape
        device = video_latent.device
        dtype = video_latent.dtype

        # Get attention mask to determine real text token count
        attention_mask = kwargs.get("attention_mask", None)

        video_tokens_batch = self._patchify(video_latent)

        outputs = []
        audio_outputs = []

        for b in range(B):
            video_tokens = video_tokens_batch[b]

            # Audio tokens: use consistent random noise matching original pipeline.
            # Cached so the same noise is used across all sampling steps.
            if has_audio and audio_latent is not None:
                audio_tok = audio_latent[b]
                if audio_tok.dim() == 1:
                    audio_tok = audio_tok.unsqueeze(0)
            else:
                num_frames = (T - 1) * 4 + 1
                cache_key = (num_frames, self.audio_in_channels, b)
                if not hasattr(self, '_audio_cache') or self._audio_cache.get('key') != cache_key:
                    gen = torch.Generator(device='cpu')
                    gen.manual_seed(1234 + b)
                    self._audio_cache = {
                        'key': cache_key,
                        'data': torch.randn(num_frames, self.audio_in_channels, generator=gen, device='cpu'),
                    }
                audio_tok = self._audio_cache['data'].to(device=device, dtype=dtype)

            # Text tokens: truncate to real length using attention mask (discard padding)
            if context is not None:
                text_tok = context[b]
                if attention_mask is not None:
                    if attention_mask.dim() > 1:
                        mask_b = attention_mask[b]
                    else:
                        mask_b = attention_mask
                    real_len = max(1, int(mask_b.sum().item()))
                    text_tok = text_tok[:real_len]
            else:
                text_tok = torch.zeros(1, self.text_in_channels, device=device, dtype=dtype)

            n_audio = audio_tok.shape[0]
            n_text = text_tok.shape[0]

            if not hasattr(self, '_call_count'):
                self._call_count = 0
            self._call_count += 1
            if self._call_count <= 2:
                import logging
                logging.info("MagiModel call %d: x_mean=%.4f x_std=%.4f, video_tok=%s, audio=%s, text=%s",
                             self._call_count,
                             video_latent[b].float().mean().item(), video_latent[b].float().std().item(),
                             list(video_tokens.shape), list(audio_tok.shape), list(text_tok.shape))

            token_seq, modality_mapping = self._build_sequence(
                video_tokens, audio_tok, text_tok, device, dtype
            )
            coords_mapping = self._build_coords(T, H, W, n_audio, n_text, device, dtype)

            video_mask = modality_mapping == MODALITY_VIDEO
            audio_mask = modality_mapping == MODALITY_AUDIO
            text_mask = modality_mapping == MODALITY_TEXT

            hidden, rope = self.adapter(token_seq, coords_mapping, video_mask, audio_mask, text_mask)

            dispatcher = ModalityDispatcher(modality_mapping, 3)
            hidden = ModalityDispatcher.permute(hidden, dispatcher.permute_mapping)

            for i, layer in enumerate(self.block.layers):
                hidden = layer(hidden, rope, dispatcher, transformer_options)

            hidden = ModalityDispatcher.inv_permute(hidden, dispatcher.inv_permute_mapping)

            x_video = hidden[video_mask].float()
            x_video = self.final_norm_video(x_video)
            x_video = self.final_linear_video(x_video.to(dtype)).float()

            x_audio = hidden[audio_mask].float()
            x_audio = self.final_norm_audio(x_audio)
            x_audio = self.final_linear_audio(x_audio.to(dtype)).float()

            outputs.append(x_video)
            audio_outputs.append(x_audio)

        video_out = torch.stack([self._unpatchify(o.unsqueeze(0), T, H, W).squeeze(0) for o in outputs], dim=0)

        video_out = video_out.to(dtype)

        if has_audio:
            audio_out = torch.stack(audio_outputs, dim=0).to(dtype)
            return [video_out, audio_out]

        return video_out
