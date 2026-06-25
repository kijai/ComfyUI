import contextlib
import math
import torch
import torch.nn as nn
from dataclasses import dataclass

from comfy import sd1_clip
import comfy.ops
import comfy.utils
import comfy.model_management
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.rmsnorm import rms_norm
from comfy.text_encoders.llama import RMSNorm, MLP, BaseLlama
from comfy.text_encoders.gemma4 import (
    GEMMA4_VISION_31B_CONFIG,
    Gemma4VisionEncoder,
    Gemma4RMSNormProjector,
    Gemma4SDTokenizer,
    Gemma4Tokenizer,
    Gemma4Model,
    ClippedLinear,
    _apply_rotary_pos_emb,
)


@dataclass
class DiffusionGemmaConfig:
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim = 256
    global_head_dim = 512
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    partial_rotary_factor: float = 0.25
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    final_logit_softcapping: float = 30.0
    canvas_length: int = 256
    unfused_experts: bool = False
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    stop_tokens = [1, 106, 50]
    pad_token_id: int = 0
    vision_config = GEMMA4_VISION_31B_CONFIG
    mm_tokens_per_image = 280


def _gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def _make_dg_scaled_embedding(ops, vocab_size, hidden_size, device, dtype):
    # Reference casts sqrt(hidden_size) to the weight dtype before multiplying.
    class ScaledEmbedding(ops.Embedding):
        def forward(self, input_ids, out_dtype=None):
            out = super().forward(input_ids, out_dtype=out_dtype)
            scale = torch.tensor(hidden_size ** 0.5, dtype=self.weight.dtype).item()
            return out * scale
    return ScaledEmbedding(vocab_size, hidden_size, device=device, dtype=dtype)


class DiffusionGemmaAttention(nn.Module):
    def __init__(self, config, head_dim, num_kv_heads, has_v_proj, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.inner_size = self.num_heads * head_dim

        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        if has_v_proj:
            self.v_proj = ops.Linear(config.hidden_size, num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        else:
            self.v_proj = None
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None, past_key_value=None,
                sliding_window=None, update_cache=True):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xq = self.q_norm(xq)
        xq = _apply_rotary_pos_emb(xq, freqs_cis)

        xk = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        if self.v_proj is not None:
            xv = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        else:
            xv = xk
        xk = self.k_norm(xk)
        xv = rms_norm(xv)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        xk = _apply_rotary_pos_emb(xk, freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            if len(past_key_value) > 0:
                past_key, past_value, cumulative_len = past_key_value
                xk = torch.cat((past_key, xk), dim=2)
                xv = torch.cat((past_value, xv), dim=2)
            else:
                cumulative_len = 0
            if update_cache:
                new_cumulative = cumulative_len + seq_length
                if sliding_window is not None and xk.shape[2] > sliding_window - 1:
                    present_key_value = (xk[:, :, -(sliding_window - 1):], xv[:, :, -(sliding_window - 1):], new_cumulative)
                else:
                    present_key_value = (xk, xv, new_cumulative)

        expand_kv = (self.num_heads != self.num_kv_heads and
                     sliding_window is not None and
                     xk.shape[2] >= sliding_window)
        if expand_kv:
            xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        gqa_kwargs = {} if expand_kv else ({"enable_gqa": True} if self.num_heads != self.num_kv_heads else {})
        output = optimized_attention_for_device(xq.device, mask=attention_mask is not None, small_input=True)(
            xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, scale=1.0, **gqa_kwargs)

        return self.o_proj(output), present_key_value


class DiffusionGemmaRouter(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.top_k = config.top_k_experts
        self.scalar_root_size = config.hidden_size ** -0.5
        self.proj = ops.Linear(config.hidden_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.scale = nn.Parameter(torch.empty(config.hidden_size, device=device, dtype=dtype))
        self.per_expert_scale = nn.Parameter(torch.empty(config.num_experts, device=device, dtype=dtype))

    def forward(self, hidden_states):
        hidden_states = rms_norm(hidden_states)
        scale = comfy.ops.cast_to_input(self.scale, hidden_states, copy=False)
        hidden_states = hidden_states * scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probabilities = torch.nn.functional.softmax(expert_scores, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        per_expert_scale = comfy.ops.cast_to_input(self.per_expert_scale, expert_scores, copy=False)
        top_k_weights = top_k_weights * per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


def _dequant_bank(module, weight, dtype):
    # Chunked over experts to cap the fp32 transient; shapes are identical every
    # layer so the allocations are reused by the caching allocator.
    qdata = weight._qdata
    params = weight._params
    E = qdata.shape[0]
    scale = params.scale
    if scale.dim() == 0:
        scale = scale.expand(E)
    out_f = module.out_features
    in_f = module.in_features
    if hasattr(params, "block_scale"):  # block-scaled (nvfp4)
        block_scale = params.block_scale
        if qdata.shape[1] == block_scale.shape[1]:
            # row-padded bank: tile-local swizzle makes one flat dequant valid
            rows = qdata.shape[1]
            if params.scale.dim() == 0:
                # shared bank scale: single kernel call straight to compute dtype
                flat_params = type(params)(scale=params.scale, orig_dtype=dtype, orig_shape=(E * rows, in_f),
                                           block_scale=block_scale.reshape(E * rows, -1))
                w = weight.layout_cls.dequantize(qdata.reshape(E * rows, -1), flat_params)
                return w.view(E, rows, in_f)[:, :out_f]
            unit = torch.ones((), device=qdata.device, dtype=torch.float32)
            flat_params = type(params)(scale=unit, orig_dtype=torch.float32, orig_shape=(E * rows, in_f),
                                       block_scale=block_scale.reshape(E * rows, -1))
            w = weight.layout_cls.dequantize(qdata.reshape(E * rows, -1), flat_params)
            w = (w.view(E, rows, in_f)[:, :out_f] * scale.view(-1, 1, 1)).to(dtype)
            return w
        w = torch.empty((E, out_f, in_f), dtype=dtype, device=qdata.device)
        for i in range(E):
            expert_params = type(params)(scale=scale[i], orig_dtype=dtype, orig_shape=(out_f, in_f),
                                         block_scale=block_scale[i])
            w[i] = weight.layout_cls.dequantize(qdata[i], expert_params)
        return w
    w = torch.empty((E, out_f, in_f), dtype=dtype, device=qdata.device)
    chunk = 32
    for i in range(0, E, chunk):
        w[i:i + chunk] = (qdata[i:i + chunk].to(torch.float32) * scale[i:i + chunk].view(-1, 1, 1)).to(dtype)
    return w


def _bank_bmm(module, x):
    # x [E, C, in] against the whole bank [E, out, in]; banks handled one at a
    # time to cap the dequant transient.
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(module, x, offloadable=True)
    try:
        w = weight
        if isinstance(w, QuantizedTensor):
            w = _dequant_bank(module, w, x.dtype)
        out = torch.bmm(x, w.transpose(1, 2))
        if bias is not None:
            out = out + bias.unsqueeze(1)
        return out
    finally:
        comfy.ops.uncast_bias_weight(module, weight, bias, offload_stream)


class DiffusionGemmaExperts(nn.Module):
    grouped_bucket = 64
    grouped_min_tokens = 64

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.unfused = config.unfused_experts
        E = config.num_experts
        H = config.hidden_size
        I = config.moe_intermediate_size
        if self.unfused:
            self.gate_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=I, bias=False, device=device, dtype=dtype)
            self.up_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=I, bias=False, device=device, dtype=dtype)
        else:
            self.gate_up_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=2 * I, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts=E, in_features=I, out_features=H, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] >= self.grouped_min_tokens:
            return self._forward_grouped(hidden_states, top_k_index, top_k_weights)
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)

    def _forward_grouped(self, hidden_states, top_k_index, top_k_weights):
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        # bucket rounded up to limit allocator size churn; bmm padding waste is
        # negligible next to the bank dequant traffic
        C = -(-int(counts.max()) // self.grouped_bucket) * self.grouped_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K

        x = hidden_states[gather_tok].view(E, C, H)
        if self.unfused:
            gate = _bank_bmm(self.gate_proj, x)
            up = _bank_bmm(self.up_proj, x)
        else:
            gate, up = _bank_bmm(self.gate_up_proj, x).chunk(2, dim=-1)
        y = _bank_bmm(self.down_proj, _gelu_tanh(gate) * up)

        # pairs are a permutation: gather the real rows back to original pair order
        # and reduce densely — no atomics, and the weight multiply touches only N*K rows
        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, -1)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_loop(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_banks = (self.gate_proj, self.up_proj) if self.unfused else (self.gate_up_proj,)
        with contextlib.ExitStack() as stack:
            banks = [stack.enter_context(b.bank_resident(hidden_states)) for b in gate_up_banks + (self.down_proj,)]
            down_bank = banks[-1]
            for ei in expert_hit:
                expert_idx = int(ei.item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                if self.unfused:
                    gate = banks[0].expert_linear(current_state, expert_idx)
                    up = banks[1].expert_linear(current_state, expert_idx)
                else:
                    gate, up = banks[0].expert_linear(current_state, expert_idx).chunk(2, dim=-1)
                current_hidden_states = _gelu_tanh(gate) * up
                current_hidden_states = down_bank.expert_linear(current_hidden_states, expert_idx)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DiffusionGemmaSelfConditioning(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(_gelu_tanh(self.gate_proj(normed)) * self.up_proj(normed))
        return rms_norm(inputs_embeds + sc_signal)


class DiffusionGemmaBlock(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        self.sliding_window = config.sliding_attention[index % len(config.sliding_attention)] or None
        is_sliding = self.sliding_window is not None
        head_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv_heads = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads

        self.self_attn = DiffusionGemmaAttention(config, head_dim=head_dim, num_kv_heads=num_kv_heads,
                                                 has_v_proj=is_sliding, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.router = DiffusionGemmaRouter(config, device=device, dtype=dtype, ops=ops)
        self.experts = DiffusionGemmaExperts(config, device=device, dtype=dtype, ops=ops)

        norm_kwargs = dict(eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))

    def forward(self, x, attention_mask=None, freqs_cis=None, past_key_value=None,
                update_cache=True, layer_scalar=None):
        freqs_cis = freqs_cis[1] if self.sliding_window is not None else freqs_cis[0]

        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, freqs_cis=freqs_cis,
            past_key_value=past_key_value, sliding_window=self.sliding_window, update_cache=update_cache)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        hidden_states_1 = self.post_feedforward_layernorm_1(h)

        flat = residual.reshape(-1, residual.shape[-1])
        top_k_weights, top_k_index = self.router(flat)
        hidden_states_2 = self.experts(self.pre_feedforward_layernorm_2(flat), top_k_index, top_k_weights)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2.reshape(residual.shape))

        x = self.post_feedforward_layernorm(hidden_states_1 + hidden_states_2)
        x = residual + x

        scalar = layer_scalar if layer_scalar is not None else self.layer_scalar
        x = x * comfy.ops.cast_to_input(scalar, x, copy=False)
        return x, present_key_value


class DiffusionGemmaDecoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.embed_tokens = _make_dg_scaled_embedding(ops, config.vocab_size, config.hidden_size, device, dtype)
        self.layers = nn.ModuleList([
            DiffusionGemmaBlock(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config, device=device, dtype=dtype, ops=ops)

        rope_angles_global = int(config.partial_rotary_factor * config.global_head_dim // 2)
        nope_global = config.global_head_dim // 2 - rope_angles_global
        global_inv = 1.0 / (config.rope_theta[0] ** (torch.arange(0, 2 * rope_angles_global, 2).float() / config.global_head_dim))
        if nope_global > 0:
            global_inv = torch.cat([global_inv, torch.zeros(nope_global)])
        self.register_buffer("_global_inv_freq", global_inv, persistent=False)
        sliding_inv = 1.0 / (config.rope_theta[1] ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("_sliding_inv_freq", sliding_inv, persistent=False)

    def _freqs_from_inv(self, inv_freq, position_ids, device, dtype):
        inv_exp = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_exp @ pos_exp).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)

    def compute_freqs_cis(self, position_ids, device, dtype=None):
        global_freqs = self._freqs_from_inv(self._global_inv_freq, position_ids, device, dtype)
        sliding_freqs = self._freqs_from_inv(self._sliding_inv_freq, position_ids, device, dtype)
        return [global_freqs, sliding_freqs]


class _EncoderLayerScalar(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))


class _EncoderLanguageModel(nn.Module):
    # Encoder text weights are tied to the decoder; only the per-layer scalars are separate.
    def __init__(self, num_layers, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayerScalar(device=device, dtype=dtype) for _ in range(num_layers)])


class DiffusionGemmaVisionTower(Gemma4VisionEncoder):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__(config, dtype=dtype, device=device, ops=ops)
        self.register_buffer("std_bias", torch.zeros(config["hidden_size"], device=device, dtype=dtype))
        self.register_buffer("std_scale", torch.ones(config["hidden_size"], device=device, dtype=dtype))
        # use_clipped_linears=False in this model: no clip buffers in the checkpoint
        for m in self.modules():
            if isinstance(m, ClippedLinear):
                for name in ("input_min", "input_max", "output_min", "output_max"):
                    m._buffers[name] = None
                m.forward = m.linear.__call__

    def forward(self, pixel_values, max_soft_tokens=None):
        x = super().forward(pixel_values, max_soft_tokens=max_soft_tokens)
        std_bias = comfy.model_management.cast_to_device(self.std_bias, x.device, torch.float32)
        std_scale = comfy.model_management.cast_to_device(self.std_scale, x.device, torch.float32)
        return ((x.float() - std_bias) * std_scale).to(x.dtype)


class DiffusionGemmaEncoderModule(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.vision_tower = DiffusionGemmaVisionTower(config.vision_config, dtype=dtype, device=device, ops=ops)
        self.embed_vision = Gemma4RMSNormProjector(config.vision_config["hidden_size"], config.hidden_size, dtype=dtype, device=device, ops=ops)
        self.language_model = _EncoderLanguageModel(config.num_hidden_layers, device=device, dtype=dtype)


class DiffusionGemmaModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.decoder = DiffusionGemmaDecoder(config, device=device, dtype=dtype, ops=ops)
        self.encoder = DiffusionGemmaEncoderModule(config, device=device, dtype=dtype, ops=ops)

    @property
    def embed_tokens(self):
        return self.decoder.embed_tokens

    def get_past_len(self, past_key_values):
        for kv in past_key_values:
            if len(kv) >= 3:
                return kv[2]
        return 0

    def _cached_kv_lens(self, past_key_values):
        sliding_len = 0
        full_len = 0
        if past_key_values is not None:
            for i, layer in enumerate(self.decoder.layers):
                kv = past_key_values[i]
                if len(kv) == 0:
                    continue
                if layer.sliding_window is not None:
                    sliding_len = kv[0].shape[2]
                else:
                    full_len = kv[0].shape[2]
        return full_len, sliding_len

    def _encoder_masks(self, q_pos, full_len, sliding_len, sliding_window, dtype, device, mm_spans=None, padding_mask=None):
        seq_len = q_pos.shape[0]
        past_len = int(q_pos[0])
        min_val = torch.finfo(dtype).min

        q = q_pos[:, None]
        k_full = torch.arange(full_len + seq_len, device=device)[None, :]
        full_bool = k_full > q
        k_sliding = torch.cat((torch.arange(past_len - sliding_len, past_len, device=device), q_pos))[None, :]
        sliding_bool = (k_sliding > q) | ((q - k_sliding) >= sliding_window)

        if mm_spans is not None:
            for start, end in mm_spans:
                q_in = (q_pos >= start) & (q_pos < end)
                full_bool = full_bool & ~(q_in[:, None] & ((k_full[0] >= start) & (k_full[0] < end))[None, :])
                sliding_bool = sliding_bool & ~(q_in[:, None] & ((k_sliding[0] >= start) & (k_sliding[0] < end))[None, :])

        def to_mask(bool_mask):
            mask = torch.zeros((1, 1) + bool_mask.shape, dtype=dtype, device=device)
            mask.masked_fill_(bool_mask[None, None], min_val)
            return mask

        masks = {"full": to_mask(full_bool), "sliding": to_mask(sliding_bool)}
        if padding_mask is not None and past_len == 0:
            pad = torch.zeros((padding_mask.shape[0], 1, 1, padding_mask.shape[1]), dtype=dtype, device=device)
            pad.masked_fill_((padding_mask == 0)[:, None, None, :], min_val)
            masks = {k: v + pad for k, v in masks.items()}
        return masks

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=None,
                past_key_values=None, input_ids=None, mode="encoder", self_conditioning_logits=None,
                mm_spans=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.decoder.embed_tokens(x, out_dtype=dtype)

        if mode == "decoder":
            embed_module = self.decoder.embed_tokens
            if self_conditioning_logits is not None:
                if embed_module.comfy_cast_weights:
                    weight, _, offload_stream = comfy.ops.cast_bias_weight(embed_module, x, offloadable=True)
                else:
                    weight, offload_stream = embed_module.weight.to(device=x.device), None
                scale = torch.tensor(self.config.hidden_size ** 0.5, dtype=weight.dtype).item()
                soft_embeddings = torch.matmul(
                    self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(weight.dtype), weight) * scale
                comfy.ops.uncast_bias_weight(embed_module, weight, None, offload_stream)
                soft_embeddings = soft_embeddings.to(x.dtype)
            else:
                soft_embeddings = torch.zeros_like(x)
            x = self.decoder.self_conditioning(x, soft_embeddings)

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.decoder.compute_freqs_cis(position_ids, x.device, dtype=x.dtype)

        if mode == "decoder":
            masks = {"full": None, "sliding": None}
        else:
            if embeds_info and mm_spans is None:
                mm_spans = [(e["index"], e["index"] + e["size"]) for e in embeds_info if e.get("type") == "image"]
            full_len, sliding_len = self._cached_kv_lens(past_key_values)
            padding_mask = attention_mask if attention_mask is not None else None
            if seq_len == 1 and past_len > 0:
                masks = {"full": None, "sliding": None}
            else:
                masks = self._encoder_masks(position_ids[0], full_len, sliding_len,
                                            self.config.sliding_attention[0], x.dtype, x.device,
                                            mm_spans=mm_spans if mm_spans else None,
                                            padding_mask=padding_mask)

        update_cache = (mode == "encoder") and past_key_values is not None
        intermediate = None
        next_key_values = []
        for i, layer in enumerate(self.decoder.layers):
            past_kv = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None
            layer_scalar = None
            if mode == "encoder":
                layer_scalar = self.encoder.language_model.layers[i].layer_scalar
            mask = masks["sliding"] if layer.sliding_window is not None else masks["full"]
            x, current_kv = layer(x, attention_mask=mask, freqs_cis=freqs_cis, past_key_value=past_kv,
                                  update_cache=update_cache, layer_scalar=layer_scalar)
            next_key_values.append(current_kv if current_kv is not None else ())
            if i == intermediate_output:
                intermediate = x.clone()

        x = self.decoder.norm(x)
        return x, intermediate, next_key_values


class _StableAndConfidentStopping:
    def __init__(self, stability_threshold, confidence_threshold):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.argmax_canvas_history = None

    def __call__(self, argmax_canvas, logits):
        if self.stability_threshold == 0:
            stable = torch.ones((logits.shape[0]), device=logits.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1, dtype=argmax_canvas.dtype, device=argmax_canvas.device)
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas

        token_entropy = torch.distributions.Categorical(logits=logits).entropy()
        confident = torch.mean(token_entropy, dim=-1) < self.confidence_threshold
        return stable & confident


def _entropy_bound_accept(current_canvas, denoiser_canvas, logits, entropy_bound):
    token_entropy = torch.distributions.Categorical(logits=logits).entropy()
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
    accepted_token_mask = torch.scatter(
        input=torch.zeros_like(sorted_selection_mask), dim=-1, index=sorted_indices, src=sorted_selection_mask)
    accepted_canvas = torch.where(accepted_token_mask, denoiser_canvas, current_canvas)
    return accepted_canvas, accepted_token_mask


class DiffusionGenerate:
    def logits(self, x):
        module = self.model.decoder.embed_tokens
        offload_stream = None
        if module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, x, offloadable=True)
        else:
            weight = module.weight.to(x)
        logits = torch.nn.functional.linear(x, weight, None)
        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)

        logits = logits.to(torch.float32)
        cap = self.model.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        return [() for _ in range(self.model.config.num_hidden_layers)]

    def generate(self, embeds=None, max_length=256, seed=42, max_denoising_steps=48, entropy_bound=0.1,
                 t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005,
                 execution_dtype=None, mm_spans=None, stop_tokens=None, **kwargs):
        device = embeds.device
        config = self.model.config

        if stop_tokens is None:
            stop_tokens = config.stop_tokens
        if execution_dtype is None:
            if comfy.model_management.should_use_bf16(device):
                execution_dtype = torch.bfloat16
            else:
                execution_dtype = torch.float32
        embeds = embeds.to(execution_dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)

        canvas_length = config.canvas_length
        vocab_size = config.vocab_size
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        eos_tensor = torch.tensor(stop_tokens, device=device)

        past_key_values = self.init_kv_cache(embeds.shape[0], 0, device, execution_dtype)
        _, _, past_key_values = self.model(None, embeds=embeds, past_key_values=past_key_values,
                                           mode="encoder", mm_spans=mm_spans)
        cur_len = embeds.shape[1]

        max_new_canvases = math.ceil(max_length / canvas_length)
        pbar = comfy.utils.ProgressBar(max_new_canvases * max_denoising_steps)
        generated_token_ids = []
        commit_canvas = None

        for canvas_idx in range(max_new_canvases):
            if commit_canvas is not None:
                commit_embeds = self.model.decoder.embed_tokens(commit_canvas).to(execution_dtype)
                _, _, past_key_values = self.model(None, embeds=commit_embeds, past_key_values=past_key_values,
                                                   mode="encoder")

            current_canvas = torch.randint(low=0, high=vocab_size, size=(1, canvas_length),
                                           device=device, generator=generator)
            self_conditioning_logits = None
            argmax_canvas = current_canvas
            decoder_position_ids = torch.arange(cur_len, cur_len + canvas_length, device=device).unsqueeze(0)
            stopping = _StableAndConfidentStopping(stability_threshold, confidence_threshold)

            steps_done = 0
            for cur_step in reversed(range(1, max_denoising_steps + 1)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                x, _, _ = self.model(current_canvas, past_key_values=past_key_values, mode="decoder",
                                     self_conditioning_logits=self_conditioning_logits,
                                     position_ids=decoder_position_ids, dtype=execution_dtype)
                raw_logits = self.logits(x)

                temperature = t_min + ((t_max - t_min) * (cur_step / max_denoising_steps))
                processed_logits = raw_logits / temperature
                probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
                denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1, generator=generator)
                denoiser_canvas = denoiser_canvas.squeeze(-1).view(1, canvas_length)
                argmax_canvas = torch.argmax(processed_logits, dim=-1)

                accepted_canvas, accepted_mask = _entropy_bound_accept(
                    current_canvas, denoiser_canvas, processed_logits, entropy_bound)
                random_canvas = torch.randint(low=0, high=vocab_size, size=(1, canvas_length),
                                              device=device, generator=generator)
                current_canvas = torch.where(accepted_mask, accepted_canvas, random_canvas)

                finished_denoising = stopping(argmax_canvas, processed_logits)
                self_conditioning_logits = processed_logits.to(execution_dtype)
                steps_done += 1
                pbar.update(1)
                if torch.all(finished_denoising):
                    break

            if steps_done < max_denoising_steps:
                pbar.update(max_denoising_steps - steps_done)

            canvas_ids = argmax_canvas[0].tolist()
            is_eos = torch.isin(argmax_canvas[0], eos_tensor)
            eos_positions = is_eos.nonzero()
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                generated_token_ids.extend(canvas_ids[:first_eos + 1])
                break
            generated_token_ids.extend(canvas_ids)
            cur_len += canvas_length
            commit_canvas = argmax_canvas

        return generated_token_ids


class DiffusionGemma26B(BaseLlama, DiffusionGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = DiffusionGemmaConfig(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = DiffusionGemmaModel(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = embed.pop("data").movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            max_soft_tokens = embed.get("max_soft_tokens", None)
            vision_out = self.model.encoder.vision_tower(image.to(device, dtype=torch.float32), max_soft_tokens=max_soft_tokens)
            return self.model.encoder.embed_vision(vision_out), None
        return None, None


class DiffusionGemmaSDTokenizer(Gemma4SDTokenizer):
    embedding_size = 2816


class DiffusionGemmaTokenizer(Gemma4Tokenizer):
    tokenizer_class = DiffusionGemmaSDTokenizer


class DiffusionGemmaClipModel(Gemma4Model):
    model_class = DiffusionGemma26B
    config_overrides = {}

    def __init__(self, device="cpu", layer="all", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        if model_options.get("custom_operations") is None:
            model_options = model_options.copy()
            quant_config = model_options.get("quantization_metadata") or {}
            model_options["custom_operations"] = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)
        self.dtypes = set()
        self.dtypes.add(dtype)
        sd1_clip.SDClipModel.__init__(self, device=device, layer=layer, layer_idx=layer_idx,
                                      textmodel_json_config=dict(self.config_overrides), dtype=dtype,
                                      special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False,
                                      model_class=self.model_class, enable_attention_masks=attention_mask,
                                      return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, **kwargs):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = sd1_clip.SDClipModel.process_tokens(self, tokens_only, self.execution_device)
        mm_spans = [(e["index"], e["index"] + e["size"]) for e in embeds_info if e.get("type") == "image"]
        for k in ("do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty"):
            kwargs.pop(k, None)
        return self.transformer.generate(embeds=embeds, mm_spans=mm_spans if mm_spans else None, **kwargs)


def diffusion_gemma_detect(clip_data):
    sd = clip_data[0]
    out = {}
    norm = sd.get("model.decoder.norm.weight", None)
    if norm is not None:
        out["dtype_llama"] = norm.dtype
    quantization_metadata = comfy.utils.detect_layer_quantization(sd, "")
    if quantization_metadata is not None:
        out["llama_quantization_metadata"] = quantization_metadata
    if "model.decoder.layers.0.experts.gate_proj.weight" in sd:
        out["unfused_experts"] = True
    return out


def diffusion_gemma_te(dtype_llama=None, llama_quantization_metadata=None, unfused_experts=False):
    class DiffusionGemmaClipModel_(DiffusionGemmaClipModel):
        config_overrides = {"unfused_experts": unfused_experts}

    class DiffusionGemmaTEModel_(sd1_clip.SD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name="gemma4", clip_model=DiffusionGemmaClipModel_, model_options=model_options)

        def memory_estimation_function(self, tokens, device=None):
            # logits/softmax fp32 buffers + dequantized expert bank + tied-embed cast
            return 4 * 1024 * 1024 * 1024
    return DiffusionGemmaTEModel_
