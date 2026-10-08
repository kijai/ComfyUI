# SkinTokens / TokenRig (MIT) automatic rigging: https://github.com/VAST-AI-Research/SkinTokens
import numpy as np
import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
from comfy.ldm.hunyuan3d.vae import FourierEmbedder, fps
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import Mlp
from comfy.text_encoders.llama import FixedKV, Qwen3_06B
from . import tokenizer as tok

TOKENS_PER_SKIN = 4
FSQ_DIMS = 5
FSQ_LEVELS = 8
VOCAB_SIZE = tok.VOCAB_SIZE + FSQ_LEVELS ** FSQ_DIMS + 1
END = VOCAB_SIZE - 1
MAX_LENGTH = 2048  # prefix embeds + generated tokens, as in the reference demo


class Attention(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.qkv = operations.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.heads, preferred_attention=self.comfy_attention)
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.kv = operations.Linear(dim, dim * 2, bias=False, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, context):
        q = self.q(x)
        k, v = self.kv(context).chunk(2, dim=-1)
        x = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.heads, preferred_attention=self.comfy_attention)
        return self.proj(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn = Attention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = Mlp(dim, dim * 4, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm_context = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn = CrossAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = Mlp(dim, dim * 4, dtype=dtype, device=device, operations=operations)

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x), self.norm_context(context))
        return x + self.mlp(self.norm2(x))


class PhaseFourierEmbedder(nn.Module):
    """Phase-modulated sin/cos embedding (use_pmpe) of the skin VAE."""

    def __init__(self, num_freqs=8):
        super().__init__()
        i = torch.arange(1, num_freqs + 1, dtype=torch.float32)
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * torch.pi
        phase = (num_freqs ** (1.0 - i / num_freqs) + i / num_freqs) * torch.pi * 2
        # upstream runs these buffers in bf16 and the decoder was trained on the rounded values
        self.register_buffer("frequencies", frequencies.bfloat16().float(), persistent=False)
        self.register_buffer("phase", phase.bfloat16().float(), persistent=False)
        self.out_dim = 3 * (num_freqs * 2 + 1)

    def forward(self, x):
        embed = (x[..., None] * self.frequencies.to(x)).flatten(-2)
        phase = (x[..., None] * (torch.pi * 0.5) + self.phase.to(x)).flatten(-2)
        return torch.cat((x, embed.sin() + phase.sin(), embed.cos() + phase.cos()), dim=-1)


def fps_queries(points, num_queries, seed):
    """Random 4x oversample then farthest point sampling, as the upstream encoders pick their query points."""
    n = points.shape[0]
    subset = np.random.default_rng(seed).choice(n, num_queries * 4, replace=num_queries * 4 > n)
    sub = points[torch.from_numpy(subset)]
    picked = fps(sub, torch.zeros(sub.shape[0], dtype=torch.long), 0.25, start_random=False)
    return torch.from_numpy(subset)[picked]


class PointEncoder(nn.Module):
    """Michelangelo perceiver encoder; its 512 latents become the decoder prefix."""

    def __init__(self, width=512, heads=8, layers=8, num_queries=512, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_queries = num_queries
        self.embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.input_proj = operations.Linear(self.embedder.out_dim + 3, width, dtype=dtype, device=device)
        self.cross_attn = CrossAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([SelfAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations) for _ in range(layers)])
        self.norm_out = operations.LayerNorm(width, dtype=dtype, device=device)

    def forward(self, points, normals, query_idx, dtype):
        data = self.input_proj(torch.cat((self.embedder(points), normals), dim=-1).to(dtype))
        x = self.cross_attn(data[:, query_idx], data)
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class CondEncoder(nn.Module):
    def __init__(self, in_channels, width=768, heads=12, layers=2, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj_in = operations.Linear(in_channels, width, dtype=dtype, device=device)
        self.cross_attn = CrossAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([SelfAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations) for _ in range(layers)])
        self.norm_out = operations.LayerNorm(width, dtype=dtype, device=device)

    def forward(self, data, query_idx):
        data = self.proj_in(data)
        x = self.cross_attn(data[:, query_idx], data)
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class SkinDecoder(nn.Module):
    def __init__(self, in_channels, width=768, heads=12, layers=10, dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations) for _ in range(layers)])
        self.proj_query = operations.Linear(in_channels, width, dtype=dtype, device=device)
        self.cross_attn = CrossAttentionBlock(width, heads, dtype=dtype, device=device, operations=operations)
        self.norm_out = operations.LayerNorm(width, dtype=dtype, device=device)
        self.proj_out = operations.Linear(width, 1, dtype=dtype, device=device)

    def forward(self, z, queries):
        for block in self.blocks:
            z = block(z)
        q = self.proj_query(queries)
        out = [self.proj_out(self.norm_out(self.cross_attn(q, z[i:i + 1]))) for i in range(z.shape[0])]
        return torch.sigmoid(torch.cat(out, dim=0)[..., 0])


class SkinVAE(nn.Module):
    """Inference half of the FSQ-CVAE: mesh condition encoder and skin decoder."""

    def __init__(self, width=768, latent_channels=512, cond_tokens=384, dtype=None, device=None, operations=None):
        super().__init__()
        self.cond_tokens = cond_tokens
        self.embedder = PhaseFourierEmbedder()
        in_channels = self.embedder.out_dim + 3
        self.cond_encoder = CondEncoder(in_channels, width, dtype=dtype, device=device, operations=operations)
        self.cond_quant = operations.Linear(width, latent_channels, dtype=dtype, device=device)
        self.post_quant = operations.Linear(latent_channels, width, dtype=dtype, device=device)
        self.decoder = SkinDecoder(in_channels, width, dtype=dtype, device=device, operations=operations)
        self.fsq_out = operations.Linear(FSQ_DIMS, latent_channels, dtype=dtype, device=device)

    def embed(self, points, normals, dtype):
        return torch.cat((self.embedder(points), normals), dim=-1).to(dtype)

    def encode_cond(self, points, normals, query_idx, dtype):
        return self.cond_quant(self.cond_encoder(self.embed(points, normals, dtype), query_idx))

    def decode(self, codes, cond, queries):
        basis = FSQ_LEVELS ** torch.arange(FSQ_DIMS, device=codes.device)
        levels = (codes[..., None] // basis) % FSQ_LEVELS
        half = FSQ_LEVELS // 2
        z = self.fsq_out(((levels - half) / half).to(cond.dtype))
        z = self.post_quant(torch.cat((z, cond.expand(z.shape[0], -1, -1)), dim=1))
        return self.decoder(z, queries)


def reorder_cache(cache, beam_idx, start):
    # in place, so captured decode graphs keep their KV buffers; rows before start are the same in every beam
    for kv in cache:
        if isinstance(kv, FixedKV):
            kv.key[:, start:kv.index] = kv.key[beam_idx, start:kv.index]
            kv.value[:, start:kv.index] = kv.value[beam_idx, start:kv.index]
        else:
            key, value, length = kv
            key[:, :, start:length] = key[beam_idx, :, start:length]
            value[:, :, start:length] = value[beam_idx, :, start:length]


def grammar_masks(device):
    """Allowed tokens per skeleton grammar state, then the skin vocabulary, then the forced end token."""
    allowed = torch.zeros((tok.EXPECT_JOINT + 3, VOCAB_SIZE), dtype=torch.bool, device=device)
    for state in range(tok.EXPECT_JOINT + 1):
        allowed[state, tok.allowed_tokens(state)] = True
    allowed[-2, tok.EOS:END] = True  # the reference opens everything from the skeleton EOS up, except the end token
    allowed[-1, END] = True
    return allowed


class SkinTokens(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.point_encoder = PointEncoder(dtype=dtype, device=device, operations=operations)
        self.point_proj = operations.Linear(512, 896, dtype=dtype, device=device)
        self.point_norm = operations.RMSNorm(896, eps=torch.finfo(torch.float32).eps, dtype=dtype, device=device)
        self.transformer = Qwen3_06B({"hidden_size": 896, "vocab_size": VOCAB_SIZE, "max_position_embeddings": 3192,
                                      "merged_qkv": True, "merged_mlp": True}, dtype, device, operations)
        llm = self.transformer.model
        llm.prefetch_dynamic_vbars = True
        llm.graph_dynamic_vbar_blocks = True
        self.skin_vae = SkinVAE(dtype=dtype, device=device, operations=operations)

    def get_dynamic_vram__units(self):
        return self.transformer.model.get_dynamic_vram__units()

    def encode(self, points, normals, device, dtype, seed):
        """points, normals: (N, 3) float32 CPU tensors in the normalized Z-up frame."""
        point_query = fps_queries(points, self.point_encoder.num_queries, 0).to(device)
        cond_query = fps_queries(points, self.skin_vae.cond_tokens, seed).to(device)
        points, normals = points[None].to(device), normals[None].to(device)

        prefix = self.point_encoder(points, normals, point_query, dtype)
        prefix = self.point_norm(self.point_proj(prefix))
        cond = self.skin_vae.encode_cond(points, normals, cond_query, dtype)
        # upstream embeds the decoder queries from compute dtype positions, which is also how the decoder was trained
        queries = self.skin_vae.embed(points.to(dtype), normals, dtype)
        return prefix, cond, queries

    @staticmethod
    def process_scores(scores, seen, grammars, masks, min_keep, do_sample, top_k, top_p, temperature, repetition_penalty):
        # transformers' order: repetition penalty, vocab switching mask, temperature, top-k, top-p
        if repetition_penalty != 1.0:
            penalized = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            scores = torch.where(seen, penalized, scores)
        rows = []
        for g in grammars:
            if g.skin_tokens is None:
                rows.append(g.state)
            elif g.skin_tokens + 1 == g.num_bones * TOKENS_PER_SKIN:
                rows.append(-1)
            else:
                rows.append(-2)
        scores = scores.masked_fill(~masks[torch.tensor(rows, device=scores.device)], -float("inf"))
        if not do_sample:
            return scores
        if temperature != 1.0:
            scores = scores / temperature
        if top_k > 0:
            kth = torch.topk(scores, min(max(top_k, min_keep), scores.shape[-1]))[0][..., -1, None]
            scores = scores.masked_fill(scores < kth, -float("inf"))
        if top_p < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=False)
            remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
            remove[..., -min_keep:] = False
            scores = scores.masked_fill(remove.scatter(1, sorted_idx, remove), -float("inf"))
        return scores

    def _decode_step(self, tokens, cache, logits, decode_buffers, dtype):
        """One token per sequence into the preallocated logits, in a memory compiler region so the layers replay as CUDA graphs."""
        llm = self.transformer
        comfy.model_prefetch.malloc_graph_begin(logits.device)
        embeds = llm.model.embed_tokens(tokens, out_dtype=dtype)
        x, _, cache = llm.model.forward(None, embeds=embeds, past_key_values=cache, decode_buffers=decode_buffers)
        logits.copy_(llm.logits(x)[:, -1])
        del x, embeds
        comfy.model_prefetch.malloc_graph_end()
        return cache

    def _sample(self, start, max_new, cache, logits, decode_buffers, dtype, masks, generator, do_sample, **warp_args):
        """transformers _sample(): the processors see raw logits."""
        grammar = tok.SkeletonGrammar()
        for t in start:
            grammar.push(t)
        seen = torch.zeros((1, VOCAB_SIZE), dtype=torch.bool, device=logits.device)
        out = []
        pbar = comfy.utils.ProgressBar(max_new)
        for _ in comfy.utils.model_trange(max_new, desc="SkinTokens decoding"):
            scores = self.process_scores(logits, seen, [grammar], masks, 1, do_sample, **warp_args)
            if do_sample:
                token = torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)
            else:
                token = scores.argmax(dim=-1, keepdim=True)
            seen[0, token[0]] = True
            out.append(token.item())
            grammar.push(out[-1])
            pbar.update(1)
            if out[-1] == END:
                break
            cache = self._decode_step(token, cache, logits, decode_buffers, dtype)
        return out

    def _beam_search(self, start, max_new, cache, logits, decode_buffers, dtype, masks, generator, num_beams, do_sample, **warp_args):
        """transformers _beam_search() for one sequence: processors see log-probs, 2 * num_beams candidates per step."""
        device = logits.device
        grammar = tok.SkeletonGrammar()
        for t in start:
            grammar.push(t)
        grammars = [grammar.copy() for _ in range(num_beams)]
        seen = torch.zeros((num_beams, VOCAB_SIZE), dtype=torch.bool, device=device)
        running = torch.full((num_beams, max_new), tok.PAD, dtype=torch.long, device=device)
        running_scores = torch.full((num_beams,), -1e9, device=device)
        running_scores[0] = 0.0
        finished = running.clone()
        finished_scores = torch.full((num_beams,), -1e9, device=device)
        finished_len = torch.zeros((num_beams,), dtype=torch.long, device=device)
        is_finished = torch.zeros((num_beams,), dtype=torch.bool, device=device)
        num_keep = 2 * num_beams
        top_mask = torch.arange(num_keep, device=device) < num_beams
        beam_range = torch.arange(num_beams, device=device)
        prefill = MAX_LENGTH - max_new  # shared by all beams, never reordered

        pbar = comfy.utils.ProgressBar(max_new)
        for cur in comfy.utils.model_trange(max_new, desc="SkinTokens decoding"):
            scores = self.process_scores(logits.log_softmax(dim=-1), seen, grammars, masks, 2, do_sample, **warp_args)
            scores = (scores + running_scores[:, None]).flatten()
            if do_sample:
                idx = torch.multinomial(scores.softmax(dim=-1), num_keep, generator=generator)
                cand_scores = scores[idx]
            else:
                cand_scores, idx = torch.topk(scores, num_keep)
            src = idx // VOCAB_SIZE
            cand = running[src]
            cand[:, cur] = idx % VOCAB_SIZE
            hits = (cand[:, cur] == END) | (cur + 1 >= max_new)

            # continue with the best candidates that did not finish
            alive_scores = cand_scores + hits * -1e9
            sel = torch.topk(alive_scores, num_beams)[1]
            beam_src = src[sel]
            running = cand[sel]
            running_scores = alive_scores[sel]

            # keep the best finished hypotheses, length normalized
            just_finished = hits & top_mask
            fin_scores = cand_scores / (cur + 1) + (~just_finished) * -1e9
            merged = torch.topk(torch.cat((finished_scores, fin_scores)), num_beams)[1]
            finished = torch.cat((finished, cand))[merged]
            finished_scores = torch.cat((finished_scores, fin_scores))[merged]
            finished_len = torch.cat((finished_len, torch.full((num_keep,), cur + 1, device=device)))[merged]
            is_finished = torch.cat((is_finished, just_finished))[merged]

            reorder_cache(cache, beam_src, prefill)
            seen = seen[beam_src]
            seen[beam_range, running[:, cur]] = True
            tokens = running[:, cur].tolist()
            grammars = [grammars[b].copy() for b in beam_src.tolist()]
            for g, t in zip(grammars, tokens):
                g.push(t)
            pbar.update(1)

            worst_finished = torch.where(is_finished, finished_scores.min(), -1e9)
            if not bool((running_scores[0] / (cur + 1) > worst_finished).any()) or bool(hits.all()):
                break
            cache = self._decode_step(running[:, cur:cur + 1], cache, logits, decode_buffers, dtype)
        return finished[0, :finished_len[0]].tolist()

    def generate(self, prefix, cls, num_beams=10, do_sample=True, top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0, seed=0):
        """Same decoding as transformers generate() with the reference VocabSwitchingLogitsProcessor. Returns token ids."""
        llm = self.transformer
        device = prefix.device
        start = [tok.BOS, tok.CLS.get(cls, tok.CLS_NONE)]
        start_embeds = llm.model.embed_tokens(torch.tensor([start], device=device), out_dtype=prefix.dtype)
        embeds = torch.cat((prefix, start_embeds), dim=1).repeat(num_beams, 1, 1)
        max_new = MAX_LENGTH - embeds.shape[1]
        llm.model.fixed_kv = prefix.dtype == torch.bfloat16  # the flash decode kernel is bf16 only
        cache = llm.init_kv_cache(num_beams, MAX_LENGTH, device, prefix.dtype)
        logits = torch.empty((num_beams, VOCAB_SIZE), device=device)
        decode_buffers = None
        if comfy.model_prefetch.malloc_graph_enabled(device) and not comfy.model_management.args.disable_cuda_graphs:
            decode_buffers = llm.model.init_decode_buffers(num_beams, device, prefix.dtype)
        masks = grammar_masks(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        warp_args = dict(top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)
        try:
            x, _, cache = llm.model.forward(None, embeds=embeds, past_key_values=cache)
            logits.copy_(llm.logits(x)[:, -1])
            del x
            if num_beams == 1:
                return start + self._sample(start, max_new, cache, logits, decode_buffers, prefix.dtype, masks, generator, do_sample, **warp_args)
            return start + self._beam_search(start, max_new, cache, logits, decode_buffers, prefix.dtype, masks, generator, num_beams, do_sample, **warp_args)
        finally:
            comfy.model_prefetch.cleanup_prefetch_queues()

    def rig(self, points, normals, device, dtype, cls="articulation", seed=0, **generate_args):
        """Returns joints (J, 3), parents (J,) and skinning weights (N, J) on the sampled points."""
        prefix, cond, queries = self.encode(points, normals, device, dtype, seed)
        ids = self.generate(prefix, cls, seed=seed, **generate_args)
        if tok.EOS not in ids:
            raise RuntimeError("SkinTokens: the skeleton did not finish within the token limit")
        split = ids.index(tok.EOS) + 1
        joints, parents = tok.detokenize(np.array(ids[:split]))
        codes = ids[split:split + len(joints) * TOKENS_PER_SKIN]
        if len(codes) != len(joints) * TOKENS_PER_SKIN:
            raise RuntimeError("SkinTokens: the skinning tokens did not finish within the token limit")
        # the reference forces the end token one step early; it lands as the last code and wraps to code 0
        codes = torch.tensor(codes, device=device).view(len(joints), TOKENS_PER_SKIN) - tok.VOCAB_SIZE
        skin = self.skin_vae.decode(codes, cond, queries)
        return joints, parents, skin.float().T.cpu().numpy()
