"""
Convert raw DiffusionGemma-26B-A4B (HuggingFace shards) to a ComfyUI text encoder file.

Keys are kept in HF naming (model.decoder.*, model.encoder.*); the only structural change
is renaming the fused expert banks to <bank>.weight to match comfy.ops.MoEExperts, plus
embedding tokenizer.json. fp8 mode quantizes the expert banks (per-expert scale) and the
large 2D text-backbone weights (per-tensor scale, max_value=416 convention).

Usage:
    python convert_diffusion_gemma.py <hf_snapshot_dir> <out.safetensors> [--bf16]
"""

import os
import sys
import json
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file

snapshot = sys.argv[1]
output = sys.argv[2]
bf16_only = "--bf16" in sys.argv

out_dtype = torch.float8_e4m3fn
inf = torch.finfo(out_dtype)
max_value = 416

EXPERT_BANK_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")


def quant_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def should_quantize_2d(key, v):
    return (key.startswith("model.decoder.layers.") and key.endswith(".weight") and v.dim() == 2
            and "norm" not in key and max(v.shape) >= 4096)


def quantize_2d(key, w):
    w = w.float()
    scale = torch.max(torch.abs(w)) / max_value
    w_q = (w / scale).clamp(min=inf.min, max=inf.max).to(dtype=out_dtype)
    return [
        (key, w_q),
        (key.replace(".weight", ".weight_scale"), scale),
        (key.replace(".weight", ".comfy_quant"), quant_tensor({"format": "float8_e4m3fn"})),
    ]


def quantize_bank(key, w):
    # [E, out, in] -> per-expert scale [E]
    w = w.float()
    scale = torch.amax(torch.abs(w), dim=(1, 2)) / max_value
    w_q = (w / scale[:, None, None]).clamp(min=inf.min, max=inf.max).to(dtype=out_dtype)
    return [
        (key + ".weight", w_q),
        (key + ".weight_scale", scale),
        (key + ".comfy_quant", quant_tensor({"format": "float8_e4m3fn", "num_experts": w.shape[0]})),
    ]


sd_new = {}
n_quant = 0
shards = sorted(glob.glob(os.path.join(snapshot, "model-*-of-*.safetensors")))
assert shards, f"no shards found in {snapshot}"
for shard in shards:
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            if k.startswith("lm_head."):
                continue
            if k.endswith(EXPERT_BANK_SUFFIXES):
                if bf16_only:
                    sd_new[k + ".weight"] = v
                else:
                    for out_k, out_v in quantize_bank(k, v):
                        sd_new[out_k] = out_v
                    n_quant += 1
            elif not bf16_only and should_quantize_2d(k, v):
                for out_k, out_v in quantize_2d(k, v):
                    sd_new[out_k] = out_v
                n_quant += 1
            else:
                sd_new[k] = v
            del v
    print(f"processed {os.path.basename(shard)}")

with open(os.path.join(snapshot, "tokenizer.json"), "rb") as tf:
    sd_new["tokenizer_json"] = torch.tensor(list(tf.read()), dtype=torch.uint8)

total = sum(t.numel() * t.element_size() for t in sd_new.values())
print(f"Quantized {n_quant} weights -> {len(sd_new)} tensors, {total / 1024**3:.2f} GB")

save_file(sd_new, output)
print(f"Saved to {output}")
