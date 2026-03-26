"""
Convert a HuggingFace T5Gemma checkpoint to ComfyUI-compatible safetensors.

Extracts the encoder weights from the full encoder-decoder checkpoint,
remaps keys to match ComfyUI's internal naming, and embeds the SentencePiece
tokenizer model.

Usage:
    python convert_t5gemma.py <hf_model_dir> <output_path>

Example:
    python convert_t5gemma.py S:/AI/comfy_models/text_encoders/t5gemma9b S:/AI/comfy_models/text_encoders/t5gemma9b_comfyui.safetensors
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file


# Key remapping: HF T5Gemma -> ComfyUI
#   model.encoder.X -> model.X
#   pre_self_attn_layernorm -> input_layernorm
#   post_self_attn_layernorm -> post_attention_layernorm
LAYERNORM_REMAP = {
    "pre_self_attn_layernorm": "input_layernorm",
    "post_self_attn_layernorm": "post_attention_layernorm",
}


def remap_key(key: str) -> str | None:
    """Remap a single HF key to ComfyUI format. Returns None if the key should be skipped."""
    # Only keep encoder weights
    if not key.startswith("model.encoder."):
        return None

    # Strip model.encoder. -> model.
    new_key = "model." + key[len("model.encoder."):]

    # Remap layer norm names
    for old, new in LAYERNORM_REMAP.items():
        new_key = new_key.replace(old, new)

    return new_key


def convert(hf_model_dir: str, output_path: str, dtype: str | None = None):
    index_path = os.path.join(hf_model_dir, "model.safetensors.index.json")
    tokenizer_path = os.path.join(hf_model_dir, "tokenizer.model")

    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer.model not found at {tokenizer_path}")
        sys.exit(1)

    # Determine which shard files contain encoder weights
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Find shards that have encoder keys
        encoder_shards = set()
        encoder_keys_by_shard: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith("model.encoder."):
                encoder_shards.add(shard)
                encoder_keys_by_shard.setdefault(shard, []).append(key)

        total_encoder_keys = sum(len(v) for v in encoder_keys_by_shard.values())
        print(f"Found {total_encoder_keys} encoder keys across {len(encoder_shards)} shards")
    else:
        # Single file
        single_path = os.path.join(hf_model_dir, "model.safetensors")
        if not os.path.exists(single_path):
            print(f"Error: no model.safetensors or index found in {hf_model_dir}")
            sys.exit(1)
        encoder_shards = {"model.safetensors"}
        encoder_keys_by_shard = None  # load all, filter later

    # Cast dtype
    cast_dtype = None
    if dtype == "fp16":
        cast_dtype = torch.float16
    elif dtype == "bf16":
        cast_dtype = torch.bfloat16
    elif dtype == "fp32":
        cast_dtype = torch.float32

    # Load and remap encoder weights
    output_sd: dict[str, torch.Tensor] = {}
    for shard_name in sorted(encoder_shards):
        shard_path = os.path.join(hf_model_dir, shard_name)
        print(f"Loading {shard_name}...")
        shard_sd = load_file(shard_path)

        for key, tensor in shard_sd.items():
            new_key = remap_key(key)
            if new_key is None:
                continue
            if cast_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(cast_dtype)
            output_sd[new_key] = tensor

    print(f"Remapped {len(output_sd)} encoder weight tensors")

    # Embed the SentencePiece tokenizer model
    with open(tokenizer_path, "rb") as f:
        spiece_data = f.read()
    output_sd["spiece_model"] = torch.ByteTensor(list(spiece_data))
    print(f"Embedded tokenizer.model ({len(spiece_data)} bytes)")

    # Verify key structure
    expected_layers = 42
    for i in range(expected_layers):
        prefix = f"model.layers.{i}."
        layer_keys = [k for k in output_sd if k.startswith(prefix)]
        if len(layer_keys) == 0:
            print(f"Warning: no keys found for layer {i}")

    norm_key = "model.norm.weight"
    if norm_key in output_sd:
        w = output_sd[norm_key]
        print(f"Final norm shape: {w.shape}, dtype: {w.dtype}")
    else:
        print(f"Warning: {norm_key} not found")

    embed_key = "model.embed_tokens.weight"
    if embed_key in output_sd:
        w = output_sd[embed_key]
        print(f"Embeddings shape: {w.shape}, dtype: {w.dtype}")

    # Save
    print(f"Saving to {output_path}...")
    save_file(output_sd, output_path)

    # Report size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Output size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert HF T5Gemma checkpoint to ComfyUI format")
    parser.add_argument("hf_model_dir", help="Path to HF T5Gemma model directory")
    parser.add_argument("output_path", help="Output .safetensors file path")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None,
                        help="Cast weights to this dtype (default: keep original)")
    args = parser.parse_args()

    convert(args.hf_model_dir, args.output_path, args.dtype)


if __name__ == "__main__":
    main()
