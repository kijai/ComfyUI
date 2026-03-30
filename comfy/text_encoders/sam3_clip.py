from comfy import sd1_clip

SAM3_CLIP_CONFIG = {
    "architectures": ["CLIPTextModel"],
    "hidden_act": "quick_gelu",
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "max_position_embeddings": 32,
    "projection_dim": 512,
    "vocab_size": 49408,
    "layer_norm_eps": 1e-5,
    "eos_token_id": 49407,
}


class SAM3ClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, max_length=32, layer="last", textmodel_json_config=SAM3_CLIP_CONFIG, special_tokens={"start": 49406, "end": 49407, "pad": 0}, return_projected_pooled=True, return_attention_masks=True, enable_attention_masks=True, model_options=model_options)
        self.transformer.text_projection = self.operations.Linear(1024, 512, bias=False, dtype=dtype, device=device)


class SAM3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(max_length=32, pad_with_end=False, pad_token=0, embedding_directory=embedding_directory, embedding_size=1024, embedding_key="sam3_clip", tokenizer_data=tokenizer_data)


class SAM3TokenizerWrapper(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="l", tokenizer=SAM3Tokenizer, name="sam3_clip")


class SAM3ClipModelWrapper(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, clip_name="l", clip_model=SAM3ClipModel, name="sam3_clip")
