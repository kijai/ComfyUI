from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
import comfy.text_encoders.llama


class T5GemmaTokenizerInner(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=3584, embedding_key='t5gemma_9b',
                         tokenizer_class=SPieceTokenizer, has_start_token=False, has_end_token=False,
                         pad_to_max_length=False, max_length=99999999, min_length=640,
                         pad_token=0, tokenizer_data=tokenizer_data,
                         tokenizer_args={"add_bos": False, "add_eos": False})

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class T5GemmaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="t5gemma_9b", tokenizer=T5GemmaTokenizerInner)


class T5GemmaClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=False, return_attention_mask=True, model_options={}):
        t5gemma_quantization_metadata = model_options.get("t5gemma_quantization_metadata", None)
        if t5gemma_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = t5gemma_quantization_metadata

        if return_attention_mask is None:
            return_attention_mask = attention_mask

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype,
                         special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False,
                         model_class=comfy.text_encoders.llama.T5Gemma_9B,
                         enable_attention_masks=attention_mask, return_attention_masks=return_attention_mask,
                         model_options=model_options)


class T5GemmaModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, name="t5gemma_9b", clip_model=T5GemmaClipModel,
                         model_options=model_options, **kwargs)


def te(dtype_t5gemma=None, t5gemma_quantization_metadata=None, attention_mask=True, return_attention_mask=True):
    class T5GemmaTEModel(T5GemmaModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5gemma_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = t5gemma_quantization_metadata
            if dtype_t5gemma is not None:
                dtype = dtype_t5gemma
            super().__init__(device=device, dtype=dtype, model_options=model_options, attention_mask=attention_mask, return_attention_mask=return_attention_mask)
    return T5GemmaTEModel
