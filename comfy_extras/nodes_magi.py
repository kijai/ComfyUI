import nodes
import torch
import comfy.nested_tensor
import comfy.model_management
from comfy_api.latest import io
from typing_extensions import override
from comfy_api.latest import ComfyExtension


class EmptyMagiAudioLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMagiAudioLatent",
            category="latent/video/magi",
            description="Generate random audio latent noise for MagiHuman. Length should match the video frame count (e.g. seconds * 25 + 1).",
            inputs=[
                io.Int.Input("length", default=251, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of audio frames (= video frames, typically seconds * 25 + 1)"),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, length, batch_size=1) -> io.NodeOutput:
        # Audio latent: (batch, num_frames, 64) - random noise, matching original pipeline
        audio_latent = torch.randn(
            [batch_size, length, 64],
            device=comfy.model_management.intermediate_device()
        )
        return io.NodeOutput({"samples": audio_latent})


class MagiConcatAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiConcatAVLatent",
            category="latent/video/magi",
            description="Combine video and audio latents for MagiHuman joint denoising",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[
                io.Latent.Output(display_name="av_latent"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        output = {}
        output.update(video_latent)
        output.update(audio_latent)

        video_noise_mask = video_latent.get("noise_mask", None)
        audio_noise_mask = audio_latent.get("noise_mask", None)
        if video_noise_mask is not None or audio_noise_mask is not None:
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_latent["samples"])
            if audio_noise_mask is None:
                audio_noise_mask = torch.ones_like(audio_latent["samples"])
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

        # VAE produces audio as (B, channels, T), model expects (B, T, channels)
        audio_samples = audio_latent["samples"]
        if audio_samples.ndim == 3 and audio_samples.shape[1] == 64:
            audio_samples = audio_samples.transpose(1, 2)
        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_samples))
        return io.NodeOutput(output)


class MagiPrepareAudioSR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiPrepareAudioSR",
            category="latent/video/magi",
            description="Prepare audio latent for MagiHuman SR: mix with noise and freeze (set denoise mask to 0)",
            inputs=[
                io.Latent.Input("audio_latent"),
                io.Float.Input("noise_scale", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Fraction of noise to mix in (0 = keep original, 1 = pure noise)"),
            ],
            outputs=[
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, audio_latent, noise_scale=0.7) -> io.NodeOutput:
        output = audio_latent.copy()
        samples = audio_latent["samples"]
        if noise_scale > 0:
            noise = torch.randn_like(samples)
            samples = noise * noise_scale + samples * (1 - noise_scale)
        output["samples"] = samples
        output["noise_mask"] = torch.zeros_like(samples)
        return io.NodeOutput(output)


class MagiSeparateAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiSeparateAVLatent",
            category="latent/video/magi",
            description="Separate combined MagiHuman latent into video and audio",
            inputs=[
                io.Latent.Input("av_latent"),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        latents = av_latent["samples"].unbind()
        video_latent = av_latent.copy()
        video_latent["samples"] = latents[0]
        audio_latent = av_latent.copy()
        # Model uses (B, T, channels), VAE uses (B, channels, T)
        audio_latent["samples"] = latents[1].transpose(1, 2)
        if "noise_mask" in av_latent:
            masks = av_latent["noise_mask"]
            if masks is not None:
                masks = masks.unbind()
                video_latent["noise_mask"] = masks[0]
                audio_latent["noise_mask"] = masks[1]
        return io.NodeOutput(video_latent, audio_latent)


class MagiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyMagiAudioLatent,
            MagiConcatAVLatent,
            MagiPrepareAudioSR,
            MagiSeparateAVLatent,
        ]

async def comfy_entrypoint() -> MagiExtension:
    return MagiExtension()
