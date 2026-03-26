import nodes
import torch
import numpy as np
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
                io.Int.Input("length", default=251, min=1, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Number of audio frames (= video frames, typically seconds * 25 + 1)"),
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

        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))
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
        audio_latent["samples"] = latents[1]
        if "noise_mask" in av_latent:
            masks = av_latent["noise_mask"]
            if masks is not None:
                masks = masks.unbind()
                video_latent["noise_mask"] = masks[0]
                audio_latent["noise_mask"] = masks[1]
        return io.NodeOutput(video_latent, audio_latent)


def _build_magi_sigmas(num_steps, shift_scale=5.0, num_timesteps=1000):
    """Build MagiHuman's ZeroSNRDDPMDiscretization sigma schedule.
    Returns ascending sigmas [~0, ..., ~0.75, 0.0] for step_ddim."""
    betas = np.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod = alphas_cumprod / (shift_scale + (1 - shift_scale) * alphas_cumprod)
    acp_sqrt = np.sqrt(alphas_cumprod)

    timesteps = np.linspace(num_timesteps - 1, 0, num_steps, endpoint=False).astype(int)[::-1]
    selected = acp_sqrt[timesteps]
    s0, sT = selected[0], selected[-1]
    normalized = (selected - sT) * s0 / (s0 - sT)

    # Flip to ascending (clean→noisy), append final 0 (denoise to clean)
    sigmas = np.flip(normalized).copy()
    sigmas = np.append(sigmas, 0.0)
    return torch.from_numpy(sigmas).float()


class MagiSigmasNode(io.ComfyNode):
    """Generate MagiHuman's ZeroSNRDDPM sigma schedule."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiSigmas",
            category="sampling/custom_sampling/schedulers",
            description="MagiHuman ZeroSNRDDPM sigma schedule (ascending clean-to-noisy-to-clean)",
            inputs=[
                io.Int.Input("steps", default=8, min=1, max=100),
                io.Float.Input("shift_scale", default=5.0, min=0.1, max=20.0, step=0.1),
            ],
            outputs=[
                io.Sigmas.Output("sigmas"),
            ],
        )

    @classmethod
    def execute(cls, steps, shift_scale=5.0) -> io.NodeOutput:
        sigmas = _build_magi_sigmas(steps, shift_scale)
        return io.NodeOutput(sigmas)


class MagiSamplerNode(io.ComfyNode):
    """Sampler matching MagiHuman's step_ddim: at each step, predict x_0 from
    velocity, then re-noise to next sigma level with fresh noise."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiSampler",
            category="sampling/custom_sampling/samplers",
            description="MagiHuman step_ddim sampler (stochastic, adds fresh noise each step)",
            inputs=[],
            outputs=[
                io.Sampler.Output("sampler"),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        from comfy.samplers import KSAMPLER
        sampler = KSAMPLER(_sample_magi_step_ddim)
        return io.NodeOutput(sampler)


def _sample_magi_step_ddim(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """MagiHuman step_ddim: stochastic sampler for model without timestep conditioning.

    Always passes sigma=1.0 to model so calculate_denoised gives x_0 = x - 1.0*v
    (the model's direct prediction). Uses the actual sigmas only for re-noising weights.
    """
    from tqdm.auto import trange
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            # Re-noise using actual schedule sigma for mixing weight
            noise = torch.randn_like(x)
            x = sigmas[i + 1] * noise + (1 - sigmas[i + 1]) * denoised
    return x


class MagiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyMagiAudioLatent,
            MagiConcatAVLatent,
            MagiSeparateAVLatent,
            MagiSigmasNode,
            MagiSamplerNode,
        ]

async def comfy_entrypoint() -> MagiExtension:
    return MagiExtension()
