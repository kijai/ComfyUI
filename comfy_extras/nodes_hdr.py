from __future__ import annotations

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


# ARRI LogC3 EI 800 constants.
_LOGC3_A = 5.555556
_LOGC3_B = 0.052272
_LOGC3_C = 0.247190
_LOGC3_D = 0.385537
_LOGC3_E = 5.367655
_LOGC3_F = 0.092809
_LOGC3_CUT = 0.010591


def _logc3_decompress(logc: torch.Tensor) -> torch.Tensor:
    logc = logc.clamp(0.0, 1.0)
    cut_log = _LOGC3_E * _LOGC3_CUT + _LOGC3_F
    lin_from_log = (10.0 ** ((logc - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A
    lin_from_lin = (logc - _LOGC3_F) / _LOGC3_E
    return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)


def _reinhard(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)


def _aces_narkowicz(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


def _apply_saturation(x: torch.Tensor, sat: float) -> torch.Tensor:
    if sat == 1.0:
        return x
    luma = 0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3]
    return luma + (x - luma) * sat


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x.clamp(min=0.0031308), 1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


# PQ (SMPTE ST.2084) OETF constants.
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875


def _linear_to_pq(linear: torch.Tensor, peak_nits: float) -> torch.Tensor:
    """PQ (ST.2084) OETF: scene-linear nits → perceptually-encoded [0, 1]."""
    normalized = (linear / peak_nits).clamp(0.0, 1.0)
    x_m1 = torch.pow(normalized, _PQ_M1)
    return torch.pow((_PQ_C1 + _PQ_C2 * x_m1) / (1.0 + _PQ_C3 * x_m1), _PQ_M2)


class HDRDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HDRDecode",
            display_name="HDR Decode",
            search_aliases=["logc3", "hdr decode", "hdr to sdr", "tonemap", "reinhard", "aces"],
            category="image/hdr",
            description=(
                "Decode HDR model output. Input is typically LogC3-encoded; "
                "select `linear` if the input is already scene-linear HDR. "
                "Output: scene-linear HDR (for EXR save), display-ready sRGB "
                "(for h264/PNG), or PQ-encoded (for HDR10 H.265 Main 10 delivery)."
            ),
            inputs=[
                io.Image.Input("image", tooltip="HDR image. For LogC3 input, values in [0, 1]."),
                io.Combo.Input("input_type", options=["logc3", "linear"], default="logc3",
                    tooltip="logc3: ARRI LogC3-encoded [0, 1] input (apply inverse curve). linear: already scene-linear HDR, passthrough.",
                ),
                io.DynamicCombo.Input("output",
                    tooltip=(
                        "srgb: exposure + tonemap + sRGB gamma for display / h264 save. "
                        "hdr_linear: scene-linear HDR for EXR save. "
                        "pq: PQ (ST.2084) encoded for HDR10 H.265 Main 10 delivery."
                    ),
                    options=[
                        io.DynamicCombo.Option("srgb", [
                            io.Combo.Input("tonemap", options=["reinhard", "aces"], default="reinhard",
                                tooltip="reinhard: x / (1+x), aces: Narkowicz 2015 filmic approximation.",
                            ),
                            io.Float.Input("exposure", default=0.0, min=-10.0, max=10.0, step=0.1,
                                tooltip="Exposure in stops. 0 = neutral, +1 = 2x brighter, -1 = half.",
                            ),
                            io.Float.Input("saturation", default=1.0, min=0.0, max=2.0, step=0.05,
                                tooltip="Post-tonemap saturation (BT.709 luma). 1.0 = neutral. Useful to compensate for tonemap desaturating highlights.",
                            ),
                        ]),
                        io.DynamicCombo.Option("hdr_linear", []),
                        io.DynamicCombo.Option("pq", [
                            io.Float.Input("reference_white", default=203.0, min=10.0, max=1000.0, step=1.0,
                                tooltip="Nits that scene-linear 1.0 maps to. 100 = BT.2100 PQ spec diffuse white. 203 = ITU-R BT.2408 / Dolby recommendation (most common HDR grading reference). Higher = brighter overall output.",
                            ),
                        ]),
                    ],
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    _MAX_LUMINANCE = 10000.0 # PQ HDR10 spec peak

    @classmethod
    def execute(cls, image: torch.Tensor, input_type: str, output: dict) -> io.NodeOutput:
        linear = image.float()
        if input_type == "logc3":
            linear = _logc3_decompress(linear)
            linear = linear.clamp(min=0.0, max=cls._MAX_LUMINANCE)
        else:
            linear = linear.clamp(min=0.0)

        if output["output"] == "hdr_linear":
            return io.NodeOutput(linear)

        if output["output"] == "pq":
            # Scene-linear is relative (1.0 = scene white). Multiply by reference_white
            # to put it in absolute nits before PQ encoding. _linear_to_pq clamps to
            # [0, 1] internally after normalizing
            nits = linear * output["reference_white"]
            return io.NodeOutput(_linear_to_pq(nits, cls._MAX_LUMINANCE))

        exposed = linear * (2.0 ** output["exposure"])
        method = output["tonemap"]
        if method == "reinhard":
            tonemapped = _reinhard(exposed)
        elif method == "aces":
            tonemapped = _aces_narkowicz(exposed)
        else:
            raise ValueError(f"Unknown tonemap method: {method}")
        tonemapped = _apply_saturation(tonemapped, output["saturation"])
        return io.NodeOutput(_linear_to_srgb(tonemapped.clamp(0.0, 1.0)))


class HDRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [HDRDecode]


async def comfy_entrypoint() -> HDRExtension:
    return HDRExtension()
