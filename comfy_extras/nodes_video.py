from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Optional
from typing_extensions import override
from fractions import Fraction
import comfy.utils
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
            search_aliases=["export webm"],
            category="image/video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if cls.hidden.prompt is not None:
            container.metadata["prompt"] = json.dumps(cls.hidden.prompt)

        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo",
            search_aliases=["export video"],
            display_name="Save Video",
            category="image/video",
            essentials_category="Basics",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=["auto", "mp4"], default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=["auto", "h264"], default="auto", tooltip="The codec to use for the video."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


def _align_mask_to_image(mask: torch.Tensor, image_batch: int) -> torch.Tensor:
    """Match mask batch to image batch; partial masks are zero-padded instead of looped."""
    mask_batch = mask.shape[0]
    if 1 < mask_batch < image_batch:
        pad = torch.zeros(image_batch - mask_batch, *mask.shape[1:], dtype=mask.dtype, device=mask.device)
        return torch.cat([mask, pad], dim=0)
    return comfy.utils.repeat_to_batch_size(mask, image_batch)


def _image_with_alpha(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate mask onto a (B, H, W, 3) image as an alpha channel."""
    mask = _align_mask_to_image(mask, image.shape[0])
    alpha = mask.to(image.dtype).to(image.device).clamp(0.0, 1.0).unsqueeze(-1)
    return torch.cat([image, alpha], dim=-1)


def _write_exr_sequence(image: torch.Tensor, path_pattern: str, frame_rate, mask: torch.Tensor | None = None) -> None:
    """Write an IMAGE tensor as a float32 EXR image sequence via FFmpeg's image2 muxer.
    When `mask` is provided (shape (B, H, W) or (1, H, W)) its values become the alpha channel;
    otherwise alpha is 1.0.
    """
    import numpy as np

    batch, height, width, _ = image.shape
    pix_fmt = "gbrapf32le"
    itemsize = 4
    default_alpha = np.ones((height, width), dtype=np.float32)
    rate = Fraction(round(float(frame_rate) * 1000), 1000)
    if mask is not None:
        mask = _align_mask_to_image(mask, batch)

    with av.open(path_pattern, mode="w", format="image2") as output:
        stream = output.add_stream("exr", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt

        for i in range(batch):
            gbr = image[i][..., [1, 2, 0]].permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32)
            if mask is not None:
                alpha = mask[i].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            else:
                alpha = default_alpha
            planes_data = [gbr[0], gbr[1], gbr[2], alpha]

            # from_ndarray doesn't support planar float formats; write planes directly.
            av_frame = av.VideoFrame(width, height, pix_fmt)
            for plane_idx in range(4):
                plane = av_frame.planes[plane_idx]
                line_samples = plane.line_size // itemsize
                if line_samples == width:
                    plane.update(planes_data[plane_idx].tobytes())
                else:
                    strided = np.zeros((height, line_samples), dtype=np.float32)
                    strided[:, :width] = planes_data[plane_idx]
                    plane.update(strided.tobytes())
            for packet in stream.encode(av_frame):
                output.mux(packet)

        for packet in stream.encode(None):
            output.mux(packet)


class SaveVideoAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoAdvanced",
            search_aliases=["save video advanced", "prores", "dnxhr", "exr", "hdr video", "hdr master"],
            display_name="Save Video (Advanced)",
            category="image/video",
            description=(
                "Save an image sequence with a pro codec. ProRes 4444 / DNxHR 444 / "
                "DNxHR HQX produce 10-bit MOV video files. H.265 Main 10 produces "
                "10-bit MP4 (HDR10-capable delivery). EXR writes a float32 image "
                "sequence (one .exr per frame) with optional mask as alpha."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI",
                    tooltip="Prefix for the output file(s). Supports %date:...% and %node.field% expansion.",
                ),
                io.DynamicCombo.Input("codec",
                    tooltip=(
                        "prores_4444 / dnxhr_444: 10-bit 4:4:4 mastering (MOV). "
                        "dnxhr_hqx: 10-bit 4:2:2 broadcast / HDR10 delivery (MOV). "
                        "h265_main10: 10-bit 4:2:0 H.265 for HDR playback / streaming (MP4). "
                        "exr: float32 linear-HDR image sequence (no clamping, values > 1.0 preserved)."
                    ),
                    options=[
                        io.DynamicCombo.Option("dnxhr_444", []),
                        io.DynamicCombo.Option("dnxhr_hqx", []),
                        io.DynamicCombo.Option("h265_main10", [
                            io.Combo.Input("container", options=["mp4", "mkv"], default="mp4",
                                tooltip="MP4 for broadest compatibility (TVs, iOS, browsers). MKV for wider format flexibility and better handling of arbitrary timestamps.",
                            ),
                            io.Combo.Input("hdr_tagging", options=["off", "hdr10", "hlg"], default="off",
                                tooltip=(
                                    "off: no color tags (SDR or untagged HDR). "
                                    "hdr10: BT.2020 + PQ (ST.2084) — standard HDR10 delivery. "
                                    "hlg: BT.2020 + HLG (ARIB STD-B67) — broadcast / YouTube HLG. "
                                    "Only set for actual HDR content; mis-tagging SDR causes wrong display on HDR TVs."
                                ),
                            ),
                        ]),
                        io.DynamicCombo.Option("prores_4444", [
                            io.Mask.Input("mask", optional=True,
                                tooltip="Optional alpha channel. Values in [0,1] become the ProRes 4444 alpha plane. Extra mask frames are sliced, missing frames are padded with zero (transparent) alpha.",
                            ),
                        ]),
                        io.DynamicCombo.Option("exr", [
                            io.Mask.Input("mask", optional=True,
                                tooltip="Optional alpha channel. Values in [0,1] become EXR alpha. Extra mask frames are sliced, missing frames are padded with zero (transparent) alpha.",
                            ),
                        ]),
                    ],
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, codec: dict) -> io.NodeOutput:
        codec_name = codec["codec"]
        mask = codec.get("mask")
        width, height = video.get_dimensions()

        if mask is not None and mask.shape[-2:] != (height, width):
            raise ValueError(
                f"mask resolution {tuple(mask.shape[-2:])} does not match video "
                f"resolution ({height}, {width})."
            )

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)

        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata

        if codec_name == "exr":
            components = video.get_components()
            file_pattern = f"{filename}_{counter:05}_%04d.exr"
            _write_exr_sequence(
                components.images.float(),
                os.path.join(full_output_folder, file_pattern),
                float(components.frame_rate),
                mask=mask.float() if mask is not None else None,
            )
            return io.NodeOutput(ui=ui.PreviewVideo([]))

        # HDR tags go into x265-params for H.265; the actual RGB→YUV matrix is
        # BT.709 (PyAV reformat doesn't expose bt2020) — subtle chroma shift
        encoder_options = None
        if codec_name == "h265_main10":
            container = Types.VideoContainer(codec.get("container", "mp4"))
            hdr_tagging = codec.get("hdr_tagging", "off")
            if hdr_tagging == "hdr10":
                # BT.2020 primaries / D65 WP; L(max,min) in 0.0001-nit units → 10000/0.0001 nits.
                encoder_options = {
                    "x265-params": (
                        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                        "hdr-opt=1:repeat-headers=1:"
                        "master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(100000000,1)"
                    ),
                }
            elif hdr_tagging == "hlg":
                encoder_options = {
                    "x265-params": "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:repeat-headers=1",
                }
        else:
            container = Types.VideoContainer.MOV

        if mask is not None:
            # Splice alpha plane onto images; preserves original audio / fps.
            orig = video.get_components()
            video = InputImpl.VideoFromComponents(Types.VideoComponents(
                images=_image_with_alpha(orig.images, mask),
                audio=orig.audio,
                frame_rate=orig.frame_rate,
            ))

        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(container)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=container,
            codec=Types.VideoCodec(codec_name),
            metadata=saved_metadata,
            encoder_options=encoder_options,
        )
        # Some codecs can't preview anything meaningful in the browser, todo: make frontend clear the preview on invalid input
        if codec_name == "h265_main10":
            return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
        return io.NodeOutput(ui=ui.PreviewVideo([]))


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
            search_aliases=["images to video"],
            display_name="Create Video",
            category="image/video",
            description="Create a video from images.",
            inputs=[
                io.Image.Input("images", tooltip="The images to create a video from."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None) -> io.NodeOutput:
        return io.NodeOutput(
            InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
            search_aliases=["extract frames", "split video", "video to images", "demux"],
            display_name="Get Video Components",
            category="image/video",
            description="Extracts all components from a video: frames, audio, and framerate.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract components from."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        components = video.get_components()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate))


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo",
            search_aliases=["import video", "open video", "video file"],
            display_name="Load Video",
            category="image/video",
            essentials_category="Basics",
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

class VideoSlice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Video Slice",
            display_name="Video Slice",
            search_aliases=[
                "trim video duration",
                "skip first frames",
                "frame load cap",
                "start time",
            ],
            category="image/video",
            essentials_category="Video Tools",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input(
                    "start_time",
                    default=0.0,
                    max=1e5,
                    min=-1e5,
                    step=0.001,
                    tooltip="Start time in seconds",
                ),
                io.Float.Input(
                    "duration",
                    default=0.0,
                    min=0.0,
                    step=0.001,
                    tooltip="Duration in seconds, or 0 for unlimited duration",
                ),
                io.Boolean.Input(
                    "strict_duration",
                    default=False,
                    tooltip="If True, when the specified duration is not possible, an error will be raised.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, start_time: float, duration: float, strict_duration: bool) -> io.NodeOutput:
        trimmed = video.as_trimmed(start_time, duration, strict_duration=strict_duration)
        if trimmed is not None:
            return io.NodeOutput(trimmed)
        raise ValueError(
            f"Failed to slice video:\nSource duration: {video.get_duration()}\nStart time: {start_time}\nTarget duration: {duration}"
        )


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            SaveVideoAdvanced,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
            VideoSlice,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
