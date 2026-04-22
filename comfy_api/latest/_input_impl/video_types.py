from __future__ import annotations
from av.container import InputContainer
from av.subtitles.stream import SubtitleStream
from fractions import Fraction
from typing import Optional
from .._input import AudioInput, VideoInput
import av
import io
import itertools
import json
import numpy as np
import math
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents


# rgb48le input is required for ProRes/DNxHR to carry real 10-bit precision;
# rgb24 pre-quantizes to 8-bit before sws_scale.
_CODEC_SETTINGS: dict[VideoCodec, dict] = {
    VideoCodec.H264: {
        "encoder": "h264",
        "input_pix_fmt": "rgb24",
        "pix_fmt": "yuv420p",
        "options": {},
        "default_container": VideoContainer.MP4,
    },
    VideoCodec.PRORES_4444: {
        "encoder": "prores_ks",
        "input_pix_fmt": "rgb48le",
        "pix_fmt": "yuv444p10le",
        "alpha_input_pix_fmt": "rgba64le",
        "alpha_pix_fmt": "yuva444p10le",
        "options": {"profile": "4"},
        "default_container": VideoContainer.MOV,
    },
    VideoCodec.DNXHR_HQX: {
        "encoder": "dnxhd",
        "input_pix_fmt": "rgb48le",
        "pix_fmt": "yuv422p10le",
        "options": {"profile": "dnxhr_hqx"},
        "default_container": VideoContainer.MOV,
    },
    VideoCodec.DNXHR_444: {
        "encoder": "dnxhd",
        "input_pix_fmt": "rgb48le",
        "pix_fmt": "yuv444p10le",
        "options": {"profile": "dnxhr_444"},
        "default_container": VideoContainer.MOV,
    },
    VideoCodec.H265_MAIN10: {
        "encoder": "libx265",
        "input_pix_fmt": "rgb48le",
        "pix_fmt": "yuv420p10le",
        "options": {"profile": "main10"},
        "default_container": VideoContainer.MP4,
    },
}


def container_to_output_format(container_format: str | None) -> str | None:
    """
    A container's `format` may be a comma-separated list of formats.
    E.g., iso container's `format` may be `mov,mp4,m4a,3gp,3g2,mj2`.
    However, writing to a file/stream with `av.open` requires a single format,
    or `None` to auto-detect.
    """
    if not container_format:
        return None  # Auto-detect

    if "," not in container_format:
        return container_format

    formats = container_format.split(",")
    return formats[0]

def get_open_write_kwargs(
    dest: str | io.BytesIO, container_format: str, to_format: str | None
) -> dict:
    """Get kwargs for writing a `VideoFromFile` to a file/stream with `av.open`"""
    open_kwargs = {
        "mode": "w",
        # If isobmff, preserve custom metadata tags (workflow, prompt, extra_pnginfo)
        "options": {"movflags": "use_metadata_tags"},
    }

    is_write_to_buffer = isinstance(dest, io.BytesIO)
    if is_write_to_buffer:
        # Set output format explicitly, since it cannot be inferred from file extension
        if to_format == VideoContainer.AUTO:
            to_format = container_format.lower()
        elif isinstance(to_format, str):
            to_format = to_format.lower()
        open_kwargs["format"] = container_to_output_format(to_format)

    return open_kwargs


class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO, *, start_time: float=0, duration: float=0):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.__file = file
        self.__start_time = start_time
        self.__duration = duration

    def get_stream_source(self) -> str | io.BytesIO:
        """
        Return the underlying file source for efficient streaming.
        This avoids unnecessary memory copies when the source is already a file path.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        return self.__file

    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            for stream in container.streams:
                if stream.type == 'video':
                    assert isinstance(stream, av.VideoStream)
                    return stream.width, stream.height
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        raw_duration = self._get_raw_duration()
        if self.__start_time < 0:
            duration_from_start = min(raw_duration, -self.__start_time)
        else:
            duration_from_start = raw_duration - self.__start_time
        if self.__duration:
            return min(self.__duration, duration_from_start)
        return duration_from_start

    def _get_raw_duration(self) -> float:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode="r") as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)

            # Fallback: calculate from frame count and frame rate
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream and video_stream.frames and video_stream.average_rate:
                return float(video_stream.frames / video_stream.average_rate)

            # Last resort: decode frames to count them
            if video_stream and video_stream.average_rate:
                frame_count = 0
                container.seek(0)
                frame_iterator = (
                    container.decode(video_stream)
                    if video_stream.codec.capabilities & 0x100
                    else container.demux(video_stream)
                )
                for packet in frame_iterator:
                    frame_count += 1
                if frame_count > 0:
                    return float(frame_count / video_stream.average_rate)

        raise ValueError(f"Could not determine duration for file '{self.__file}'")

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video without materializing them as
        torch tensors.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # 1. Prefer the frames field if available and usable
            if (
                video_stream.frames
                and video_stream.frames > 0
                and not self.__start_time
                and not self.__duration
            ):
                return int(video_stream.frames)

            # 2. Try to estimate from duration and average_rate using only metadata
            if (
                getattr(video_stream, "duration", None) is not None
                and getattr(video_stream, "time_base", None) is not None
                and video_stream.average_rate
            ):
                raw_duration = float(video_stream.duration * video_stream.time_base)
                if self.__start_time < 0:
                    duration_from_start = min(raw_duration, -self.__start_time)
                else:
                    duration_from_start = raw_duration - self.__start_time
                duration_seconds = min(self.__duration, duration_from_start)
                estimated_frames = int(round(duration_seconds * float(video_stream.average_rate)))
                if estimated_frames > 0:
                    return estimated_frames

            # 3. Last resort: decode frames and count them (streaming)
            if self.__start_time < 0:
                start_time = max(self._get_raw_duration() + self.__start_time, 0)
            else:
                start_time = self.__start_time
            frame_count = 1
            start_pts = int(start_time / video_stream.time_base)
            end_pts = int((start_time + self.__duration) / video_stream.time_base)
            container.seek(start_pts, stream=video_stream)
            frame_iterator = (
                container.decode(video_stream)
                if video_stream.codec.capabilities & 0x100
                else container.demux(video_stream)
            )
            for frame in frame_iterator:
                if frame.pts >= start_pts:
                    break
            else:
                raise ValueError(f"Could not determine frame count for file '{self.__file}'\nNo frames exist for start_time {self.__start_time}")
            for frame in frame_iterator:
                if frame.pts >= end_pts:
                    break
                frame_count += 1
            return frame_count

    def get_frame_rate(self) -> Fraction:
        """
        Returns the average frame rate of the video using container metadata
        without decoding all frames.
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)

        with av.open(self.__file, mode="r") as container:
            video_stream = self._get_first_video_stream(container)
            # Preferred: use PyAV's average_rate (usually already a Fraction-like)
            if video_stream.average_rate:
                return Fraction(video_stream.average_rate)

            # Fallback: estimate from frames + duration if available
            if video_stream.frames and container.duration:
                duration_seconds = float(container.duration / av.time_base)
                if duration_seconds > 0:
                    return Fraction(video_stream.frames / duration_seconds).limit_denominator()

            # Last resort: match get_components_internal default
            return Fraction(1)

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)
        with av.open(self.__file, mode='r') as container:
            return container.format.name

    def get_components_internal(self, container: InputContainer) -> VideoComponents:
        video_stream = self._get_first_video_stream(container)
        if self.__start_time < 0:
            start_time = max(self._get_raw_duration() + self.__start_time, 0)
        else:
            start_time = self.__start_time
        # Get video frames
        frames = []
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + self.__duration) / video_stream.time_base)
        container.seek(start_pts, stream=video_stream)
        for frame in container.decode(video_stream):
            if frame.pts < start_pts:
                continue
            if self.__duration and frame.pts >= end_pts:
                break
            img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
            img = torch.from_numpy(img) / 255.0  # shape: (H, W, 3)
            frames.append(img)

        images = torch.stack(frames) if len(frames) > 0 else torch.zeros(0, 3, 0, 0)

        # Get frame rate
        frame_rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)

        # Get audio if available
        audio = None
        container.seek(start_pts, stream=video_stream)
        # Use last stream for consistency
        if len(container.streams.audio):
            audio_stream = container.streams.audio[-1]
            audio_frames = []
            resample = av.audio.resampler.AudioResampler(format='fltp').resample
            frames = itertools.chain.from_iterable(
                map(resample, container.decode(audio_stream))
            )

            has_first_frame = False
            for frame in frames:
                offset_seconds = start_time - frame.pts * audio_stream.time_base
                to_skip = max(0, int(offset_seconds * audio_stream.sample_rate))
                if to_skip < frame.samples:
                    has_first_frame = True
                    break
            if has_first_frame:
                audio_frames.append(frame.to_ndarray()[..., to_skip:])

            for frame in frames:
                if self.__duration and frame.time > start_time + self.__duration:
                    break
                audio_frames.append(frame.to_ndarray())  # shape: (channels, samples)
            if len(audio_frames) > 0:
                audio_data = np.concatenate(audio_frames, axis=1)  # shape: (channels, total_samples)
                if self.__duration:
                    audio_data = audio_data[..., :int(self.__duration * audio_stream.sample_rate)]

                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, channels, total_samples)
                audio = AudioInput({
                    "waveform": audio_tensor,
                    "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
                })

        metadata = container.metadata
        return VideoComponents(images=images, audio=audio, frame_rate=frame_rate, metadata=metadata)

    def get_components(self) -> VideoComponents:
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            return self.get_components_internal(container)
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        encoder_options: Optional[dict] = None,
    ):
        if isinstance(self.__file, io.BytesIO):
            self.__file.seek(0)  # Reset the BytesIO object to the beginning
        with av.open(self.__file, mode='r') as container:
            container_format = container.format.name
            video_encoding = container.streams.video[0].codec.name if len(container.streams.video) > 0 else None
            reuse_streams = True
            if format != VideoContainer.AUTO and format not in container_format.split(","):
                reuse_streams = False
            if codec != VideoCodec.AUTO and codec != video_encoding and video_encoding is not None:
                reuse_streams = False
            if self.__start_time or self.__duration:
                reuse_streams = False
            # encoder_options only applies to the re-encode path; forcing it
            # would otherwise require re-encoding when we'd rather just remux.
            if encoder_options:
                reuse_streams = False

            if not reuse_streams:
                components = self.get_components_internal(container)
                video = VideoFromComponents(components)
                return video.save_to(
                    path,
                    format=format,
                    codec=codec,
                    metadata=metadata,
                    encoder_options=encoder_options,
                )

            streams = container.streams

            open_kwargs = get_open_write_kwargs(path, container_format, format)
            with av.open(path, **open_kwargs) as output_container:
                # Copy over the original metadata
                for key, value in container.metadata.items():
                    if metadata is None or key not in metadata:
                        output_container.metadata[key] = value

                # Add our new metadata
                if metadata is not None:
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            output_container.metadata[key] = value
                        else:
                            output_container.metadata[key] = json.dumps(value)

                # Add streams to the new container
                stream_map = {}
                for stream in streams:
                    if isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream)):
                        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
                        stream_map[stream] = out_stream

                # Write packets to the new container
                for packet in container.demux():
                    if packet.stream in stream_map and packet.dts is not None:
                        packet.stream = stream_map[packet.stream]
                        output_container.mux(packet)

    def _get_first_video_stream(self, container: InputContainer):
        if len(container.streams.video):
            return container.streams.video[0]
        raise ValueError(f"No video stream found in file '{self.__file}'")

    def as_trimmed(
        self, start_time: float = 0, duration: float = 0, strict_duration: bool = True
    ) -> VideoInput | None:
        trimmed = VideoFromFile(
            self.get_stream_source(),
            start_time=start_time + self.__start_time,
            duration=duration,
        )
        if trimmed.get_duration() < duration and strict_duration:
            return None
        return trimmed


class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, components: VideoComponents):
        self.__components = components

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate,
        )

    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        encoder_options: Optional[dict] = None,
    ):
        """Save the video to a file path or BytesIO buffer.

        `encoder_options` merges into the codec's baseline options on the video
        stream (e.g. `x265-params` for HDR10 tagging).
        """
        if isinstance(codec, str):
            codec = VideoCodec(codec)
        if codec == VideoCodec.AUTO:
            codec = VideoCodec.H264
        settings = _CODEC_SETTINGS.get(codec)
        if settings is None:
            raise ValueError(f"Unsupported codec: {codec}")

        if isinstance(format, str):
            format = VideoContainer(format)
        explicit_format = format != VideoContainer.AUTO
        if format == VideoContainer.AUTO:
            format = settings["default_container"]

        # Force the muxer explicitly for BytesIO (no extension to infer from)
        # and when the caller asked for a specific container. For AUTO + file
        # path, let ffmpeg infer from the extension so callers that rely on
        # extension-based muxer selection keep working.
        extra_kwargs = {}
        if isinstance(path, io.BytesIO) or explicit_format:
            extra_kwargs["format"] = format.to_ffmpeg_format()
        with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}, **extra_kwargs) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)
            # Create a video stream
            video_stream = output.add_stream(settings["encoder"], rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]
            # Route 4-channel input to the codec's alpha pix_fmts if it supports them.
            has_alpha = self.__components.images.shape[-1] == 4 and "alpha_pix_fmt" in settings
            if has_alpha:
                input_pix_fmt = settings["alpha_input_pix_fmt"]
                stream_pix_fmt = settings["alpha_pix_fmt"]
            else:
                input_pix_fmt = settings["input_pix_fmt"]
                stream_pix_fmt = settings["pix_fmt"]
            video_stream.pix_fmt = stream_pix_fmt
            merged_options = {**settings["options"], **(encoder_options or {})}
            if merged_options:
                video_stream.options = merged_options

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                audio_sample_rate = int(self.__components.audio['sample_rate'])
                waveform = self.__components.audio['waveform']
                waveform = waveform[0, :, :math.ceil((audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                audio_stream = output.add_stream('aac', rate=audio_sample_rate, layout=layout)

            for i, frame in enumerate(self.__components.images):
                if input_pix_fmt in ("rgb48le", "rgba64le"):
                    img = frame.clamp(0.0, 1.0).mul_(65535.0).round_().cpu().numpy().astype(np.uint16)
                else:
                    img = (frame * 255).clamp_(0, 255).byte().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(img, format=input_pix_fmt)
                frame = frame.reformat(format=stream_pix_fmt)
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.__components.audio:
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                frame.sample_rate = audio_sample_rate
                frame.pts = 0
                output.mux(audio_stream.encode(frame))

                # Flush encoder
                output.mux(audio_stream.encode(None))

    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = True,
    ) -> VideoInput | None:
        if self.get_duration() < start_time + duration:
            return None
        #TODO Consider tracking duration and trimming at time of save?
        return VideoFromFile(self.get_stream_source(), start_time=start_time, duration=duration)
