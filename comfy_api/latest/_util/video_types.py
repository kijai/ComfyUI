from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional
from .._input import ImageInput, AudioInput

class VideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"
    H265_MAIN10 = "h265_main10"
    PRORES_4444 = "prores_4444"
    DNXHR_HQX = "dnxhr_hqx"
    DNXHR_444 = "dnxhr_444"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]

class VideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"
    MOV = "mov"
    MKV = "mkv"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of container names that can be used as node input.
        """
        return [member.value for member in cls]

    @classmethod
    def get_extension(cls, value) -> str:
        """
        Returns the file extension for the container.
        """
        if isinstance(value, str):
            value = cls(value)
        if value == VideoContainer.MOV:
            return "mov"
        if value == VideoContainer.MKV:
            return "mkv"
        if value == VideoContainer.MP4 or value == VideoContainer.AUTO:
            return "mp4"
        return ""

    def to_ffmpeg_format(self) -> str:
        """Format string accepted by av.open()/ffmpeg. MKV's user-facing name
        differs from the libavformat muxer name (matroska).
        """
        if self == VideoContainer.MKV:
            return "matroska"
        return self.value

@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """

    images: ImageInput
    frame_rate: Fraction
    audio: Optional[AudioInput] = None
    metadata: Optional[dict] = None


