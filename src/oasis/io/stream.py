from __future__ import annotations
import os
os.environ["DECORD_EOF_RETRY_MAX"] = "204800"
import time
import numpy as np
from typing import Iterator, Optional
from ..utils.logging import get_logger
import torch
import decord
from ..types import FramePacket, AudioPacket
from ..config import OasisConfig
from torchvision import io, transforms

logger = get_logger(__name__)


class VideoStreamer:
    def __init__(self, path: str, pace: float = 1.0, fps: Optional[float] = 1):
        try:
            self.video = _read_video_decord(path, fps)
        except Exception as e1:
            logger.warning(f"decord failed, fallback to torchvision: {e1}")
            self.video = _read_video_torchvision(path, fps)
        self.frame_lst = [frame for frame in self.video]
        self.fps = fps
        self.dt = 1.0 / self.fps
        self.pace = pace
        self.idx = 0
        self._t0 = time.time()
        self._sim_t = 0.0

    @classmethod
    def from_config(cls, video_path: str, cfg: OasisConfig) -> VideoStreamer:
        return cls(video_path, pace=cfg.stream.pace, fps=cfg.stream.fps)

    def __iter__(self) -> Iterator[FramePacket]:
        return self

    def __len__(self) -> int:
        return len(self.frame_lst)

    def __next__(self) -> FramePacket:
        if self.idx >= len(self.frame_lst):
            raise StopIteration
        frame = self.frame_lst[self.idx]
        t = self.idx * self.dt
        pkt = FramePacket(t=t, idx=self.idx, frame=frame)
        self.idx += 1
        if self.pace > 0:
            elapsed = time.time() - self._t0
            target = t / self.pace
            if target > elapsed:
                time.sleep(target - elapsed)
        return pkt


class AudioStreamer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.t = 0.0

    @classmethod
    def from_config(cls, cfg: OasisConfig) -> AudioStreamer:
        return cls(sample_rate=cfg.asr.sample_rate)

    def pull(self, seconds: float) -> AudioPacket:
        n = int(self.sample_rate * seconds)
        samples = np.zeros(n, dtype=np.float32)
        pkt = AudioPacket(t=self.t, samples=samples)
        self.t += seconds
        return pkt


def _read_video_decord(video_path: str, fps: float) -> torch.Tensor:
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = int(total_frames / video_fps * fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    logger.info(f"decord: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    logger.info(f"sample_fps: {fps}, sample_frames: {nframes}")
    return video


def _read_video_torchvision(video_path: str, fps: float) -> torch.Tensor:
    st = time.time()
    video, audio, info = io.read_video(
        video_path, pts_unit="sec", output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = int(total_frames / video_fps * fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    logger.info(f"sample_fps: {fps}, sample_frames: {nframes}")
    return video
