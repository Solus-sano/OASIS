from __future__ import annotations
from typing import List
from ..types import FramePacket
from ..utils.logging import get_logger
from ..config import OasisConfig

logger = get_logger(__name__)


class ShortMemory:
    def __init__(
        self,
        buffer_frames_limit: int = 64,
        now_window_frames_limit: int = 32,
        buffer_fps: float = 1.0,
    ):
        self.buffer_frames_limit = buffer_frames_limit
        self.now_window_frames_limit = now_window_frames_limit
        self.buffer_fps = buffer_fps
        self.now_window: List[FramePacket] = []
        self.buf: List[FramePacket] = []
        self.buf_for_event: List[FramePacket] = []

    @classmethod
    def from_config(cls, cfg: OasisConfig) -> ShortMemory:
        return cls(
            buffer_frames_limit=cfg.short_memory.buffer_frames_limit,
            now_window_frames_limit=cfg.short_memory.now_window_frames_limit,
            buffer_fps=cfg.short_memory.buffer_fps,
        )

    def reset(self):
        self.buf = []
        self.buf_for_event = []
        self.now_window = []

    def push(self, pkt: FramePacket) -> List[List[FramePacket]]:
        self.now_window.append(pkt)

        dt = 99999
        if len(self.buf) > 0:
            dt = pkt.t - self.buf[-1].t
        if dt >= 1.0 / self.buffer_fps:
            self.buf.append(pkt)
            self.buf_for_event.append(pkt)
        cuts = []

        if len(self.buf) >= self.buffer_frames_limit:
            self.buf.pop(0)

        if len(self.now_window) >= self.now_window_frames_limit:
            self.now_window.pop(0)

        if len(self.buf_for_event) >= self.buffer_frames_limit:
            cuts.append(self.buf_for_event)
            self.buf_for_event = []

        return cuts
