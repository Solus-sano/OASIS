import cv2
import numpy as np
from typing import List
from ..utils.logging import get_logger
from ..types import FramePacket

logger = get_logger(__name__)


def uniform_keyframes(pkt_lst: List[FramePacket], max_frames: int) -> List[FramePacket]:
    if len(pkt_lst) <= max_frames:
        return pkt_lst
    step = max(1, len(pkt_lst) // max_frames)
    return pkt_lst[::step][:max_frames]
