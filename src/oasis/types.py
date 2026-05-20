from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import numpy as np
from dataclasses import dataclass
import torch


@dataclass
class FramePacket:
    t: float
    idx: int
    frame: np.ndarray


@dataclass
class AudioPacket:
    t: float
    samples: np.ndarray


class EventNode(BaseModel):
    id: str
    t_start: float
    t_end: float
    level: int = 0
    summary: str = ""
    summary_embedding: torch.Tensor = None
    keywords: List[str] = Field(default_factory=list)
    frames: list = Field(default_factory=list)
    asr_text: str = ""
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("frames")
    def _serialize_frames(self, frames):
        return [p.idx for p in frames]

    @field_serializer("summary_embedding")
    def _serialize_summary_embedding(self, summary_embedding):
        return summary_embedding.shape


class QANode(BaseModel):
    id: str
    t: float
    question: str
    answer: str
    QA_embedding: torch.Tensor = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("QA_embedding")
    def _serialize_QA_embedding(self, QA_embedding):
        return QA_embedding.shape


class MergeRecord(BaseModel):
    parent_id: str
    left_id: str
    right_id: str
    t_start: float
    t_end: float
    reason: str


class ForestSnapshot(BaseModel):
    timestamp: float
    root_ids: List[str]
    num_nodes: int
    QA_ids: List[str]
    num_QAs: int
    QA_summary_all: str
