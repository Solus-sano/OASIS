from pydantic import BaseModel, Field
from typing import Optional

class StreamConfig(BaseModel):
    pace: float = Field(1.0, description=">1.0 = faster than realtime; 0 for as-fast-as-possible")
    fps: Optional[float] = Field(1.0, description="the fps to extract frames")

class EventNodeConfig(BaseModel):
    max_frames_per_node: int = None
    tokens_per_frame: int = None

class ShortMemoryConfig(BaseModel):
    buffer_frames_limit: int = None
    buffer_fps: float = None
    now_window_frames_limit: int = None

class EventForestConfig(BaseModel):
    root_cnt_limit: int = None
    rag_event_retrieve_limit: int = 4
    rag_qa_retrieve_limit: int = 4

class ASRConfig(BaseModel):
    backend: str = "none"
    sample_rate: int = 16000
    chunk_seconds: float = 2.0

class MLLMConfig(BaseModel):
    model_path: str = ""
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 0.001

class LLMConfig(BaseModel):
    model_path: str = ""
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 0.001

class EmbeddingModelConfig(BaseModel):
    model_path: str = ""

class OasisConfig(BaseModel):
    stream: StreamConfig = StreamConfig()
    event_node: EventNodeConfig = EventNodeConfig()
    short_memory: ShortMemoryConfig = ShortMemoryConfig()
    event_forest: EventForestConfig = EventForestConfig()
    asr: ASRConfig = ASRConfig()
    mllm: MLLMConfig = MLLMConfig()
    llm: LLMConfig = LLMConfig()
    embedding_model: EmbeddingModelConfig = EmbeddingModelConfig()
