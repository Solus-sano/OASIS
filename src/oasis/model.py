from __future__ import annotations
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .config import OasisConfig
from .event.forest import EventForest, time_to_hhmmss
from .types import EventNode, QANode
from .utils.logging import get_logger
from .event.segmenter import ShortMemory
from .types import FramePacket, AudioPacket
from typing import List, Optional
import torch
from qwen_vl_utils import smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import re
import json
import time

logger = get_logger(__name__)

QUERY_SYSTEM_PROMPT = """
You are an expert multimodal assistant on virtual reality headset for streaming video QA.
You are looking at a video of a real-world scene, and you are answering real-time questions for user

## Inputs (separate fields)
- now_window_frames: fine-grained frames from NowWindow, very recent video frames that the headset is looking at.
- short_term_frames: fine-grained frames from ShortTermWindow.
- long_term_events: textual summaries (and/or coarse frames) of longer segments.
- qa_history_summary: summary of prior Q/A.

## Decision Policy
- Read the user question + provided frames/summaries, briefly think step-by-step about the question
- After thinking, call `rag_retrieval` with a precise interest description, the tool will return the specific video clips and question-and-answer history that are most relevant to your description.
- Briefly think step-by-step based on all the information, answer the question inside <answer> and </answer>.

## Forming the Retrieval Query
- Be specific and brief (≤ 20 words), prefer noun phrases.
- Include disambiguators when available: who/what, action, location/region, salient attributes (color/count).
- Avoid vague queries ("more info", "look again").

## Tool
You may fetch missing details by issuing a concise interest description (entity, action, time anchor, location, attributes).
<tools>
{
  "type": "function",
  "function": {
    "name_for_human": "rag_retrieval",
    "name": "rag_retrieval",
    "description": "Retrieve details based on a concise interest description.",
    "parameters": {
      "type": "object",
      "properties": {
        "text_input": {
          "type": "string",
          "description": "Short, specific description to retrieve (e.g., 'woman picks up red bottle near fridge')."
        }
      },
      "required": ["text_input"]
    }
  }
}
</tools>

## Tool Call Format (example)
<tool_call>
{"name": "rag_retrieval", "arguments": {"text_input": "man opens car trunk, parking lot"}}
</tool_call>
"""

TOOL_CALL_PROMPT = """
Based on all the above information, answer the question inside <answer> and </answer>. DO NOT call function tool again.
"""

SUMMARY_PROMPT = """
You are an event summarizer for a Short-Term Memory (STM) video window.

Goal
- Produce ONE self-contained summary describing what happens inside this STM window only.

Inputs
- STM frames (authoritative evidence).

Hard Rules
1) Chronology: Narrate in temporal order within the STM window. No reordering across time.
2) No guessing: Do not infer intentions/causes not shown. If something is unclear, state "unidentified/unclear" rather than guessing.

Content Focus
- Who did what to whom/what, where, with what tool/object, and the immediate result.
- Include objects visible in STM 

Style
- Active voice; present or simple past; concrete, observable facts.
- No titles, lists, timestamps, metadata, or markup.
- Length ≤ 300 words.

Output
- Output ONLY the summary text.
"""


class OasisModel:
    def __init__(self, cfg: OasisConfig):
        self.cfg = cfg
        self.event_forest = EventForest.from_config(cfg)
        self.short_memory = ShortMemory.from_config(cfg)

        self.max_frames_per_node = cfg.event_node.max_frames_per_node
        self.tokens_per_frame = cfg.event_node.tokens_per_frame
        self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.mllm.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.mllm_gen_kwargs = dict(
            max_new_tokens=self.cfg.mllm.max_new_tokens,
            do_sample=self.cfg.mllm.do_sample,
            temperature=self.cfg.mllm.temperature,
            top_p=self.cfg.mllm.top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        self.mllm.eval()
        self.mllm_processor = AutoProcessor.from_pretrained(cfg.mllm.model_path)

        self.event_forest.mllm = self.mllm
        self.event_forest.mllm_processor = self.mllm_processor

    def reset(self):
        self.event_forest.reset()
        self.short_memory.reset()

    def uniform_keyframes(self, pkt_lst: List[FramePacket], cnt: int) -> List[FramePacket]:
        if len(pkt_lst) <= cnt:
            return pkt_lst
        step = max(1, len(pkt_lst) // cnt)
        return pkt_lst[::step][:cnt]

    def frame_to_video(self, pkt_lst: List[FramePacket]) -> tuple:
        event_video = torch.concat(
            [pkt.frame.unsqueeze(0) for pkt in pkt_lst], dim=0
        )

        resize_h, resize_w = smart_resize(
            event_video.shape[2],
            event_video.shape[3],
            factor=32,
            min_pixels=4 * 32 * 32,
            max_pixels=self.tokens_per_frame * 32 * 32,
        )
        event_video = transforms.functional.resize(
            event_video,
            [resize_h, resize_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        raw_fps = (pkt_lst[-1].t - pkt_lst[0].t) / len(pkt_lst)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=list(range(event_video.shape[0])),
            total_num_frames=event_video.shape[0],
        )
        return event_video, video_metadata

    def _mllm_generate(self, prompts, videos, video_metadata):
        mllm_inputs = self.mllm_processor(
            text=prompts,
            images=None,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )
        mllm_inputs = mllm_inputs.to(self.mllm.device)
        with torch.no_grad():
            output_dict = self.mllm.generate(**mllm_inputs, **self.mllm_gen_kwargs)
            generated_ids = output_dict.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(mllm_inputs.input_ids, generated_ids)
            ]
            return self.mllm_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    def summarize_event(self, asr_text: str, pkt_lst: List[FramePacket]) -> tuple:
        event_video, video_metadata = self.frame_to_video(pkt_lst)

        msg = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is some short-term memory frames: \n"},
                    {"type": "video", "video": []},
                    {"type": "text", "text": "Summarize the short-term memory video."},
                ],
            },
        ]

        prompts = [
            self.mllm_processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        ]
        output_text = self._mllm_generate(prompts, event_video, video_metadata)

        summary_embedding = torch.tensor(
            self.event_forest.embedding_model.encode(output_text)
        )
        summary_embedding = F.normalize(summary_embedding, dim=0)

        return output_text, summary_embedding

    def process_stream_input(
        self,
        frame_packet: FramePacket,
        audio_packet: Optional[AudioPacket] = None,
    ) -> None:
        event_sources: List[List[FramePacket]] = self.short_memory.push(frame_packet)
        asr_text = ""

        for source in event_sources:
            summary, summary_embedding = self.summarize_event(
                asr_text=asr_text, pkt_lst=source
            )
            node = EventNode(
                id=f"ev_{source[0].idx}_{source[-1].idx}",
                t_start=source[0].t,
                t_end=source[-1].t,
                level=0,
                summary=summary,
                summary_embedding=summary_embedding,
                keywords=[],
                frames=self.uniform_keyframes(source, self.max_frames_per_node),
                asr_text=asr_text,
            )
            self.event_forest.insert_root(node)
            self.event_forest.snapshot(timestamp=node.t_end)

    def process_query(self, query: str) -> str:
        short_mem_t_start = self.short_memory.buf[0].t
        short_mem_t_end = self.short_memory.buf[-1].t
        total_videos, total_video_metadata = [], []

        init_user_content = [
            {
                "type": "text",
                "text": (
                    f"Here is the NowWindow, very recent video frames that the headset "
                    f"is looking at, time[{time_to_hhmmss(self.short_memory.now_window[0].t)} "
                    f"- {time_to_hhmmss(self.short_memory.now_window[-1].t)}]: \n"
                ),
            },
            {"type": "video", "video": []},
            {
                "type": "text",
                "text": (
                    f"Here is short-term memory video window, "
                    f"time[{time_to_hhmmss(short_mem_t_start)} - {time_to_hhmmss(short_mem_t_end)}]: \n"
                ),
            },
            {"type": "video", "video": []},
            {"type": "text", "text": "Here is some long-term events summary: \n"},
        ]

        nowwindow_video, nowwindow_meta = self.frame_to_video(self.short_memory.now_window)
        total_videos.append(nowwindow_video)
        total_video_metadata.append(nowwindow_meta)

        short_mem_video, short_mem_meta = self.frame_to_video(self.short_memory.buf)
        total_videos.append(short_mem_video)
        total_video_metadata.append(short_mem_meta)

        for root_node in self.event_forest.get_root_node_lst():
            init_user_content.append({
                "type": "text",
                "text": (
                    f"time[{time_to_hhmmss(root_node.t_start)} - "
                    f"{time_to_hhmmss(root_node.t_end)}]: {root_node.summary}\n"
                ),
            })

        init_user_content.extend([
            {
                "type": "text",
                "text": f"Here is QA history summary: \n{self.event_forest.QA_summary_all}",
            },
            {
                "type": "text",
                "text": f"Now process the question at time[{time_to_hhmmss(short_mem_t_end)}]: {query}",
            },
        ])

        msg = [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": init_user_content},
        ]

        prompts = [
            self.mllm_processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        ]
        logger.info(f"prompts: {prompts[0]}")

        output_text = self._mllm_generate(prompts, total_videos, total_video_metadata)
        final_output = output_text

        tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        tool_call_match = tool_call_pattern.findall(output_text)

        if len(tool_call_match) > 0:
            try:
                tool_call = json.loads(tool_call_match[0])
                query_text = tool_call["arguments"]["text_input"]

                retrieved_events = self.event_forest.retrieve_event(query_text)
                logger.debug(f"Retrieved events: {[e.id for e in retrieved_events]}")

                retrieved_QAs = self.event_forest.retrieve_QA(query)
                logger.debug(f"Retrieved QAs: {[qa.id for qa in retrieved_QAs]}")

                msg.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                })

                user_content_lst = [
                    {"type": "text", "text": "The following are the details retrieved from the tool: \n"},
                    {"type": "text", "text": "Here is some QA history retrieved from the tool: "},
                ]
                for rag_QA in retrieved_QAs:
                    user_content_lst.append({
                        "type": "text",
                        "text": (
                            f"time[{time_to_hhmmss(rag_QA.t)}]: "
                            f"Question: {rag_QA.question}, Answer: {rag_QA.answer}\n"
                        ),
                    })

                user_content_lst.append({
                    "type": "text",
                    "text": "Here is some event details retrieved from the tool: ",
                })
                for rag_event_node in retrieved_events:
                    user_content_lst.append({
                        "type": "text",
                        "text": (
                            f"time[{time_to_hhmmss(rag_event_node.t_start)} - "
                            f"{time_to_hhmmss(rag_event_node.t_end)}]: "
                        ),
                    })
                    user_content_lst.append({"type": "video", "video": []})
                    event_video, video_metadata = self.frame_to_video(rag_event_node.frames)
                    total_videos.append(event_video)
                    total_video_metadata.append(video_metadata)

                user_content_lst.append({"type": "text", "text": TOOL_CALL_PROMPT})
                msg.append({"role": "user", "content": user_content_lst})

                prompts = [
                    self.mllm_processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                ]
                output_text_2 = self._mllm_generate(
                    prompts, total_videos, total_video_metadata
                )
                final_output = (
                    output_text
                    + "\n\n"
                    + f" Retrieved events: {[e.id for e in retrieved_events]}\n"
                    + f" Retrieved QAs: {[qa.id for qa in retrieved_QAs]}\n"
                    + output_text_2
                )
            except json.JSONDecodeError:
                final_output = output_text

        qa_emb = torch.tensor(
            self.event_forest.embedding_model.encode(
                f"Question: {query}, Answer: {final_output}"
            )
        )
        self.event_forest.insert_QA(
            QANode(
                id=f"qa_{short_mem_t_end}",
                t=short_mem_t_end,
                question=query,
                answer=final_output,
                QA_embedding=qa_emb,
            )
        )
        self.event_forest.snapshot(timestamp=short_mem_t_end)
        return final_output
