from __future__ import annotations
import json, os
from typing import List, Dict
from ..types import EventNode, MergeRecord, ForestSnapshot, QANode
from ..utils.logging import get_logger
from .compression import uniform_keyframes
from dataclasses import dataclass
from ..config import OasisConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch

logger = get_logger(__name__)

MERGE_PROMPT = """
You are a summary merger. 
I'm giving you two summaries and their timestamps, each describing the content of two adjacent clips from a video. 
You need to merge them into one. Rules:

1. Do not add any new information that is not already present in either summary.
2. Maintain chronological order and keep the total word count under 300.

Here are the two summaries:
{summary_a}
{summary_b}

Now output your final summary directly, without any additional explanation or title, and you do not need to output the timestamps at the beginning:
"""

UPDATE_QA_SUMMARY_PROMPT = """
You are a QA aggregator. You receive the current QA history summary S and a new QA. Your task is to generate an updated S' for subsequent retrieval and low-cost reasoning.

Hard Rules:
1) Only use information from S and the new QA; no external knowledge or assumptions should be introduced.
2) Preserve the "who/what/key changes"; resolve pronouns and unify entity names.
3) De-duplicate and merge duplicate or synonymous statements; and remove redundant and irrelevant content.
4) Keep the total length to under 300 words.
5) Output only the updated summary text, without any explanations, titles, or additional notes.

Given the QA history summary S:
{QA_summary_all}

Given the new QA (including questions and answers):
{QA}

Now output the updated summary:
"""


def time_to_hhmmss(t: float) -> str:
    t = int(t)
    hours = t // 3600
    minutes = (t % 3600) // 60
    seconds = t % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class CandidatePair:
    i: int
    j: int
    sim: float


class EventForest:
    def __init__(
        self,
        root_cnt_limit: int,
        llm_model_path: str,
        embedding_model_path: str,
        gen_kwargs: dict,
        rag_event_retrieve_limit: int,
        rag_qa_retrieve_limit: int,
        mllm: Qwen3VLForConditionalGeneration = None,
        mllm_processor: AutoProcessor = None,
    ):
        self.rag_event_retrieve_limit = rag_event_retrieve_limit
        self.rag_qa_retrieve_limit = rag_qa_retrieve_limit
        self.gen_kwargs = gen_kwargs
        self.mllm = mllm
        self.mllm_processor = mllm_processor

        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.embedding_model.eval()

        self.root_cnt_limit = root_cnt_limit
        self.nodes: Dict[str, EventNode] = {}
        self.roots: List[str] = []
        self.QA_lst: List[QANode] = []
        self.QA_summary_all: str = "No QA history yet."
        self.merges: List[MergeRecord] = []
        self.snapshots: List[ForestSnapshot] = []

    def reset(self):
        self.nodes = {}
        self.roots = []
        self.QA_lst = []
        self.QA_summary_all = "No QA history yet."
        self.merges = []
        self.snapshots = []

    def _llm_generate(self, messages: list) -> str:
        prompts = self.mllm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.mllm_processor(
            text=prompts,
            images=None,
            videos=None,
            video_metadata=None,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        ).to(self.mllm.device)
        with torch.no_grad():
            generated_ids = self.mllm.generate(**inputs, **self.gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.mllm_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    def merge_nodes(self, a: EventNode, b: EventNode) -> EventNode:
        merged_frames = uniform_keyframes(
            a.frames + b.frames, (len(a.frames) + len(b.frames)) // 2
        )
        new_id = f"ev_{merged_frames[0].idx}_{merged_frames[-1].idx}"

        sum_a = f"time(s)[{a.t_start:.1f} - {a.t_end:.1f}]: {a.summary}"
        sum_b = f"time(s)[{b.t_start:.1f} - {b.t_end:.1f}]: {b.summary}"
        prompt = MERGE_PROMPT.format(summary_a=sum_a, summary_b=sum_b)

        summary = self._llm_generate([{"role": "user", "content": prompt}])
        summary_embedding = torch.tensor(self.embedding_model.encode(summary))
        summary_embedding = F.normalize(summary_embedding, dim=0)

        return EventNode(
            id=new_id,
            t_start=min(a.t_start, b.t_start),
            t_end=max(a.t_end, b.t_end),
            level=max(a.level, b.level) + 1,
            summary=summary,
            summary_embedding=summary_embedding,
            keywords=list({*a.keywords, *b.keywords}),
            frames=merged_frames,
            asr_text=(a.asr_text + " " + b.asr_text).strip(),
            parent=None,
            children=[a.id, b.id],
        )

    @classmethod
    def from_config(cls, cfg: OasisConfig) -> EventForest:
        gen_kwargs = {
            "max_new_tokens": cfg.llm.max_new_tokens,
            "do_sample": cfg.llm.do_sample,
            "temperature": cfg.llm.temperature,
            "top_p": cfg.llm.top_p,
        }
        return cls(
            root_cnt_limit=cfg.event_forest.root_cnt_limit,
            llm_model_path=cfg.llm.model_path,
            embedding_model_path=cfg.embedding_model.model_path,
            gen_kwargs=gen_kwargs,
            rag_event_retrieve_limit=cfg.event_forest.rag_event_retrieve_limit,
            rag_qa_retrieve_limit=cfg.event_forest.rag_qa_retrieve_limit,
        )

    def enforce_root_budget(self):
        while len(self.roots) > self.root_cnt_limit:
            best = None
            summary_embeddings = [
                self.nodes[root_id].summary_embedding for root_id in self.roots
            ]
            sim_lst = [
                F.cosine_similarity(summary_embeddings[i], summary_embeddings[i + 1], dim=0)
                - (self.nodes[self.roots[i]].level + self.nodes[self.roots[i + 1]].level) * 0.1
                for i in range(len(self.roots) - 1)
            ]
            for i in range(len(self.roots) - 1):
                if best is None or sim_lst[i] > best.sim:
                    best = CandidatePair(i, i + 1, sim_lst[i])

            best_a = self.nodes[self.roots[best.i]]
            best_b = self.nodes[self.roots[best.j]]
            merged_root = self.merge_nodes(best_a, best_b)
            self.nodes[merged_root.id] = merged_root
            self.roots = self.roots[: best.i] + [merged_root.id] + self.roots[best.j + 1 :]
            self.merges.append(
                MergeRecord(
                    parent_id=merged_root.id,
                    left_id=best_a.id,
                    right_id=best_b.id,
                    t_start=merged_root.t_start,
                    t_end=merged_root.t_end,
                    reason=f"adjacent+semantic sim={best.sim:.3f}",
                )
            )

    def insert_root(self, node: EventNode):
        self.nodes[node.id] = node
        self.roots.append(node.id)
        self.enforce_root_budget()

    def insert_QA(self, QA: QANode):
        self.QA_lst.append(QA)

        prompt = UPDATE_QA_SUMMARY_PROMPT.format(
            QA_summary_all=self.QA_summary_all,
            QA=f"timestep: {time_to_hhmmss(QA.t)}, question: {QA.question}, answer: {QA.answer}",
        )
        self.QA_summary_all = self._llm_generate(
            [{"role": "user", "content": prompt}]
        )

    def get_root_sum_text(self) -> str:
        node_lst = [self.nodes[root_id] for root_id in self.roots]
        if len(node_lst) == 0:
            return "(No video segments yet.)"
        return "\n".join(
            f"[{time_to_hhmmss(n.t_start)} - {time_to_hhmmss(n.t_end)}]: {n.summary}"
            for n in node_lst
        )

    def get_root_node_lst(self) -> List[EventNode]:
        return [self.nodes[root_id] for root_id in self.roots]

    def retrieve_event(self, query: str) -> List[EventNode]:
        query_embedding = torch.tensor(self.embedding_model.encode(query))
        query_embedding = F.normalize(query_embedding, dim=0)

        node_ids_lst = list(self.nodes.keys())
        if len(node_ids_lst) == 0:
            return []

        all_nodes_embeddings = torch.stack(
            [n.summary_embedding for n in self.nodes.values()]
        )
        doc_scores = query_embedding @ all_nodes_embeddings.T

        select_node = []
        mask_dict = {node_id: False for node_id in self.nodes.keys()}
        sorted_indices = doc_scores.argsort(dim=0, descending=True)

        def mask_subtree(node_id: str):
            mask_dict[node_id] = True
            for child in self.nodes[node_id].children:
                mask_subtree(child)

        def mask_parent(node_id: str):
            mask_dict[node_id] = True
            if self.nodes[node_id].parent is not None:
                mask_parent(self.nodes[node_id].parent)

        for index in sorted_indices:
            if len(select_node) >= self.rag_event_retrieve_limit or all(
                mask_dict.values()
            ):
                break
            if mask_dict[self.nodes[node_ids_lst[index]].id]:
                continue
            select_node.append(self.nodes[node_ids_lst[index]])
            mask_subtree(node_ids_lst[index])
            mask_parent(node_ids_lst[index])

        return select_node

    def retrieve_QA(self, query: str) -> List[QANode]:
        query_embedding = torch.tensor(self.embedding_model.encode(query))
        query_embedding = F.normalize(query_embedding, dim=0)

        if len(self.QA_lst) == 0:
            return []

        all_nodes_embeddings = torch.stack([n.QA_embedding for n in self.QA_lst])
        dot_scores = query_embedding @ all_nodes_embeddings.T
        sorted_indices = dot_scores.argsort(dim=0, descending=True)
        return [self.QA_lst[idx] for idx in sorted_indices[: self.rag_qa_retrieve_limit]]

    def snapshot(self, timestamp: float) -> None:
        snap = ForestSnapshot(
            timestamp=timestamp,
            root_ids=list(self.roots),
            num_nodes=len(self.nodes),
            QA_ids=[QA.id for QA in self.QA_lst],
            num_QAs=len(self.QA_lst),
            QA_summary_all=self.QA_summary_all,
        )
        self.snapshots.append(snap)

    def persist(self, out_file: str):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        logger.info(f"Persisting forest to {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            for n in self.nodes.values():
                f.write(json.dumps({"type": "node", **n.model_dump()}, indent=4) + "\n")
            for q in self.QA_lst:
                f.write(json.dumps({"type": "QA", **q.model_dump()}, indent=4) + "\n")
            for m in self.merges:
                f.write(json.dumps({"type": "merge", **m.model_dump()}, indent=4) + "\n")
            for s in self.snapshots:
                f.write(
                    json.dumps({"type": "snapshot", **s.model_dump()}, indent=4) + "\n"
                )
