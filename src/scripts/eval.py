"""
Unified evaluation script for OASIS across all benchmarks.

Supports any dataset in the unified info+breakpoint JSON format.
- Multiple-choice questions: extracts answer letter, compares with gt, reports accuracy.
- Open-ended questions: saves model response without scoring.

Usage examples:

  # OVO-Bench
  CUDA_VISIBLE_DEVICES=0 python src/scripts/eval.py \
      --metadata metadata/eval_ovo_bench.json \
      --dataset_root datasets/ \
      --output_dir output/ovobench

  # StreamBench
  CUDA_VISIBLE_DEVICES=0 python src/scripts/eval.py \
      --metadata metadata/eval_streambench.json \
      --dataset_root datasets/ \
      --output_dir output/streambench

  # StreamingBench (one sub-task)
  CUDA_VISIBLE_DEVICES=0 python src/scripts/eval.py \
      --metadata metadata/StreamingBench/Anomaly_Context_Understanding.json \
      --dataset_root datasets/ \
      --output_dir output/streamingbench_ACU
"""

import argparse
import copy
import json
import os
import re
from string import Template
from typing import Dict, List

from tqdm import tqdm

from src.oasis.config import OasisConfig
from src.oasis.io.stream import VideoStreamer
from src.oasis.model import OasisModel
from src.oasis.utils.logging import get_logger, setup_logging_from_config

logger = get_logger("eval")

QUESTION_TEMPLATE_MC = Template(
    "$Question\n\n"
    "Please briefly think step-by-step about this question. Keep your reasoning under 100 words.\n"
    "After thinking, You SHOULD CALL `rag_retrieval` to retrieve specific details from long-term "
    "memory to ensure the accuracy of your answers, unless the question explicitly asks about the "
    "content at the current moment(now, currently, etc.) or background(encyclopedic) knowledge "
    "unrelated to the video scene.\n"
    "Once you confirm your final answer, place the final answer inside <answer> and </answer>.\n"
    "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the "
    "<answer> </answer> tags."
)

QUESTION_TEMPLATE_OPEN = Template(
    "Here is the question:\n$Question\n\n"
    "Please briefly think step-by-step about this question. Keep your reasoning under 100 words.\n"
    "After thinking, You SHOULD CALL `rag_retrieval` to retrieve specific details from long-term "
    "memory to ensure the accuracy of your answers, unless the question explicitly asks about the "
    "content at the current moment(now, currently, etc.) or the question is related to general knowledge.\n"
    "Once you confirm your final answer, place the final answer inside <answer> and </answer>."
)


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_query(bp: dict) -> str:
    q_type = bp.get("type", "open_ended")
    question_text = bp["question"]

    if q_type == "multiple_choice" and "options" in bp:
        options_str = "\n".join(bp["options"])
        question_text = question_text + "\nOptions:\n" + options_str
        return QUESTION_TEMPLATE_MC.substitute(Question=question_text)
    else:
        return QUESTION_TEMPLATE_OPEN.substitute(Question=question_text)


def judge_answer(bp: dict, prediction: str) -> dict:
    """Return judgement dict. For MC, compare with gt; for open-ended, skip."""
    q_type = bp.get("type", "open_ended")
    result = {"prediction": prediction}

    if q_type == "multiple_choice" and "gt" in bp:
        gt_letter = bp["gt"].strip().upper()
        pred_letter = prediction.strip().upper()
        result["correct"] = pred_letter == gt_letter
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="OASIS unified evaluation script")
    ap.add_argument("--metadata", type=str, required=True, help="Path to dataset JSON (unified format)")
    ap.add_argument("--dataset_root", type=str, default="datasets/", help="Root dir for video files")
    ap.add_argument("--output_dir", type=str, default="output/eval", help="Output directory")
    ap.add_argument("--log_file", type=str, default="eval.log")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--pace", type=float, default=0.0)
    ap.add_argument("--shortmem_frames_limit", type=int, default=32)
    ap.add_argument("--now_window_frames_limit", type=int, default=16)
    ap.add_argument("--buffer_fps", type=float, default=1.0)
    ap.add_argument("--frames_per_node", type=int, default=16)
    ap.add_argument("--tokens_per_frame", type=int, default=256)
    ap.add_argument("--root_cnt_limit", type=int, default=4)
    ap.add_argument("--asr", type=str, default="none", choices=["none", "whisper"])
    ap.add_argument("--rag_event_retrieve_limit", type=int, default=2)
    ap.add_argument("--rag_qa_retrieve_limit", type=int, default=1)
    ap.add_argument("--mllm_path", type=str, default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--embedding_path", type=str, default="models/Qwen3-Embedding-0.6B")
    args = ap.parse_args()

    cfg = OasisConfig()
    cfg.stream.pace = args.pace
    cfg.stream.fps = args.fps
    cfg.event_node.max_frames_per_node = args.frames_per_node
    cfg.event_node.tokens_per_frame = args.tokens_per_frame
    cfg.short_memory.buffer_frames_limit = args.shortmem_frames_limit
    cfg.short_memory.now_window_frames_limit = args.now_window_frames_limit
    cfg.short_memory.buffer_fps = args.buffer_fps
    cfg.event_forest.root_cnt_limit = args.root_cnt_limit
    cfg.event_forest.rag_event_retrieve_limit = args.rag_event_retrieve_limit
    cfg.event_forest.rag_qa_retrieve_limit = args.rag_qa_retrieve_limit
    cfg.asr.backend = args.asr
    cfg.mllm.model_path = args.mllm_path
    cfg.embedding_model.model_path = args.embedding_path

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging_from_config(
        cfg, name="eval", filename=args.log_file, overwrite=True, out_dir=args.output_dir
    )
    logger.info(f"Config: {cfg}")

    oasis_model = OasisModel(cfg)

    dataset_name = os.path.splitext(os.path.basename(args.metadata))[0]
    output_path = os.path.join(args.output_dir, f"{dataset_name}_output.json")
    forest_dir = os.path.join(args.output_dir, f"{dataset_name}_forest")

    with open(args.metadata, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_output: List[dict] = []
    start_idx = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                logger.info(f"Resuming from sample index {start_idx}")
        except Exception as e:
            logger.exception(f"Error reading existing output: {e}")

    for item_idx, item in enumerate(data[start_idx:], start=start_idx):
        result_dict = copy.deepcopy(item)
        result_dict["breakpoint"] = []

        video_rel_path = item["info"]["video_path"]
        video_path = os.path.join(args.dataset_root, video_rel_path)

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}, skipping")
            continue

        vs = VideoStreamer.from_config(video_path, cfg)
        breakpoints = sorted(item["breakpoint"], key=lambda x: x["time"])
        pending_bps = list(breakpoints)

        oasis_model.reset()

        for pkt in tqdm(vs, desc=f"[{item_idx}] {video_rel_path}"):
            while pending_bps and pkt.t >= pending_bps[0]["time"]:
                bp = pending_bps.pop(0)
                try:
                    query = build_query(bp)
                    response = oasis_model.process_query(query)
                    bp_result = copy.deepcopy(bp)
                    bp_result["response"] = response

                    prediction = extract_answer(response)
                    if not prediction:
                        prediction = response
                    bp_result.update(judge_answer(bp, prediction))
                    result_dict["breakpoint"].append(bp_result)
                except Exception as e:
                    bp_result = copy.deepcopy(bp)
                    bp_result["prediction"] = "Error"
                    bp_result["error"] = str(e)
                    result_dict["breakpoint"].append(bp_result)
                    logger.exception(f"Failed on breakpoint: {e}")

            oasis_model.process_stream_input(pkt)

        # Handle remaining breakpoints after video ends (e.g. OVO-Bench with time=99999)
        for bp in pending_bps:
            try:
                query = build_query(bp)
                response = oasis_model.process_query(query)
                bp_result = copy.deepcopy(bp)
                bp_result["response"] = response

                prediction = extract_answer(response)
                if not prediction:
                    prediction = response
                bp_result.update(judge_answer(bp, prediction))
                result_dict["breakpoint"].append(bp_result)
            except Exception as e:
                bp_result = copy.deepcopy(bp)
                bp_result["prediction"] = "Error"
                bp_result["error"] = str(e)
                result_dict["breakpoint"].append(bp_result)
                logger.exception(f"Failed on breakpoint: {e}")

        video_id = os.path.splitext(os.path.basename(video_rel_path))[0]
        oasis_model.event_forest.persist(
            os.path.join(forest_dir, f"forest_{video_id}.jsonl")
        )

        final_output.append(result_dict)
        _save_results(output_path, final_output)

    _print_summary(final_output)


def _save_results(output_path: str, results: list):
    mc_total, mc_correct = 0, 0
    for item in results:
        for bp in item.get("breakpoint", []):
            if "correct" in bp:
                mc_total += 1
                if bp["correct"]:
                    mc_correct += 1

    summary = {
        "results": results,
        "total_videos": len(results),
        "mc_total": mc_total,
        "mc_correct": mc_correct,
        "mc_accuracy": mc_correct / mc_total if mc_total > 0 else 0.0,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_summary(results: list):
    mc_total, mc_correct = 0, 0
    open_total = 0
    task_stats: Dict[str, dict] = {}

    for item in results:
        for bp in item.get("breakpoint", []):
            task = bp.get("task", bp.get("class", "unknown"))
            if task not in task_stats:
                task_stats[task] = {"total": 0, "correct": 0}

            if "correct" in bp:
                mc_total += 1
                task_stats[task]["total"] += 1
                if bp["correct"]:
                    mc_correct += 1
                    task_stats[task]["correct"] += 1
            else:
                open_total += 1

    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    if mc_total > 0:
        logger.info(f"Multiple-choice: {mc_correct}/{mc_total} = {mc_correct / mc_total * 100:.2f}%")
        for task, stats in sorted(task_stats.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                logger.info(f"  {task}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
    if open_total > 0:
        logger.info(f"Open-ended questions: {open_total} (no auto-scoring)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
