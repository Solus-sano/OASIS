# OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2604.17052"><img src="https://img.shields.io/badge/arXiv-2604.17052-b31b1b.svg" alt="arXiv"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/CVPR-2026-6b42a0.svg" alt="CVPR 2026">
</p>

> Streaming video reasoning requires models to operate in a setting where history grows without bound while meaningful evidence remains scarce. In such a landscape, relevant signal is like an oasis — small, critical, and easily lost in a desert of redundancy. Enlarging memory only widens the desert; aggressive compression dries up the oasis. **The real difficulty lies in discovering *where to look*, not *how much to remember*.**

OASIS is a **training-free**, **plug-and-play** framework that organizes streaming history into hierarchical events and performs reasoning as controlled refinement — short-context inference first, followed by semantically grounded retrieval only when uncertainty arises.

## Data Flow

```
VideoStreamer ─► ShortMemory.push()
                  ├─ NowWindow  (recent fine-grained frames)
                  ├─ Buffer     (downsampled short-term context)
                  └─ Segment ready? ──► MLLM summarize ──► EventForest.insert_root()
                                                               └─ enforce_root_budget() ──► merge if #roots > K

User Question ──► process_query()
                   ├─ Coarse: NowWindow + Buffer + root summaries + QA summary ──► MLLM
                   └─ Fine (if tool call): retrieve events & QAs ──► MLLM with retrieved clips
                        └─ Store QA, update QA summary
```

## Project Structure

```
OASIS/
├── setup.sh                           # Environment setup
├── src/
│   ├── configs/default.yaml           # Reference configuration
│   ├── oasis/                         # Core library
│   │   ├── config.py                  # OasisConfig (Pydantic)
│   │   ├── types.py                   # FramePacket, EventNode, QANode, etc.
│   │   ├── model.py                   # OasisModel — stream processing & two-stage QA
│   │   ├── event/
│   │   │   ├── forest.py              # EventForest — insert, merge, retrieve, persist
│   │   │   ├── segmenter.py           # ShortMemory — NowWindow + buffer management
│   │   │   └── compression.py         # uniform_keyframes
│   │   ├── io/stream.py               # VideoStreamer (decord/torchvision), AudioStreamer
│   │   └── utils/logging.py           # Colored logger
│   └── scripts/
│       └── eval.py                    # Unified evaluation script
├── metadata/                          # Benchmark metadata (unified JSON format)
│   ├── eval_ovo_bench.json
│   ├── eval_streambench.json
│   └── StreamingBench/
│       ├── Sequential_Question_Answering.json
│       ├── Misleading_Context_Recognition.json
│       ├── Anomaly_Context_Understanding.json
│       └── Real_Time_Visual_Understanding.json
└── datasets/                          # Video files (user-provided)
```

## Getting Started

### Requirements

- Python 3.12
- CUDA 12.1

### Installation

```bash
conda create -n OASIS python=3.12
conda activate OASIS

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install qwen-vl-utils[decord]
pip install accelerate==1.10.1
pip install opencv-python pydantic sentence-transformers
pip install transformers==4.57.6

# Flash Attention (Linux x86_64, CUDA 12, PyTorch 2.5)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### Models

Download the following models and place them under `./models/`:

| Model | HuggingFace | Path |
|-------|-------------|------|
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `models/Qwen3-VL-8B-Instruct` |
| Qwen3-Embedding-0.6B | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `models/Qwen3-Embedding-0.6B` |

### Datasets

Download the benchmark datasets and place the video files under `./datasets/`:

| Benchmark | Source | Expected path under `datasets/` |
|-----------|--------|---------------------------------|
| OVO-Bench | [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) | `OVO-Bench/chunked_videos/` |
| StreamBench | [Barry-12138/StreamBench_v0.3](https://huggingface.co/datasets/Barry-12138/StreamBench_v0.3) | `StreamBench/{Ego,Movie,WebVideo}/` |
| StreamingBench | [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench) | `StreamingBench/{task_name}/` |

The `metadata/` directory already contains our pre-processed annotation files in a unified format.

## Evaluation

All benchmarks are evaluated through a single script. Run from the project root:

```bash
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

# StreamingBench (per sub-task)
CUDA_VISIBLE_DEVICES=0 python src/scripts/eval.py \
    --metadata metadata/StreamingBench/Anomaly_Context_Understanding.json \
    --dataset_root datasets/ \
    --output_dir output/streamingbench_ACU
```

## Citation

```bibtex
@inproceedings{liang2026oasis,
  title     = {OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning},
  author    = {Liang, Zhijia and Li, Jiaming and Chen, Weikai and Zhang, Yanhao and Lu, Haonan and Li, Guanbin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
