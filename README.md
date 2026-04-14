# DMAC: Dynamic Multi-Agent Collaboration for LLM Interaction

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![RL](https://img.shields.io/badge/Algorithm-GRPO-orange.svg)](#rl-training)
[![Engine](https://img.shields.io/badge/Serving-vLLM%200.8.3-purple.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

### A reinforcement learning framework where a lightweight policy model learns to dynamically coordinate large-scale LLMs through multi-turn role-aware dialogue

[Setup](#setup) | [Training](#rl-training) | [Evaluation](#evaluation) | [Project Layout](#project-layout)

</div>

---

## What is DMAC?

Large language models are powerful, but their output quality is often constrained by how well the input prompt is formulated. Most users lack the expertise to craft optimal instructions for complex tasks.

**DMAC** addresses this gap with a two-tier architecture:

- A **compact policy model** that acts as the *orchestrator* — it decomposes problems, formulates targeted prompts, and selects appropriate agent personas
- A **large-scale worker model** (e.g., GPT-class) that serves as the *executor* — it receives refined instructions and produces high-quality responses

The orchestrator is trained via **end-to-end reinforcement learning** (GRPO) to maximize downstream answer quality. At each dialogue turn, it can **dynamically specify an agent role** for the worker (domain expert, critiquer, summarizer, etc.), creating a flexible multi-agent collaboration pattern within a single interaction loop.

```
User Question
     │
     ▼
┌──────────────────┐         <interaction_prompt>          ┌─────────────────┐
│   Policy Model   │  ─────── prompt + agent_role ───────> │   Worker LLM    │
│   (Qwen3-4B)     │                                       │   (GPT / etc.)  │
│                  │  <──── <interaction_response> ─────── │                 │
│   Thinks, plans, │         structured answer             │   Executes the  │
│   assigns roles  │                                       │   given task    │
└──────────────────┘    ... repeat up to N turns ...       └─────────────────┘

```

## Setup

### Hardware Requirements

- Linux server with NVIDIA GPUs (tested with 4-GPU and 8-GPU configurations)
- Conda installed

### Install Dependencies

```bash
conda create -n mas python==3.12 -y
conda activate mas
cd verl
pip3 install -e .
pip3 install vllm==0.8.3
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install FlagEmbedding faiss-cpu
pip3 install debugpy==1.8.0 "ray[default]" debugpy
```

> Pre-compiled `flash-attn` wheels: [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

### Prepare the Data

Training and evaluation data should be organized as Parquet files:

```
dataset/
├── train_data/
│   ├── train.parquet
│   └── validation.parquet
└── test_data/
    ├── hotpot.parquet
    ├── PopQA.parquet
    ├── MusiQue.parquet
    ├── triviaqa_ood.parquet
    ├── squad_v2_ood.parquet
    ├── gsm8k.parquet
    ├── MathQA_ood.parquet
    ├── 2WikiMultihopQA.parquet
    ├── BookSum.parquet
    ├── WritingPrompts.parquet
    ├── Xsum_ood.parquet
    └── DAPO-Math.parquet
```

Use `process_data.py` to convert raw datasets into the expected Parquet schema.

## Setting Up the Worker LLM

The orchestrator communicates with the worker LLM via an **OpenAI-compatible API**. You have two options:

### Option A: Cloud / Third-Party API

Open `dmac/tool/tools/LLM_tool_dynamic_agent.py` and set:

```python
API_KEY = "your_key_here"
MODEL = "model_name"
BASE_URL = "https://your-endpoint"
```

### Option B: Local Deployment via vLLM

Set up a dedicated serving environment:

```bash
conda create -n vllmapi python=3.12 -y
conda activate vllmapi
pip3 install transformers accelerate huggingface_hub vllm
```

Start the API server:

```bash
nohup bash vllm_api.sh > worker_api.out 2>&1 &
```

Then update `LLM_tool_dynamic_agent.py` accordingly:

```python
BASE_URL = "http://<YOUR_SERVER>:8006/v1"
```

## RL Training

Launch GRPO training with Ray + FSDP + vLLM rollout:

```bash
nohup bash run_dmac.sh > training.out 2>&1 &
```

Notable training parameters in `run_dmac.sh`:

| Parameter | Value | Purpose |
|---|---|---|
| `data.train_batch_size` | 128 | Samples per gradient step |
| `actor_rollout_ref.rollout.n_repeat` | 5 | Rollouts per prompt for variance reduction |
| `tool.max_turns` | 5 | Max orchestrator-worker dialogue rounds |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Policy learning rate |
| `trainer.save_freq` | 20 | Checkpoint interval (steps) |
| `trainer.total_epochs` | 1 | Training epochs |

Checkpoints are saved to `checkpoints/Dynamic-MAS/<experiment_name>/`.

Training progress can be tracked via **Weights & Biases** (enabled by default in the config).

## Evaluation

The evaluation pipeline consists of four stages:

### Step 1 — Merge the RL Checkpoint

Convert FSDP shards back to a standard HuggingFace model. Edit `model_merge.sh`:

```bash
export CHECKPOINT_DIR='checkpoints/Dynamic-MAS/<experiment>/global_step_300/actor'
export HF_MODEL_PATH='Qwen/Qwen3-4B'
export TARGET_DIR='./merge_model/MAS-300'
```

```bash
bash model_merge.sh
```

### Step 2 — Deploy the Trained Model

Edit `vllm_serve.sh` to point to the merged model:

```bash
export MODEL_NAME='./merge_model/MAS-300'
```

```bash
nohup bash vllm_serve.sh > serve.out 2>&1 &
```

### Step 3 — Run Inference

**Interactive single-question mode:**

```bash
python inference.py
```

**Batch evaluation on all benchmarks:**

```bash
python batch_inference.py
```

**Selective dataset evaluation:**

```bash
python batch_inference.py --datasets PopQA MusiQue gsm8k
```

### Step 4 — Compute Metrics

```bash
python eval_scores.py --res-dir <results_folder>
```

Metrics produced for each dataset:
- **Exact Match (EM)** — strict string equality after normalization
- **Substring Match** — checks containment in either direction
- **F1 Score** — token-level precision/recall
- **Semantic Similarity** — embedding-based cosine similarity (optional)

Results are saved as per-dataset `eval.json` and an aggregated CSV summary.

## Project Layout

```
dmac/
├── dmac/                        # Core framework
│   ├── src/                     # RL training logic (trainer, actor, critic, reward)
│   ├── tool/
│   │   ├── envs/                # Execution environments (NousToolEnv, etc.)
│   │   └── tools/               # Worker LLM tool (LLM_tool_dynamic_agent.py)
│   ├── llm_agent/               # Tensor helpers & generation utilities
│   └── vllm_infer/config.py     # Model serving & inference config
├── verl/                        # veRL library (editable install)
├── dataset/                     # Train / test Parquet files
├── DMAC_Qwen3_chat_template.jinja  # Custom chat template for the policy model
├── run_dmac.sh                  # RL training entry point
├── model_merge.sh               # FSDP checkpoint merger
├── vllm_serve.sh                # Serve the trained policy model
├── vllm_api.sh                  # Serve the worker LLM locally
├── inference.py                 # Single-question interactive demo
├── batch_inference.py           # Full benchmark evaluation
├── eval_scores.py               # Metric aggregation
└── process_data.py              # Raw data → Parquet converter
```
