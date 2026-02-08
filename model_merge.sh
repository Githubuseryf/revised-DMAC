#!/bin/bash

export CHECKPOINT_DIR='/workspace/DMAC/checkpoints/Dynamic-MAS/Dynamic-MAS-qwen3-4b-gpt/global_step_300/actor'
export HF_MODEL_PATH='Qwen/Qwen3-4B'
export TARGET_DIR='/workspace/DMAC/merge_model/MAS-300'

python3 verl/scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path "$HF_MODEL_PATH" \
  --local_dir "$CHECKPOINT_DIR" \
  --target_dir "$TARGET_DIR"
