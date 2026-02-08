#!/bin/bash

# 指定 GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 指定模型路径（就是你 merge 完后的 HF 模型目录）
export MODEL_NAME='/workspace/DMAC/merge_model/MAS-300'

# 启动 vLLM 服务
vllm serve $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name agent \
  --port 8082 \
  --tensor-parallel-size 4
