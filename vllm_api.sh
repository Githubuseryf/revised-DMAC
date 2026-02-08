export CUDA_VISIBLE_DEVICES=0,1,2,3
/venv/vllmapi/bin/python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --port 8006 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
#   --dtype bfloat16 \
#   --max-model-len 32768 \
#   --max-num-seqs 32