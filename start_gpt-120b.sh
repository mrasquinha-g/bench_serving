#!/bin/bash

# Source CUDA paths
source ~/cuda-paths.txt

# Set library paths for PyTorch and CUDA
export LD_LIBRARY_PATH=$HOME/.conda/envs/vllm_env/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_HARMONY_CACHE_DIR=/home/mrasquinha/gptoss-tokenizer
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_RPC_TIMEOUT=600000

export VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0

with-proxy vllm serve ~/checkpoints/gpt-oss-120b \
  --tokenizer /home/mrasquinha/gptoss-tokenizer \
  --chat-template ~/checkpoints/gpt-oss-120b/chat_template.jinja \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 512 \
  --async-scheduling \
  --host :: \
  --port 8081 \
  --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce \
  --served-model-name gpt-oss-120b 2>&1 | tee server_`date +%m_%d_%H%M`.log
