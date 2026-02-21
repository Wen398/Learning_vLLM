#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. 設定 CUDA Runtime 路徑 (解決 libcudart.so.12 缺失問題)
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "Starting vLLM OpenAI API Server..."
echo "Model: Qwen/Qwen2.5-0.5B-Instruct"
echo "Port: 8000"

# 2. 啟動 Server
# --model: 模型名稱
# --port: 服務端口 (預設 8000)
# --enforce-eager: 針對 Blackwell 架構的穩定性修正
# --trust-remote-code: 允許執行模型自定義代碼
# --max-model-len: 設定上下文長度限制 (視顯存調整)
"$PROJECT_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --port 8000 \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 2048
