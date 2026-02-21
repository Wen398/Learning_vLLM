#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Setup CUDA Runtime path
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "Starting vLLM OpenAI API Server (Production Configuration Simulation)..."

# 說明：
# 1. --gpu-memory-utilization 0.5: 
#    強制 vLLM 只使用 50% 的 GPU 顯存。這在以下情況非常有用：
#    - 單卡部署多個模型 (例如同時跑一個 Embedding 模型)
#    - 預留顯存給其他應用程式
#
# 2. --max-model-len 4096:
#    限制模型的最大上下文長度。
#    較小的長度可以顯著降低 KV Cache 的顯存佔用。
#
# 3. --quantization (註解中):
#    若使用 AWQ/GPTQ 量化模型 (如 Qwen2.5-72B-Instruct-AWQ)，需加上此參數。
#    量化能將顯存需求降低至 1/2 或 1/3。
#
# 4. --tensor-parallel-size 1 (註解中):
#    若有多張 GPU，可設定此數值來進行模型並行 (Model Parallelism)。
#    例如: 70B 模型通常需要 2~4 張卡。

"$PROJECT_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --port 8000 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096
    # --quantization awq \
    # --tensor-parallel-size 2 \
