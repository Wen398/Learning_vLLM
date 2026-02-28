#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 設定 LD_LIBRARY_PATH 指向 .venv 中的 nvidia runtime
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "Running News Classifier Demo..."
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/05_news_classifier.py"
