# vLLM 學習筆記

## 0. 環境建置 (Environment Setup) (2026/02/18)

### 1. 系統環境
- **GPU**: NVIDIA GB10 (Blackwell Architecture)
- **CUDA**: 13.0
- **Python**: 3.12.3

### 2. 專案結構
```
/home/khstudent3/Learning_vLLM/
├── .gitignore          # 忽略 venv, __pycache__, models 等
├── requirements.txt    # 記錄依賴 (vllm, openai)
├── .venv/              # 虛擬環境 (Python 3.12)
└── learning_vllm_notes.md
```

### 3. 初始化指令
```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 第一階段：基礎部署與 API 化 (快速上手)

**目標**：將 LLM 轉變為高吞吐、低延遲的「可調用服務」，並掌握 Agent 開發所需的控制參數與接口。

---

### 1. 核心部署模式 (Deployment Modes)

#### Offline Inference (離線批處理)
- **核心組件**：`vllm.LLM` 類別。
- **用途**：對大量數據進行高吞吐量的文本生成（如數據清洗、生成訓練數據、批量評測）。
- **特點**：不僅是為了跑通，而是為了最大化 GPU 利用率。

#### Online Serving (在線服務)
- **核心組件**：`vllm.entrypoints.openai.api_server`。
- **用途**：啟動一個與 OpenAI API 完全兼容的 HTTP 伺服器。
- **重要性**：這是 Agent 的核心。你的 Agent 程式碼 (Client) 將通過標準 OpenAI 協議與 vLLM (Server) 通訊，這意味著你的代碼可以無縫切換 backend。

---

### 2. 關鍵參數調優 (Key Parameters)

#### 基礎生成參數
- `temperature`: 控制輸出的隨機性。Agent 執行任務通常設低 (0.0 - 0.2) 以獲得穩定輸出；創意寫作則設高。
- `top_p`: 核採樣 (Nucleus Sampling)。
- `max_tokens`: 限制生成的最大長度，防止模型廢話連篇。

#### Agent 流程控制 (Crucial for Agents)
- **`stop` / `stop_token_ids`**：
  - **作用**：強制模型在生成特定字串（如 "Observation:", "User:"）時停止。
  - **應用**：這是 ReAct Agent 或 Tool Use 的核心機制，讓模型在需要執行外部動作時停下來。

---

### 3. 硬體適配與量化 (Hardware & Quantization)

- **顯存管理**：
  - `--gpu-memory-utilization`: 設定 vLLM 佔用 GPU 顯存的比例（預設 0.9）。多卡部署或與其他服務共存時需調整。
- **量化 (Quantization)**：
  - 使用 `--quantization awq` 或 `gptq` 來載入量化模型。
  - **優勢**：在有限顯存下運行更大參數的模型（例如在單卡跑 70B 模型），這對 Agent 的推理能力至關重要。

---

### 4. 介面與模板 (Interface & Templates)

- **Chat Templates (對話模板)**：
  - 確保模型能正確理解 System, User, Assistant 的對話結構。
  - 學習如何讓 vLLM 自動應用 `tokenizer_config.json` 中的模板。
- **Client 端實作**：
  - 使用官方 `openai` Python SDK 連接本地 vLLM 服務，而非使用 raw HTTP requests。

---

## 實作記錄 (Hands-on Log)

*(在此處記錄後續的實作代碼、遇到的錯誤與解決方案)*

## 4. 實作記錄：Offline Inference (離線批次推理)

我們成功完成了第一步的離線推理實作，並解決了幾個關鍵的環境問題。

### 遇到的問題與解決方案 (Troubleshooting)

1.  **CUDA Library Mismatch (`libcudart.so.12` missing)**
    *   **原因**: 系統環境 CUDA 13.0，但 vLLM (0.15.1) 預編譯包依賴 CUDA 12 的 `libcudart.so.12`。
    *   **解決**: 安裝與 vLLM 兼容的 CUDA Runtime 並設定 `LD_LIBRARY_PATH`。
    *   指令：
        ```bash
        pip install nvidia-cuda-runtime-cu12
        ```
    *   **工具腳本 (`run_offline.sh`)**: 由於 `LD_LIBRARY_PATH` 每次重啟 terminal 都會失效，我們建立了一個啟動腳本來自動處理。

2.  **Compilation Crash (`EngineCore died unexpectedly`)** (2026/02/18 更新)
    *   **原因**: 即便安裝了 `python3-dev` 解決了 `Python.h` 缺失問題，在 NVIDIA GB10 (Blackwell) + ARM64 架構下，Triton/Inductor 的 CUDA Graph 編譯仍會產生大量 PTX 錯誤並導致進程崩潰。
    *   **解決**: 在 `LLM` 初始化時加入 `enforce_eager=True`，強制使用 PyTorch Eager 模式，跳過編譯以換取穩定性。

### 程式碼範例 (`01_offline_inference.py`)

```python
from vllm import LLM, SamplingParams

# 1. 準備輸入
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "What is the future of AI?",
]

# 2. 設定參數
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 3. 初始化引擎
# enforce_eager=True 是關鍵：避免在新架構(Blackwell)下因 CUDA Graph 編譯失敗而崩潰
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, enforce_eager=True)

# 4. 生成
outputs = llm.generate(prompts, sampling_params)

# 5. 顯示結果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
```

### 方便的啟動腳本 (`run_offline.sh`)
```bash
#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 設定 LD_LIBRARY_PATH 指向 .venv 中的 nvidia runtime
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/01_offline_inference.py"
```

## 5. 實作記錄：Online Serving (在線服務)

在線服務將 vLLM 轉變為一個常駐的 API Server，讓多個 Client 可以同時發送請求。這對於構建 Agent 系統至關重要。

### Server 端啟動 (`run_server.sh`)

類似於 Offline Inference，我們需要處理 `LD_LIBRARY_PATH` 與 `enforce_eager` 問題。這次我們使用 vLLM 內建的 OpenAI API 入口點。

```bash
#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. 設定 CUDA Runtime 路徑
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "Starting vLLM OpenAI API Server..."
# 2. 啟動 Server
# --enforce-eager: 針對 Blackwell 架構的穩定性修正
"$PROJECT_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --port 8000 \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 2048
```

在 Terminal 1 執行：
```bash
bash run_server.sh
```

### Client 端調用 (`02_online_client.py`)

使用標準的 `openai` Python SDK 進行連接。這證明了我們的 Server 與 OpenAI 協議完全兼容。

```python
from openai import OpenAI

# 1. 初始化 Client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY", # 本地部署通常不需要 Key
)

# 2. 發送請求 (Streaming)
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "你好，請自我介紹。"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

在 Terminal 2 執行：
```bash
# 確保已進入 venv
source .venv/bin/activate
python 02_online_client.py
```

## 6. 實作記錄：關鍵參數調優 (Key Parameters)

在 AI Agent 開發中，控制模型輸出的穩定性與格式至關重要。我們透過 `03_sampling_params.py` 來實驗這些參數。

### 核心參數說明

1.  **Temperature (溫度)**:
    *   `0.0`: 幾乎無隨機性，適合邏輯推導、數學運算、程式碼生成。
    *   `0.7-1.0`: 增加多樣性，適合創意寫作。
2.  **Max Tokens**:
    *   限制生成的最大長度，避免模型生成過多無意義的內容，或陷入無限循環。
3.  **Stop Sequences (停止序列)**:
    *   **Agent 的靈魂**。讓模型在生成特定字串（如 `Observation:` 或 `User:`）時立即停止，將控制權交還給程式碼（Function Calling）。

### 實驗代碼 (`03_sampling_params.py`)

這個腳本展示了如何動態調整這些參數：

```python
# 設定參數範例
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.0,      # 穩定輸出
    max_tokens=50,        # 限制長度
    stop=["Observation:"] # Agent 停止信號
)
```

### 執行方式
確保 `run_server.sh` 正在另一個 Terminal 運行中，然後執行：
```bash
python 03_sampling_params.py
```

## 7. 實作記錄：硬體適配與量化 (Hardware & Quantization)

在實際生產環境中，尤其是資源有限的情況下，直接部署全參數模型往往不切實際。我們透過 `run_production_server.sh` 來學習如何控制硬體資源。

### 1. 顯存控制 (VRAM Management)

即便是小模型，vLLM 預設也會佔用 **90%** 的顯存來最大化 KV Cache (為了吞吐量)。這在單卡多模型部署時會造成問題。

*   **`--gpu-memory-utilization 0.5`**: 強制 vLLM 只使用 50% 的顯存。
    *   **應用場景**:
        *   同時部署 LLM 與 Embedding 模型。
        *   預留顯存給作業系統或其他應用程式。
*   **`--max-model-len 4096`**: 限制最大上下文長度。
    *   **原理**: 上下文越長，KV Cache 佔用的顯存越多。將其限制在合理範圍 (如 4k 或 8k) 可以顯著節省顯存。

### 2. 量化 (Quantization)

雖然本次實驗使用非量化模型，但在部署 70B 等大模型時，量化是必須的。

*   **常規用法**:
    ```bash
    # 假設我們下載了 Qwen2.5-72B-Instruct-AWQ
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-72B-Instruct-AWQ \
        --quantization awq \
        --tensor-parallel-size 4
    ```
*   **關鍵參數**:
    *   `--quantization awq`: 指定量化格式 (AWQ, GPTQ, SqueezeLLM)。
    *   `--tensor-parallel-size`: 指定使用幾張 GPU 進行張量並行 (Tensor Parallelism)。

### 3. 實作腳本 (`run_production_server.sh`)

我們建立了一個模擬生產環境配置的腳本：

```bash
"$PROJECT_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --gpu-memory-utilization 0.5 \  # 只用一半顯存
    --max-model-len 4096            # 限制上下文長度
```

### 驗證方式
1.  執行 `bash run_production_server.sh`。
2.  觀察 Terminal 輸出，確認顯存佔用率 (vLLM 會顯示 `GPU Memory Usage` 相關資訊)。
3.  再次執行 `python 02_online_client.py` 確保服務仍能正常運作。


