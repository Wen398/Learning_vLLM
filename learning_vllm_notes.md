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

1.  **CUDA Library Mismatch (**`**libcudart.so.12**` **missing)**
    *   **原因**: 系統環境 CUDA 13.0，但 vLLM (0.15.1) 預編譯包依賴 CUDA 12 的 `libcudart.so.12`。
    *   **解決**: 安裝與 vLLM 兼容的 CUDA Runtime 並設定 `LD_LIBRARY_PATH`。
    *   指令：
        ```bash
        pip install nvidia-cuda-runtime-cu12
        export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
        ```

2.  **Missing Python.h (**`**fatal error: Python.h**`**)**
    *   **原因**: vLLM 嘗試使用 `torch.compile` (Inductor) 來優化模型執行，這需要 Python 開發頭文件 (`python3-dev`)。但在無 root 權限且系統未安裝該套件的情況下會編譯失敗。
    *   **解決**: 在 `LLM` 初始化時加入 `enforce_eager=True`，強制使用 PyTorch Eager 模式，跳過即時編譯。

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
# enforce_eager=True 是關鍵：避免在無 python-dev 環境下因編譯失敗而崩潰
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
