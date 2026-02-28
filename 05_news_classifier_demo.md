# vLLM 實戰專案：極速中文新聞分類器 (High-Throughput News Classifier)

本專案旨在展示如何利用 vLLM 的 **Offline Batch Inference (離線批次推理)** 功能，構建一個高效率的中文新聞分類與摘要工具。

---

## 1. 專案目標 (Goal)

模擬真實世界的 **ETL (Extract, Transform, Load)** 數據處理場景：
- **Input**: 大量非結構化的中文新聞文本。
- **Requirements**: 需要自動識別新聞類別（如科技、財經），並生成一句話摘要。
- **Constraint**: 輸出必須是嚴格的 **JSON 格式**，以便後續程式自動寫入資料庫或 API。

## 2. 核心技術亮點 (Key Features)

### A. 高吞吐量 (High Throughput)
與傳統的逐篇 API 呼叫不同，本專案使用 `vLLM` 的批處理引擎。
- **優勢**：vLLM 的 PagedAttention 與 Continuous Batching 技術能最大化 GPU 利用率，同時處理數百甚至數千篇文章，速度比傳統 `HuggingFace Transformers` 快數倍。

### B. 結構化輸出 (Structured Output)
展示了如何透過 **Prompt Engineering** 與 **Sampling Parameters** 控制，讓 LLM 穩定輸出機器可讀的 JSON 格式，這是構建 AI Agent 或自動化流程的關鍵。

### C. 強大的中文能力 (Multilingual Capability)
選用 `Qwen/Qwen2.5-7B-Instruct` 模型，展示了其在繁體中文理解與指令遵循上的卓越表現。

---

## 3. 環境與模型 (Environment)

- **Library**: `vllm`, `transformers`
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (70億參數，指令微調版)
- **Hardware**: NVIDIA GPU (支援 Blackwell/Hopper/Ampere 等架構)

---

## 4. 程式碼解析 (Code Walkthrough)

主要邏輯位於 `05_news_classifier.py`。

### 步驟 1: 準備數據與 Prompt
我們定義了一個 **System Prompt**，明確規範了角色的任務與輸出格式。

```python
system_prompt = """
你是一個新聞分類助手。
對於每篇文章，請提供：
1. 'category' (類別，例如：科技、體育、健康、財經、政治)。
2. 'summary' (一句話摘要)。

請嚴格使用有效的 JSON 格式回傳...
"""
```

### 步驟 2: 聊天模板處理 (Chat Templates)
vLLM 的 `LLM.generate` 接口主要接收 String。為了讓 Instruct 模型發揮效果，我們必須正確套用其訓練時的對話格式（如 `<|im_start|>`）。

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 將 List[Dict] 轉換為模型可讀的 Prompt String
text_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
```

### 步驟 3: 初始化 vLLM 引擎
```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    enforce_eager=True,  # 針對部分新顯卡架構(如Blackwell)的穩定性優化
    max_model_len=2048
)
```

### 步驟 4: 設定採樣參數 (Sampling Params)
為了確保 JSON 格式穩定，我們降低了隨機性。
```python
sampling_params = SamplingParams(
    temperature=0.1,    # 低溫：各次輸出結果一致，適合邏輯/格式化任務
    max_tokens=200,     # 限制輸出長度
    stop=["<|im_end|>"] # 停止詞：確保模型講完就停
)
```

---

## 5. 如何執行 (How to Run)

我們提供了一個啟動腳本 `run_news_classifier.sh`，它會自動處理環境變數 (如 `LD_LIBRARY_PATH`)。

在 Terminal 中執行：

```bash
# 賦予執行權限 (若尚未設定)
chmod +x run_server.sh

bash run_news_classifier.sh
```

---

## 6. 預期輸出 (Sample Output)

程式執行完畢後，您將看到每篇文章被轉換為標準的 JSON 物件：

```text
Processing 4 articles...
Initializing vLLM Engine...
...
Finished in 2.54 seconds

--- Article 1 ---
Original Length: 120 字符
{
    "category": "科技",
    "summary": "蘋果推出AI驅動的iPhone新功能，股價上漲3%，分析師看好未來銷售。"
}

--- Article 2 ---
Original Length: 105 字符
{
    "category": "體育",
    "summary": "本地籃球隊在延長賽中以102-99險勝奪冠，球星強森攻下40分。"
}
...
```

---

## 7. 總結 (Conclusion)

這個範例展示了 vLLM 在 **「非同步、高併發數據處理」** 上的潛力。如果不使用 Offline Batch 模式，而是架設 API Server 再透過 HTTP 逐筆請求，在處理百萬級別的數據時，時間成本將會是巨大的。

掌握這個模式，您就可以將 LLM 整合到任何後端數據流水線 (Data Pipeline) 中，實現自動標註、清洗或分析。
