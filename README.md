# Learning vLLM

這是一個關於 vLLM (Versatile Large Language Model) 的學習與實作專案，包含離線推理 (Offline Inference)、API Server 架設以及一個新聞分類器的範例應用。

## 專案結構
- `01_offline_inference.py`: 基礎的離線批量推理範例
- `02_online_client.py`: 呼叫 vLLM API 的客戶端範例
- `03_sampling_params.py`: 探索不同的採樣參數 (Temperature, Top-P 等)
- `04_chat_templates.py`: 聊天模板 (Chat Template) 的使用範例
- `05_news_classifier.py`: 實戰應用：新聞分類器
- `tests/`: 單元測試程式碼

## 環境建置

建議使用 Python 3.10+ 環境：

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```