# 開發討論與問題記錄 (Development Discussion Log)

這份文件用於記錄在開發 vLLM 專案過程中遇到的技術瓶頸、未解決問題以及需要與團隊進一步討論的架構決策。

---

## 2026/02/18 - vLLM 編譯模式與硬體適配問題

### 問題描述 (Issue Description)
在 **NVIDIA GB10 (Blackwell Architecture, ARM64)** 環境下運行 vLLM 時，預設的 CUDA Graph 編譯模式會導致進程崩潰。

### 環境細節
- **OS**: Ubuntu 24.04 (ARM64)
- **Hardware**: NVIDIA GB10
- **CUDA**: 13.0
- **Python**: 3.12
- **vLLM Version**: 0.15.1 (Pre-compiled or pip installed)

### 嘗試過程
1. **Initial Error**: `fatal error: Python.h: No such file or directory`
   - **嘗試**: 安裝 `python3-dev` (`sudo apt-get install python3-dev`)。
   - **結果**: 解決了標頭檔缺失問題，`torch.compile` (Inductor) 開始執行。

2. **Secondary Error (Critical)**:
   - **現象**: 程式在模型載入後，進入 CUDA Graph capturing 或 Triton 編譯階段時崩潰。
   - **Log**: 出現大量的 PTX (Parallel Thread Execution) 組合語言轉儲 (Dump)，隨後進程直接終止 (Engine core died unexpectedly)。
   - **推測**: 目前的 PyTorch/Triton 版本與 CUDA 13 或 Blackwell 架構的 PTX 指令集可能存在相容性問題，導致自動生成的 Kernel 無法正確執行。

### 臨時解決方案 (Workaround)
- **強制使用 Eager Mode**:
  - 在 `vllm.LLM` 初始化或 API Server 啟動參數中加入 `enforce_eager=True`。
  - **效果**: 跳過圖編譯，直接執行。雖然犧牲了部分推理性能 (Performance penalty on small batch sizes)，但保證了服務的穩定性。

### 待討論事項 (To Discuss)
1. 是否有針對 Blackwell/CUDA 13 優化的 vLLM/PyTorch nightly 版本可以嘗試？
2. 在生產環境 (Production) 中，若必須追求極致效能，是否需要等待官方修復 Triton 編譯器對新架構的支援？
3. 目前的 `enforce_eager=True` 對於我們的 Agent 場景 (通常 Batch Size 較小) 會帶來多大的延遲影響？是否需要進行基準測試 (Benchmark)？

---
