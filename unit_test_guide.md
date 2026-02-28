# AI Agent 專案：單元測試規範與討論文件

> 本文件用於團隊內部討論與對齊，請所有組員在第一次測試討論會議前閱讀。

---

## 一、為什麼需要單元測試

在多人協作的大型 AI Agent 專案中，單元測試扮演著非常重要的角色：

- **快速發現 bug**：每次改動後立刻確認有沒有破壞既有功能
- **降低溝通成本**：測試本身就是活文件，說明每個模組預期的行為
- **安全重構**：有測試的保護，重構或優化程式碼時不怕出錯
- **CI/CD 自動化**：測試可整合到自動化流程，PR 合併前強制確認

> 💡 **AI Agent 專案的特殊考量**
>
> 我們的 agent 使用本地 LLM，雖然避免了 API 費用問題，但單元測試仍建議使用 Mock。原因是 LLM 輸出具有隨機性（相同 input 不保證相同 output），直接呼叫模型會讓測試結果不穩定。單元測試的目標是測「我們自己寫的程式碼」，而不是測 LLM 本身。

---

## 二、單元測試核心原則

### 2.1 AAA 模式（業界標準寫法）

每一個測試函式都應該遵循 **Arrange → Act → Assert** 三個步驟：

```python
def test_agent_returns_greeting_when_input_is_hello():
    # Arrange — 準備測試資料和環境
    agent = MyAgent()
    user_input = "hello"

    # Act — 執行要測試的功能
    result = agent.respond(user_input)

    # Assert — 驗證結果是否符合預期
    assert "hi" in result.lower()
```

### 2.2 五大核心原則

**① 每個測試只測一件事**
一個 test function 只驗證一個行為。失敗時你能立刻知道是哪個功能壞了，不需要再往裡面挖。

**② 測試要能獨立執行**
不依賴外部資料庫、不依賴其他測試的執行順序。每個測試自己能跑、自己能 pass，任意順序執行結果都一樣。

**③ 命名要說清楚在測什麼**
好的測試名稱本身就是文件。推薦格式：`test_[功能]_[條件]_[預期結果]`

```python
# ❌ 不好的命名
def test_agent_1():

# ✅ 好的命名
def test_agent_returns_error_when_input_is_empty():
```

**④ 用 Mock 隔離外部依賴**
將 LLM 呼叫、資料庫查詢、外部 API 等全部 mock 掉，讓測試專注在自己的邏輯上。

```python
from unittest.mock import patch

def test_agent_calls_tool_when_llm_requests_it():
    with patch("myproject.llm_client.call") as mock_llm:
        mock_llm.return_value = '{"action": "search", "query": "AI news"}'
        agent = MyAgent()
        result = agent.process("告訴我 AI 的最新消息")
        mock_llm.assert_called_once()      # 確認有呼叫 LLM
        assert result["tool"] == "search"  # 確認有選到正確 tool
```

**⑤ 測試要夠快**
單元測試應該在幾秒內跑完。如果某個測試很慢，通常是因為它依賴了真實的外部資源，這時候就需要 mock。

---

## 三、AI Agent 的分層測試策略

我們建議將測試分成三個層次，各層有不同的目的和使用時機：

| 測試層次 | 使用對象 | 執行時機 |
|---|---|---|
| 單元測試 | Mock（模擬 LLM） | 每次 commit 都跑 |
| 整合測試 | 本地 LLM 模型 | 每天或 PR 合併前跑 |
| 端對端測試 | 本地 LLM 模型 | 重大版本或上線前跑 |

各層重點說明：

- **單元測試**：測你寫的每一個函式和 class，LLM 一律 mock，速度快、結果穩定
- **整合測試**：測 agent 的完整流程是否能正確接收 LLM 輸出並完成任務，需要真實模型
- **端對端測試**：模擬真實使用情境，從 user input 到最終輸出完整跑過一遍

### Agent 偵錯工具（適用於整合測試與端對端測試）

在整合測試和端對端測試層次，agent 的行為變得複雜，一旦測試失敗，光看 assert 的錯誤訊息很難知道是哪個環節出了問題。這時候適合引入 **agent 專用的偵錯與可觀測性工具**，例如：

- **LangSmith**：LangChain 生態系的追蹤工具，可視覺化每一步的輸入輸出
- **LangFuse**：開源的 LLM observability 平台，可自架
- **LangGraph 內建 tracing**：如果使用 LangGraph 建構 agent，內建就有節點追蹤功能

這類工具能讓你清楚看到：agent 走了哪些節點、每一步花了多少時間、哪個 tool 被呼叫、LLM 的輸出是什麼。在單元測試層次因為範圍很小、使用 mock，不太需要這些工具；但在更複雜的測試層次，它們是 debug 的重要幫手。

### 在 Agent 專案中特別需要測的東西

- **Tool 呼叫邏輯**：LLM 決定要用某個 tool 時，agent 是否正確執行？
- **錯誤處理**：LLM 回傳格式錯誤或 tool 執行失敗時，有沒有正確 fallback？
- **Prompt 組裝**：送給 LLM 的 prompt 是否根據不同情境正確組裝？
- **Memory 管理**：對話歷史是否正確儲存和傳遞？
- **輸出解析**：LLM 的輸出是否被正確 parse，異常格式有沒有處理？

---

## 四、快速開始：用現有程式寫第一個測試

### Step 1：安裝 pytest

```bash
pip install pytest pytest-mock
```

### Step 2：找一個「最單純的函式」開始

不要從最複雜的 agent 主流程開始，先找一個輸入輸出最清楚的函式，例如：

- `parse_llm_response(text)` → 把 LLM 的文字輸出轉成結構化資料
- `build_prompt(history, query)` → 根據對話歷史和新問題組出 prompt
- `validate_tool_input(params)` → 驗證 tool 的輸入參數是否合法

### Step 3：按照 AAA 模式寫測試

假設你有一個 `parse_llm_response` 函式：

```python
# 檔案：tests/test_parser.py
import pytest
from myproject.parser import parse_llm_response

def test_parse_returns_action_and_query_when_valid_json():
    # Arrange
    raw = '{"action": "search", "query": "AI news"}'

    # Act
    result = parse_llm_response(raw)

    # Assert
    assert result["action"] == "search"
    assert result["query"] == "AI news"

def test_parse_raises_exception_when_input_is_invalid_json():
    # Arrange
    raw = "this is not valid json"

    # Act & Assert
    with pytest.raises(ValueError):
        parse_llm_response(raw)
```

### Step 4：執行測試

```bash
pytest tests/ -v
# -v 會顯示每個測試的名稱和結果，方便 debug
# 加上 --cov=myproject 可以看測試覆蓋率
```

> ✅ **第一個測試的目標**
>
> 不需要一開始就追求高覆蓋率，目標是讓每個人都親手跑過一次測試、確認環境設定正確，並且對 AAA 模式有實際的感受。建議每個組員在討論會議前，針對自己負責的模組至少寫 2-3 個測試。

---

## 五、推薦工具與概念說明

### 工具一覽

| 工具 | 用途 | 安裝方式 |
|---|---|---|
| pytest | 主要測試框架，語法簡潔 | `pip install pytest` |
| pytest-mock | 讓 mock 在 pytest 更好用 | `pip install pytest-mock` |
| coverage.py | 檢查測試覆蓋率 | `pip install pytest-cov` |
| unittest.mock | Python 內建 mock 工具 | （不需安裝） |

### 什麼是 CI/CD？

**CI（Continuous Integration，持續整合）** 和 **CD（Continuous Deployment，持續部署）** 是一套讓測試和部署流程自動化的機制。

實際運作方式：當有人把程式碼 push 到 repo 或發出 PR，系統會自動幫你跑所有測試、檢查程式碼品質，全部通過才允許合併。這樣就不需要靠人工提醒，也不會有人「忘記跑測試就合併」的問題。

常見工具有 **GitHub Actions**、**GitLab CI** 等，通常只需要在 repo 裡加一個設定檔就能啟用。

```yaml
# 範例：GitHub Actions 設定（.github/workflows/test.yml）
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install pytest pytest-mock
      - run: pytest tests/ -v
```

### 什麼是 Fixture？

Fixture 是 pytest 提供的機制，用來準備「測試需要的前置資料或物件」，並且可以在多個測試之間重複使用。

沒有 fixture 的寫法（重複的 Arrange）：

```python
def test_agent_responds():
    agent = MyAgent(config="test_config")  # 每次都要重複這行
    result = agent.respond("hello")
    assert result is not None

def test_agent_handles_empty():
    agent = MyAgent(config="test_config")  # 每次都要重複這行
    result = agent.respond("")
    assert result == "input is empty"
```

使用 fixture 之後：

```python
import pytest

@pytest.fixture
def agent():
    return MyAgent(config="test_config")

def test_agent_responds(agent):       # pytest 自動注入，不需要自己建立
    result = agent.respond("hello")
    assert result is not None

def test_agent_handles_empty(agent):  # 同一個 fixture 重複用
    result = agent.respond("")
    assert result == "input is empty"
```

Fixture 也很適合用來管理 mock 好的 LLM client 或測試用假資料，讓測試程式碼保持乾淨。

---

## 六、需要與組員討論並取得共識的事項

以下事項建議在第一次正式 coding 前透過會議討論並決定：

### 📁 6.1 檔案結構與命名規則

- 測試資料夾要放在哪裡？（常見：`tests/` 與 `src/` 平行；或放在各模組內）
- 測試檔案的命名格式：建議統一用 `test_*.py`（pytest 預設識別方式）
- 測試函式命名格式：建議採用 `test_[功能]_[條件]_[預期結果]` 格式

### 📊 6.2 測試覆蓋率目標

- 是否設定最低覆蓋率門檻？（常見做法：70% 或 80%）
- 哪些程式碼不需要測？（例如 `main.py` 入口、config 檔）
- 是否在 CI/CD 中加入覆蓋率檢查，低於門檻就阻擋 PR？

### 🔄 6.3 CI/CD 整合方式

- 使用哪個 CI/CD 平台？（GitHub Actions、GitLab CI 或其他）
- 是否強制：每次 PR 都要通過所有單元測試才能 merge？
- 整合測試（跑本地模型的）是否也加入 CI？還是只手動跑？

### 🛠️ 6.4 Mock 的使用規範

- LLM 的 mock response 要集中管理嗎？（例如統一放在 `tests/fixtures/` 資料夾）
- 同一個 LLM 呼叫的 mock，不同組員是否要使用一致的格式？
- 如何處理 tool 呼叫的 mock？（直接 mock function，或用 pytest fixture）

### 👥 6.5 職責分工

- 誰負責寫自己模組的測試？還是有人專門負責寫測試？（建議：自己的模組自己測）
- Code review 時，是否要求 reviewer 確認測試是否足夠？
- 測試失敗時的處理流程：是誰負責修？多快要修好？

### 📐 6.6 測試的品質標準

- 一個測試函式是否限制只能有一個 assert？（較嚴格）或允許多個相關 assert？
- 是否要求每個 test function 都有 docstring 說明測試意圖？
- 如何處理測試資料（test data）：寫死在測試裡，還是集中在 fixture 管理？

> 🗣️ **建議的討論流程**
>
> 1. 每個組員在會議前先閱讀此文件並想好自己的意見
> 2. 第一次會議：聚焦討論 6.1 ~ 6.3（最影響整體流程的部分）
> 3. 確認後立即寫成 `CONTRIBUTING.md` 或 `TEST_GUIDE.md` 放進 repo
> 4. 第一週結束前，每個人完成自己模組的第一批測試
> 5. 之後每次 sprint 回顧時，檢視測試覆蓋率和品質

---

## 七、開始前的 Checklist

在正式開始寫測試之前，請確認以下事項都已完成：

1. 所有組員都已閱讀並理解本文件
2. 第六節的討論事項已完成討論並記錄結論
3. pytest 和相關套件已在所有人的開發環境中安裝完畢
4. `tests/` 資料夾結構和命名規則已建立並 push 到 repo
5. 每個組員至少跑過一次 `pytest` 確認環境正常
6. CI/CD 流程已設定（或已決定何時設定）

---

*文件版本 v1.0 ｜ 請根據團隊討論結果持續更新此文件*
