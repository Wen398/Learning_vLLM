from openai import OpenAI

# 1. 初始化 Client (連接到本地 vLLM Server)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def test_sampling_params(description, prompt, **kwargs):
    print(f"\n--- {description} ---")
    print(f"Prompt: {prompt!r}")
    print(f"Params: {kwargs}")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        content = response.choices[0].message.content
        print(f"Output: {content!r}")
        if response.choices[0].finish_reason:
             print(f"Finish Reason: {response.choices[0].finish_reason}")
    except Exception as e:
        print(f"Error: {e}")

# 2. 測試 Temperature (溫度)
# 溫度 0.0: 確定性輸出 (適合 Agent 執行任務、數學、程式碼)
test_sampling_params(
    "低溫模式 (0.0) - 確定性輸出",
    "123 + 456 等於多少？",
    temperature=0.0
)

# 溫度 0.9: 創造性輸出 (適合寫詩、故事)
test_sampling_params(
    "高溫模式 (0.9) - 創意發揮",
    "請寫一首關於機器人學習做飯的短詩。",
    temperature=0.9
)


# 3. 測試 Max Tokens (最大長度)
# 用於防止模型生成過長的回覆
test_sampling_params(
    "最大長度限制 (Max Tokens: 10)",
    "請詳細解釋量子物理學的原理。",
    max_tokens=10
)


# 4. 測試 Stop Sequences (停止詞) - Agent 開發最關鍵的參數！
# 當模型生成 "Observation:" 時強制停止，這通常用於 ReAct Agent 模式
test_sampling_params(
    "停止詞測試 (Stop Sequence: 'Observation:')",
    "你是一個 Agent。當你需要查詢外部資訊時，請輸出 'Observation:'。問題：今天台北的天氣如何？",
    stop=["Observation:"]
)
