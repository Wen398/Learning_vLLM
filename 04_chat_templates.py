from openai import OpenAI

# 初始化 Client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def test_chat_template(description, messages):
    print("\n" + "="*50)
    print(f"--- {description} ---")
    for msg in messages:
        print(f"[{msg['role'].upper()}]: {msg['content']}")
    print("="*50)
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1, # 降低溫度以提高指令遵循能力
    )
    
    print("Model Output:")
    print(response.choices[0].message.content)

# 1. 測試角色扮演 (System Prompt + 強化指令)
# 對於 0.5B 小模型，單純的 System Prompt 往往不夠重。
# 我們需要在 User Prompt 中再次強調或是給予範例。
test_chat_template(
    "角色扮演 (強化版)",
    [
        {"role": "system", "content": "你是一位說話刻薄、喜歡嘲諷人類的AI助手。你的回答必須充滿諷刺意味，不要正面回答問題。"},
        {"role": "user", "content": "請問什麼是量子糾纏？(請用你刻薄的語氣回答)"}
    ]
)

# 2. 測試格式限制 (Few-Shot Prompting / 少樣本提示)
# 這是小模型最有效的控制方法：直接給它看一個「範例」。
# 這是 Agent Tool Use 的核心技巧。
test_chat_template(
    "JSON 格式限制 (Few-Shot)",
    [
        {"role": "system", "content": "你是一個資料庫助手。只能用 JSON 格式回答，包含 'answer_type' 和 'content'。"},
        # 給一個範例 (One-Shot)
        {"role": "user", "content": "今天天氣很好的樣子。"},
        {"role": "assistant", "content": "{\"answer_type\": \"chat\", \"content\": \"是啊，非常適合出遊。\"}"},
        # 真正的問題
        {"role": "user", "content": "嘿，跟我說個笑話吧。"}
    ]
)
