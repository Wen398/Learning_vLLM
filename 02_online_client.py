from openai import OpenAI
import time

def main():
    print("Connecting to vLLM OpenAI API Server at http://localhost:8000/v1...")
    
    # 1. 初始化 OpenAI Client
    # vLLM 預設不需要 API Key，但不填可能會報錯，填 "EMPTY" 即可
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # 2. 準備對話歷史 (Chat History)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "寫一首關於人工智慧未來的短詩。"},
    ]

    print("\n[User]:", messages[-1]["content"])
    print("\n[Assistant]: ", end="", flush=True)

    try:
        # 3. 發送請求 (Streaming Response)
        # stream=True 可以讓回應像打字機一樣逐字出現，體驗更好
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # 必須與 Server 啟動的模型名稱一致
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n\n[Done]")

    except Exception as e:
        print(f"\nErrorCode: {e}")
        print("提示：請確認 run_server.sh 是否已經在另一個 Terminal 成功啟動？")

if __name__ == "__main__":
    main()
