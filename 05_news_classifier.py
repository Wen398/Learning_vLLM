from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import time

# 1. 準備模擬的新聞數據 (長文本 - 中文版)
articles = [
    """
    (科技) 蘋果公司今日宣佈推出一項全新的 AI 驅動 iPhone 功能後，股價上漲了 3%。
    分析師預測，這將顯著提升下一季度的銷售額。
    新功能整合了裝置端的大型語言模型，以增強 Siri 的能力，允許主動建議，並加強隱私保護。
    預計三星和 Google 等競爭對手將迅速跟進。
    """,
    """
    (體育) 本地籃球隊昨晚在激動人心的延長賽中贏得了冠軍賽。
    最終比分是 102-99。明星球員強森拿下了 40 分，其中包括致勝的三分球。
    球迷湧上街頭慶祝，造成輕微交通延誤，但沒有重大事故報告。
    """,
    """
    (健康) 一項新研究表明，每天喝三杯咖啡可能會降低心臟病的風險。
    這項發表在《心臟病學雜誌》上的研究追蹤了 50,000 名參與者長達 10 年。
    然而，專家警告說，添加過多的糖或奶油可能會抵消這些好處。
    適度是關鍵，個人結果可能因遺傳而異。
    """,
    """
    (財經) 中央銀行在最近的會議上決定將利率維持在 5.25% 不變。
    通貨膨脹顯示出降溫跡象，但在降息之前，決策者希望看到更持續的進展。
    市場反應輕微下跌，因為一些投資者原本希望能提前降息。
    由於抵押貸款利率居高不下，房地產數據仍然疲軟。
    """
]

# 2. 定義系統提示詞 (System Prompt) - 強制要求 JSON 格式
system_prompt = """
你是一個新聞分類助手。
對於每篇文章，請提供：
1. 'category' (類別，例如：科技、體育、健康、財經、政治)。
2. 'summary' (一句話摘要)。

請嚴格使用有效的 JSON 格式回傳，格式如下：
{
    "category": "科學",
    "summary": "科學家發現了一顆新行星。"
}
"""

def build_prompts(tokenizer, articles, system_prompt):
    """
    將文章列表轉換為模型可接受的 prompt 格式
    """
    prompts = []
    for article in articles:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是新聞文章：\n{article}"}
        ]
        # apply_chat_template 會幫我們加上 <|im_start|>... 這些標籤
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text_prompt)
    return prompts

def parse_llm_output(generated_text):
    """
    解析 LLM 的輸出，處理可能的 markdown 格式並回傳 JSON 物件
    """
    try:
        # 嘗試尋找 JSON 物件的開頭與結尾，過濾掉前後的雜訊 (包括 Markdown 標籤)
        start_idx = generated_text.find("{")
        end_idx = generated_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
             clean_text = generated_text[start_idx : end_idx + 1]
        else:
             # 如果找不到大括號，退回使用簡單的 replace 作為 fallback
             clean_text = generated_text.replace("```json", "").replace("```", "").strip()
             
        data = json.loads(clean_text)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"無法解析 JSON: {generated_text}") from e

def main():
    # 改用 7B 模型，這是目前 CP 值最高的中文模型選擇，指令遵循能力遠強於 0.5B
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 3. 準備 Prompts (使用 Tokenizer 套用 Chat Template)
    # vLLM 的 LLM 類別主要接受 string list，所以我們先用 tokenizer 轉好
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompts = build_prompts(tokenizer, articles, system_prompt)

    # 4. 初始化 vLLM 引擎
    # 注意: 根據你的環境 (Blackwell)，我們保留 allow_user_override_backend 或 enforce_eager 設定
    # 為了腳本通用性，這裡示範最常見的配置，若報錯可加 enforce_eager=True
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True, # 針對你的環境特別加入
        max_model_len=2048,
        gpu_memory_utilization=0.6, # 降低顯存佔用率 (預設 0.9)，避免 OOM
    )

    # 5. 設定採樣參數
    sampling_params = SamplingParams(
        temperature=0.1,       # 低溫以獲得穩定的 JSON 格式
        top_p=0.95,
        max_tokens=200,        # 摘要不需要太長
        stop=["<|im_end|>"]    # 確保模型講完就停
    )

    # 6. 執行批次推理
    print(f"Processing {len(prompts)} articles...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    
    # 7. 顯示結果
    print("\n" + "="*50)
    print(f"Finished in {end_time - start_time:.2f} seconds")
    print("="*50 + "\n")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"--- Article {i+1} ---")
        try:
            data = parse_llm_output(generated_text)
            print(f"Original Length: {len(articles[i])} 字符")
            # 直接印出解析後的標準 JSON
            print(json.dumps(data, ensure_ascii=False, indent=4))
        except ValueError as e:
            print(e)
        print()

if __name__ == "__main__":
    main()
