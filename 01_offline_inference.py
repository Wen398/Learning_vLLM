from vllm import LLM, SamplingParams

# 1. 準備輸入提示 (Prompts)
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "What is the future of AI?",
]

# 2. 設定採樣參數 (Sampling Parameters)
# temperature:控制隨機性 (0.8 稍高，適合創意；Agent 通常設 0.0-0.2)
# top_p: 核採樣 (0.95 表示只考慮累積機率 95% 的 token)
# max_tokens: 限制生成長度
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 3. 初始化 LLM 引擎 (Initialize LLM)
# 將自動從 HuggingFace 下載模型 Qwen/Qwen2.5-0.5B-Instruct
# trust_remote_code=True: 有些模型架構需要執行遠端代碼 (Qwen 建議開啟)
# 如果沒有 enforce_eager=True，vLLM 會使用 CUDA Graphs。
# 若在你的環境下 CUDA Graph 編譯仍有問題，請取消註解下面的 enforce_eager=True
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, enforce_eager=True)

# 4. 執行生成 (Generate)
# 這是一個阻塞操作 (Blocking)，會等到所有 prompts 都生成完畢
outputs = llm.generate(prompts, sampling_params)

# 5. 輸出結果 (Print Results)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)
