import pytest
import json
import os
import sys
import importlib.util
from unittest.mock import MagicMock

# --- Step 0: Mock external dependencies to avoid loading heavy libs or CUDA ---
# We mock 'vllm' and 'transformers' BEFORE importing the module under test.
# This prevents the "ImportError: libcudart.so.12" if we are in a CPU-only env
# or simply want to speed up tests by not loading the real libraries.
sys.modules["vllm"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# --- Step 1: Helper to import a module with a numeric prefix ---
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot find file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# We need to load the module dynamically because '05_news_classifier' is not a valid Python identifier
# Assumes the test is run from the project root or we can find the file relative to this test file.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
module_path = os.path.join(project_root, "05_news_classifier.py")

news_classifier = load_module_from_path("news_classifier", module_path)

# Extract functions for easier testing
parse_llm_output = news_classifier.parse_llm_output
build_prompts = news_classifier.build_prompts

# --- Step 2: Unit Tests for parse_llm_output ---

def test_parse_llm_output_when_input_is_valid_json_returns_dict():
    """
    測試標準 JSON 字串能否正確解析為 Python dict
    """
    # Arrange
    raw_json = '{"category": "科技", "summary": "這是一篇科技新聞"}'
    
    # Act
    result = parse_llm_output(raw_json)
    
    # Assert
    assert isinstance(result, dict)
    assert result["category"] == "科技"
    assert result["summary"] == "這是一篇科技新聞"

def test_parse_llm_output_when_input_has_markdown_blocks_returns_clean_json():
    """
    測試包含 markdown code block 的字串能否被清洗並解析
    """
    # Arrange
    raw_text = """
    Here is the result:
    ```json
    {
        "category": "體育",
        "summary": "球隊贏了比賽"
    }
    ```
    """
    
    # Act
    result = parse_llm_output(raw_text)
    
    # Assert
    assert result["category"] == "體育"
    assert result["summary"] == "球隊贏了比賽"

def test_parse_llm_output_when_input_is_invalid_raises_value_error():
    """
    測試無效格式是否會拋出 ValueError
    """
    # Arrange
    invalid_text = "這不是 JSON"
    
    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        parse_llm_output(invalid_text)
    
    assert "無法解析 JSON" in str(excinfo.value)

# --- Step 3: Unit Tests for build_prompts (using Mock) ---

def test_build_prompts_when_given_articles_calls_tokenizer_for_each_article():
    """
    測試 build_prompts 是否正確呼叫 tokenizer.apply_chat_template
    """
    # Arrange
    mock_tokenizer = MagicMock()
    # 模擬 apply_chat_template 的回傳值
    mock_tokenizer.apply_chat_template.return_value = "<mock_prompt>"
    
    articles = ["文章一", "文章二"]
    system_prompt = "你是分類助手"
    
    # Act
    prompts = build_prompts(mock_tokenizer, articles, system_prompt)
    
    # Assert
    # 1. 確認回傳的 prompt 數量與文章數量一致
    assert len(prompts) == 2
    assert prompts == ["<mock_prompt>", "<mock_prompt>"]
    
    # 2. 確認 tokenizer 被呼叫了兩次 (因為有兩篇文章)
    assert mock_tokenizer.apply_chat_template.call_count == 2
    
    # 3. 檢查第一次呼叫的參數是否正確 (包含 system prompt 和 article)
    # call_args_list[0] 是第一次呼叫，args[0] 是第一個參數 (messages list)
    first_call_messages = mock_tokenizer.apply_chat_template.call_args_list[0].args[0]
    
    assert first_call_messages[0]["role"] == "system"
    assert first_call_messages[0]["content"] == system_prompt
    assert first_call_messages[1]["role"] == "user"
    assert "文章一" in first_call_messages[1]["content"]
