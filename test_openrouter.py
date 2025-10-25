#!/usr/bin/env python3
"""
测试 OpenRouter API 连接
"""
from openai import OpenAI

# OpenRouter 配置
OPENROUTER_API_KEY = "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"

# 尝试多个可能的模型名称
POSSIBLE_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-sonnet-20240229",
    "openai/gpt-3.5-turbo",  # 作为备选测试
]

MODEL_NAME = "anthropic/claude-sonnet-4.5"

def test_basic_api():
    """测试基本的 API 调用"""
    print("=" * 60)
    print("测试 1: 基本 API 调用")
    print("=" * 60)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/ruc-datalab/DeepAnalyze",
                "X-Title": "DeepAnalyze",
            },
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, I am Claude via OpenRouter!' in one sentence."
                }
            ],
            max_tokens=100,
            temperature=0.5,
        )

        response = completion.choices[0].message.content
        print(f"✅ API 调用成功！")
        print(f"📝 响应: {response}")
        print(f"💰 使用情况: {completion.usage}")
        return True

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def test_data_science_prompt():
    """测试数据科学相关的提示"""
    print("\n" + "=" * 60)
    print("测试 2: 数据科学任务提示")
    print("=" * 60)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    prompt = """You are a data science agent. Generate Python code to analyze this data:

Data: sales_data.csv with columns: date, product, revenue, quantity

Task: Calculate total revenue by product.

Use these XML tags:
<Code>```python
# your code here
```</Code>

Generate only the code block."""

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/ruc-datalab/DeepAnalyze",
                "X-Title": "DeepAnalyze",
            },
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.3,
            stop=["</Code>"],  # 测试自定义 stop sequence
        )

        response = completion.choices[0].message.content
        print(f"✅ 数据科学提示测试成功！")
        print(f"📝 响应:\n{response}")
        print(f"🛑 停止原因: {completion.choices[0].finish_reason}")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_streaming():
    """测试流式响应"""
    print("\n" + "=" * 60)
    print("测试 3: 流式响应")
    print("=" * 60)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    try:
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/ruc-datalab/DeepAnalyze",
                "X-Title": "DeepAnalyze",
            },
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 5, one number per line."
                }
            ],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        print("✅ 流式响应测试:")
        print("📝 响应: ", end="", flush=True)
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")
        return True

    except Exception as e:
        print(f"❌ 流式测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试 OpenRouter API")
    print(f"🔑 API Key: {OPENROUTER_API_KEY[:20]}...")
    print(f"🤖 模型: {MODEL_NAME}\n")

    # 运行所有测试
    results = []
    results.append(("基本 API 调用", test_basic_api()))
    results.append(("数据科学提示", test_data_science_prompt()))
    results.append(("流式响应", test_streaming()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！可以继续集成到 DeepAnalyze")
    else:
        print("\n⚠️  部分测试失败，请检查配置")
