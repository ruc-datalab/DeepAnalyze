#!/usr/bin/env python3
"""
测试不同的模型名称
"""
from openai import OpenAI

OPENROUTER_API_KEY = "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"

# 可能的 Claude 模型名称
POSSIBLE_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-sonnet-3.5",
    "openai/gpt-3.5-turbo",  # 作为备选
    "google/gemini-2.0-flash-thinking-exp",  # 免费模型
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

print("🔍 测试不同的模型名称...")
print("=" * 60)

for model in POSSIBLE_MODELS:
    print(f"\n测试模型: {model}")
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/ruc-datalab/DeepAnalyze",
                "X-Title": "DeepAnalyze",
            },
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        response = completion.choices[0].message.content
        print(f"✅ 成功! 响应: {response}")
        print(f"   模型: {model} 可用!")
        break
    except Exception as e:
        error_msg = str(e)
        if "Access denied" in error_msg:
            print(f"❌ 拒绝访问 (可能是 API key 问题)")
        elif "not found" in error_msg.lower():
            print(f"❌ 模型不存在")
        else:
            print(f"❌ 错误: {error_msg[:100]}")

print("\n" + "=" * 60)
