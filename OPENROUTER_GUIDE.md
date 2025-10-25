# 🌐 OpenRouter 集成指南

本指南说明如何使用 OpenRouter API 替换本地模型，实现对 100+ 模型的统一访问。

## 📋 目录

- [什么是 OpenRouter](#什么是-openrouter)
- [快速开始](#快速开始)
- [支持的模型](#支持的模型)
- [使用方法](#使用方法)
- [成本对比](#成本对比)
- [常见问题](#常见问题)

---

## 什么是 OpenRouter

OpenRouter 是一个统一的 LLM API 网关，提供：

- ✅ **100+ 模型**：Claude、GPT-4、Gemini、DeepSeek、Llama 等
- ✅ **OpenAI 兼容**：无需修改代码，直接切换
- ✅ **自动 Fallback**：主模型失败时自动切换备用模型
- ✅ **统一计费**：一个账号访问所有模型
- ✅ **免费额度**：某些模型提供免费配额

官网：https://openrouter.ai/

---

## 🚀 快速开始

### 1. 获取 API Key

1. 访问 [OpenRouter](https://openrouter.ai/) 注册账号
2. 前往 [Keys 页面](https://openrouter.ai/keys) 创建 API Key
3. （可选）添加信用卡以获得更高配额

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 启用 OpenRouter
USE_OPENROUTER=true

# 设置 API Key
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# 选择模型
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### 3. 运行 Demo

```bash
cd demo
bash start.sh
```

访问 http://localhost:4000，开始使用！

---

## 📊 支持的模型

### 推荐用于数据科学任务的模型

| 模型 | OpenRouter ID | 成本 | 特点 |
|------|--------------|------|------|
| **Claude 3.5 Sonnet** | `anthropic/claude-3.5-sonnet` | $3/$15 per 1M | 🏆 最强推理，200K 上下文 |
| **Claude Sonnet 4.5** | `anthropic/claude-sonnet-4.5` | $3/$15 per 1M | 🆕 最新 Claude |
| **GPT-4o** | `openai/gpt-4o` | $2.5/$10 per 1M | ⚡ 快速，多模态 |
| **DeepSeek R1** | `deepseek/deepseek-r1` | $0.55/$2.19 per 1M | 💰 性价比最高 |
| **Gemini 2.0 Flash** | `google/gemini-2.0-flash-thinking-exp` | 免费（限额） | 🆓 免费测试 |
| **Qwen2.5 72B** | `qwen/qwen-2.5-72b-instruct` | $0.35/$0.40 per 1M | 🌟 开源，便宜 |

完整列表：https://openrouter.ai/models

### 如何选择模型

**追求最佳质量**：
```bash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

**追求性价比**：
```bash
OPENROUTER_MODEL=deepseek/deepseek-r1
```

**免费测试**：
```bash
OPENROUTER_MODEL=google/gemini-2.0-flash-thinking-exp
```

---

## 💻 使用方法

### 方法 1：命令行使用

```python
from deepanalyze import DeepAnalyzeOpenRouter

# 初始化（从环境变量读取 API key）
deepanalyze = DeepAnalyzeOpenRouter(
    model_name="anthropic/claude-3.5-sonnet"
)

# 或者直接传入 API key
deepanalyze = DeepAnalyzeOpenRouter(
    model_name="anthropic/claude-3.5-sonnet",
    api_key="sk-or-v1-your-api-key-here"
)

# 使用
prompt = """# Instruction
Analyze this dataset and create visualizations.

# Data
File 1: {"name": "sales.csv", "size": "10.6KB"}
"""

workspace = "/path/to/workspace"
result = deepanalyze.generate(prompt, workspace=workspace)
print(result["reasoning"])
```

### 方法 2：Demo 后端使用

启动后端时设置环境变量：

```bash
# 方式 1：在 .env 文件中设置
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-v1-xxxxx
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# 方式 2：命令行设置
USE_OPENROUTER=true \
OPENROUTER_API_KEY=sk-or-v1-xxxxx \
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet \
python demo/backend.py
```

### 方法 3：动态切换模型

```python
from deepanalyze import DeepAnalyzeOpenRouter

# 使用不同的模型处理不同任务
claude = DeepAnalyzeOpenRouter(model_name="anthropic/claude-3.5-sonnet")
deepseek = DeepAnalyzeOpenRouter(model_name="deepseek/deepseek-r1")
gpt4 = DeepAnalyzeOpenRouter(model_name="openai/gpt-4o")

# 复杂任务用 Claude
complex_result = claude.generate(complex_prompt, workspace)

# 简单任务用 DeepSeek（省钱）
simple_result = deepseek.generate(simple_prompt, workspace)
```

---

## 💰 成本对比

### 示例：处理 1000 个数据分析任务

假设每个任务：
- 输入：2000 tokens（数据描述）
- 输出：4000 tokens（分析结果 + 代码）
- 总计：6000 tokens/任务 = 6M tokens

| 方案 | 输入成本 | 输出成本 | 总成本 |
|------|---------|---------|--------|
| **本地 DeepAnalyze-8B** | GPU 租赁 $50-100/天 | - | **$50-100**（无限次） |
| **Claude 3.5 Sonnet** | $3 × 2 = $6 | $15 × 4 = $60 | **$66** |
| **DeepSeek R1** | $0.55 × 2 = $1.1 | $2.19 × 4 = $8.76 | **$9.86** ⭐ |
| **GPT-4o** | $2.5 × 2 = $5 | $10 × 4 = $40 | **$45** |
| **Gemini 2.0 Flash** | 免费（限额内） | 免费（限额内） | **$0** 🆓 |

**结论**：
- 🏆 **偶尔使用**：OpenRouter + DeepSeek R1 最划算
- 💼 **大量使用**：自建 DeepAnalyze-8B 最经济
- 🎯 **高质量要求**：Claude 3.5 Sonnet 最优

---

## 🔧 高级功能

### 1. 自动 Fallback

配置备用模型列表，主模型失败时自动切换：

```python
import requests

payload = {
    "model": "anthropic/claude-3.5-sonnet",
    "route": "fallback",  # 启用 fallback
    "models": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "deepseek/deepseek-r1"
    ],
    "messages": messages,
    # ...
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload
)
```

### 2. 成本跟踪

OpenRouter 响应中包含详细的成本信息：

```python
response_data = response.json()
usage = response_data.get("usage", {})

print(f"输入 tokens: {usage.get('prompt_tokens')}")
print(f"输出 tokens: {usage.get('completion_tokens')}")
print(f"总成本: ${usage.get('total_cost', 0):.4f}")
```

### 3. 选择最便宜的模型

```python
payload = {
    "model": "openai/gpt-4o",  # 偏好模型
    "route": "cheapest",  # 选择满足要求的最便宜模型
    # ...
}
```

---

## ❓ 常见问题

### Q1: 如何在本地模型和 OpenRouter 之间切换？

**A:** 修改 `.env` 文件中的 `USE_OPENROUTER` 变量：

```bash
# 使用 OpenRouter
USE_OPENROUTER=true

# 使用本地模型
USE_OPENROUTER=false
```

### Q2: 支持哪些模型？

**A:** 100+ 模型，包括：
- Anthropic: Claude 3/3.5/Sonnet 4.5
- OpenAI: GPT-4o, GPT-4 Turbo
- Google: Gemini 2.0
- DeepSeek: DeepSeek R1, DeepSeek Coder
- Meta: Llama 3.1/3.2
- Qwen, Mistral, Phi, 等等

完整列表：https://openrouter.ai/models

### Q3: OpenRouter 比直接调用 Claude API 有什么优势？

**A:**
- ✅ 一个 API key 访问所有模型
- ✅ 自动 fallback 保证高可用性
- ✅ 统一计费和成本监控
- ✅ 某些模型比官方 API 便宜
- ✅ 无需多个 SDK

### Q4: 如何获得免费额度？

**A:**
1. 某些模型（如 Gemini）提供免费配额
2. OpenRouter 新用户可能有初始赠金
3. 查看 https://openrouter.ai/credits

### Q5: API key 安全吗？

**A:**
- ✅ 使用环境变量，不要提交到 Git
- ✅ 定期轮换 API key
- ✅ 设置使用限额
- ✅ 监控异常调用

### Q6: 出现 "Access denied" 错误怎么办？

**A:** 检查：
1. API key 是否正确
2. 是否已激活（访问 openrouter.ai 检查）
3. 是否有足够的余额
4. 模型名称是否正确

### Q7: 如何限制成本？

**A:**
1. 在 OpenRouter 后台设置月度限额
2. 使用更便宜的模型（DeepSeek, Qwen）
3. 实施混合策略（简单任务用便宜模型）
4. 启用成本跟踪和告警

---

## 📚 更多资源

- **OpenRouter 文档**: https://openrouter.ai/docs
- **模型列表**: https://openrouter.ai/models
- **定价**: https://openrouter.ai/models（每个模型页面）
- **DeepAnalyze 项目**: https://github.com/ruc-datalab/DeepAnalyze

---

## 🤝 贡献

发现问题或有改进建议？欢迎提交 Issue 或 Pull Request！

---

**祝使用愉快！** 🚀
