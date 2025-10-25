# 🚀 快速开始：使用 OpenRouter

5 分钟内让 DeepAnalyze 接入 100+ 大模型！

## 📋 前提条件

- Python 3.12+
- OpenRouter API Key（免费注册：https://openrouter.ai/）

## ⚡ 3 步完成集成

### 1️⃣ 获取 API Key

访问 [OpenRouter Keys](https://openrouter.ai/keys)，创建一个新的 API key。

### 2️⃣ 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

设置以下内容：

```bash
# 启用 OpenRouter
USE_OPENROUTER=true

# 设置你的 API Key
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# 选择模型（可选，默认使用 Claude 3.5 Sonnet）
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### 3️⃣ 运行！

#### 方式 A：使用 Demo（推荐新手）

```bash
cd demo
bash start.sh
```

访问 http://localhost:4000，上传数据开始分析！

#### 方式 B：Python 脚本

```python
from deepanalyze import DeepAnalyzeOpenRouter

# 初始化（会自动读取 .env 中的配置）
deepanalyze = DeepAnalyzeOpenRouter()

# 分析数据
prompt = """# Instruction
分析这个数据集并生成报告。

# Data
File 1: {"name": "sales.csv", "size": "10KB"}
"""

result = deepanalyze.generate(prompt, workspace="./data")
print(result["reasoning"])
```

#### 方式 C：运行示例

```bash
# 编辑示例文件，设置你的 API key
nano example_openrouter.py

# 运行
python example_openrouter.py
```

## 🎯 模型选择

根据需求选择合适的模型：

### 最佳质量（推荐）

```bash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
# 成本: $3/$15 per 1M tokens
```

### 最佳性价比

```bash
OPENROUTER_MODEL=deepseek/deepseek-r1
# 成本: $0.55/$2.19 per 1M tokens (便宜 80%)
```

### 免费测试

```bash
OPENROUTER_MODEL=google/gemini-2.0-flash-thinking-exp
# 成本: 免费（有配额限制）
```

### 平衡选择

```bash
OPENROUTER_MODEL=openai/gpt-4o
# 成本: $2.5/$10 per 1M tokens
```

更多模型：https://openrouter.ai/models

## 🔄 在本地模型和 OpenRouter 之间切换

修改 `.env` 文件：

```bash
# 使用 OpenRouter
USE_OPENROUTER=true

# 切换回本地模型
USE_OPENROUTER=false
```

无需修改代码，重启服务即可！

## 📊 成本估算

假设一个典型的数据分析任务：
- 输入：2000 tokens（数据描述 + 指令）
- 输出：4000 tokens（分析 + 代码 + 结果）
- 总计：6000 tokens

| 模型 | 单次成本 | 100 次 | 1000 次 |
|------|---------|-------|---------|
| **Claude 3.5 Sonnet** | $0.066 | $6.6 | $66 |
| **DeepSeek R1** | $0.010 | $1.0 | $10 💰 |
| **GPT-4o** | $0.045 | $4.5 | $45 |
| **Gemini Flash** | $0 | $0 | $0 (限额内) 🆓 |

💡 **提示**：混合使用！简单任务用便宜模型，复杂任务用高端模型。

## ⚙️ 高级用法

### 在代码中动态切换模型

```python
from deepanalyze import DeepAnalyzeOpenRouter

# 复杂任务：使用 Claude
claude = DeepAnalyzeOpenRouter(model_name="anthropic/claude-3.5-sonnet")
complex_result = claude.generate(complex_prompt, workspace)

# 简单任务：使用 DeepSeek（省 80% 成本）
deepseek = DeepAnalyzeOpenRouter(model_name="deepseek/deepseek-r1")
simple_result = deepseek.generate(simple_prompt, workspace)
```

### 自定义参数

```python
deepanalyze = DeepAnalyzeOpenRouter(
    model_name="anthropic/claude-3.5-sonnet",
    api_key="sk-or-v1-xxxxx",  # 或从环境变量读取
    max_rounds=20,  # 最大推理轮数
    site_url="https://your-app.com",  # 可选
)

result = deepanalyze.generate(
    prompt,
    workspace="./data",
    temperature=0.3,  # 更确定性的输出
    max_tokens=8000,  # 限制长度
    top_p=0.9,
)
```

## 🐛 常见问题

### ❌ "Access denied" 错误

**原因**：API key 无效或未激活

**解决**：
1. 检查 API key 是否正确复制
2. 访问 https://openrouter.ai 确认账号状态
3. 确认是否有足够余额（某些模型需要充值）

### ❌ "Model not found" 错误

**原因**：模型名称错误

**解决**：
- 检查模型名称格式：`provider/model-name`
- 查看可用模型列表：https://openrouter.ai/models
- 常见错误：
  - ❌ `claude-3.5-sonnet`
  - ✅ `anthropic/claude-3.5-sonnet`

### ❌ 响应很慢

**原因**：API 延迟或模型负载高

**解决**：
1. 选择 "fast" 类型的模型（如 gpt-4o, gemini-flash）
2. 启用 fallback 自动切换到可用模型
3. 考虑使用本地模型

### 💰 成本控制

1. **设置月度限额**：在 OpenRouter 后台设置
2. **使用便宜模型**：DeepSeek R1 比 Claude 便宜 80%
3. **监控使用**：查看 https://openrouter.ai/usage
4. **混合策略**：简单任务用便宜模型

## 📚 下一步

- 📖 阅读完整指南：[OPENROUTER_GUIDE.md](./OPENROUTER_GUIDE.md)
- 🔍 查看更多示例：[example_openrouter.py](./example_openrouter.py)
- 🌐 浏览 OpenRouter 文档：https://openrouter.ai/docs
- 💬 加入讨论：查看项目 Issues

## 🆘 需要帮助？

- 🐛 提交 Issue：https://github.com/ruc-datalab/DeepAnalyze/issues
- 💬 讨论区：https://github.com/ruc-datalab/DeepAnalyze/discussions
- 📧 OpenRouter 支持：https://openrouter.ai/support

---

**开始探索吧！** 🎉
