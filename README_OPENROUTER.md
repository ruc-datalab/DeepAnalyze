# 🌐 OpenRouter 集成更新

## 🎉 新功能：DeepAnalyze 现已支持 OpenRouter！

通过 OpenRouter API，您现在可以使用 100+ 大模型，包括：
- 🤖 **Claude 3.5 / Sonnet 4.5** (Anthropic)
- 💬 **GPT-4o / GPT-4 Turbo** (OpenAI)
- 🌟 **Gemini 2.0** (Google)
- 🚀 **DeepSeek R1** (最佳性价比)
- 🦙 **Llama 3.1 / 3.2** (Meta)
- 以及更多...

### ✨ 主要特性

- ✅ **统一接口**：一个 API key 访问所有模型
- ✅ **灵活切换**：环境变量即可切换本地/云端模型
- ✅ **自动降级**：主模型失败时自动切换备用
- ✅ **成本优化**：根据任务复杂度选择合适模型
- ✅ **零代码修改**：兼容现有 DeepAnalyze API

## 📦 新增文件

```
DeepAnalyze/
├── deepanalyze.py              # 新增 DeepAnalyzeOpenRouter 类
├── demo/backend.py             # 支持 OpenRouter 配置
├── .env.example                # 环境变量配置模板
├── example_openrouter.py       # 完整使用示例
├── QUICKSTART_OPENROUTER.md    # 5分钟快速开始
├── OPENROUTER_GUIDE.md         # 详细集成指南
└── README_OPENROUTER.md        # 本文件
```

## 🚀 快速开始

### 1. 获取 API Key

访问 [OpenRouter](https://openrouter.ai/keys) 创建免费账号并获取 API key。

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，设置：
# USE_OPENROUTER=true
# OPENROUTER_API_KEY=sk-or-v1-your-key-here
# OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### 3. 使用

#### 命令行使用

```python
from deepanalyze import DeepAnalyzeOpenRouter

deepanalyze = DeepAnalyzeOpenRouter()
result = deepanalyze.generate(prompt, workspace="./data")
print(result["reasoning"])
```

#### Demo 界面

```bash
cd demo
bash start.sh
# 访问 http://localhost:4000
```

完整教程请查看：[QUICKSTART_OPENROUTER.md](./QUICKSTART_OPENROUTER.md)

## 🎯 推荐模型

| 需求 | 推荐模型 | 成本 |
|------|---------|------|
| 🏆 最佳质量 | `anthropic/claude-3.5-sonnet` | $3/$15 per 1M |
| 💰 最佳性价比 | `deepseek/deepseek-r1` | $0.55/$2.19 per 1M |
| ⚡ 最快速度 | `openai/gpt-4o` | $2.5/$10 per 1M |
| 🆓 免费测试 | `google/gemini-2.0-flash-thinking-exp` | 免费 |

更多模型：https://openrouter.ai/models

## 💡 使用场景

### 场景 1：开发测试（使用免费模型）

```bash
OPENROUTER_MODEL=google/gemini-2.0-flash-thinking-exp
```

### 场景 2：生产环境（高质量）

```bash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### 场景 3：批量处理（省钱）

```bash
OPENROUTER_MODEL=deepseek/deepseek-r1
```

### 场景 4：混合策略

```python
# 复杂任务用 Claude
claude = DeepAnalyzeOpenRouter(model_name="anthropic/claude-3.5-sonnet")

# 简单任务用 DeepSeek（省 80% 成本）
deepseek = DeepAnalyzeOpenRouter(model_name="deepseek/deepseek-r1")
```

## 📊 成本对比

处理 1000 个数据分析任务（每个 6000 tokens）：

| 方案 | 总成本 | 说明 |
|------|-------|------|
| **本地 DeepAnalyze-8B** | $50-100 | GPU 租赁成本，无限次使用 |
| **Claude 3.5 Sonnet** | $66 | 最高质量 |
| **DeepSeek R1** | $10 | 💰 性价比之王 |
| **GPT-4o** | $45 | 平衡选择 |
| **Gemini Flash** | $0 | 🆓 免费（限额内） |

## 🔄 迁移指南

### 从本地模型迁移

**之前**：
```python
from deepanalyze import DeepAnalyzeVLLM
deepanalyze = DeepAnalyzeVLLM("DeepAnalyze-8B")
```

**现在**：
```python
from deepanalyze import DeepAnalyzeOpenRouter
deepanalyze = DeepAnalyzeOpenRouter()  # 自动读取环境变量
```

### Demo 迁移

只需修改 `.env` 文件：
```bash
USE_OPENROUTER=true
OPENROUTER_API_KEY=your-key
```

无需修改任何代码！

## 🆚 本地模型 vs OpenRouter

| 特性 | 本地 DeepAnalyze-8B | OpenRouter |
|------|-------------------|-----------|
| **成本** | GPU 租赁 | 按 token 计费 |
| **质量** | 优化的 8B 模型 | 可选 Claude/GPT-4 等 |
| **延迟** | ~100ms | ~500-1000ms |
| **上下文** | 32K tokens | 最高 200K tokens |
| **维护** | 需要管理服务器 | 零维护 |
| **灵活性** | 单一模型 | 100+ 模型随意切换 |

**建议**：
- 🏠 **大量使用**：本地模型更经济
- 🌐 **偶尔使用**：OpenRouter 更方便
- 🔀 **混合使用**：根据任务选择

## 📚 文档

- 📖 [快速开始](./QUICKSTART_OPENROUTER.md) - 5 分钟上手
- 📘 [完整指南](./OPENROUTER_GUIDE.md) - 详细文档
- 💻 [代码示例](./example_openrouter.py) - 可运行的示例
- 🌐 [OpenRouter 官方文档](https://openrouter.ai/docs)

## ❓ 常见问题

**Q: 是否会影响现有功能？**
A: 不会！完全向后兼容，现有代码无需修改。

**Q: 如何在两种方案间切换？**
A: 修改 `.env` 中的 `USE_OPENROUTER` 变量即可。

**Q: 哪个模型最好？**
A:
- 质量优先：Claude 3.5 Sonnet
- 成本优先：DeepSeek R1
- 平衡选择：GPT-4o
- 免费测试：Gemini Flash

**Q: 成本如何？**
A: 典型数据分析任务约 $0.01-0.07/次，比雇佣分析师便宜 1000 倍！

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

与主项目相同许可。

---

**立即体验 100+ 大模型的强大能力！** 🚀

查看快速开始：[QUICKSTART_OPENROUTER.md](./QUICKSTART_OPENROUTER.md)
