# DeepAnalyze General
**设计了专用提示词，让通用大模型可以仿照DeepAnalyze-8b `<Analyze>`-> `<Code>` -> `<Execute>`的执行轨迹，完成数据分析任务**


## 🚀 快速体验支持

提供专用 `system prompt`，并全面兼容 OpenAI API（如阿里云百炼等平台），让用户能够基于通用大模型快速上手 DeepAnalyze 的 Agentic 范式，无需复杂配置即可体验DeepAnalyze。

## 💬 新增 `<Ask>` 交互模式

在原有 `<Action>` 标签体系基础上，新增可选的 `<Ask>` 标签。  
DeepAnalyze 可在任务执行过程中主动向用户提问，以获取其无法自行推断的关键背景信息：从而：

- 提升任务完成质量  
- 避免因信息缺失导致的方向性错误  
- 防止错误累积、积重难返  

## ⚙️ `generate` 流程优化

为提升执行效率与用户体验，对代码生成与执行流程进行了部分优化：

- **自动过滤警告信息**：执行 `<Code>` 块时自动屏蔽 `warning` 日志，避免占用宝贵的上下文窗口。
- **共享代码执行环境**：修复多个 `<Code>` 块之间变量隔离的问题。通过统一命名空间（namespace），实现跨代码块的变量与状态共享（例如前一个 `<Code>` 中定义的 `df` 可在后续 `<Code>` 中直接使用），显著简化逻辑编写。
- **预导入常用依赖**：执行环境已预加载常用 Python 库（如 `pandas`等），无需在每个 `<Code>` 块中重复 `import`，进一步精简用户输入。（根据使用模型的能力而定，不一定会遵循提示词）

## 🧪 模拟场景示例

在 [`example/financial_insights_and_api_usage_analytics`](./example/financial_insights_and_api_usage_analytics) 目录下，我们提供了两个贴近真实使用场景的测试文件，帮助开发者快速理解、验证和集成 DeepAnalyze 的各项功能。


## 具体使用方式

# DeepAnalyze General
**设计了专用提示词，让通用大模型可以仿照DeepAnalyze-8b `<Analyze>`-> `<Code>` -> `<Execute>`的执行轨迹，完成数据分析任务**


## 🚀 快速体验支持

提供专用 `system prompt`，并全面兼容 OpenAI API（如阿里云百炼等平台），让用户能够基于通用大模型快速上手 DeepAnalyze 的 Agentic 范式，无需复杂配置即可体验核心能力。

## 💬 新增 `<Ask>` 交互模式

在原有 `<Action>` 标签体系基础上，新增可选的 `<Ask>` 标签。  
DeepAnalyze 可在任务执行过程中主动向用户提问，以获取其无法自行推断的关键背景信息：从而：

- 提升任务完成质量  
- 避免因信息缺失导致的方向性错误  
- 防止错误累积、积重难返  

## ⚙️ `generate` 流程优化

为提升执行效率与用户体验，对代码生成与执行流程进行了部分优化：

- **自动过滤警告信息**：执行 `<Code>` 块时自动屏蔽 `warning` 日志，避免占用宝贵的上下文窗口。
- **共享代码执行环境**：修复多个 `<Code>` 块之间变量隔离的问题。通过统一命名空间（namespace），实现跨代码块的变量与状态共享（例如前一个 `<Code>` 中定义的 `df` 可在后续 `<Code>` 中直接使用），显著简化逻辑编写。
- **预导入常用依赖**：执行环境已预加载常用 Python 库（如 `pandas`等），无需在每个 `<Code>` 块中重复 `import`，进一步精简用户输入。（根据使用模型的能力而定，不一定会遵循提示词）

## 🧪 模拟场景示例

在 [`example/financial_insights_and_api_usage_analytics`](./example/financial_insights_and_api_usage_analytics) 目录下，我们提供了两个贴近真实使用场景的测试文件，帮助开发者快速理解、验证和集成 DeepAnalyze 的各项功能。


## 具体使用方式

1. 设置环境变量DASHSCOPE_API_KEY

2. 创建DeepAnalyzeVLLM实例：
   ```python
   deepanalyze = DeepAnalyzeVLLM(
       model_name="qwen3-coder-30b-a3b-instruct",  # 指定使用的模型
       is_interactive=True  # 是否启用交互模式（允许使用<Ask>标签）
   )
   ```

3. 准备数据分析任务描述和工作目录：
   ```python
   task = "数据分析任务描述"
   workspace = "example/financial_insights_and_api_usage_analytics"  # 工作目录路径
   ```

4. 调用generate方法执行数据分析任务：
   ```python
   task_execute_trace = deepanalyze.generate(
       prompt=task,
       workspace=workspace,
       temperature=0.3,  # 控制生成的随机性
       top_p=1.0,        # 控制生成的多样性
   )
   ```

5. 查看任务执行结果：
   ```python
   print(task_execute_trace["reasoning"])  # 输出完整的推理过程,另外执行过程中控制台也会逐步打印动作标签
   ```

### 参数说明

- `model_name`: 指定使用的模型名称，如"qwen3-coder-30b-a3b-instruct"
- `is_interactive`: 是否启用交互模式，默认为False
  - True: 启用<Ask>标签，允许模型在需要时向用户提问
  - False: 使用基础模式，仅支持`<Analyze>`->`<Code>`->`<Execute>`流程
- `prompt`: 数据分析任务的具体描述
- `workspace`: 工作目录路径，包含待分析的数据文件
- `temperature`: 控制生成文本的随机性，值越低结果越确定
- `top_p`: 控制生成文本的多样性，值越高结果越多样化

### 支持的动作标签

- `<Analyze>`: 任务规划、推理、假设、结果解读或反思
- `<Understand>`: 主动表达对数据源结构与语义的探索意图
- `<Code>`: 生成可执行的Python代码(pandas/numpy/matplotlib等)
- `<Ask>`: (仅在交互模式下可用)向用户提问获取关键背景信息
- `<Execute>`: 系统自动注入，包含代码执行结果或用户回答
- `<Finish>`: 输出最终结论、报告或建议 

### 参考任务示例

```plaintext
数据分析任务说明：

你是一名银行数据分析师，现在需要对一批个人贷款客户的数据进行分析。请使用提供的Excel文件（bank_data.xlsx）完成以下任务：

高价值客户特征分析：
   - 分析高价值客户的年龄分布特点
   - 分析高价值客户的收入水平特点
   - 分析高价值客户的贷款金额特点
```

```plaintext
数据分析任务 - 接口调用情况分析

请基于interface_calls.xlsx文件中的数据，回答以下问题：

1. 应用来源分析：不同应用来源（网页应用, 移动应用等）对接口的调用分布如何？哪种应用来源的调用量最大？  
```
