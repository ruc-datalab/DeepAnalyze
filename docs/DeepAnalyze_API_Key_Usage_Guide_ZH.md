# DeepAnalyze API Key 使用指南（中文版本）

本文档介绍如何使用 DeepAnalyze API 进行在线调试，以及如何在本地部署并运行 Gradio 应用前端。英文版本请见：`docs/DeepAnalyze API Key Usage Guide (English Version).md`。

如需申请 7 天有效的 DeepAnalyze API Key，请填写该 Google Form：https://forms.gle/YxVkCzczqq8jeciw9

**🧑‍💻 文档作者 & Demo 开发者**：李浩鸣，现任上海合津信息科技有限公司（HeyWhale Technology）AI 数据工程师，香港城市大学数据科学硕士。专注大语言模型（LLM）与 Agent 系统开发，具备端到端 LLM 应用部署经验。负责 DeepAnalyze 生态中 ModelWhale 平台的模型部署、前端交互与应用迭代。曾参与高并发智能编程助手服务建设，技术栈包含 FastAPI、LangChain、LangGraph 等，实现 LLM 接入、流式输出与状态管理。熟悉 Hadoop/Spark 数据生态，具备大规模数据 ETL 开发与数仓运营经验，擅长从算法研发到业务交付的全流程工程化落地，致力于推动数据智能系统的稳健部署与性能优化。GitHub：https://github.com/LHMQ878

# 1. API 接口调用

提供两种调用方式：

## 1.1 纯 Prompt 请求示例

```
curl -X POST https://www.heywhale.com/api/model/services/691d42c36c6dda33df0bf645/app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"messages": [{"role": "user", "content": "Who are you?"}]}'
```

## 1.2 Prompt + 文件上传请求示例

首先下载 zip 包：[https://open-cdn.kesci.com/admin/t7v5mzpmw/da_quick_start.zip](https://open-cdn.kesci.com/admin/t7v5mzpmw/da_quick_start.zip?referrer=grok.com)

然后在终端执行：

```
python test_streaming.py
```

### 1.2.1 交互式使用

启动脚本后，按提示依次操作：

**Step 1**：输入 API Key
```
Enter API Key: your_api_key_here
```

**Step 2**：选择对话类型
```
Select dialog type:
  1. No-file dialog
  2. Dialog with files

Enter choice (1 or 2):
```

**Step 3**：输入文件路径（选择 2 时需要）

```
Enter file paths (comma separated):
```

**Step 4**：输入分析指令（可选）

```
Enter analysis instruction (blank for default):
```

### 1.2.2 使用示例

**示例 1**：分析 CSV 文件

**场景**：分析 `Simpson.csv` 数据文件

**步骤**：

1. 运行脚本：
   - python test_streaming.py
2. 输入 API Key：
   - Enter API Key: your_api_key
3. 选择对话类型：
   - Enter choice (1 or 2): 2
4. 输入文件路径：
   - Enter file paths (comma separated): Simpson.csv
   - 或使用绝对路径：
   - Enter file paths (comma separated): D:\da_gradio\test\Simpson.csv
5. 输入分析指令（可选）：
   - Enter analysis instruction (blank for default):
   - 若留空，将使用默认指令：分析数据文件，进行 EDA，并生成可视化结果。

**期望输出**：

- 脚本会自动将文件上传到 API 服务端
- 实时流式展示分析结果
- 若生成了可视化文件，会显示生成文件数量

**示例 2**：使用 ZIP 压缩包

**场景**：分析 `example.zip` 内部的文件

**步骤**：

1. 运行脚本：
   - python test_streaming.py
2. 输入 API Key：
   - Enter API Key: your_api_key
3. 选择对话类型：
   - Enter choice (1 or 2): 2
4. 输入 ZIP 文件路径：
   - Enter file paths (comma separated): example.zip
5. 输入自定义分析指令（可选）：
   - Enter analysis instruction (blank for default): Please analyze all files in the archive, identify key patterns and outliers

**说明**：

- ZIP 文件会自动解压
- 仅处理支持的文件格式（见下方支持格式说明）
- 若 ZIP 内含多个文件，将上传并分析其中所有受支持的文件

**示例 3**：无文件对话

**场景**：不上传文件的普通对话

**步骤**：

1. 运行脚本：
   - python test_streaming.py
2. 输入 API Key：
   - Enter API Key: your_api_key
3. 选择对话类型：
   - Enter choice (1 or 2): 1
4. 输入对话内容：
   - Enter analysis instruction (blank for default): Please explain what machine learning is

# 2. Gradio 应用本地部署

**目的**：帮助你快速在本地部署、启动并体验 DeepAnalyze 的 Gradio 前端应用。

## 2.1 环境与准备

**环境要求**：

- 操作系统：Windows / macOS / Linux
- Python：3.8 或更高版本
- 依赖：gradio、openai、fastapi、uvicorn、pandas、requests、python-multipart

**所需文件**：

- 代码文件：[https://open-cdn.kesci.com/admin/t7v5tp139f/DeepAnalyze_Gradio.zip](https://open-cdn.kesci.com/admin/t7v5tp139f/DeepAnalyze_Gradio.zip?referrer=grok.com)
- 测试样例文件：
  - [Simpson.csv](https://open-cdn.kesci.com/admin/t7r80krss/Simpson.csv?referrer=grok.com)（示例数据集）
  - [example.zip](https://open-cdn.kesci.com/admin/t62le3tdx/example.zip?referrer=grok.com)（包含多个文件的 ZIP 示例）

## 2.2 安装步骤

1. 进入项目目录
```
cd project
```

2. 安装依赖（二选一）
```
pip install gradio openai fastapi uvicorn pandas requests python-multipart
```

或
```
pip install -r requirements.txt
```

## 2.3 启动 Gradio 前端

提供两种界面版本——任选其一：

- 英文界面：python app.py
- 中文界面：python app_ZH.py

启动后程序将：

1. 在后台启动 API 服务
   - 默认：http://localhost:8200
   - 文件服务：http://localhost:8100
   - 健康检查：http://localhost:8200/health
2. 启动 Gradio 前端（默认：http://localhost:8080）

如果端口被占用，可在 `config.py` 或 `app.py` 中调整端口。

## 2.4 前端使用流程

1. 打开浏览器访问：http://localhost:8080
2. 输入 API Key（必填）
3. 可选上传文件：支持单/多文件；ZIP 会自动解压。常见格式包括 CSV/Excel/PDF/图片/代码/日志等
4. 输入分析指令，或使用预置按钮（如 Data Overview、Trend Analysis）
5. 点击 “Start Analysis”，右侧观察流式输出，同时可预览/下载文件

## 2.5 快速测试示例

- 直接上传 [Simpson.csv](https://open-cdn.kesci.com/admin/t7r80krss/Simpson.csv?referrer=grok.com)，并输入类似 “Please analyze and visualize the relationships between main variables” 的指令
- 上传 [example.zip](https://open-cdn.kesci.com/admin/t62le3tdx/example.zip?referrer=grok.com) 以验证自动解压与多文件分析
