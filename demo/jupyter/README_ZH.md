# DeepAnalyze Jupyter 前端

中文版 | [English](./README.md)

## 项目简介

DeepAnalyze Jupyter 前端是 DeepAnalyze 数据科学智能体的 Jupyter Notebook 接口实现。本项目提供了一个交互式环境，让用户可以通过熟悉的 Jupyter Notebook 界面与 DeepAnalyze 模型进行交互，实现数据准备、分析、建模、可视化和报告生成等完整数据科学流程。

## 快速开始

### 环境要求
- uv (统一管理 Python 运行与依赖)
- Node.js (用于运行 Jupyter MCP 服务器)
- OpenAI 兼容的 API 服务器 (如 vLLM 部署的 DeepAnalyze-8B)

### 安装步骤

1. 安装必要环境
```bash
# 安装node.js
curl -qL https://www.npmjs.com/install.sh | sh
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

具体可详见 [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/) 与 [Node.js 安装文档](https://nodejs.org/en/download/package-manager)。

2. 克隆项目并进入目录
```bash
git clone https://github.com/ruc-datalab/DeepAnalyze.git
cd demo/jupyter
```

3. 安装依赖
```bash
# 使用 uv sync 完成所有环境依赖一键安装
uv sync
```

如遇到安装网络问题,可以考虑在`pyproject.toml`末尾添加镜像：
```toml
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

可以使用`uv add`命令添加新的依赖,例如：
```bash
uv add xgboost
```

4. 配置环境变量与配置文件
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 配置
# 编辑 config.toml 文件，填入相关配置
```

5. 启动终端交互式聊天
```bash
uv run CLI.py
```

### 使用示例

运行`uv run CLI.py`后，会在`jupyter`目录下创建`workspace`目录作为Jupyter工作目录,并创建`deep_analyze.ipynb`作为分析文件,你可以在Jupyter Lab中上传所需的数据集(或者直接复制到`workspace`目录下)。

在终端中输入以下命令,即可开始与 DeepAnalyze 模型交互：
```
>>> 分析这个数据集并生成可视化图表
>>> 对这些数据进行回归分析并解释结果
>>> 创建一个预测模型并评估其性能
```

所有的分析任务都将在`deep_analyze.ipynb`中执行,你可以在Jupyter Lab中查看结果并导出报告。

## 项目结构

```
jupyter/
├── CLI.py          # 命令行交互界面
├── server.py       # 核心服务器逻辑
├── mcp_tools.py    # MCP 工具集，用于与 Jupyter 交互
├── utils.py        # 实用工具函数
├── config.toml     # 配置文件
├── .env.example    # 环境变量示例
└── test/           # 测试文件
```

该项目主要依赖[datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server)实现与 Jupyter Notebook 的交互。

Jupyter MCP 是目前Github上Jupyter相关MCP项目最主流的实现,欢迎大家使用与Star支持。