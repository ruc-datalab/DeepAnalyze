# Chat Demo

`demo/chat_v2` 是 DeepAnalyze 的浏览器交互 Demo，包含后端 API、workspace 文件层、前端界面，以及本地和 Docker 两种执行模式。

[English README](./README.md)

## 功能概览

- 支持在 workspace 中上传和管理表格、数据库、文本、日志与文档
- 支持常见 workspace 文件的在线预览
- 支持流式展示 `<Analyze> / <Understand> / <Code> / <Execute> / <File> / <Answer>` 区块
- 支持在 workspace 中执行 Python 分析代码
- 支持导出 Markdown 和 PDF 报告
- 支持中英文界面切换
- 支持 local / docker 两种代码执行模式

## 运行前准备

### 1. 模型服务

先启动 DeepAnalyze 模型服务，例如：

```bash
vllm serve DeepAnalyze-8B
```

默认连接 `http://localhost:8000` 附近的 OpenAI 兼容接口。

### 2. Python 与 Node.js

建议：

- Python 使用现有 DeepAnalyze 环境，例如 `deepanalyze`
- Node.js 使用可运行当前 Next.js 前端的版本

首次运行前端前先安装依赖：

```bash
cd demo/chat_v2/frontend
npm install
cd ..
```

### 3. 环境变量

使用示例配置文件：

```bash
cd demo/chat_v2
cp .env.example .env
```

Windows：

```powershell
cd demo/chat_v2
Copy-Item .env.example .env
```

## 执行模式

### local 模式

这是默认推荐模式，适合本机已经具备 Python 数据分析依赖的场景：

```env
DEEPANALYZE_EXECUTION_MODE=local
```

### docker 模式

适合希望隔离执行环境的场景：

```env
DEEPANALYZE_EXECUTION_MODE=docker
```

注意：

- 系统不会自动构建 Docker 镜像
- 如果目标机器没有镜像，Docker 执行会直接失败
- 需要先手动构建镜像

示例：

```bash
cd demo/chat_v2
docker build -t deepanalyze-chat-exec:latest -f Dockerfile.exec .
```

## 如何运行

### Linux / macOS

```bash
cd demo/chat_v2
bash start.sh
```

停止：

```bash
cd demo/chat_v2
bash stop.sh
```

### Windows

```bat
cd demo\chat
start.bat
```

停止：

```bat
cd demo\chat
stop.bat
```

默认地址：

- 前端：`http://localhost:4000`
- 后端 API：`http://localhost:8200`
- 文件服务：`http://localhost:8100`

## PDF 导出

PDF 导出依赖：

- `pypandoc`
- `pandoc`
- `xelatex`

行为说明：

- 如果缺少 `pandoc`，后端会尝试自动下载（默认开启）。
- `xelatex` 仍是必需项，需要手动安装。
- 可通过以下环境变量控制：
  - `DEEPANALYZE_PDF_AUTO_DOWNLOAD_PANDOC`（默认 `true`）
  - `DEEPANALYZE_PDF_PANDOC_CACHE_DIR`（可选，指定 pandoc 缓存目录）

## 目录说明

- `backend.py`：后端启动入口
- `backend_app/`：FastAPI 后端实现
- `frontend/`：Next.js 前端
- `Dockerfile.exec`：执行镜像定义
- `workspace/`：按 session 隔离的工作区
- `logs/`：运行日志
