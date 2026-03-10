# Chat Demo

`demo/chat` 是 DeepAnalyze 的浏览器交互 Demo，包含后端 API、文件工作区、前端界面，以及本地/容器两种代码执行模式。

## 功能概览

- 支持上传表格、数据库、文本等多种数据文件到 workspace
- 支持图片、日志、文档等文件统一管理与预览
- 支持流式展示 `<Analyze> / <Understand> / <Code> / <Execute> / <File> / <Answer>` 区块
- 支持在 workspace 中执行 Python 分析代码
- 支持本地执行模式和 Docker 执行模式
- 支持导出 Markdown 报告和 PDF 报告
- 支持中英文界面切换

## 运行前准备

### 1. 模型服务

先启动 DeepAnalyze 的模型服务，例如：

```bash
vllm serve DeepAnalyze-8B
```

默认会连接 `http://localhost:8000` 附近的 OpenAI 风格接口。若你有自定义地址，请同步修改前端/后端配置。

### 2. Python 与 Node 环境

建议：

- Python 使用项目已有环境，例如 `deepanalyze`
- Node.js 使用可运行 Next.js 的版本

前端首次运行前需要安装依赖：

```bash
cd demo/chat/frontend
npm install
cd ..
```

### 3. 配置环境变量

`demo/chat/.env.example` 提供了执行后端相关示例配置。

可按需复制为本地配置文件：

```bash
cd demo/chat
cp .env.example .env
```

Windows:

```powershell
cd demo/chat
Copy-Item .env.example .env
```

## 执行模式说明

### local 模式

适合本机已经具备 Python 数据分析依赖的场景：

```env
DEEPANALYZE_EXECUTION_MODE=local
```

### docker 模式

适合隔离执行环境的场景：

```env
DEEPANALYZE_EXECUTION_MODE=docker
```

注意：

- 系统不会自动构建 Docker 镜像
- 如果目标机器没有镜像，启动分析执行时会直接报错
- 需要先手动构建镜像

示例命令：

```bash
cd demo/chat
docker build -t deepanalyze-chat-exec:latest -f Dockerfile.exec .
```

## 如何运行

### Linux / macOS

```bash
cd demo/chat
bash start.sh
```

停止：

```bash
cd demo/chat
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

启动后默认地址：

- 前端：`http://localhost:4000`
- 后端 API：`http://localhost:8200`
- 文件服务：`http://localhost:8100`

## PDF 导出说明

PDF 导出依赖以下组件：

- `pypandoc`
- `pandoc`
- `xelatex`

如果缺少 `pandoc` 或 `xelatex`，界面会给出明确提示。当前不会自动安装这些依赖。

## 目录说明

- `backend.py`：后端启动入口
- `backend_app/`：FastAPI 后端实现
- `frontend/`：Next.js 前端界面
- `Dockerfile.exec`：Docker 执行镜像定义
- `workspace/`：会话级工作区
- `logs/`：运行日志

## 常见问题

### 1. 换到另一台机器后 Docker 模式不能直接运行

这是当前设计行为。因为系统不会自动构建镜像，你需要先手动执行：

```bash
docker build -t deepanalyze-chat-exec:latest -f Dockerfile.exec .
```

### 2. PDF 导出失败

优先检查：

- 是否安装 `pandoc`
- 是否安装 `xelatex`
- 后端日志 `demo/chat/logs/backend.log`

### 3. 只想快速体验，不想准备 Docker

把 `.env` 中的执行模式改为：

```env
DEEPANALYZE_EXECUTION_MODE=local
```

前提是本机 Python 环境已经具备运行数据分析代码所需依赖。
