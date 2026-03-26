# Chat Demo

`demo/chat_v2` is the browser-based DeepAnalyze demo. It includes the backend API, the workspace/file layer, the frontend UI, and both local and Docker execution modes.

[Chinese Version](./README_ZH.md)

## Features

- Upload and manage tables, databases, text files, logs, and documents in the workspace
- Preview common workspace files directly in the UI
- Stream structured `<Analyze> / <Understand> / <Code> / <Execute> / <File> / <Answer>` blocks
- Execute Python analysis code inside the workspace
- Export Markdown and PDF reports
- Switch between Chinese and English UI
- Run code either locally or inside Docker
- Multiple LLM providers: Local (vLLM), HeyWhale API, and [MiniMax AI](https://www.minimaxi.com)

## Prerequisites

### 1. Model service

Start a DeepAnalyze model service first, for example:

```bash
vllm serve DeepAnalyze-8B
```

By default the chat demo connects to an OpenAI-compatible endpoint around `http://localhost:8000`.

Alternatively, you can switch the model provider in the UI to **MiniMax AI** and use cloud-hosted models (MiniMax-M2.7) without running a local model service. Just select "MiniMax AI" from the Model Provider dropdown and enter your API key from [MiniMax platform](https://www.minimaxi.com).

### 2. Python and Node.js

Recommended setup:

- Python: use your existing DeepAnalyze environment, for example `deepanalyze`
- Node.js: use a version that can run the bundled Next.js frontend

Install frontend dependencies once:

```bash
cd demo/chat_v2/frontend
npm install
cd ..
```

### 3. Environment variables

Use the sample config file:

```bash
cd demo/chat_v2
cp .env.example .env
```

Windows:

```powershell
cd demo/chat_v2
Copy-Item .env.example .env
```

## Execution Modes

### Local mode

Recommended as the default if the local machine already has the required Python data-analysis dependencies.

```env
DEEPANALYZE_EXECUTION_MODE=local
```

### Docker mode

Use this if you want an isolated execution environment.

```env
DEEPANALYZE_EXECUTION_MODE=docker
```

Important:

- The system does not auto-build the Docker image
- If the target machine has no image, Docker execution will fail immediately
- You must build the image manually first

Example:

```bash
cd demo/chat_v2
docker build -t deepanalyze-chat-exec:latest -f Dockerfile.exec .
```

## Run

### Linux / macOS

```bash
cd demo/chat_v2
bash start.sh
```

Stop:

```bash
cd demo/chat_v2
bash stop.sh
```

### Windows

```bat
cd demo\chat
start.bat
```

Stop:

```bat
cd demo\chat
stop.bat
```

Default addresses after startup:

- Frontend: `http://localhost:4000`
- Backend API: `http://localhost:8200`
- File service: `http://localhost:8100`

## PDF Export

PDF export depends on:

- `pypandoc`
- `pandoc`
- `xelatex`

If `pandoc` or `xelatex` is missing, the UI will show an explicit error message. The system does not install these dependencies automatically.

## Directory Overview

- `backend.py`: backend startup entry
- `backend_app/`: FastAPI backend implementation
- `frontend/`: Next.js frontend
- `Dockerfile.exec`: Docker image for code execution
- `workspace/`: per-session workspace
- `logs/`: runtime logs
