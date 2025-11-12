# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepAnalyze is an agentic Large Language Model for autonomous data science. It can perform end-to-end data science tasks including data preparation, analysis, modeling, visualization, and report generation without human intervention.

## Architecture

### Core Components

1. **Main Library (`deepanalyze.py`)**: Core DeepAnalyzeVLLM class that provides multi-round reasoning and code execution capabilities using vLLM API
2. **Demo Application (`demo/`)**: Full-stack web interface with:
   - `backend.py`: FastAPI server providing OpenAI-compatible API endpoints
   - `chat/`: Next.js frontend with React components
   - WebSocket-based real-time communication
3. **Training Infrastructure (`deepanalyze/`)**: Model training components using:
   - `ms-swift`: Fine-tuning framework
   - `SkyRL`: Reinforcement learning training framework
4. **Evaluation Framework (`playground/`)**: Comprehensive evaluation system for data science benchmarks

### Key Features

- **Multi-round Reasoning**: Supports iterative code execution and refinement
- **File Management**: Automatic workspace handling with file uploads and organization
- **Code Execution**: Safe sandboxed Python code execution with timeout protection
- **Report Generation**: Automated PDF and Markdown report creation
- **Session Management**: Multi-session workspace isolation

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -n deepanalyze python=3.12 -y
conda activate deepanalyze

# Install basic dependencies
pip install -r requirements.txt

# For training environments
(cd ./deepanalyze/ms-swift/ && pip install -e .)
(cd ./deepanalyze/SkyRL/ && pip install -e .)
```

### Running the Application

1. **Start vLLM Server** (required first):
```bash
vllm serve DeepAnalyze-8B
```

2. **Start Demo Application**:
```bash
cd demo
bash start.sh  # Starts both backend and frontend
# Or stop with:
bash stop.sh
```

3. **Manual Backend Start**:
```bash
python demo/backend.py  # Runs on ports 8200 (API) and 8100 (files)
```

4. **Manual Frontend Start**:
```bash
cd demo/chat
npm install
npm run dev  # Default port 4000
```

### Model Training

```bash
# Single-ability fine-tuning
bash scripts/single.sh

# Multi-ability agentic training (cold start)
bash scripts/multi_coldstart.sh

# Multi-ability agentic training (RL)
bash scripts/multi_rl.sh
```

### Code Usage Examples

**Basic Library Usage**:
```python
from deepanalyze import DeepAnalyzeVLLM

deepanalyze = DeepAnalyzeVLLM("DeepAnalyze-8B")
answer = deepanalyze.generate(prompt, workspace="/path/to/data")
print(answer["reasoning"])
```

**API Usage**:
```bash
curl -X POST http://localhost:8200/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [{"role": "user", "content": "Generate a data science report."}],
           "workspace": "example/student_loan/"
         }'
```

## Important File Locations

- **Main Library**: `deepanalyze.py`
- **Demo Backend**: `demo/backend.py`
- **Frontend**: `demo/chat/` (Next.js application)
- **Training Scripts**: `scripts/*.sh`
- **Requirements**: `requirements.txt`
- **Examples**: `.example/` directory

## Configuration

- **Model Path**: Update `MODEL_PATH` in `demo/backend.py` to point to your DeepAnalyze model
- **API Ports**: Backend uses 8200 for API, 8100 for file serving, vLLM uses 8000
- **Frontend Port**: Default 4000 (configurable via `FRONTEND_PORT` environment variable)

## Testing and Evaluation

Use the evaluation framework in `./playground` to test DeepAnalyze against various data science benchmarks. The framework supports vLLM-based evaluation with multiple benchmarks continuously being added.

## Development Notes

- The codebase supports both inference and training environments (recommend separating them)
- The demo application includes session-based workspace management
- Code execution is sandboxed with timeout protection for safety
- The system automatically organizes generated files into `generated/` subdirectories
- Multi-round reasoning continues until `<Answer>` tag is encountered or max rounds reached