# SkyAgent

SkyAgent is a generic agent layer for training and evaluating agents.

SkyAgent is designed primarly for researchers to have a unified interface around implementing agentic tasks. A modular design allows researchers to 
1. bring in their own tasks
2. use any training backend or simply run evaluation
3. modify runtime implementation for a given task
4. improve dispatching logic for a batch of trajectories
5. and more ...

SkyAgent is still under development. We are actively working on expanding available tasks and runtime implementations.


## Getting Started


The first step is the clone the repository. `skyagent` is it's own subpackage in the SkyRL repository.

```bash
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL.git 
# our working directory
cd skyagent
```

### Installation

We use [uv](https://docs.astral.sh/uv/) to manage the dependencies.

```bash
uv venv
uv sync
```

### Running evaluation

We support running evaluation with any OpenAI-compatible server, 

For example: 

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```
You also need set up the sandbox fusion following instructions here: https://github.com/bytedance/SandboxFusion.

```bash
uv run --isolated --directory . --env-file .env --frozen python ./tests/react_task_tests/test.py --yaml tests/react_task_tests/react_interpreter.yaml --dataset NovaSky-AI/AIME-Repeated-8x-240 --split test
```