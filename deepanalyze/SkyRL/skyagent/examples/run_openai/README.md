# SkyAgent + OpenAI

You can run evaluation with any OpenAI-compatible server.


## Running evaluation with Sandbox Fusion

### Spin up a vllm server

You can spin up a vllm server with: 

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```

### Setup Sandbox Fusion

Make sure to set up sandbox fusion following instructions here: https://github.com/bytedance/SandboxFusion. You should create a `.env` file with the `SANDBOX_FUSION_URL` environment variable, as shown in the [.env.example](../../.env.example) file.

### Download the dataset

```bash
uv run huggingface-cli download NovaSky-AI/AIME-Repeated-8x-240 --repo-type dataset --local-dir <path_to_dataset>
```

make sure to update the path in the script to the local dataset path. 

### Run the evaluation

You can run the evaluation with:

```bash
uv run --isolated --directory . --env-file .env --frozen python run_openai_react.py
```

### Switching to a different model

At the moment, you need to change the model name in the yaml file and in the run script to grab the right tokenizer. We are actively working on making this experience better.