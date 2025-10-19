from skyagent import AutoAgentRunner
from skyagent.integrations.openai import OpenAIBackend
from transformers import AutoTokenizer
import datasets
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import AsyncEngineArgs
import logging

logging.basicConfig(level=logging.DEBUG)

# model = "NovaSky-AI/SWE-Gym-OpenHands-7B-Agent"
model = "qwen/qwen2-0.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
dataset = "/mnt/shared_storage/datasets/filtered_sft/train.parquet"
# read a few samples from the dataset
dataset = datasets.load_dataset("parquet", data_files=dataset)["train"].select(range(2))
print(dataset[0])


backend = OpenAIBackend(
    infer_engine=None, cfg={"model_name": model, "api_url": "http://localhost:6002"}
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, how are you?"}],
    tokenize=True,
    add_generation_prompt=True,
)
output = asyncio.run(
    backend.async_generate_ids(
        prompt, sampling_params={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}
    )
)
print(output)
