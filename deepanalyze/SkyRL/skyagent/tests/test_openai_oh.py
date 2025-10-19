from skyagent import AutoAgentRunner
from skyagent.integrations.openai import OpenAIBackend
from transformers import AutoTokenizer
from datasets import load_dataset
import asyncio
import logging
import os

# logging.basicConfig(level=logging.DEBUG)
# OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "sc"
# model = "Qwen/Qwen3-32B"
model = "qwen/qwen2-0.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
# dataset_file = "/data/sycao/r2e/train.parquet"
dataset_file = "/mnt/shared_storage/datasets/r2e-1000/validation.parquet"
# read a few samples from the dataset
dataset = load_dataset("parquet", data_files=dataset_file)["train"].select(range(1, 2))

agent_generator = AutoAgentRunner.from_task(
    "./tests/test_openai_oh.yaml", infer_engine=None, tokenizer=tokenizer
)

output = asyncio.run(agent_generator.run(dataset))

print(output["rewards"])
print(output["rollout_metrics"])
