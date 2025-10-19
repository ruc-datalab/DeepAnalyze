import os
from skyagent import AutoAgentRunner
from pathlib import Path
from transformers import AutoTokenizer
import datasets
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import AsyncEngineArgs

os.environ["OPENAI_API_KEY"] = (
    "sc"  # dummy key, assumes an unath'ed vLLM service running locally
)
model = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
dataset = "/reasoning_data/train_filtered_80/math__combined_10.1k.parquet"
# read a few samples from the dataset
dataset = datasets.load_dataset("parquet", data_files=dataset)["train"].select(
    range(10)
)
print(dataset[0])

yaml_path = str(Path(__file__).parent / "openai_react.yaml")

agent_generator = AutoAgentRunner.from_task(
    yaml_path,
    # no explicit inference engine with OpenAI
    infer_engine=None,
    tokenizer=tokenizer,
)

output = asyncio.run(agent_generator.run(dataset))
print(output["rewards"])
