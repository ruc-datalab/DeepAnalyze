import os
import re
import io
import sys
import json
import time
import argparse
import traceback
import contextlib
from multiprocessing.pool import Pool
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI


client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="xxxxx",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate data science reports via OpenAI API"
    )
    parser.add_argument("--input_path", required=True, help="Path to input JSONL file.")
    parser.add_argument(
        "--output_path", required=True, help="Directory to save evaluated reports."
    )
    parser.add_argument(
        "--model_name",
        default="ep-20250928102838-9fj8x",
        help="Model name or endpoint ID.",
    )
    parser.add_argument(
        "--num_tasks", default=1, type=int, help="Number of parallel tasks."
    )
    return parser.parse_args()


PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction. Your task is to comprehensively evaluate the quality of the generated data science report, based on the provided user instruction [INSTRUCTION], a checklist offering reference points for an ideal report [CHECKLIST], and the generated report [REPORT].

Evaluate across two dimensions (1–5 scale):

- **Content**: Relevance, comprehensiveness, and insightfulness.
- **Format**: Structure, readability, and professionalism.

### [INSTRUCTION]:
{instruction}

### [CHECKLIST]:
{checklist}

### [REPORT]:
{report}

Return your evaluation strictly as JSON:
```json
{{
"Content": <score>,
"Format": <score>
}}
```"""


def evaluate_reports(model_name, conversation_set, output_dir):
    for conv in tqdm(conversation_set, desc="Evaluating reports"):
        task_id = conv.get("task_id", f"task_{int(time.time())}")
        for attempt in range(4):
            try:
                prompt = PROMPT.format(
                    instruction=conv.get("question", ""),
                    checklist=conv.get("checklist", ""),
                    report=conv.get("agent_answer", ""),
                )

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4000,
                )

                reply = completion.choices[0].message.content.strip()
                match = re.search(r"```(?:json)?(.*?)```", reply, re.DOTALL)
                score_str = match.group(1).strip() if match else reply
                score_dict = json.loads(score_str)

                conv["evaluation"] = reply
                conv["score"] = score_dict
                conv["avg_score"] = (
                    float(score_dict["Content"]) + float(score_dict["Format"])
                ) / 2

                output_file = os.path.join(output_dir, f"{task_id}.jsonl")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

                break  # success, exit retry loop

            except Exception as e:
                print(f"[Retry {attempt+1}/4] Error in '{task_id}': {e}")
                time.sleep(1)
        else:
            print(f"Failed to process task: {task_id} after 4 retries.")


def main():
    args = parse_args()

    # Prepare output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load input JSONL
    with open(args.input_path, "r", encoding="utf-8") as file:
        conversations = [json.loads(line) for line in file if line.strip()]

    # Skip already processed files
    completed = set(os.listdir(args.output_path))
    pending = [
        conv for conv in conversations if f"{conv['task_id']}.jsonl" not in completed
    ]

    if not pending:
        print("All tasks already processed.")
        return

    # Split workload and launch multiprocessing
    part_len = max(1, len(pending) // args.num_tasks)
    parts = [pending[i : i + part_len] for i in range(0, len(pending), part_len)]
    task_args = [(args.model_name, part, args.output_path) for part in parts]

    with Pool(processes=args.num_tasks) as pool:
        pool.starmap(evaluate_reports, task_args)

    print("All reports have been evaluated and saved successfully.")


if __name__ == "__main__":
    main()
