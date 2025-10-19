import os
import re
import json
from pathlib import Path
from multiprocessing import Pool
from deepanalyze import DeepAnalyzeVLLM


def collect_file_info(directory: str) -> str:
    """
    Collects file metadata from a given directory for supported file types.
    Returns a formatted string containing file info.
    """
    all_file_info_str = ""
    dir_path = Path(directory)
    allowed_suffixes = {".csv", ".xlsx", ".sqlite", ".md", ".json"}

    files = sorted(
        [
            f
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]
    )

    for idx, file_path in enumerate(files, start=1):
        size_bytes = file_path.stat().st_size
        size_kb = size_bytes / 1024
        file_info = {"name": file_path.name, "size": f"{size_kb:.1f}KB"}
        file_info_str = json.dumps(file_info, indent=4, ensure_ascii=False)
        all_file_info_str += f"File {idx}:\n{file_info_str}\n\n"

    return all_file_info_str.strip()


def extract_last_answer(text: str) -> str | None:
    """
    Extracts the content of the last <Answer> block from a text.
    Returns None if no <Answer> block is found.
    """
    last_pos = text.rfind("<Answer>")
    if last_pos == -1:
        return None

    remaining_text = text[last_pos:]
    match = re.search(r"<Answer>(.*?)</Answer>", remaining_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def run_single_task(
    task: dict, output_dir: Path, agent: DeepAnalyzeVLLM, context_files: str
):
    """
    Executes a single benchmark task using DeepAnalyzeVLLM.
    Saves the result to a JSONL file.
    """
    task_id = str(task["id"])
    output_file = output_dir / f"{task_id}.jsonl"

    if output_file.exists():
        return  # Skip already completed tasks

    prompt = f"# Instruction\n{task['question']}\n\n# Data\n{context_files}"

    for attempt in range(1):
        try:
            print(f"[Task {task_id}] Attempt {attempt + 1}: {task['question']}")
            answer = agent.generate(
                prompt, workspace=str(output_dir)  # Using output dir as workspace
            )

            reasoning = answer.get("reasoning", "./workspace")
            if not reasoning or "</Answer>" not in reasoning:
                raise ValueError("Incomplete reasoning or missing <Answer> block.")

            agent_answer = extract_last_answer(reasoning)

            result = {
                "task_id": task_id,
                "agent_answer": agent_answer,
                "reasoning_trace": reasoning,
                "question": task["question"],
                "checklist": task.get("checklist", ""),
                "checklist": task.get("checklist", ""),
                "type": task.get("type", ""),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=True) + "\n")

            print(f"[Task {task_id}] Success.")
            return

        except Exception as e:
            print(f"[Task {task_id}] Attempt {attempt + 1} failed: {e}")

    # All attempts failed → save error record
    error_result = {
        "task_id": task_id,
        "agent_answer": "error",
        "reasoning_trace": {},
        "question": task["question"],
        "checklist": task.get("checklist", ""),
        "type": task.get("type", ""),
        "prompt": prompt,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(error_result, ensure_ascii=True) + "\n")


def run_benchmark(
    task_list: list[dict],
    agent: DeepAnalyzeVLLM,
    output_dir: Path,
    context_files: str,
    num_processes: int = 4,
):
    """
    Executes multiple benchmark tasks in parallel using multiprocessing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_ids = {f.stem for f in output_dir.glob("*.jsonl")}
    incomplete_tasks = [t for t in task_list if str(t["id"]) not in completed_ids]

    print(
        f"Total tasks: {len(task_list)} | Completed: {len(completed_ids)} | Pending: {len(incomplete_tasks)}"
    )

    if not incomplete_tasks:
        print("All tasks are already completed.")
        return

    # Prepare arguments for parallel execution
    args = [(task, output_dir, agent, context_files) for task in incomplete_tasks]

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_single_task, args)


if __name__ == "__main__":
    # ===============================
    # Configuration
    # ===============================
    CONTEXT_DIR = "./context"
    JSONL_PATH = "dabstep_research.jsonl"
    OUTPUT_DIR = Path("./runs/deepanalyze")  # Generalized experiment folder
    NUM_PROCESSES = 4

    # ===============================
    # Collect file info for context
    # ===============================
    CONTEXT_FILES = collect_file_info(CONTEXT_DIR)

    # ===============================
    # Load tasks
    # ===============================
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]

    print(f"Loaded {len(tasks)} tasks from {JSONL_PATH}")

    # ===============================
    # Initialize agent
    # ===============================
    AGENT_MODEL_PATH = "/path/to/deepanalyze_model"
    agent = DeepAnalyzeVLLM(AGENT_MODEL_PATH)

    # ===============================
    # Run benchmark
    # ===============================
    run_benchmark(tasks, agent, OUTPUT_DIR, CONTEXT_FILES, num_processes=NUM_PROCESSES)
