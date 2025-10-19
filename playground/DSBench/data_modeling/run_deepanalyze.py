import os
import json
import time
import shutil
import signal
from contextlib import contextmanager
from multiprocessing import Pool
from deepanalyze import DeepAnalyzeVLLM

# Global configuration
WORKING_DIR = "Absolute path_to_DSBench/data_modeling/"
SAVE_PATH = "./output_model/"
MODEL = "DeepAnalyze-8B"
TASK_PROMPT = "Save the final results as 'submission.csv'."

# Initialize the agent
agent = DeepAnalyzeVLLM("path_to_DeepAnalyze-8B")


# Timeout context manager
@contextmanager
def timeout(seconds: int):
    """Raise TimeoutError if block execution exceeds `seconds`."""

    def signal_handler(signum, frame):
        raise TimeoutError("Operation timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def process_task(task: dict):
    """
    Process a single task using DeepAnalyzeVLLM.
    Saves both CSV output and JSON metadata.
    """
    name = task["name"]

    output_file_csv = os.path.join(SAVE_PATH, MODEL, f"{name}.csv")
    output_file_json = os.path.join(SAVE_PATH, MODEL, f"{name}.json")

    if os.path.exists(output_file_csv):
        print(f"[SKIP] {name} already processed.")
        return

    # Read task description
    description_file = os.path.join("./data/task/", f"{name}.txt")
    with open(description_file, "r", encoding="utf-8") as f:
        description = f.read()

    result = {}

    try:
        os.chdir(WORKING_DIR)
        with timeout(3600):
            result = agent.generate(
                f"{description}\n\n{TASK_PROMPT}", f"./data/data_resplit/{name}"
            )

        os.chdir(WORKING_DIR)
        src = f"./data/data_resplit/{name}/submission.csv"
        dst = output_file_csv

        # Ensure target directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Copy file
        shutil.copyfile(src, dst)
        os.remove(src)
        print(dst)
        success = True

    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        result = {}

    # Save JSON metadata
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "model": MODEL,
                "description": description,
                "result": result,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def main():
    # Ensure output directory exists
    os.makedirs(os.path.join(SAVE_PATH, MODEL), exist_ok=True)

    # Load tasks
    data_file = "./data.json"
    tasks = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    # Filter pending tasks
    pending_tasks = [
        t
        for t in tasks
        if not os.path.exists(os.path.join(SAVE_PATH, MODEL, f"{t['name']}.csv"))
    ]

    num_pending = len(pending_tasks)
    if num_pending == 0:
        print("All tasks have been processed.")
        return

    print(f"Processing {num_pending} pending tasks using {MODEL}...")

    # Use multiprocessing for parallel execution
    max_processes = min(num_pending, 2)
    with Pool(processes=max_processes) as pool:
        pool.map(process_task, pending_tasks)


if __name__ == "__main__":
    main()
