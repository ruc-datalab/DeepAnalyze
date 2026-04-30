import os
import json
import re
import subprocess
import sys
from tqdm import tqdm
import time

data = []
with open("./data.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(data)

# model = 'gpt-4-turbo'
# model = 'gpt-4o-2024-05-13'
model = "gpt-3.5-turbo-0125"
# model = 'baseline'
# model = 'gpt-3.5-turbo-0125-autoagent'
# model = 'gpt-4o-2024-05-13-autoagent'
# model = 'llama3-autoagent'

gt_path = "./data/answers/"
# pred_path = gt_path
python_path = "./evaluation/"
pred_path = f"./output_model/{model}/"
save_path = f"./save_performance/{model}"

for line in data:
    name = line["name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid dataset name: {name!r}")
    # print(line['name'])
    answer_file = gt_path + name + "/test_answer.csv"
    pred_file = pred_path + name + ".csv"
    # pred_file = pred_path + line['name'] + '/test_answer.csv'
    # print(pred_file)
    if os.path.exists(pred_file):
        # print(pred_file)
        if not os.path.exists(os.path.join(save_path, name)):
            os.makedirs(os.path.join(save_path, name))
        print(f"compute performance for {name}")
        subprocess.run(
            [
                sys.executable,
                os.path.join(python_path, f"{name}_eval.py"),
                "--answer_file",
                answer_file,
                "--predict_file",
                pred_file,
                "--path",
                save_path,
                "--name",
                name,
            ],
            check=False,
        )
