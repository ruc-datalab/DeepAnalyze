import os
import json
from tqdm import tqdm
import time

data = []
with open("./data.json", "r") as f:
    for line in f:
        data.append(eval(line))

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
    # print(line['name'])
    answer_file = gt_path + line["name"] + "/test_answer.csv"
    pred_file = pred_path + line["name"] + ".csv"
    # pred_file = pred_path + line['name'] + '/test_answer.csv'
    # print(pred_file)
    if os.path.exists(pred_file):
        # print(pred_file)
        if not os.path.exists(os.path.join(save_path, line["name"])):
            os.makedirs(os.path.join(save_path, line["name"]))
        print(f"compute performance for {line['name']}")
        os.system(
            f"python {python_path}{line['name']}_eval.py --answer_file {answer_file} --predict_file {pred_file} --path {save_path} --name {line['name']}"
        )
