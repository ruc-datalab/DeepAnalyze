import os
import json
from tqdm.notebook import tqdm
import time

data = []
with open("./data.json", "r") as f:
    for line in f:
        data.append(eval(line))


# model = 'gpt-4-turbo'
# model = 'gpt-4o'
# model = 'gpt-3.5-turbo-0125'
# model = 'baseline'

model = "gpt-3.5-turbo-0125"
# model = 'gpt-4o-2024-05-13'
# model = 'gpt-3.5-turbo-0125-autoagent'
# model = 'gpt-4o-2024-05-13-autoagent'
# model = 'llama3-autoagent'

path = "./save_performance/"
baseline_path = f"{path}baseline"
save_path = f"{path}{model}"
gt_path = f"{path}GT"

output_path = f"./output_model/{model}"

task_complete = 0

scores = []
all_costs = []
all_times = []
for line in data:
    flag = False  ## whetehr bigger is better
    with open(os.path.join(gt_path, line["name"], "result.txt"), "r") as f:
        gt = eval(f.read().strip())
    with open(os.path.join(output_path, f"{line['name']}.json"), "r") as f:
        record = eval(f.read().strip())
    all_costs.append(record["cost"])
    all_times.append(record["time"])
    with open(os.path.join(baseline_path, line["name"], "result.txt"), "r") as f:
        bl = eval(f.read().strip())
    if gt > bl:
        flag = True
    # print(f"{line['name']} gt {gt} baseline {bl}")
    # print(line['name'])

    if not os.path.exists(os.path.join(save_path, line["name"], "result.txt")):
        scores.append(0)
        show_pre = "not exists"
    else:
        task_complete += 1
        with open(os.path.join(save_path, line["name"], "result.txt"), "r") as f:
            pre = f.read().strip()
        if pre == "nan":
            show_pre = "nan"
            scores.append(0)
        else:
            pre = eval(pre)
            sc = max(0, (pre - bl) / (gt - bl))
            scores.append(sc)
            show_pre = pre

            # print(f"For the challenge {line['name']}, the performance of the agent {pre}, the performance of GT is {gt}. {sc}")
# print(scores)
print(f"Task completion rate is {task_complete/len(scores)}")
print(f"All the cost is {sum(all_costs)}")
print(f"The total time consuming is {sum(all_times)}")
print(f"The performance is {sum(scores)/len(scores)}")
