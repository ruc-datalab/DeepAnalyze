import os.path

import numpy as np
import pandas as pd
import argparse


# 计算RMSLE
def rmsle(predicted, actual):
    sum_log_diff = np.sum((np.log(predicted + 1) - np.log(actual + 1)) ** 2)
    mean_log_diff = sum_log_diff / len(predicted)
    return np.sqrt(mean_log_diff)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="count")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = rmsle(predictions[args.value], answers[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
