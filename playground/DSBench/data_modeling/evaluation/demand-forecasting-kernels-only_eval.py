import os.path

import numpy as np
import pandas as pd
import argparse


# 定义 SMAPE 计算函数
def smape(y_true, y_pred):
    return (
        100
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="sales")

args = parser.parse_args()


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = smape(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
