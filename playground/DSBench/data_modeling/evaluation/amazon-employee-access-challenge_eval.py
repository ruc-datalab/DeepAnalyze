import os.path

import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="ACTION")

args = parser.parse_args()


# 定义 RMSLE 计算函数
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

performance = roc_auc_score(actual[args.value], submission[args.value])


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
