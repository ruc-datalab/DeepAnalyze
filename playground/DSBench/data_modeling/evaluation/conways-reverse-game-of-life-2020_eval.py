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

parser.add_argument("--value", type=str, default="generated")

args = parser.parse_args()

actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

# 移除id列，剩下的是矩阵的值
submission_values = submission.drop(columns=["id"]).values
actual_values = actual.drop(columns=["id"]).values

# 计算平均绝对误差
performance = np.mean(np.abs(submission_values - actual_values))


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
