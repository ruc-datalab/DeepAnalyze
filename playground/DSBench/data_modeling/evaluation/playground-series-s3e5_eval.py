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

parser.add_argument("--value", type=str, default="quality")

args = parser.parse_args()


actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

actual.sort_values(by=["Id"])
submission.sort_values(by=["Id"])


def quadratic_weighted_kappa(actual, predicted, N):
    O = np.zeros((N, N), dtype=int)
    for a, p in zip(actual, predicted):
        O[a][p] += 1

    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w[i][j] = ((i - j) ** 2) / ((N - 1) ** 2)

    actual_hist = np.zeros(N)
    for a in actual:
        actual_hist[a] += 1

    pred_hist = np.zeros(N)
    for p in predicted:
        pred_hist[p] += 1

    E = np.outer(actual_hist, pred_hist)
    E = E / E.sum() * O.sum()

    num = (w * O).sum()
    den = (w * E).sum()

    return 1 - num / den


# 计算平均错误率
performance = quadratic_weighted_kappa(actual[args.value], submission[args.value], 10)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
