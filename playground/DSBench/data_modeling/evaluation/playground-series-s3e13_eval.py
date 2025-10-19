import os.path

import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", default="", type=str, required=False)
parser.add_argument("--name", default="", type=str, required=False)
parser.add_argument(
    "--answer_file",
    default="/Users/tencentintern/PycharmProjects/DSBench/kaggle_data/data_filted_csv/answers/playground-series-s3e13/test_answer.csv",
    type=str,
    required=False,
)
parser.add_argument(
    "--predict_file",
    default="/Users/tencentintern/PycharmProjects/DSBench/kaggle_data/data_filted_csv/answers/playground-series-s3e13/test_answer.csv",
    type=str,
    required=False,
)

parser.add_argument("--value", type=str, default="prognosis")

args = parser.parse_args()


actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

actual.sort_values(by=["id"])
submission.sort_values(by=["id"])


def mpa_at_3(actual, predictions):
    """
    Calculate Mean Percentage Agreement at 3 (MPA@3).

    Parameters:
    actual (list): List of actual prognosis values.
    predictions (list of lists): List of lists containing up to 3 predicted prognosis values.

    Returns:
    float: The MPA@3 score.
    """
    total = len(actual)
    score = 0.0

    for act, preds in zip(actual, predictions):
        preds = preds.split()
        if act in preds[:3]:
            score += 1

    return score / total


# 计算平均错误率
performance = mpa_at_3(actual[args.value], submission[args.value])
print(performance)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
