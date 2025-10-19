import os.path

import numpy as np
import pandas as pd
import argparse
from scipy.stats import spearmanr

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="place_id")

args = parser.parse_args()

actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)


def mean_spearmanr(y_true, y_pred):
    """
    计算每列的Spearman's rank correlation coefficient，并取平均值
    """
    assert (
        y_true.shape == y_pred.shape
    ), "The shapes of true and predicted values do not match"
    correlations = []
    for col in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, col], y_pred[:, col])
        correlations.append(corr)
    return sum(correlations) / len(correlations)


# 提取实际标签和预测结果
actual_values = actual.iloc[
    :, 1:
].values  # 假设实际标签文件中第一列是qa_id，后面是实际标签值
predicted_values = submission.iloc[
    :, 1:
].values  # 假设提交文件中第一列是qa_id，后面是预测标签值
# 计算MAP@3
performance = mean_spearmanr(actual_values, predicted_values)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
