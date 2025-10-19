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
parser.add_argument("--value", type=str, default="place_id")

args = parser.parse_args()

actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)


def mcrmse(y_true, y_pred):
    """
    计算Mean Columnwise Root Mean Squared Error (MCRMSE)
    """
    assert (
        y_true.shape == y_pred.shape
    ), "The shapes of true and predicted values do not match"
    columnwise_rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    return columnwise_rmse.mean()


# 提取实际标签和预测结果
actual_values = actual.iloc[
    :, 1:
].values  # 假设实际标签文件中第一列是text_id，后面是实际标签值
predicted_values = submission.iloc[
    :, 1:
].values  # 假设提交文件中第一列是text_id，后面是预测标签值

# 计算MAP@3
performance = mcrmse(actual_values, predicted_values)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
