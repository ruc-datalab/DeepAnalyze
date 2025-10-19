import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score


# 计算多类对数损失
def multiclass_logloss(actuals, predictions):
    epsilon = 1e-15  # 避免对数运算中的数值问题
    predictions = np.clip(
        predictions, epsilon, 1 - epsilon
    )  # 限制预测概率的范围，防止对数为无穷
    predictions /= predictions.sum(axis=1)[:, np.newaxis]  # 归一化确保总和为1
    log_pred = np.log(predictions)
    loss = -np.sum(actuals * log_pred) / len(actuals)
    return loss


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="NObeyesdad")

args = parser.parse_args()


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = accuracy_score(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
