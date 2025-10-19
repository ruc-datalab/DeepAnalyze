import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="FloodProbability")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# 提取预测值和实际值
y_pred_views = predictions["num_views"]
y_true_views = answers["num_views"]
y_pred_votes = predictions["num_votes"]
y_true_votes = answers["num_votes"]
y_pred_comments = predictions["num_comments"]
y_true_comments = answers["num_comments"]

# 计算RMSLE
rmsle_views = rmsle(y_true_views, y_pred_views)
rmsle_votes = rmsle(y_true_votes, y_pred_votes)
rmsle_comments = rmsle(y_true_comments, y_pred_comments)


average_rmsle = (rmsle_views + rmsle_votes + rmsle_comments) / 3

performance = average_rmsle

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
