import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import log_loss

from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="score")

args = parser.parse_args()


# Compute MAE
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


actual = pd.read_csv(args.answer_file)
submission = pd.read_csv(args.predict_file)

# 提取实际值和预测值
actual_values = actual[["winner_model_a", "winner_model_b", "winner_tie"]].values
predicted_values = submission[["winner_model_a", "winner_model_b", "winner_tie"]].values


performance = log_loss(actual_values, predicted_values)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
