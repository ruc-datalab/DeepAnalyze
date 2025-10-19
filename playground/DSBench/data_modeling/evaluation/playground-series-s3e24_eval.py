import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="smoking")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

# 提取预测概率和实际标签
predicted_probabilities = predictions["smoking"].values
actual_labels = answers["smoking"].values  #

performance = roc_auc_score(actual_labels, predicted_probabilities)
with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
